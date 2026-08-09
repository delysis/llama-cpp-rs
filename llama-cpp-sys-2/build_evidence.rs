#![cfg_attr(test, allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, Metadata};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest as _, Sha256};

pub(crate) const EVIDENCE_FORMAT: &str = "llama.native-build-evidence.v1";
const MAX_ARTIFACTS: usize = 64;
const MAX_LOGICAL_NAME_BYTES: usize = 96;
const HASH_BUFFER_BYTES: usize = 1024 * 1024;

const KNOWN_FEATURES: &[(&str, &str)] = &[
    ("CARGO_FEATURE_COMMON", "common"),
    ("CARGO_FEATURE_CUDA", "cuda"),
    ("CARGO_FEATURE_CUDA_NO_VMM", "cuda-no-vmm"),
    ("CARGO_FEATURE_DEFAULT", "default"),
    ("CARGO_FEATURE_DYNAMIC_BACKENDS", "dynamic-backends"),
    ("CARGO_FEATURE_DYNAMIC_LINK", "dynamic-link"),
    ("CARGO_FEATURE_METAL", "metal"),
    ("CARGO_FEATURE_MKL", "mkl"),
    ("CARGO_FEATURE_MTMD", "mtmd"),
    ("CARGO_FEATURE_OPENCL", "opencl"),
    ("CARGO_FEATURE_OPENMP", "openmp"),
    ("CARGO_FEATURE_ROCM", "rocm"),
    ("CARGO_FEATURE_SHARED_STDCXX", "shared-stdcxx"),
    ("CARGO_FEATURE_STATIC_OPENMP", "static-openmp"),
    ("CARGO_FEATURE_STATIC_STDCXX", "static-stdcxx"),
    ("CARGO_FEATURE_SYSTEM_GGML", "system-ggml"),
    ("CARGO_FEATURE_SYSTEM_GGML_STATIC", "system-ggml-static"),
    ("CARGO_FEATURE_VULKAN", "vulkan"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeLinkage {
    Static,
    Shared,
}

impl NativeLinkage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Shared => "shared",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GgmlOrigin {
    Vendored,
    System,
}

impl GgmlOrigin {
    fn as_str(self) -> &'static str {
        match self {
            Self::Vendored => "vendored",
            Self::System => "system",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendLinkage {
    Linked,
    DynamicModules,
}

impl BackendLinkage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Linked => "linked",
            Self::DynamicModules => "dynamic-modules",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LinkageEvidence {
    pub(crate) local: NativeLinkage,
    pub(crate) ggml_origin: GgmlOrigin,
    pub(crate) ggml: NativeLinkage,
    pub(crate) backends: BackendLinkage,
}

impl LinkageEvidence {
    fn canonical(self) -> String {
        format!(
            "local={};ggml-origin={};ggml={};backends={}",
            self.local.as_str(),
            self.ggml_origin.as_str(),
            self.ggml.as_str(),
            self.backends.as_str()
        )
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ArtifactInput {
    pub(crate) logical_name: String,
    pub(crate) path: PathBuf,
    pub(crate) external: bool,
}

impl ArtifactInput {
    pub(crate) fn produced(logical_name: impl Into<String>, path: PathBuf) -> Self {
        Self {
            logical_name: logical_name.into(),
            path,
            external: false,
        }
    }

    pub(crate) fn selected_external(logical_name: impl Into<String>, path: PathBuf) -> Self {
        Self {
            logical_name: logical_name.into(),
            path,
            external: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArtifactEvidence {
    logical_name: String,
    byte_len: u64,
    sha256: String,
}

impl ArtifactEvidence {
    fn canonical(&self) -> String {
        format!("{}|{}|{}", self.logical_name, self.byte_len, self.sha256)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct BuildEvidence {
    features: Vec<String>,
    linkage: LinkageEvidence,
    artifacts: Vec<ArtifactEvidence>,
    digest: String,
}

impl BuildEvidence {
    pub(crate) fn collect(
        features: Vec<String>,
        linkage: LinkageEvidence,
        artifacts: Vec<ArtifactInput>,
    ) -> Result<Self, String> {
        if artifacts.is_empty() {
            return Err("native build evidence has no artifacts".to_owned());
        }
        if artifacts.len() > MAX_ARTIFACTS {
            return Err(format!(
                "native build evidence has {} artifacts; maximum is {MAX_ARTIFACTS}",
                artifacts.len()
            ));
        }

        validate_features(&features)?;
        let mut by_name = BTreeMap::new();
        for input in artifacts {
            validate_logical_name(&input.logical_name)?;
            if input.external {
                // An external artifact can change without any vendored source changing.
                // This is intentionally a Cargo rerun directive, not links metadata.
                println!("cargo:rerun-if-changed={}", input.path.display());
            }
            let (byte_len, sha256) = hash_stable_file(&input.path).map_err(|error| {
                format!(
                    "cannot fingerprint required native artifact {}: {error}",
                    input.logical_name
                )
            })?;
            let evidence = ArtifactEvidence {
                logical_name: input.logical_name.clone(),
                byte_len,
                sha256,
            };
            if by_name
                .insert(input.logical_name.clone(), evidence)
                .is_some()
            {
                return Err(format!(
                    "duplicate native artifact logical name: {}",
                    input.logical_name
                ));
            }
        }
        let artifacts: Vec<_> = by_name.into_values().collect();
        let digest = evidence_digest(&features, linkage, &artifacts);
        Ok(Self {
            features,
            linkage,
            artifacts,
            digest,
        })
    }

    pub(crate) fn emit_cargo_metadata(&self) {
        for (key, value) in self.metadata_entries() {
            println!("cargo:{key}={value}");
        }
    }

    fn metadata_entries(&self) -> Vec<(String, String)> {
        let mut entries = vec![
            (
                "build_evidence_format".to_owned(),
                EVIDENCE_FORMAT.to_owned(),
            ),
            (
                "build_evidence_features".to_owned(),
                self.features.join(","),
            ),
            (
                "build_evidence_linkage".to_owned(),
                self.linkage.canonical(),
            ),
            (
                "build_evidence_artifact_count".to_owned(),
                self.artifacts.len().to_string(),
            ),
        ];
        entries.extend(self.artifacts.iter().enumerate().map(|(index, artifact)| {
            (
                format!("build_evidence_artifact_{index:02}"),
                artifact.canonical(),
            )
        }));
        entries.push(("build_evidence_sha256".to_owned(), self.digest.clone()));
        entries
    }
}

pub(crate) fn effective_features() -> Result<Vec<String>, String> {
    let known: BTreeMap<_, _> = KNOWN_FEATURES.iter().copied().collect();
    let active_env: BTreeSet<_> = std::env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .filter(|key| key.starts_with("CARGO_FEATURE_"))
        .collect();

    let unknown: Vec<_> = active_env
        .iter()
        .filter(|key| !known.contains_key(key.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "unrecognized active Cargo features in native build evidence: {}",
            unknown.join(",")
        ));
    }

    let mut features: Vec<_> = active_env
        .iter()
        .filter_map(|key| known.get(key.as_str()).copied())
        .map(str::to_owned)
        .collect();
    features.sort_unstable();
    Ok(features)
}

pub(crate) fn system_ggml_artifacts(
    cmake_cache: &Path,
    require_cpu: bool,
) -> Result<Vec<ArtifactInput>, String> {
    let contents = std::fs::read_to_string(cmake_cache)
        .map_err(|error| format!("cannot read system GGML CMake cache: {error}"))?;
    let mut found = BTreeMap::new();
    for line in contents.lines() {
        let Some((declaration, value)) = line.split_once('=') else {
            continue;
        };
        let Some((key, kind)) = declaration.split_once(':') else {
            continue;
        };
        if kind != "FILEPATH" || !is_system_ggml_library_key(key) {
            continue;
        }
        if value.is_empty() || value.ends_with("-NOTFOUND") {
            return Err(format!("required system GGML artifact {key} was not found"));
        }
        let logical_key = key
            .trim_end_matches("_LIBRARY")
            .to_ascii_lowercase()
            .replace('_', "-");
        let logical_name = format!("system-ggml/{logical_key}/link");
        if found
            .insert(
                logical_name.clone(),
                ArtifactInput::selected_external(logical_name.clone(), PathBuf::from(value)),
            )
            .is_some()
        {
            return Err(format!("duplicate system GGML artifact {logical_name}"));
        }
    }

    let mut required = vec!["system-ggml/ggml/link", "system-ggml/ggml-base/link"];
    if require_cpu {
        required.push("system-ggml/ggml-cpu/link");
    }
    for required in required {
        if !found.contains_key(required) {
            return Err(format!(
                "system GGML CMake cache did not identify required artifact {required}"
            ));
        }
    }
    Ok(found.into_values().collect())
}

fn is_system_ggml_library_key(key: &str) -> bool {
    key == "GGML_LIBRARY" || (key.starts_with("GGML_") && key.ends_with("_LIBRARY"))
}

fn validate_features(features: &[String]) -> Result<(), String> {
    let known: BTreeSet<_> = KNOWN_FEATURES.iter().map(|(_, feature)| *feature).collect();
    let mut previous: Option<&str> = None;
    for feature in features {
        if !known.contains(feature.as_str()) {
            return Err(format!("unknown native build feature: {feature}"));
        }
        if previous.is_some_and(|value| value >= feature.as_str()) {
            return Err("native build features are not strictly sorted and unique".to_owned());
        }
        previous = Some(feature);
    }
    Ok(())
}

fn validate_logical_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_LOGICAL_NAME_BYTES {
        return Err(format!(
            "invalid native artifact logical name length: {name:?}"
        ));
    }
    if !name.bytes().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'-' | b'_' | b'/' | b'.')
    }) {
        return Err(format!(
            "native artifact logical name is not canonical ASCII: {name:?}"
        ));
    }
    if name
        .split('/')
        .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(format!(
            "native artifact logical name has an invalid component: {name:?}"
        ));
    }
    Ok(())
}

fn hash_stable_file(path: &Path) -> io::Result<(u64, String)> {
    let mut file = File::open(path)?;
    let before = file.metadata()?;
    if !before.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "artifact is not a regular file",
        ));
    }
    let first = hash_open_file(&mut file)?;
    file.seek(SeekFrom::Start(0))?;
    let second = hash_open_file(&mut file)?;
    let after = file.metadata()?;
    if !same_file_snapshot(&before, &after) || first != second {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "artifact changed while it was being fingerprinted",
        ));
    }
    Ok((after.len(), first))
}

fn hash_open_file(file: &mut File) -> io::Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn same_file_snapshot(before: &Metadata, after: &Metadata) -> bool {
    before.len() == after.len()
        && modified_time(before) == modified_time(after)
        && platform_file_identity(before) == platform_file_identity(after)
}

fn modified_time(metadata: &Metadata) -> Option<SystemTime> {
    metadata.modified().ok()
}

#[cfg(unix)]
fn platform_file_identity(metadata: &Metadata) -> (u64, u64, i64, i64) {
    use std::os::unix::fs::MetadataExt as _;

    (
        metadata.dev(),
        metadata.ino(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    )
}

#[cfg(not(unix))]
fn platform_file_identity(_metadata: &Metadata) {}

fn evidence_digest(
    features: &[String],
    linkage: LinkageEvidence,
    artifacts: &[ArtifactEvidence],
) -> String {
    let mut digest = Sha256::new();
    digest.update(EVIDENCE_FORMAT.as_bytes());
    digest.update([0]);
    for feature in features {
        digest.update(feature.as_bytes());
        digest.update([0]);
    }
    digest.update([0]);
    digest.update(linkage.canonical().as_bytes());
    digest.update([0]);
    for artifact in artifacts {
        digest.update(artifact.canonical().as_bytes());
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn temporary_file(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llama-build-evidence-{}-{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("artifact.a");
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        path
    }

    #[test]
    fn evidence_is_sorted_and_content_bound() {
        let first = temporary_file("sorted-a", b"alpha");
        let second = temporary_file("sorted-b", b"beta");
        let linkage = LinkageEvidence {
            local: NativeLinkage::Static,
            ggml_origin: GgmlOrigin::Vendored,
            ggml: NativeLinkage::Static,
            backends: BackendLinkage::Linked,
        };
        let evidence = BuildEvidence::collect(
            vec!["common".to_owned(), "metal".to_owned()],
            linkage,
            vec![
                ArtifactInput::produced("llama/z/link", second),
                ArtifactInput::produced("ggml/a/link", first),
            ],
        )
        .unwrap();

        assert_eq!(evidence.artifacts[0].logical_name, "ggml/a/link");
        assert_eq!(evidence.artifacts[0].byte_len, 5);
        assert_eq!(
            evidence.artifacts[0].sha256,
            "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"
        );
        assert_eq!(evidence.digest.len(), 64);
    }

    #[test]
    fn duplicate_logical_names_fail_closed() {
        let first = temporary_file("duplicate-a", b"alpha");
        let second = temporary_file("duplicate-b", b"beta");
        let error = BuildEvidence::collect(
            vec![],
            LinkageEvidence {
                local: NativeLinkage::Shared,
                ggml_origin: GgmlOrigin::System,
                ggml: NativeLinkage::Shared,
                backends: BackendLinkage::DynamicModules,
            },
            vec![
                ArtifactInput::produced("llama/core/link", first),
                ArtifactInput::produced("llama/core/link", second),
            ],
        )
        .unwrap_err();
        assert!(error.contains("duplicate"));
    }

    #[test]
    fn system_cache_collects_every_ggml_library() {
        let path = temporary_file(
            "system-cache",
            b"GGML_LIBRARY:FILEPATH=/opt/lib/libggml.so\n\
              GGML_BASE_LIBRARY:FILEPATH=/opt/lib/libggml-base.so\n\
              GGML_CPU_LIBRARY:FILEPATH=/opt/lib/libggml-cpu.so\n\
              OTHER_LIBRARY:FILEPATH=/secret/nope.so\n",
        );
        let artifacts = system_ggml_artifacts(&path, true).unwrap();
        let names: Vec<_> = artifacts
            .iter()
            .map(|artifact| artifact.logical_name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "system-ggml/ggml-base/link",
                "system-ggml/ggml-cpu/link",
                "system-ggml/ggml/link",
            ]
        );
        assert!(artifacts.iter().all(|artifact| artifact.external));
    }

    #[test]
    fn malformed_names_and_feature_order_fail_closed() {
        assert!(validate_logical_name("../../private").is_err());
        assert!(validate_logical_name("ggml/CPU/link").is_err());
        assert!(validate_features(&["metal".to_owned(), "common".to_owned()]).is_err());
        assert!(validate_features(&["future-feature".to_owned()]).is_err());
    }

    #[test]
    fn links_metadata_never_contains_source_paths() {
        let sentinel = "private-path-must-not-escape";
        let artifact = temporary_file(sentinel, b"content");
        assert!(artifact.to_string_lossy().contains(sentinel));
        let evidence = BuildEvidence::collect(
            vec![],
            LinkageEvidence {
                local: NativeLinkage::Static,
                ggml_origin: GgmlOrigin::Vendored,
                ggml: NativeLinkage::Static,
                backends: BackendLinkage::Linked,
            },
            vec![ArtifactInput::produced("llama/core/link", artifact)],
        )
        .unwrap();
        let serialized = evidence
            .metadata_entries()
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!serialized.contains(sentinel));
        assert!(!serialized.contains(std::env::temp_dir().to_string_lossy().as_ref()));
    }
}
