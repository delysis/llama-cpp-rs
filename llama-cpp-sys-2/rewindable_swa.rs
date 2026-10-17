//! Reviewed overlay for upstream 5f55650a78f92aff4d48d671423e888fac0469ff.
//! Keep dependency checkouts immutable; compile an owned copy in OUT_DIR.
use std::{
    fs,
    path::{Path, PathBuf},
};

fn replace_once(path: &Path, before: &str, after: &str) {
    let source = fs::read_to_string(path).expect("read pinned native source");
    assert_eq!(
        source.matches(before).count(),
        1,
        "native SWA patch drift: {}",
        path.display()
    );
    fs::write(path, source.replacen(before, after, 1)).expect("write native SWA overlay");
}

pub fn prepare(manifest: &Path, out: &Path) -> PathBuf {
    let root = out.join("rewindable-native-source");
    // This directory is owned exclusively by this build script, never user data.
    if root.exists() {
        fs::remove_dir_all(&root).expect("replace stale native overlay");
    }
    let source = manifest.join("llama.cpp");
    for entry in walkdir::WalkDir::new(&source)
        .into_iter()
        .filter_entry(|entry| entry.file_name() != ".git")
    {
        let entry = entry.expect("enumerate native source");
        let destination = root
            .join("llama.cpp")
            .join(entry.path().strip_prefix(&source).unwrap());
        if entry.file_type().is_dir() {
            fs::create_dir_all(destination).expect("create native overlay");
        } else if entry.file_type().is_file() {
            fs::copy(entry.path(), destination).expect("copy native source");
        } else {
            panic!(
                "unsupported native source entry: {}",
                entry.path().display()
            );
        }
    }
    for wrapper in [
        "wrapper.h",
        "wrapper_common.h",
        "wrapper_common.cpp",
        "wrapper_utils.h",
        "wrapper_mtmd.h",
    ] {
        fs::copy(manifest.join(wrapper), root.join(wrapper)).expect("copy native wrapper");
    }
    // Upstream GGML unconditionally discovers the enclosing Git repository.
    // Honor the build script's pinned source revision instead; the overlay has
    // no .git directory and lives inside an unrelated consuming repository.
    let ggml_cmake = root.join("llama.cpp/ggml/CMakeLists.txt");
    replace_once(
        &ggml_cmake,
        "find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)\nif(GIT_EXE)",
        "if(NOT DEFINED GGML_BUILD_COMMIT)\nfind_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)\nif(GIT_EXE)",
    );
    replace_once(
        &ggml_cmake,
        "endif()\n\nset(GGML_VERSION \"${GGML_VERSION_BASE}\")",
        "endif()\nendif()\n\nset(GGML_VERSION \"${GGML_VERSION_BASE}\")",
    );
    replace_once(&root.join("llama.cpp/include/llama.h"),
        "#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2",
        "#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2\n\n// Retain all resident SWA cells so a saved sequence can be cropped to an earlier prefix.\n// Requires a full-size SWA context to retain the complete history.\n#define LLAMA_STATE_SEQ_FLAGS_KEEP_ALL_SWA_CELLS 4");
    replace_once(&root.join("llama.cpp/src/llama-kv-cache.cpp"),
        "if (add_cell && seq_id != -1) {\n                const bool is_masked",
        "if (add_cell && seq_id != -1 && !(flags & LLAMA_STATE_SEQ_FLAGS_KEEP_ALL_SWA_CELLS)) {\n                const bool is_masked");
    root
}

/// Bindgen includes generated headers; watch their pristine inputs instead.
#[derive(Debug)]
pub struct SourceCallbacks {
    pub overlay: PathBuf,
    pub manifest: PathBuf,
}
impl bindgen::callbacks::ParseCallbacks for SourceCallbacks {
    fn header_file(&self, filename: &str) {
        self.include_file(filename);
    }
    fn include_file(&self, filename: &str) {
        let path = fs::canonicalize(filename).unwrap_or_else(|_| PathBuf::from(filename));
        let overlay = fs::canonicalize(&self.overlay).expect("canonical native overlay");
        let source = path
            .strip_prefix(&overlay)
            .map(|relative| self.manifest.join(relative))
            .unwrap_or_else(|_| path.to_path_buf());
        println!("cargo:rerun-if-changed={}", source.display());
    }
    fn read_env_var(&self, key: &str) {
        println!("cargo:rerun-if-env-changed={key}");
    }
}
