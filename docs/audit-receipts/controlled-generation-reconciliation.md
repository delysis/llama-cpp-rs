# Controlled-generation reconciliation receipt

## Identity and scope

- Repository: `delysis/llama-cpp-rs`
- Branch: `codex/reconcile-native-controls`
- Canonical base: `01e48b7c1e7de39c3e5e8a67cd9efac498f8da1f`
- Compared head: `a74dbb79f96e0ebad8b0737ee1d3c9c1deb185af`
- llama.cpp submodule retained: `5f55650a78f92aff4d48d671423e888fac0469ff`
- Status: successor candidate pending pull-request review

The three stale-branch commits are enumerated and classified exactly once in
`docs/reconciliation/a74db-to-01e48.md`. A checked-in shell gate rejects a
missing, duplicate, or unclassified row. No stale commit was merged or
cherry-picked, and no production hunk was needed: the canonical base already
contains the pooling fix and LoRA ownership contract, while `01e48b7`
supersedes the older controlled-wrapper implementation with stronger
validation and deterministic build evidence.

## Local gates

The first feature command initially failed because this isolated worktree's
submodule was uninitialized and `llama.cpp/include/llama.h` was absent. After
checking out the exact recorded submodule SHA, the following passed:

```text
./scripts/check-reconciliation-ledger.sh
cargo test -p llama-cpp-2 --no-default-features
  82 passed, 3 ignored doctests/examples requiring runtime setup
cargo test -p llama-cpp-2 --no-default-features --features sampler
  82 passed, 3 ignored
cargo test -p llama-cpp-2 --no-default-features --features common,sampler,mtmd
  91 passed, 3 ignored
cargo test -p llama-cpp-sys-2 --test build-evidence
  6 passed
cargo test --doc -p llama-cpp-2
  82 passed, 3 ignored
cargo fmt --all -- --check
git diff --check
```

The packet's spelling `cargo test -p llama-cpp-sys-2 build_evidence` runs zero
tests in this repository because the integration target is named
`build-evidence`; the corrected command above is the evidence-bearing gate.

## Open promotion blocker

Strict Clippy is not green on the canonical base:

```text
cargo clippy -p llama-cpp-2 --all-targets --features common,sampler,mtmd -- -D warnings
```

It fails first in `llama-cpp-2/build.rs` on missing crate documentation and an
uninlined format argument. A temporary local correction exposed substantial
additional pre-existing pedantic/documentation lint debt, so it was reverted
rather than mixing a broad stylistic rewrite into semantic reconciliation. A
dedicated lint/CI hardening change must make the selected feature matrix
strict-green before promotion.

No real LoRA adapter artifact was available for a load/activate/drop runtime
test. The Rustdoc compile-fail lifetime proof and non-dropping-handle unit test
passed; runtime LoRA activation remains an explicit platform evidence gate.
