# Workflow hardening receipt

## Scope

- Every third-party action reference in `.github/workflows` is an exact
  40-hex revision.
- The Rust jobs use the reviewed `dtolnay/rust-toolchain` snapshot with
  toolchain `1.88.0`; no MSRV is declared by this repository, so this matches
  the native consumer stack without claiming a lower compatibility floor.
- Repository-owned policy and fixture checks run before the main Linux job.
- The semantic reconciliation ledger is checked in CI.
- Existing platform separation is retained: Linux checks/tests, macOS native
  build, Windows native build/test, and Docker/QEMU architecture builds.

No vendored source, llama.cpp submodule revision, crate dependency, or runtime
behavior changes in this commit.

## Local evidence

```text
./scripts/check-workflow-policy.sh
  pass
./tests/workflow_policy.sh
  pass: exact action, local action, and Docker action accepted;
        mutable tag and ssh-keyscan rejected
./scripts/check-reconciliation-ledger.sh
  pass
cargo fmt --all -- --check
  pass
git diff --check
  pass
```

## Deliberate remaining blocker

The existing Clippy job retains the repository's warning semantics. Promotion
to strict `-D warnings` remains blocked by broad pre-existing pedantic and
documentation lint debt in the upstream-tracking crate, recorded in
`controlled-generation-reconciliation.md`. This workflow change does not
silence that debt or create a knowingly red required job.
