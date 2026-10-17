# llama-cpp-sys

Raw bindings to llama.cpp with cuda support.

See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a safe API.

### Rewindable sliding-window cache exports

The pinned native source's default sequence export drops SWA cells masked at the
last saved position. Such a snapshot cannot safely be cropped to an earlier
shared prompt prefix: keys needed at that earlier position have been discarded.
`KEEP_ALL_SWA_CELLS` (bit 4) preserves all resident cells on export. Use a full-size
SWA context to retain the complete history. Import uses the ordinary state format;
the cell count and positions already describe the expanded payload. The default
compact export remains unchanged. Snapshot memory and disk usage grow with the
retained context, so consumers must keep their existing capacity bounds.

`rewindable_swa.rs` applies this two-site reviewed change to an owned build copy of
upstream commit `5f55650a78f92aff4d48d671423e888fac0469ff`. It rejects patch drift,
never modifies Cargo's shared checkout, and maps generated-header dependencies
back to their pristine sources. Artifact hashes in build evidence include the
patched native library.
