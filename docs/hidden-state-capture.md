# Bounded hidden-state capture

## Contract

`LlamaModel::new_context_with_hidden_state_capture` installs a context-owned
callback. Configuration and output types live in `context::hidden_states`:

```rust,ignore
let config = HiddenStateCaptureConfig::new(vec![1, 15, 31]);
let mut ctx = model.new_context_with_hidden_state_capture(&backend, params, config)?;
ctx.control_vector_clear()?;
ctx.set_hidden_state_capture_enabled(false)?;
// Decode the prefix with capture disabled.
ctx.decode(&mut prefix)?;
ctx.set_hidden_state_capture_enabled(true)?;
ctx.decode(&mut final_singleton_token)?;
let states = ctx.take_hidden_states()?;
// states[i].layer: zero-based u32; states[i].values: owned Vec<f32>
```

Capture starts enabled. Either enable/disable call clears the readout, even if
the flag does not change. Each decode invalidates the previous readout. Disabled
decodes request no tensor reads and produce `NoCapture`. Successful take consumes
the readout. A failed capture never returns partial or stale vectors. Native
decode and capture have separate results; both must be checked. Encode is not
supported. New contexts start without controls; toggling capture does not change
controls, KV state, or other context policy.

Selections are nonempty, unique zero-based layers, at most 64. Residual width is
`model.n_embd()`, at most 32768, with at most 4 MiB of vector storage per context.
Taking a snapshot allocates up to another 4 MiB; caller-retained snapshots are
caller-owned. Storage is allocated outside the callback. There are no caller
pointers, callback closures, borrowed tensor views, gradients, or autograd APIs.

Only a single sequence with increasing positions in one physical microbatch is
accepted, with logits requested for its final token. For longer prompts, disable
capture for the prefix and enable it for the final singleton token. In a supported
multi-token batch, only the last row is read: final-layer output-row pruning does
not provide a full-sequence row index. Tensor rows must equal either the batch
token count or requested output count. Each layer must appear exactly once.

The accepted native format is contiguous 2D f32 with the expected width and a
live backend buffer. Other layouts, absent/duplicate layers, and NaN/infinity
fail closed. Unsupported architectures fail at capture time rather than claiming
support based on model metadata.

## Pinned Native Semantics And Safety

At llama.cpp `5f55650a78f92aff4d48d671423e888fac0469ff`:

- `src/llama-context.cpp::graph_get_cb` names nodes exactly `l_out-{layer}`.
- `src/models/gemma4.cpp` emits this after block scaling and `build_cvec`, before
  final normalization. Active controls therefore affect the captured residual.
- `ggml/src/ggml-backend.cpp` synchronizes the backend before `ask=false`.
  `ggml_backend_tensor_get` copies only the last row, including from device memory.
- Returning false from the read callback only breaks one graph split and can
  still report graph success. Capture errors are latched while computation
  continues; they are not used to abort model evaluation.

A stable box holds mutex-protected callback state. It outlives construction and
all evaluation, and is dropped after `llama_free` destroys the native context.
No tensor pointer survives a callback. Rust callback panics are caught and poison
the capture; no unwind crosses the ABI. Native assertions and process-wide OOM
aborts are not recoverable Rust errors. Disabled capture still incurs native
callback dispatch overhead and can synchronize at scheduler split boundaries,
but requests no per-layer readback.

## Opt-in Regression Protocol

Preregistered before the local model run: use a synthetic fixed prompt, no
private data, and no downloads. Select first, middle, and final model layers.
Require finite model-width vectors, identical ordering, last-token agreement
between whole-prompt and disabled-prefix/singleton evaluation, changed results
for a changed final token, and no readout during disabled capture. Compare plain
context logits with captured logits and both zero-control and cleared-control
runs. Falsification is any absent/duplicate layer, stale result, non-finite value,
wrong width, or difference exceeding `1e-4 + 1e-4 * max(abs(a), abs(b))`.

The backend-specific extension also applies a final-layer-only `+0.25` control
in coordinate zero. Earlier layer vectors must stay unchanged, and the final
layer must include exactly that addition within the same tolerance. This directly
checks the claimed post-control capture stage.

The initial full-Metal run failed whole-batch versus singleton equivalence at
the original tolerance (one residual coordinate was -0.04613984 versus
-0.045972243). The backend-specific protocol therefore keeps CPU cross-shape
equivalence as a gate, reports GPU cross-shape differences as diagnostics, and
adds a capture-free split-token reference plus enabled/disabled-prefix capture
with identical batch shapes. Same-shape comparisons retain the original
tolerance. This is a recorded protocol revision, not a widened tolerance or a
claim of cross-shape Metal equivalence.

```sh
LLAMA_HIDDEN_STATE_MODEL=/absolute/path/model.gguf \
  cargo test -p llama-cpp-2 --no-default-features --test hidden_states \
  -- --ignored --nocapture
```

CPU is the default. Set `LLAMA_HIDDEN_STATE_GPU_LAYERS=99` to exercise a configured
GPU backend. No scientific or training-quality result is implied by this test.

## Local Verification (2026-09-16)

Base wrapper commit: `34444e2`; native submodule unchanged at the SHA above.
Validated with Gemma 4 E4B IT Q8_0, snapshot
`2714b5519c6c3516b1000e7c5e1eba998dfe1fe8`, using synthetic prompts only.

- `cargo test -p llama-cpp-2 --no-default-features`: 34 library tests (including
  8 capture tests), 4 existing integration tests, and 82 doctests passed;
  2 opt-in integration tests and 3 doctests ignored by the ordinary gate.
- The initial CPU-only real-model protocol passed, including original
  cross-shape comparisons and the final-layer nonzero control check.
- The revised same-shape real-model protocol passed with
  `LLAMA_HIDDEN_STATE_GPU_LAYERS=99`; the loader reported 43/43 layers offloaded.
  Captured layers `[0, 21, 41]` each had 2560 finite values. Plain/captured logits,
  zero/clear controls, enabled/disabled-prefix capture, and the final-layer
  `+0.25` control all met the original same-shape tolerance.
- Metal cross-shape diagnostics: capture-free logits maximum absolute delta
  `0.004171133`; residual deltas at layers 0/21/41 were respectively
  `0.0000038146973`, `0.00068295`, and `0.003967285`. Cross-shape Metal
  equivalence is explicitly not qualified.
- Formatting and `git diff --check` passed. Clippy completed with existing
  repository warnings; no warning referenced the new capture module/test.

Other models, CUDA, and other feature combinations were not runtime-qualified.
No upstream/native-kit training quality or integrated acceptance is claimed.
