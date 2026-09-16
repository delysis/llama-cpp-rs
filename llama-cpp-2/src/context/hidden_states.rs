//! Bounded, owned snapshots of the last token's post-block residual stream.
//!
//! The pinned llama.cpp graph names these nodes exactly `l_out-{layer}`. On
//! Gemma 4 this is after block scaling and `build_cvec` (including any active
//! control vector), but before final output normalization. Other architectures
//! must emit the same nodes; missing nodes are errors, not empty observations.

use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

use crate::llama_batch::LlamaBatch;
use llama_cpp_sys_2 as sys;

/// Maximum number of selected layers in one context.
pub const MAX_CAPTURE_LAYERS: usize = 64;
/// Maximum supported residual width.
pub const MAX_CAPTURE_WIDTH: usize = 32_768;
/// Maximum vector storage in a context (4 MiB). A taken snapshot may add as much.
pub const MAX_CAPTURE_BYTES: usize = 4 * 1024 * 1024;

/// Selected zero-based model layers. Validated against the model at construction.
#[derive(Debug, Clone)]
pub struct HiddenStateCaptureConfig {
    layers: Vec<u32>,
}

impl HiddenStateCaptureConfig {
    /// Select layers; order is retained in snapshots. Empty, duplicate, excessive,
    /// and out-of-range selections are rejected by the context constructor.
    #[must_use]
    pub fn new(layers: Vec<u32>) -> Self {
        Self { layers }
    }
}

/// One owned last-token residual vector, before final output normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerHiddenState {
    /// Zero-based model layer index (not a one-based control-vector row).
    pub layer: u32,
    /// Finite f32 values, exactly `model.n_embd()` wide.
    pub values: Vec<f32>,
}

/// Construction or capture failure. Partial observations are never returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HiddenStateCaptureError {
    /// No capture callback was installed on this context.
    #[error("hidden-state capture is not enabled")]
    NotEnabled,
    /// No completed decode, or the snapshot has already been taken.
    #[error("no hidden-state snapshot is available")]
    NoCapture,
    /// Empty, duplicate, excessive, or out-of-range layer selection.
    #[error("invalid hidden-state layer selection")]
    InvalidLayers,
    /// Unsupported model width or capture memory budget exceeded.
    #[error("hidden-state dimensions exceed capture limits")]
    CapacityExceeded,
    /// Allocation failed before entering the callback.
    #[error("could not allocate hidden-state storage")]
    AllocationFailed,
    /// Capture requires a decoder context without embedding pooling.
    #[error("hidden-state capture requires a decoder-only context without embeddings")]
    UnsupportedContext,
    /// llama.cpp could not create the context.
    #[error("llama.cpp could not create the capture context")]
    ContextCreationFailed,
    /// Requires one sequence, increasing positions, one physical microbatch,
    /// and a requested output for its last token. Encode is not supported.
    #[error("capture requires one sequence in one microbatch with last-token logits")]
    UnsupportedBatch,
    /// Native evaluation failed; any partial observations were discarded.
    #[error("native evaluation failed during hidden-state capture")]
    EvaluationFailed,
    /// The selected graph node was absent.
    #[error("missing residual tensor for layer {0}")]
    MissingLayer(u32),
    /// More than one matching node appeared; token identity is ambiguous.
    #[error("duplicate residual tensor for layer {0}")]
    DuplicateLayer(u32),
    /// Not a contiguous, allocated 2D f32 residual of the expected width/rows.
    #[error("unsupported residual tensor format for layer {0}")]
    UnsupportedTensor(u32),
    /// A selected residual contained NaN or infinity.
    #[error("non-finite residual value in layer {0}")]
    NonFinite(u32),
    /// Callback panic or poisoned state. No unwind crosses the C ABI.
    #[error("hidden-state callback failed")]
    CallbackFailed,
}

#[derive(Debug)]
struct Slot {
    snapshot: LayerHiddenState,
    seen: bool,
}

#[derive(Debug)]
struct State {
    slots: Vec<Slot>,
    rows: usize,
    outputs: usize,
    active: bool,
    enabled: bool,
    ready: bool,
    error: Option<HiddenStateCaptureError>,
}

/// The box's address is stable, including when the owning context moves.
/// The mutex prevents aliasing if native callback execution changes threads.
#[derive(Debug)]
pub(crate) struct Capture {
    state: Mutex<State>,
}

impl Capture {
    pub(crate) fn new(
        config: HiddenStateCaptureConfig,
        width: i32,
        layers: u32,
    ) -> Result<Box<Self>, HiddenStateCaptureError> {
        use HiddenStateCaptureError as E;
        if config.layers.is_empty() || config.layers.len() > MAX_CAPTURE_LAYERS {
            return Err(E::InvalidLayers);
        }
        for (i, layer) in config.layers.iter().enumerate() {
            if *layer >= layers || config.layers[..i].contains(layer) {
                return Err(E::InvalidLayers);
            }
        }
        let width = usize::try_from(width).map_err(|_| E::CapacityExceeded)?;
        if width == 0
            || width > MAX_CAPTURE_WIDTH
            || width * config.layers.len() * size_of::<f32>() > MAX_CAPTURE_BYTES
        {
            return Err(E::CapacityExceeded);
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(config.layers.len())
            .map_err(|_| E::AllocationFailed)?;
        for layer in config.layers {
            let mut values = Vec::new();
            values
                .try_reserve_exact(width)
                .map_err(|_| E::AllocationFailed)?;
            values.resize(width, 0.0);
            slots.push(Slot {
                snapshot: LayerHiddenState { layer, values },
                seen: false,
            });
        }
        Ok(Box::new(Self {
            state: Mutex::new(State {
                slots,
                rows: 0,
                outputs: 0,
                active: false,
                enabled: true,
                ready: false,
                error: None,
            }),
        }))
    }

    pub(crate) fn begin(&self, batch: &LlamaBatch<'_>, n_ubatch: u32) {
        if let Ok(mut state) = self.state.lock() {
            state.ready = false;
            state.error = None;
            state.active = false;
            for slot in &mut state.slots {
                slot.seen = false;
            }
            if !state.enabled {
                return;
            }
            match batch_shape(batch, n_ubatch) {
                Ok((rows, outputs)) => {
                    state.rows = rows;
                    state.outputs = outputs;
                    state.active = true;
                }
                Err(error) => state.error = Some(error),
            }
        }
    }

    pub(crate) fn finish(&self, success: bool) {
        if let Ok(mut state) = self.state.lock() {
            state.active = false;
            if !state.enabled {
                return;
            }
            state.ready = true;
            if !success {
                state.error = Some(HiddenStateCaptureError::EvaluationFailed);
            } else if state.error.is_none() {
                state.error = state
                    .slots
                    .iter()
                    .find(|slot| !slot.seen)
                    .map(|slot| HiddenStateCaptureError::MissingLayer(slot.snapshot.layer));
            }
        }
    }

    pub(crate) fn unsupported(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.active = false;
            state.ready = state.enabled;
            state.error = Some(HiddenStateCaptureError::UnsupportedBatch);
        }
    }

    pub(crate) fn set_enabled(&self, enabled: bool) -> Result<(), HiddenStateCaptureError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| HiddenStateCaptureError::CallbackFailed)?;
        state.enabled = enabled;
        state.active = false;
        state.ready = false;
        state.error = None;
        for slot in &mut state.slots {
            slot.seen = false;
        }
        Ok(())
    }

    pub(crate) fn take(&self) -> Result<Vec<LayerHiddenState>, HiddenStateCaptureError> {
        use HiddenStateCaptureError as E;
        let mut state = self.state.lock().map_err(|_| E::CallbackFailed)?;
        if !state.ready {
            return Err(E::NoCapture);
        }
        if let Some(error) = state.error {
            return Err(error);
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(state.slots.len())
            .map_err(|_| E::AllocationFailed)?;
        for slot in &state.slots {
            let mut values = Vec::new();
            values
                .try_reserve_exact(slot.snapshot.values.len())
                .map_err(|_| E::AllocationFailed)?;
            values.extend_from_slice(&slot.snapshot.values);
            result.push(LayerHiddenState {
                layer: slot.snapshot.layer,
                values,
            });
        }
        state.ready = false;
        Ok(result)
    }
}

// Restrict token identity rather than guessing from scheduler/sequence ordering.
fn batch_shape(
    batch: &LlamaBatch<'_>,
    n_ubatch: u32,
) -> Result<(usize, usize), HiddenStateCaptureError> {
    let invalid = HiddenStateCaptureError::UnsupportedBatch;
    let raw = &batch.llama_batch;
    let rows = usize::try_from(raw.n_tokens).map_err(|_| invalid)?;
    if rows == 0 || rows > n_ubatch as usize || raw.token.is_null() {
        return Err(invalid);
    }
    let mut sequence = None;
    let mut previous_pos = None;
    let mut outputs = 0;
    for i in 0..rows {
        // SAFETY: LlamaBatch owns (or borrows through get_one) initialized arrays
        // for n_tokens. Null arrays encode llama.cpp's single-sequence defaults.
        unsafe {
            if !raw.n_seq_id.is_null() {
                if *raw.n_seq_id.add(i) != 1 || raw.seq_id.is_null() {
                    return Err(invalid);
                }
                let ids = *raw.seq_id.add(i);
                if ids.is_null() {
                    return Err(invalid);
                }
                let id = *ids;
                if sequence.is_some_and(|previous| previous != id) {
                    return Err(invalid);
                }
                sequence = Some(id);
            } else if !raw.seq_id.is_null() {
                return Err(invalid);
            }
            if !raw.pos.is_null() {
                let pos = *raw.pos.add(i);
                if pos < 0 || previous_pos.is_some_and(|previous| previous >= pos) {
                    return Err(invalid);
                }
                previous_pos = Some(pos);
            }
            let output = if raw.logits.is_null() {
                i == rows - 1
            } else {
                *raw.logits.add(i) != 0
            };
            outputs += usize::from(output);
            if i == rows - 1 && !output {
                return Err(invalid);
            }
        }
    }
    Ok((rows, outputs))
}

// Parse a bounded fixed-size name, never an unbounded C string or prefix match.
fn layer_from_name(name: &[c_char]) -> Option<u32> {
    let end = name.iter().position(|c| *c == 0)?;
    let name = &name[..end];
    let prefix = b"l_out-";
    if name.len() <= prefix.len()
        || !name
            .iter()
            .zip(prefix)
            .all(|(a, b)| i32::from(*a) == i32::from(*b))
    {
        return None;
    }
    let digits = &name[prefix.len()..];
    if digits.len() > 1 && digits[0] == 48 {
        return None;
    }
    digits.iter().try_fold(0_u32, |value, c| {
        let digit = u32::try_from(*c).ok()?.checked_sub(u32::from(b'0'))?;
        if digit > 9 {
            return None;
        }
        value.checked_mul(10)?.checked_add(digit)
    })
}

fn row_offset(t: &sys::ggml_tensor, width: usize, rows: usize, outputs: usize) -> Option<usize> {
    let tensor_rows = usize::try_from(t.ne[1]).ok()?;
    let row_bytes = width.checked_mul(size_of::<f32>())?;
    let total = row_bytes.checked_mul(tensor_rows)?;
    if t.type_ != sys::GGML_TYPE_F32
        || usize::try_from(t.ne[0]).ok()? != width
        || tensor_rows == 0
        || (tensor_rows != rows && tensor_rows != outputs)
        || t.ne[2] != 1
        || t.ne[3] != 1
        || t.nb != [size_of::<f32>(), row_bytes, total, total]
    {
        return None;
    }
    total.checked_sub(row_bytes)
}

/// Called only by the native scheduler. No allocation or user code in this path.
///
/// SAFETY: `user_data` points to the context-owned Capture box, alive until after
/// `llama_free` synchronizes/destroys the scheduler. A nonnull tensor is owned by
/// the scheduler and alive for this invocation. On ask=false the scheduler has
/// synchronized its backend. No tensor/data pointers escape this function.
pub(crate) unsafe extern "C" fn callback(
    t: *mut sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return !ask;
    }
    // SAFETY: installed only by new_context_with_hidden_state_capture.
    let capture = unsafe { &*user_data.cast::<Capture>() };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Ok(mut state) = capture.state.lock() else {
            return !ask;
        };
        if !state.active || state.error.is_some() {
            return !ask;
        }
        // SAFETY: the scheduler provides a valid tensor, or we fail closed.
        let Some(t) = (unsafe { t.as_ref() }) else {
            state.error = Some(HiddenStateCaptureError::CallbackFailed);
            return !ask;
        };
        let Some(layer) = layer_from_name(&t.name) else {
            return !ask;
        };
        let rows = state.rows;
        let outputs = state.outputs;
        let Some(slot) = state.slots.iter_mut().find(|s| s.snapshot.layer == layer) else {
            return !ask;
        };
        if ask {
            return true;
        }
        let error = if slot.seen {
            Some(HiddenStateCaptureError::DuplicateLayer(layer))
        } else if let Some(offset) = row_offset(t, slot.snapshot.values.len(), rows, outputs) {
            // SAFETY: view_src, when present, belongs to this live native graph.
            let buffer = unsafe { t.view_src.as_ref() }.map_or(t.buffer, |src| src.buffer);
            if t.data.is_null() || buffer.is_null() {
                Some(HiddenStateCaptureError::UnsupportedTensor(layer))
            } else {
                // SAFETY: checked contiguous f32 layout and last-row offset;
                // destination is an initialized, width-sized owned f32 buffer.
                // The backend API handles device memory, not a host dereference.
                unsafe {
                    sys::ggml_backend_tensor_get(
                        t,
                        slot.snapshot.values.as_mut_ptr().cast(),
                        offset,
                        slot.snapshot.values.len() * size_of::<f32>(),
                    );
                }
                if slot.snapshot.values.iter().all(|v| v.is_finite()) {
                    slot.seen = true;
                    None
                } else {
                    Some(HiddenStateCaptureError::NonFinite(layer))
                }
            }
        } else {
            Some(HiddenStateCaptureError::UnsupportedTensor(layer))
        };
        state.error = error;
        // Do not abort computation: in the pinned scheduler false only breaks
        // one split and may still report success with uncomputed output nodes.
        true
    }));
    match result {
        Ok(value) => value,
        Err(payload) => {
            // A panic payload may itself panic on drop. This exceptional leak
            // ensures even such a panic cannot unwind through the C ABI.
            std::mem::forget(payload);
            if let Ok(mut state) = capture.state.lock() {
                state.error = Some(HiddenStateCaptureError::CallbackFailed);
            }
            !ask
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::LlamaToken;
    use HiddenStateCaptureError as E;

    #[allow(clippy::unnecessary_box_returns)] // Mirror the stable callback owner.
    fn capture() -> Box<Capture> {
        Capture::new(HiddenStateCaptureConfig::new(vec![0, 2]), 4, 3).unwrap()
    }

    fn batch() -> LlamaBatch<'static> {
        let mut batch = LlamaBatch::new(2, 1);
        batch.add(LlamaToken(1), 0, &[0], false).unwrap();
        batch.add(LlamaToken(2), 1, &[0], true).unwrap();
        batch
    }

    fn tensor(name: &str) -> sys::ggml_tensor {
        // SAFETY: ggml_tensor is a C POD with integer type tags and nullable
        // pointers. Backend reads require a separate buffer allocation below.
        let mut tensor: sys::ggml_tensor = unsafe { std::mem::zeroed() };
        tensor.type_ = sys::GGML_TYPE_F32;
        tensor.ne = [4, 2, 1, 1];
        tensor.nb = [4, 16, 32, 32];
        for (dest, src) in tensor.name.iter_mut().zip(name.bytes()) {
            *dest = c_char::try_from(src).unwrap();
        }
        tensor
    }

    #[test]
    fn bounds_and_selection_are_checked_before_allocation() {
        for layers in [vec![], vec![0, 0], vec![3], vec![0; MAX_CAPTURE_LAYERS + 1]] {
            assert_eq!(
                Capture::new(HiddenStateCaptureConfig::new(layers), 4, 3).unwrap_err(),
                E::InvalidLayers
            );
        }
        for width in [-1, 0, i32::MAX] {
            assert_eq!(
                Capture::new(HiddenStateCaptureConfig::new(vec![0]), width, 3).unwrap_err(),
                E::CapacityExceeded
            );
        }
        assert_eq!(
            Capture::new(HiddenStateCaptureConfig::new((0..64).collect()), 32_768, 64).unwrap_err(),
            E::CapacityExceeded
        );
        assert!(Capture::new(HiddenStateCaptureConfig::new((0..32).collect()), 32_768, 64).is_ok());
    }

    #[test]
    fn names_must_match_exactly_and_be_terminated() {
        for (name, expected) in [
            ("l_out-0", Some(0)),
            ("l_out-31", Some(31)),
            ("l_out", None),
            ("l_out-01", None),
            ("l_out-1extra", None),
            ("l_out-4294967296", None),
            ("l_out--1", None),
            ("other-1", None),
        ] {
            assert_eq!(layer_from_name(&tensor(name).name), expected, "{name}");
        }
        assert_eq!(
            layer_from_name(&[c_char::try_from(b'1').unwrap(); 64]),
            None
        );
    }

    #[test]
    fn accepts_only_unambiguous_batches() {
        assert_eq!(batch_shape(&batch(), 2), Ok((2, 1)));
        assert_eq!(batch_shape(&batch(), 1), Err(E::UnsupportedBatch));
        let tokens = [LlamaToken(1), LlamaToken(2)];
        assert_eq!(
            batch_shape(&LlamaBatch::get_one(&tokens).unwrap(), 2),
            Ok((2, 1))
        );
        for (seq, pos, output) in [(1, 1, true), (0, 0, true), (0, 1, false)] {
            let mut b = LlamaBatch::new(2, 1);
            b.add(LlamaToken(1), 0, &[0], true).unwrap();
            b.add(LlamaToken(2), pos, &[seq], output).unwrap();
            assert_eq!(batch_shape(&b, 2), Err(E::UnsupportedBatch));
        }
        assert_eq!(
            batch_shape(&LlamaBatch::new(2, 1), 2),
            Err(E::UnsupportedBatch)
        );
    }

    #[test]
    fn validates_type_dimensions_and_strides() {
        let mut t = tensor("l_out-0");
        assert_eq!(row_offset(&t, 4, 2, 1), Some(16));
        t.ne[1] = 1;
        t.nb = [4, 16, 16, 16];
        assert_eq!(row_offset(&t, 4, 2, 1), Some(0));
        for ne in [
            [3, 1, 1, 1],
            [4, 0, 1, 1],
            [4, 1, 2, 1],
            [4, 3, 1, 1],
            [4, -1, 1, 1],
        ] {
            t.ne = ne;
            assert_eq!(row_offset(&t, 4, 2, 1), None);
        }
        t = tensor("l_out-0");
        t.type_ = sys::GGML_TYPE_F16;
        assert_eq!(row_offset(&t, 4, 2, 1), None);
        t.type_ = sys::GGML_TYPE_F32;
        t.nb[1] = 32;
        assert_eq!(row_offset(&t, 4, 2, 1), None);
    }

    #[test]
    fn complete_owned_snapshot_is_consumed_and_never_reused() {
        let c = capture();
        assert_eq!(c.take(), Err(E::NoCapture));
        c.begin(&batch(), 2);
        {
            let mut s = c.state.lock().unwrap();
            for slot in &mut s.slots {
                slot.seen = true;
                slot.snapshot.values.fill(7.0);
            }
        }
        c.finish(true);
        let owned = c.take().unwrap();
        assert_eq!(
            owned.iter().map(|s| s.layer).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(c.take(), Err(E::NoCapture));
        c.begin(&batch(), 2);
        c.finish(true);
        assert_eq!(c.take(), Err(E::MissingLayer(0)));
        c.begin(&batch(), 2);
        c.finish(false);
        assert_eq!(c.take(), Err(E::EvaluationFailed));
        c.begin(&batch(), 1);
        c.finish(true);
        assert_eq!(c.take(), Err(E::UnsupportedBatch));
        assert_eq!(owned[0].values, vec![7.0; 4]);
    }

    #[test]
    fn callback_selects_exact_layers_and_rejects_unallocated_tensors() {
        let mut c = capture();
        c.begin(&batch(), 2);
        let ptr = std::ptr::from_mut(c.as_mut()).cast();
        // SAFETY: these live tensors and capture storage outlive each callback;
        // rejected tensors are never passed to a backend read.
        unsafe {
            assert!(!callback(&mut tensor("l_out-1"), true, ptr));
            assert!(!callback(&mut tensor("l_out-0extra"), true, ptr));
            assert!(callback(&mut tensor("l_out-0"), true, ptr));
            assert!(callback(&mut tensor("l_out-0"), false, ptr));
        }
        c.finish(true);
        assert_eq!(c.take(), Err(E::UnsupportedTensor(0)));
    }

    #[test]
    fn poisoned_state_is_a_closed_capture_error() {
        let c = capture();
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _guard = c.state.lock().unwrap();
            panic!("simulated callback panic");
        }));
        c.finish(true);
        assert_eq!(c.take(), Err(E::CallbackFailed));
    }

    #[test]
    fn backend_read_copies_only_last_row_and_rejects_duplicates_and_nonfinite() {
        struct Buffer(sys::ggml_backend_buffer_t);
        impl Drop for Buffer {
            fn drop(&mut self) {
                // SAFETY: sole owner; every tensor access has finished.
                unsafe { sys::ggml_backend_buffer_free(self.0) }
            }
        }
        // SAFETY: allocate a native CPU buffer, then allocate a correctly sized
        // tensor in it. The RAII guard outlives all callback reads, and the host
        // source passed to tensor_set is valid for the full copied byte count.
        unsafe {
            let buffer = Buffer(sys::ggml_backend_buft_alloc_buffer(
                sys::ggml_backend_cpu_buffer_type(),
                32,
            ));
            assert!(!buffer.0.is_null());
            let mut t = tensor("l_out-0");
            assert_eq!(
                sys::ggml_backend_tensor_alloc(
                    buffer.0,
                    &raw mut t,
                    sys::ggml_backend_buffer_get_base(buffer.0)
                ),
                sys::GGML_STATUS_SUCCESS
            );
            let mut c = Capture::new(HiddenStateCaptureConfig::new(vec![0]), 4, 3).unwrap();
            let ptr = std::ptr::from_mut(c.as_mut()).cast();
            let data: [f32; 8] = [f32::NAN, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
            sys::ggml_backend_tensor_set(&raw mut t, data.as_ptr().cast(), 0, size_of_val(&data));
            c.begin(&batch(), 2);
            assert!(callback(&raw mut t, false, ptr));
            c.finish(true);
            assert_eq!(c.take().unwrap()[0].values, vec![1.0, 2.0, 3.0, 4.0]);

            c.begin(&batch(), 2);
            callback(&raw mut t, false, ptr);
            callback(&raw mut t, false, ptr);
            c.finish(true);
            assert_eq!(c.take(), Err(E::DuplicateLayer(0)));

            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut data = data;
                data[7] = invalid;
                sys::ggml_backend_tensor_set(
                    &raw mut t,
                    data.as_ptr().cast(),
                    0,
                    size_of_val(&data),
                );
                c.begin(&batch(), 2);
                callback(&raw mut t, false, ptr);
                c.finish(true);
                assert_eq!(c.take(), Err(E::NonFinite(0)));
            }

            c.set_enabled(false).unwrap();
            c.begin(&batch(), 1); // disabled prefill is not shape-validated
            assert!(!callback(&raw mut t, true, ptr));
            c.finish(true);
            assert_eq!(c.take(), Err(E::NoCapture));
            c.set_enabled(true).unwrap();
            c.begin(&batch(), 2);
            c.finish(true);
            assert_eq!(c.take(), Err(E::MissingLayer(0)));
            c.set_enabled(true).unwrap();
            assert_eq!(c.take(), Err(E::NoCapture));
        }
    }
}
