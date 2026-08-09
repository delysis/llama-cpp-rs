//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;

use crate::context::params::LlamaPoolingType;
use crate::llama_batch::LlamaBatch;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::sampling::LlamaSampler;
use crate::timing::LlamaTimings;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaControlVectorError,
    LlamaLoraAdapterRemoveError, LlamaLoraAdapterSetError,
};

pub mod kv_cache;
pub mod params;
pub mod session;

/// Safe wrapper around `llama_context`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaContext<'a> {
    pub(crate) context: NonNull<llama_cpp_sys_2::llama_context>,
    /// a reference to the contexts model.
    pub model: &'a LlamaModel,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
    /// Backend samplers kept alive for the context's lifetime.
    _backend_samplers: Vec<(i32, LlamaSampler)>,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    pub(crate) fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_sys_2::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
            embeddings_enabled,
            _backend_samplers: Vec::new(),
        }
    }

    pub(crate) fn with_samplers(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_sys_2::llama_context>,
        embeddings_enabled: bool,
        backend_samplers: Vec<(i32, LlamaSampler)>,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
            embeddings_enabled,
            _backend_samplers: backend_samplers,
        }
    }

    /// Gets the max number of logical tokens that can be submitted to decode. Must be greater than or equal to [`Self::n_ubatch`].
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch. Must be less than or equal to [`Self::n_batch`].
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_ubatch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Returns the context's live embedding pooling mode.
    ///
    /// This reports the resolved mode used by llama.cpp, which can differ from
    /// the value originally requested through context parameters when that
    /// value was [`LlamaPoolingType::Unspecified`].
    #[must_use]
    pub fn pooling_type(&self) -> LlamaPoolingType {
        let pooling = unsafe { llama_cpp_sys_2::llama_pooling_type(self.context.as_ptr()) };
        LlamaPoolingType::from(pooling)
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result =
            unsafe { llama_cpp_sys_2::llama_decode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let result =
            unsafe { llama_cpp_sys_2::llama_encode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(EncodeError::from(error)),
        }
    }

    /// Get the embeddings for the `i`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size is the pooling-derived output width: `n_cls_out` for RANK,
    /// `n_embd_out` otherwise — NOT `n_embd` (llama.h:1029 /
    /// llama-context.cpp's extraction switch).
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_sys_2::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        unsafe {
            let embedding = llama_cpp_sys_2::llama_get_embeddings_seq(self.context.as_ptr(), i);

            // Technically also possible whenever `i >= max(batch.n_seq)`, but can't check that here.
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, self.embeddings_out_len()))
            }
        }
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size is the pooling-derived output width: `n_cls_out` for RANK,
    /// `n_embd_out` otherwise — NOT `n_embd` (llama.h:1029 /
    /// llama-context.cpp's extraction switch).
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        unsafe {
            let embedding = llama_cpp_sys_2::llama_get_embeddings_ith(self.context.as_ptr(), i);
            // Technically also possible whenever `i >= batch.n_tokens`, but no good way of checking `n_tokens` here.
            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, self.embeddings_out_len()))
            }
        }
    }

    /// The correct output width for an embeddings read, keyed on the context's
    /// LIVE pooling type rather than the model's `n_embd`.
    ///
    /// RANK reads return `float[n_cls_out]` (default 1) per llama.h:1029; every
    /// other pooling mode extracts at `n_embd_out` per llama-context.cpp's
    /// extraction switch (which diverges from `n_embd` whenever
    /// `{arch}.embedding_length_out` is present).
    fn embeddings_out_len(&self) -> usize {
        let pooling = unsafe { llama_cpp_sys_2::llama_pooling_type(self.context.as_ptr()) };
        if pooling == llama_cpp_sys_2::LLAMA_POOLING_TYPE_RANK {
            usize::try_from(self.model.n_cls_out()).expect("n_cls_out does not fit into a usize")
        } else {
            usize::try_from(self.model.n_embd_out()).expect("n_embd_out does not fit into a usize")
        }
    }

    /// Get the logits for the last token in the context.
    ///
    /// # Returns
    /// An iterator over unsorted `LlamaTokenData` containing the
    /// logits for the last token in the context.
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    pub fn candidates(&self) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits()).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the last token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates(), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    #[must_use]
    pub fn token_data_array(&self) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates(), false)
    }

    /// Token logits obtained from the last call to `decode()`.
    /// The logits for which `batch.logits[i] != 0` are stored contiguously
    /// in the order they have appeared in the batch.
    /// Rows: number of tokens for which `batch.logits[i] != 0`
    /// Cols: `n_vocab`
    ///
    /// # Returns
    ///
    /// A slice containing the logits for the last decoded token.
    /// The size corresponds to the `n_vocab` parameter of the context's model.
    ///
    /// # Panics
    ///
    /// - `n_vocab` does not fit into a usize
    /// - token data returned is null
    #[must_use]
    pub fn get_logits(&self) -> &[f32] {
        let data = unsafe { llama_cpp_sys_2::llama_get_logits(self.context.as_ptr()) };
        assert!(!data.is_null(), "logits data for last token is null");
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the ith token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates_ith(i), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn token_data_array_ith(&self, i: i32) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates_ith(i), false)
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `i` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        assert!(
            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
            "n_ctx ({}) must be greater than i ({})",
            self.n_ctx(),
            i
        );

        let data = unsafe { llama_cpp_sys_2::llama_get_logits_ith(self.context.as_ptr(), i) };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_sys_2::llama_perf_context_reset(self.context.as_ptr()) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_sys_2::llama_perf_context(self.context.as_ptr()) };
        LlamaTimings { timings }
    }

    /// Sets a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter<'_>,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        self.lora_adapters_set(&[(&*adapter, scale)])
    }

    /// Atomically replaces every `LoRA` adapter active on this context.
    ///
    /// Passing an empty slice clears the stack. All adapters and scales are
    /// validated before llama.cpp is called, so invalid input cannot leave a
    /// partially changed stack. llama.cpp stores non-owning adapter pointers in
    /// the context, while the borrowed [`LlamaModel`] owns their allocations;
    /// dropping an [`LlamaLoraAdapter`] handle after this call is therefore safe
    /// and does not unload the active adapter.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaLoraAdapterSetError`] if a scale is non-finite, an
    /// adapter belongs to another model, an adapter is duplicated, or
    /// llama.cpp rejects the replacement.
    pub fn lora_adapters_set(
        &self,
        adapters: &[(&LlamaLoraAdapter<'_>, f32)],
    ) -> Result<(), LlamaLoraAdapterSetError> {
        validate_lora_adapters(self.model, adapters)?;

        let mut adapter_ptrs: Vec<_> = adapters
            .iter()
            .map(|(adapter, _)| adapter.lora_adapter.as_ptr())
            .collect();
        let mut scales: Vec<_> = adapters.iter().map(|(_, scale)| *scale).collect();

        let adapter_ptr = if adapter_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            adapter_ptrs.as_mut_ptr()
        };
        let scales_ptr = if scales.is_empty() {
            std::ptr::null_mut()
        } else {
            scales.as_mut_ptr()
        };

        let err_code = unsafe {
            llama_cpp_sys_2::llama_set_adapters_lora(
                self.context.as_ptr(),
                adapter_ptr,
                adapter_ptrs.len(),
                scales_ptr,
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
        }

        tracing::debug!(adapter_count = adapters.len(), "Replaced lora adapters");
        Ok(())
    }

    /// Remove all lora adapters.
    ///
    /// Note: The upstream API now replaces all adapters at once via
    /// `llama_set_adapters_lora`. This clears all adapters from the context.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        _adapter: &mut LlamaLoraAdapter<'_>,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            llama_cpp_sys_2::llama_set_adapters_lora(
                self.context.as_ptr(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
        }

        tracing::debug!("Remove lora adapter");
        Ok(())
    }

    /// Sets a static additive control vector on an inclusive range of model
    /// layers.
    ///
    /// `data` must contain exactly one `n_embd`-wide row for every non-zero
    /// model layer, ordered from layer 1 through `model.n_layer() - 1`. Requiring
    /// the complete buffer prevents a shorter replacement from retaining stale
    /// rows in llama.cpp's already allocated control-vector storage.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaControlVectorError`] for an invalid model dimension,
    /// layer range, buffer length, non-finite value, or llama.cpp error.
    pub fn control_vector_set(
        &self,
        data: &[f32],
        layer_start: u32,
        layer_end: u32,
    ) -> Result<(), LlamaControlVectorError> {
        let (n_embd, layer_start, layer_end) = validate_control_vector(
            self.model.n_embd(),
            self.model.n_layer(),
            data,
            layer_start,
            layer_end,
        )?;

        let err_code = unsafe {
            llama_cpp_sys_2::llama_set_adapter_cvec(
                self.context.as_ptr(),
                data.as_ptr(),
                data.len(),
                n_embd,
                layer_start,
                layer_end,
            )
        };
        if err_code != 0 {
            return Err(LlamaControlVectorError::ErrorResult(err_code));
        }

        tracing::debug!(layer_start, layer_end, "Set control vector");
        Ok(())
    }

    /// Clears the static additive control vector from this context.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaControlVectorError::ErrorResult`] if llama.cpp rejects
    /// the operation.
    pub fn control_vector_clear(&self) -> Result<(), LlamaControlVectorError> {
        let err_code = unsafe {
            llama_cpp_sys_2::llama_set_adapter_cvec(
                self.context.as_ptr(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
            )
        };
        if err_code != 0 {
            return Err(LlamaControlVectorError::ErrorResult(err_code));
        }

        tracing::debug!("Cleared control vector");
        Ok(())
    }

    /// Get the backend-sampled token at the given index.
    ///
    /// This is part of the experimental backend sampling API. Only usable
    /// when the context was created with at least one `llama_sampler_seq_config`.
    ///
    /// Returns `None` if no token was sampled at the given index
    /// (i.e. the C API returned `LLAMA_TOKEN_NULL`).
    ///
    /// # Arguments
    ///
    /// * `i` - The token index, matching the order from the batch.
    #[must_use]
    pub fn sampled_token_ith(&self, i: i32) -> Option<LlamaToken> {
        let token =
            unsafe { llama_cpp_sys_2::llama_get_sampled_token_ith(self.context.as_ptr(), i) };
        // LLAMA_TOKEN_NULL is #define'd as -1 in llama.h (not exposed by bindgen)
        if token == -1 {
            None
        } else {
            Some(LlamaToken(token))
        }
    }

    /// Print a breakdown of per-device memory use to the default logger.
    #[cfg(feature = "common")]
    pub fn print_memory_breakdown(&self) {
        unsafe { llama_cpp_sys_2::llama_rs_memory_breakdown_print(self.context.as_ptr()) }
    }
}

fn validate_lora_adapters(
    context_model: &LlamaModel,
    adapters: &[(&LlamaLoraAdapter<'_>, f32)],
) -> Result<(), LlamaLoraAdapterSetError> {
    for (index, (adapter, scale)) in adapters.iter().enumerate() {
        if !scale.is_finite() {
            return Err(LlamaLoraAdapterSetError::NonFiniteScale { index });
        }
        if !std::ptr::eq(adapter.model, context_model) {
            return Err(LlamaLoraAdapterSetError::ModelMismatch { index });
        }
        if let Some(first_index) = adapters[..index]
            .iter()
            .position(|(prior, _)| prior.lora_adapter == adapter.lora_adapter)
        {
            return Err(LlamaLoraAdapterSetError::DuplicateAdapter {
                first_index,
                duplicate_index: index,
            });
        }
    }
    Ok(())
}

fn validate_control_vector(
    n_embd: i32,
    layer_count: u32,
    data: &[f32],
    layer_start: u32,
    layer_end: u32,
) -> Result<(i32, i32, i32), LlamaControlVectorError> {
    let n_embd_usize = usize::try_from(n_embd)
        .ok()
        .filter(|width| *width > 0)
        .ok_or(LlamaControlVectorError::InvalidEmbeddingWidth(n_embd))?;

    if layer_count < 2 {
        return Err(LlamaControlVectorError::InsufficientLayers(layer_count));
    }
    if layer_start == 0 || layer_start > layer_end || layer_end >= layer_count {
        return Err(LlamaControlVectorError::InvalidLayerRange {
            start: layer_start,
            end: layer_end,
            layer_count,
        });
    }

    let row_count =
        usize::try_from(layer_count - 1).map_err(|_| LlamaControlVectorError::LengthOverflow)?;
    let expected = n_embd_usize
        .checked_mul(row_count)
        .ok_or(LlamaControlVectorError::LengthOverflow)?;
    if data.len() != expected {
        return Err(LlamaControlVectorError::LengthMismatch {
            expected,
            actual: data.len(),
        });
    }
    if let Some(index) = data.iter().position(|value| !value.is_finite()) {
        return Err(LlamaControlVectorError::NonFiniteValue { index });
    }

    let layer_start = i32::try_from(layer_start)
        .map_err(|_| LlamaControlVectorError::LayerIndexOutOfRange(layer_start))?;
    let layer_end = i32::try_from(layer_end)
        .map_err(|_| LlamaControlVectorError::LayerIndexOutOfRange(layer_end))?;
    Ok((n_embd, layer_start, layer_end))
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod lora_adapter_tests {
    use std::mem::ManuallyDrop;
    use std::ptr::NonNull;

    use super::validate_lora_adapters;
    use crate::model::{LlamaLoraAdapter, LlamaModel};
    use crate::LlamaLoraAdapterSetError;

    #[test]
    fn rejects_invalid_stacks_before_ffi() {
        let model = ManuallyDrop::new(LlamaModel {
            model: NonNull::dangling(),
        });
        let other_model = ManuallyDrop::new(LlamaModel {
            model: NonNull::dangling(),
        });
        let adapter = LlamaLoraAdapter {
            lora_adapter: NonNull::dangling(),
            model: &model,
        };
        let other_adapter = LlamaLoraAdapter {
            lora_adapter: NonNull::dangling(),
            model: &other_model,
        };

        assert_eq!(validate_lora_adapters(&model, &[]), Ok(()));
        assert_eq!(
            validate_lora_adapters(&model, &[(&adapter, f32::NAN)]),
            Err(LlamaLoraAdapterSetError::NonFiniteScale { index: 0 })
        );
        assert_eq!(
            validate_lora_adapters(&model, &[(&other_adapter, 1.0)]),
            Err(LlamaLoraAdapterSetError::ModelMismatch { index: 0 })
        );
        assert_eq!(
            validate_lora_adapters(&model, &[(&adapter, 1.0), (&adapter, 0.5)]),
            Err(LlamaLoraAdapterSetError::DuplicateAdapter {
                first_index: 0,
                duplicate_index: 1,
            })
        );
    }
}

#[cfg(test)]
mod control_vector_tests {
    use super::validate_control_vector;
    use crate::LlamaControlVectorError;

    #[test]
    fn accepts_complete_finite_nonzero_layer_matrix() {
        let data = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(validate_control_vector(2, 3, &data, 1, 2), Ok((2, 1, 2)));
    }

    #[test]
    fn rejects_short_matrix_that_could_retain_stale_layers() {
        assert_eq!(
            validate_control_vector(2, 3, &[0.0, 1.0], 1, 1),
            Err(LlamaControlVectorError::LengthMismatch {
                expected: 4,
                actual: 2,
            })
        );
    }

    #[test]
    fn rejects_zero_and_out_of_bounds_layers() {
        let data = [0.0, 1.0, 2.0, 3.0];
        for (start, end) in [(0, 1), (2, 1), (1, 3)] {
            assert_eq!(
                validate_control_vector(2, 3, &data, start, end),
                Err(LlamaControlVectorError::InvalidLayerRange {
                    start,
                    end,
                    layer_count: 3,
                })
            );
        }
    }

    #[test]
    fn rejects_nonfinite_values_before_ffi() {
        assert_eq!(
            validate_control_vector(2, 3, &[0.0, 1.0, f32::NAN, 3.0], 1, 2),
            Err(LlamaControlVectorError::NonFiniteValue { index: 2 })
        );
    }
}
