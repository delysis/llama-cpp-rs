//! Opt-in local GGUF integration check. No downloads and no user data.
use std::num::NonZeroU32;

use llama_cpp_2::context::hidden_states::{HiddenStateCaptureConfig, HiddenStateCaptureError};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

fn params(gpu_layers: u32) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(256))
        .with_n_batch(128)
        .with_n_ubatch(128)
        .with_offload_kqv(gpu_layers != 0)
        .with_op_offload(gpu_layers != 0)
}

#[track_caller]
fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (i, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 1e-4 + 1e-4 * a.abs().max(b.abs()),
            "index {i}: {a} != {b}"
        );
    }
}

#[test]
#[ignore = "requires LLAMA_HIDDEN_STATE_MODEL pointing to a local GGUF"]
#[allow(clippy::too_many_lines)] // One backend/model lifetime for the full protocol.
fn local_model_capture_and_noop_controls() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("LLAMA_HIDDEN_STATE_MODEL")?;
    let gpu_layers = std::env::var("LLAMA_HIDDEN_STATE_GPU_LAYERS")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(
        &backend,
        path,
        &LlamaModelParams::default().with_n_gpu_layers(gpu_layers),
    )?;
    let layers = vec![0, model.n_layer() / 2, model.n_layer() - 1];
    let width = usize::try_from(model.n_embd())?;
    let tokens = model.str_to_token(
        "A small reproducible residual capture test.",
        AddBos::Always,
    )?;
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    batch.add_sequence(&tokens, 0, false)?;

    let mut plain = model.new_context(&backend, params(gpu_layers))?;
    assert_eq!(
        plain.take_hidden_states(),
        Err(HiddenStateCaptureError::NotEnabled)
    );
    plain.decode(&mut batch)?;
    let plain_logits = plain.get_logits().to_vec();
    let mut prefix = LlamaBatch::new(tokens.len() - 1, 1);
    prefix.add_sequence(&tokens[..tokens.len() - 1], 0, false)?;
    let mut last = LlamaBatch::new(1, 1);
    last.add(
        *tokens.last().unwrap(),
        i32::try_from(tokens.len() - 1)?,
        &[0],
        true,
    )?;
    plain.clear_kv_cache();
    plain.decode(&mut prefix)?;
    plain.decode(&mut last)?;
    let plain_split_logits = plain.get_logits().to_vec();
    let native_batch_shape_drift = plain_logits
        .iter()
        .zip(&plain_split_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("capture-free batch/singleton logits max_abs_delta={native_batch_shape_drift}");
    drop(plain);

    let mut ctx = model.new_context_with_hidden_state_capture(
        &backend,
        params(gpu_layers),
        HiddenStateCaptureConfig::new(layers.clone()),
    )?;
    ctx.decode(&mut batch)?;
    let baseline = ctx.take_hidden_states()?;
    assert_eq!(baseline.iter().map(|s| s.layer).collect::<Vec<_>>(), layers);
    for snapshot in &baseline {
        assert_eq!(snapshot.values.len(), width);
        assert!(snapshot.values.iter().all(|v| v.is_finite()));
    }
    assert_close(&plain_logits, ctx.get_logits());
    assert_eq!(
        ctx.take_hidden_states(),
        Err(HiddenStateCaptureError::NoCapture)
    );
    for zero_control in [false, true, false] {
        eprintln!("checking whole-batch controls: zero={zero_control}");
        ctx.clear_kv_cache();
        if zero_control {
            ctx.control_vector_set(
                &vec![0.0; width * usize::try_from(model.n_layer() - 1)?],
                1,
                model.n_layer() - 1,
            )?;
        } else {
            ctx.control_vector_clear()?;
        }
        ctx.decode(&mut batch)?;
        let captured = ctx.take_hidden_states()?;
        for (a, b) in baseline.iter().zip(&captured) {
            assert_close(&a.values, &b.values);
        }
        assert_close(&plain_logits, ctx.get_logits());
    }
    // Disabled prefix capture, then singleton final-token capture. No row index
    // from a pruned final layer is interpreted as a full-sequence token index.
    ctx.clear_kv_cache();
    ctx.decode(&mut prefix)?;
    ctx.decode(&mut last)?;
    let enabled_prefix = ctx.take_hidden_states()?;
    assert_close(&plain_split_logits, ctx.get_logits());
    ctx.clear_kv_cache();
    ctx.set_hidden_state_capture_enabled(false)?;
    ctx.decode(&mut prefix)?;
    assert_eq!(
        ctx.take_hidden_states(),
        Err(HiddenStateCaptureError::NoCapture)
    );
    ctx.set_hidden_state_capture_enabled(true)?;
    ctx.decode(&mut last)?;
    let singleton = ctx.take_hidden_states()?;
    eprintln!("checking disabled-prefix singleton equivalence");
    for (a, b) in enabled_prefix.iter().zip(&singleton) {
        assert_close(&a.values, &b.values);
    }
    assert_close(&plain_split_logits, ctx.get_logits());
    // CPU also qualifies cross-shape equivalence at the original tolerance.
    // Metal chooses different batched/singleton kernels; this is diagnostic,
    // not a claim that capture can make those kernels numerically identical.
    for (a, b) in baseline.iter().zip(&singleton) {
        if gpu_layers == 0 {
            assert_close(&a.values, &b.values);
        } else {
            let drift = a
                .values
                .iter()
                .zip(&b.values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "batch/singleton residual layer={} max_abs_delta={drift}",
                a.layer
            );
        }
    }
    // A final-layer-only control must be included in that layer's capture,
    // while earlier layer captures remain unmodified.
    ctx.clear_kv_cache();
    let mut controls = vec![0.0; width * usize::try_from(model.n_layer() - 1)?];
    controls[(usize::try_from(model.n_layer())? - 2) * width] = 0.25;
    ctx.control_vector_set(&controls, model.n_layer() - 1, model.n_layer() - 1)?;
    ctx.decode(&mut batch)?;
    let controlled = ctx.take_hidden_states()?;
    eprintln!("checking final-layer +0.25 control");
    for (a, b) in baseline.iter().zip(&controlled) {
        let mut expected = a.values.clone();
        if a.layer == model.n_layer() - 1 {
            expected[0] += 0.25;
        }
        assert_close(&expected, &b.values);
    }
    ctx.control_vector_clear()?;
    // Changing the final token must replace, not reuse, the previous snapshot.
    ctx.clear_kv_cache();
    let changed_tokens = model.str_to_token("A different final token!", AddBos::Always)?;
    let mut changed = LlamaBatch::new(changed_tokens.len(), 1);
    changed.add_sequence(&changed_tokens, 0, false)?;
    ctx.decode(&mut changed)?;
    let changed = ctx.take_hidden_states()?;
    assert_ne!(baseline[0].values, changed[0].values);
    eprintln!(
        "capture verified: layers={layers:?}, width={width}, zero/clear controls and logits agree"
    );
    Ok(())
}
