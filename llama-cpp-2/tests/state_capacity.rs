use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::LlamaStateSeqFlags;
use std::num::NonZeroU32;

#[test]
#[ignore = "requires LLAMA_TEST_MODEL pointing to a real GGUF model"]
fn sequence_export_respects_destination_capacity() {
    let path = std::env::var_os("LLAMA_TEST_MODEL").expect("LLAMA_TEST_MODEL is required");
    let backend = LlamaBackend::init().expect("backend");
    let model = LlamaModel::load_from_file(
        &backend,
        path,
        &LlamaModelParams::default().with_n_gpu_layers(0),
    )
    .expect("model");
    let mut context = model
        .new_context(
            &backend,
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(128)),
        )
        .expect("context");
    let tokens = model.str_to_token("Hello", AddBos::Always).expect("tokens");
    let mut batch = LlamaBatch::new(128, 1);
    for (index, token) in tokens.iter().enumerate() {
        batch
            .add(*token, i32::try_from(index).expect("position"), &[0], true)
            .expect("batch");
    }
    context.decode(&mut batch).expect("decode");
    let flags = LlamaStateSeqFlags::empty();
    let size = context.state_seq_get_size_ext(0, flags);
    assert!(size > 1);
    // Keep the backing allocation large enough even under the original bug.
    // Bytes outside the offered slice must remain untouched.
    for capacity in [0, 1, size - 1] {
        let mut guarded = vec![0xa5; size + 32];
        assert_eq!(
            context.state_seq_get_data_ext(&mut guarded[..capacity], 0, flags),
            0
        );
        assert!(guarded[capacity..].iter().all(|byte| *byte == 0xa5));
    }
    let mut exact = vec![0; size];
    assert_eq!(context.state_seq_get_data_ext(&mut exact, 0, flags), size);
    let state = context.state_seq_get(0, flags).expect("opaque export");
    context.state_seq_set(&state, 0).expect("restore");
    assert_eq!(
        context
            .state_seq_get(0, flags)
            .expect("reexport")
            .byte_len(),
        size
    );
}
