//! Propagates native backend discovery from the sys crate into the safe wrapper.

fn main() {
    if let Ok(dir) = std::env::var("DEP_LLAMA_BACKENDS_DIR") {
        println!("cargo:rustc-env=GGML_BACKENDS_DIR={dir}");
    }
}
