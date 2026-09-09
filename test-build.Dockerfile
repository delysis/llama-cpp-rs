ARG CUDA_VERSION=12.3.1
ARG UBUNTU_VERSION=22.04
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base-cuda

# Install requirements for rustup install + bindgen: https://rust-lang.github.io/rust-bindgen/requirements.html
RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y curl llvm-dev libclang-dev clang pkg-config libssl-dev cmake git
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --profile minimal --default-toolchain 1.92.0
ENV PATH=/root/.cargo/bin:$PATH

COPY . .
# This is a compile check for one representative GPU target, not a distributable
# fat binary for every CUDA architecture supported by llama.cpp.
ARG CUDA_ARCHITECTURES=75
RUN CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} cargo build --locked --bin simple --features cuda

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base-cuda-runtime

COPY --from=base-cuda /target/debug/simple /usr/local/bin/simple

ENTRYPOINT ["/usr/local/bin/simple"]
