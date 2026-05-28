FROM rust:1.82 as builder

ENV MLIR_SYS_220_PREFIX=/usr/lib/llvm-22
ENV LLVM_SYS_220_PREFIX=/usr/lib/llvm-22

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      cmake \
      gnupg \
      lsb-release \
      ninja-build \
      software-properties-common \
      wget \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 22 all \
    && apt-get install -y --no-install-recommends \
      clang-22 \
      libmlir-22-dev \
      mlir-22-tools \
      llvm-22-dev \
    && rm -rf /var/lib/apt/lists/* llvm.sh

# Install and use the nightly toolchain to support Edition 2024 dependencies
RUN rustup toolchain install nightly
RUN rustup default nightly
WORKDIR /app

# Pre-cache deps
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src/bin && echo "fn main(){}" > src/bin/dummy.rs && cargo build --release || true

# Build
COPY . .
RUN cargo build --release --bin server

FROM debian:bookworm-slim

ENV MLIR_SYS_220_PREFIX=/usr/lib/llvm-22
ENV LLVM_SYS_220_PREFIX=/usr/lib/llvm-22

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      gnupg \
      lsb-release \
      software-properties-common \
      wget \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 22 all \
    && apt-get install -y --no-install-recommends \
      libmlir-22 \
      libllvm22 \
    && useradd --system --create-home --uid 10001 quill \
    && rm -rf /var/lib/apt/lists/* llvm.sh

USER quill
WORKDIR /app
COPY --from=builder /app/target/release/server /usr/local/bin/server
COPY --from=builder /app/public /app/public
COPY --from=builder /app/docs /app/docs
ENV QUILL_HTTP_ADDR=0.0.0.0:8080
EXPOSE 8080
CMD ["/usr/local/bin/server"]

