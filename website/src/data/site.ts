export const repositoryUrl = "https://github.com/feichai0017/loom-kernels";

export const navigation = [
  { label: "Overview", href: "/" },
  { label: "Operators", href: "/docs/operators/" },
  { label: "Integration", href: "/docs/integration/" },
  { label: "Evidence", href: "/benchmarks/" },
];

export const supportedOperators = [
  {
    name: "RMSNorm",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Standalone normalization",
    status: "supported",
  },
  {
    name: "Add + RMSNorm",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Residual update + normalization",
    status: "supported",
  },
  {
    name: "RMSNorm + dynamic FP8",
    dtypes: "F32 · FP16 · BF16 → E4M3FN",
    boundary: "Normalization + GEMM input quantization",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Split-half SwiGLU activation",
    status: "supported",
  },
  {
    name: "SiLU-and-Mul + block FP8",
    dtypes: "FP16 · BF16 → E4M3FN",
    boundary: "Activation + group-64/128 quantization",
    status: "supported",
  },
  {
    name: "RoPE + paged-KV write",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Packed Q/K rotation + native cache write",
    status: "supported",
  },
  {
    name: "Greedy + sampled logprob",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Argmax + logsumexp + gather + tie rank",
    status: "supported",
  },
  {
    name: "Selected-token logprob + rank",
    dtypes: "F32 · FP16 · BF16",
    boundary: "Engine-owned sampling + one-pass normalization",
    status: "supported",
  },
  {
    name: "Min-P filtering",
    dtypes: "F32 · FP16 · BF16",
    boundary: "In-place row-max threshold; shape-gated in vLLM",
    status: "supported",
  },
];

export const nextOperators = [
  {
    name: "Loom-owned sampling",
    reason: "Build from shape-gated Min-P into top-k/top-p and deterministic RNG where profiling pays.",
  },
  {
    name: "MoE routing + movement",
    reason: "Own the launch-heavy path around vendor grouped GEMM.",
  },
  {
    name: "Paged decode attention",
    reason: "Rust contract and oracle are fixed; next qualify CUDA against engine-owned paged KV.",
  },
];

export const evidence = [
  {
    operator: "Add + RMSNorm",
    shape: "BF16 · 8 × 4096",
    result: "2.914 µs",
    detail: "Raw H20 kernel median",
  },
  {
    operator: "RMSNorm + FP8",
    shape: "BF16 · 8 × 4096",
    result: "1.057–1.076×",
    detail: "CUDA Graph ratio vs vLLM",
  },
  {
    operator: "SiLU + Mul + FP8",
    shape: "BF16 · 8 × 11008 · G128",
    result: "1.037–1.082×",
    detail: "CUDA Graph ratio vs vLLM fused",
  },
  {
    operator: "Qwen2.5 FP8 engine",
    shape: "0.5B · batches 1 / 8 / 32",
    result: "0.999–1.004×",
    detail: "Exact-token path hit; end-to-end parity",
  },
  {
    operator: "RoPE + paged-KV write",
    shape: "BF16 · Qwen2.5-style · 1–512 tokens",
    result: "2.30–2.40×",
    detail: "Dispatcher ratio vs separate vLLM ops",
  },
  {
    operator: "Greedy + sampled logprob",
    shape: "Qwen2.5-0.5B · batches 1 / 8 / 32",
    result: "1.129–1.250×",
    detail: "Order-stable real-engine batch-latency ratio",
  },
  {
    operator: "Selected-token logprob + rank",
    shape: "Qwen2.5 top-k/top-p · batches 1 / 8 / 32",
    result: "1.044–1.125×",
    detail: "vLLM-owned sampling; order-stable engine ratio",
  },
  {
    operator: "Min-P filtering",
    shape: "F32 · 151,936 vocab · 128 rows",
    result: "1.885×",
    detail: "0 tensor-sized temp; smaller batches route back to vLLM",
  },
];
