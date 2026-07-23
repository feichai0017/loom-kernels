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
  {
    name: "Paged MQA/GQA decode",
    dtypes: "F32 · FP16 · BF16",
    boundary: "GQA packing + local split-K/LSE; short shapes route into vLLM",
    status: "supported",
  },
];

export const nextOperators = [
  {
    milestone: "K4.5 · P0",
    name: "Speculative decoding",
    reason: "Prepare batched verification metadata, deterministic accept/reject, and token compaction around vendor attention and GEMM.",
  },
  {
    milestone: "K3 · P0",
    name: "KV-cache compression + movement",
    reason: "Add FP8 cache scales and scheduler-facing block movement that improve admitted context, batch size, or prefix/preemption cost.",
  },
  {
    milestone: "K4 · P0",
    name: "Complete sampling tail",
    reason: "Own penalties, top-k/top-p, deterministic RNG, and top-k logprobs without host round trips.",
  },
  {
    milestone: "K2.5 · P1",
    name: "Quantization plumbing",
    reason: "Remove scale, pack/unpack, dequant/requant, and layout passes around an unchanged vendor GEMM.",
  },
  {
    milestone: "K5 · P1",
    name: "MoE routing + movement",
    reason: "Own routing, histogram/prefix sum, permutation, and combine while grouped GEMM stays vendor-owned.",
  },
  {
    milestone: "K8 · Proof",
    name: "Rust decode step",
    reason: "Prove zero-copy engine-neutral orchestration over borrowed tensors and streams without building an inference engine.",
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
  {
    operator: "Paged MQA/GQA decode",
    shape: "FP16/BF16 · Hq/Hkv 32/8 · context ≤ 32",
    result: "1.154–2.374×",
    detail: "24/24 routed vLLM backend cases win; other shapes fall back to FA3",
  },
  {
    operator: "Paged decode split-K/LSE",
    shape: "BF16 · batch 1–8 · context 128–1,024",
    result: "1.14–6.22×",
    detail: "CUDA Graph ratio vs legacy Loom; FA3 remains the engine fallback",
  },
];
