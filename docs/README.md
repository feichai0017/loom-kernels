# Documentation

- [Operator library design](design/operator-library.md): architecture and
  admission gates.
- [Paged decode attention contract](design/paged-decode-attention.md): the
  engine-owned KV boundary, base semantics, exclusions, and qualification plan.
- [LLM inference operator catalog](operator-catalog.md): complete intended
  common-operator surface, scope, priority, and current state.
- [Roadmap](roadmap.md): prioritized operator sequence and exit criteria.
- [Implementation status](status.md): what is implemented and validated now.
- [vLLM IR provider guide](guides/vllm-ir-provider.md): build, load, select,
  test, and benchmark the engine adapter.
- [H20 F32 RMSNorm report](results/h20-rms-norm-f32-smoke-20260721.json):
  hardware-qualified bring-up evidence.
- [H20 FP16/BF16 RMSNorm report](results/h20-rms-norm-low-precision-20260721.json):
  pair-vectorized and odd-size fallback evidence.
- [H20 fused Add+RMSNorm report](results/h20-add-rms-norm-20260721.json):
  double in-place, multi-dtype, and odd-size evidence.
- [H20 vLLM IR integration report](results/h20-vllm-ir-add-rms-norm-20260721.json):
  named baseline, PyTorch bridge, CUDA Graph, and engine-run evidence.
- [H20 RMSNorm+dynamic-FP8 report](results/h20-rms-norm-dynamic-fp8-20260721.json):
  multi-dtype bitwise compatibility, raw CUDA, and order-reversed vLLM evidence.
- [H20 SiLU-and-Mul report](results/h20-silu-and-mul-20260721.json):
  multi-dtype compatibility, graph parity, eager instability, and vLLM engine
  smoke evidence.
- [H20 SiLU-and-Mul+dynamic-block-FP8 report](results/h20-silu-and-mul-dynamic-fp8-20260721.json):
  exact fused-vLLM compatibility, raw CUDA, compiler-boundary, and
  order-reversed named-baseline evidence.
- [H20 Qwen2.5 FP8 engine report](results/h20-vllm-qwen25-05b-fp8-engine-20260722.json):
  pinned pretrained checkpoint, compiler and launch path evidence, exact-token
  generation, and order-reversed end-to-end parity.
- [H20 greedy sampled-logprob operator report](results/h20-greedy-sample-logprobs-20260722.json):
  exact token/rank gates and the 1-128 row Qwen-vocabulary microbenchmark.
- [H20 greedy sampled-logprob engine report, baseline first](results/h20-vllm-greedy-logprobs-baseline-first-20260722.json):
  pinned Qwen2.5 path hits, exact outputs, and real-engine latency/TPOT benefit.
- [H20 greedy sampled-logprob engine report, Loom first](results/h20-vllm-greedy-logprobs-loom-first-20260722.json):
  reverse-order confirmation of the same correctness and performance result.
- [H20 selected-token logprob operator report](results/h20-selected-token-logprobs-20260722.json):
  arbitrary selected ranks, exact vLLM rank parity, and 1-128 row latency.
- [H20 min-p filter report](results/h20-min-p-filter-20260722.json): exact F32
  masks, temporary-memory removal, crossover point, and vLLM routing decision.
- [H20 min-p 65,536-vocabulary boundary report](results/h20-min-p-filter-vocab65536-20260722.json):
  direct evidence for the lower vocabulary gate at 32 and 128 rows.
- [H20 paged decode-attention report](results/h20-paged-decode-attention-20260722.json):
  randomized correctness plus the batch/context crossover against vLLM 0.24
  FlashAttention FA3.
- [H20 top-k/top-p selected-logprob engine report, baseline first](results/h20-vllm-selected-logprobs-baseline-first-20260722.json):
  vLLM-owned sampling with exact tokens/ranks and end-to-end latency evidence.
- [H20 top-k/top-p selected-logprob engine report, Loom first](results/h20-vllm-selected-logprobs-loom-first-20260722.json):
  reverse-order exact-output and end-to-end latency evidence.

Only results under `docs/results` count as performance evidence. A CPU test, a
successful CUDA launch, or an isolated number without a named baseline is not a
speedup claim.
