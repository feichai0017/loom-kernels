# Greedy Speculative Verification

Loom's first speculative-decoding boundary is deterministic verification and
token compaction. Draft generation, target-model execution, attention, GEMM,
sampling policy, and KV-cache ownership remain in the inference engine.

## Contract

The operator matches the flattened ragged layout used by vLLM 0.24/0.25:

| Buffer | Dtype and layout | Ownership |
| --- | --- | --- |
| draft token IDs | contiguous int32 `[total_drafts]` | caller-owned, read-only |
| target argmax IDs | contiguous int64 `[total_drafts]` | caller-owned, read-only |
| bonus token IDs | contiguous int32 `[requests, 1]` | caller-owned, read-only |
| cumulative draft lengths | inclusive contiguous int32 `[requests]` | caller-owned, read-only |
| output token IDs | contiguous int32 `[requests, max_drafts + 1]` | caller-owned, written by Loom |
| accepted lengths | contiguous int32 `[requests]` | caller-owned, written by Loom |
| emitted lengths | contiguous int32 `[requests]` | caller-owned, written by Loom |

The flattened batch must contain at least one draft token, while individual
requests may contain zero. Each cumulative segment is nondecreasing, no segment
exceeds `max_drafts`, and the final boundary equals `total_drafts`. Token IDs
must fit int32 because the engine output contract is int32.

For each request, Loom emits:

1. the longest draft prefix equal to the target argmax IDs;
2. the first mismatching target ID, if a mismatch exists; otherwise
3. the target-model bonus token.

Unused output positions are `-1`. `accepted_lengths` excludes the mismatch or
bonus token; `emitted_lengths` includes it.

## One Execution Path

```text
PyTorch/vLLM tensors
  -> boxed LibTorch Stable ABI mutation op
  -> versioned loom-cuda-bridge entrypoint
  -> checked borrowed Rust views
  -> safe CudaBackend method
  -> internal CUDA launch ABI
  -> one-warp-per-request CUDA kernel
```

The Python convenience API makes one combined int32 allocation and exposes
three disjoint views for output IDs and the two length arrays. The bridge
validates shape-derived spans, alignment, and non-overlap before asynchronous
launch on PyTorch's current stream. The Rust CPU oracle validates cumulative
metadata and every output buffer before mutation.

The CUDA kernel fills each output row, reduces the first mismatch within one
warp, compacts accepted tokens, and writes both lengths. It allocates no hidden
workspace and performs no host synchronization.

## vLLM Boundary

`register_vllm_greedy_speculative_verify()` replaces only vLLM's standard
`rejection_sample` all-greedy branch. vLLM still computes target argmax,
constructs ragged metadata, chooses the bonus token, and owns every
non-greedy or synthetic path.

Registration is explicit. Current evidence closes correctness, framework
compatibility, operator latency, and real draft/target engine invocation, but
not a model-level speedup. Unsupported policy or tensor contracts execute
vLLM's original implementation.

## Deliberate Exclusions

- stochastic residual-distribution rejection and RNG;
- tree or branch attention-mask construction;
- draft or target model execution;
- verification attention or any GEMM;
- KV-cache commit, rollback, or remapping;
- scheduler policy and host-visible sequence state.

Those boundaries become Loom operators only when a named engine call site can
preserve ownership and show an end-to-end benefit.

## Evidence

The [H20 result](../results/h20-greedy-speculative-verify-20260723.json)
compares the public Loom call with vLLM 0.24's exact greedy Triton verifier,
including output allocation and Python dispatch. All 15 batch/draft shapes are
bit-exact; the measured ratio is `1.101-1.128x`. This is operator-level
evidence.

The order-reversed [native-first](../results/h20-vllm-qwen25-speculative-native-first-20260723.json)
and [Loom-first](../results/h20-vllm-qwen25-speculative-loom-first-20260723.json)
reports use a pinned Qwen2.5-1.5B target, Qwen2.5-0.5B draft, four draft
tokens, natural prompts, and vLLM 0.24. Native and Loom speculative providers
match every output token and acceptance counter. Each Loom process records
`714/714` measured verifier launches, while the other processes record zero.
Loom reduces the profiled verifier boundary by `2.6-11.7%`, but that boundary
is only `0.048-0.200%` of batch latency. End-to-end native/Loom ratios cross
parity when provider order reverses, and both speculative providers are
`3.18-4.97x` slower than target-only for these cases. The real-engine path is
therefore proven, while the milestone's performance exit remains open.

Target-only uses a different target-model execution shape. Two of 32
target-only requests diverge from both speculative providers after token 51
or 53; the speculative providers remain mutually exact in both orders. The
reports preserve this numerical-path difference instead of treating
target-only as the verifier-replacement correctness oracle.

vLLM's dummy sampler warm-up deliberately sets `all_greedy=False`, so lifetime
fallback telemetry contains one unsupported-contract observation. The gate
uses the measured-call delta after warm-up: every measured rejection call
reaches Loom. Optional CUDA-event profiling runs only after primary timing.
