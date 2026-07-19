# Distributed Attention MVP

## Supported Shape

- decode only;
- Llama-style MHA/GQA;
- FP16 or BF16 KV;
- one model process and one remote attention process;
- two GPUs in one host;
- CUDA Graph disabled;
- complete sealed prefix blocks plus one local active tail.

The M2 remote-prefix request is
`Attend(Q, KvView, no-append, layout, causal-mask, scale) -> partial`. Historical
KV remains on GPU 1. K_new/V_new stay with the local active tail; the exact
merge combines its partial with the remote sealed-prefix partial.

## M1 Engine-Local Baseline

The first engine adapter is an out-of-tree vLLM V1 `CUSTOM` attention backend.
It wraps vLLM FlashAttention, validates the tensor and head layout on the first
forward, records process-local call telemetry, and delegates the operation
unchanged. Therefore the real attention computation remains on the GPU inside
vLLM; the Rust `f32` implementation is only a correctness reference.

The adapter also wraps `FlashAttentionMetadataBuilder`. Once per metadata build,
it records request boundaries from vLLM's existing CPU offsets and opaque
descriptors for the device-side block table, slot mapping, sequence lengths,
and query offsets. Block-table updates advance a snapshot generation. Device
tensor values are never copied to CPU by this observer.

The current adapter does not map vLLM physical block IDs to external
`PoolObjectRef` values or install the snapshot in the Rust runtime yet, and it
has not decoded a real model. The `attnarc-vllm-smoke` command is the GPU
acceptance harness: it runs native and delegated backends in isolated processes,
requires exact generated token equality, checks sampled logprobs within a fixed
tolerance, and writes a hardware/version-qualified JSON report. The harness is
implemented, but no CUDA report has been produced on the current macOS host.
Those remain M1 exit conditions. Remote attention and split-KV execution begin
at M2.

## M2a Two-GPU Data-Path Gate

`attnarc-two-gpu-smoke` launches two exclusive CUDA processes with an NCCL
process group. Rank 0 acts as the model worker and owns Q plus the active tail.
Rank 1 owns the sealed prefix. The Route-Q path sends Q to rank 1, returns
float32 `(max, exp_sum, weighted_value)` partials, and merges them with the
local-tail partial. The result is compared with full attention over the
concatenated prefix and tail.

The same processes then run a Stage-KV baseline that sends prefix K/V from rank
1 to rank 0. The JSON report records p50/p99 latency, payload bytes, GPU/NCCL
versions, peer-access capability, workload shape, and correctness error.

The harness performs real CUDA computation and NCCL transfers, but its
attention kernel is a PyTorch `einsum` online-softmax reference. It validates
the protocol and performance crossover, not production kernel throughput. M2b
replaces that executor with a paged FlashInfer or native vLLM kernel.

## Correctness Gate

For fixed Q/K/V tensors, compare:

1. one local reference attention over all KV;
2. local-tail partial plus remote-prefix partial;
3. exact online-softmax merge.

The test must cover unequal shard lengths, extreme logits, multiple heads,
batched decode rows, GQA head mapping, empty tail, lease expiry, worker restart,
and layout mismatch.

## Performance Gate

Measure context lengths from 4K through the largest feasible configuration and
report p50/p99 TTFT, p50/p99 TPOT, tokens/s, SLO goodput, Q/O bytes, KV bytes
avoided, remote queue time, kernel time, merge time, and GPU utilization.

Baselines are local-only attention, fetch-KV-then-local, static route-Q, and the
dynamic planner under the same model, batch, prefix trace, and hardware.
