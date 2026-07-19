# Implementation Status

This document is the source of truth for what the `main` branch implements and
what still requires hardware evidence. Future work belongs in
[roadmap.md](roadmap.md).

## Supported Shape

- decode only;
- Llama-style MHA/GQA;
- FP16 or BF16 KV;
- one model process and one remote attention process;
- two GPUs in one host;
- CUDA Graph disabled;
- complete sealed prefix blocks plus one local active tail.

The M2 remote-prefix request is
`Attend(Q, KvView, no-append, layout, causal-mask, scale) -> (O, LSE)`. Historical
KV remains on GPU 1. K_new/V_new stay with the local active tail; the exact
merge combines its attention state with the remote sealed-prefix state.

## Implementation Map

| Area | Implemented | Missing evidence or integration |
| --- | --- | --- |
| Rust runtime | `KvPool`, Holt catalog, planner, leases, generation-pinned `KvView`, transport handles | production pool and device transport adapters |
| vLLM | local `CUSTOM` delegate, metadata observer, and real-model Modal L4 report | physical block to `PoolObjectRef` mapping |
| Attention state | Rust reference, contiguous PyTorch/FlashInfer paths, and generation-pinned FlashInfer paged executor | external engine/pool page-table binding |
| One-node data path | NCCL Route-Q and Stage-KV harness with contiguous and paged modes; phase-instrumented Modal L4 prefix sweep | Nsight attribution and topology-comparable bare-metal sweep |
| Cross-node data path | contracts only | NIXL/UCX/GPUDirect RDMA implementation and measurements |

The installable Python package lives under `python/src/loom_attention`, while
tests live under `python/tests`. Reusable attention-state kernels stay in the
core package. CUDA smoke tests and benchmark orchestration live under
`python/tests/integration` and are excluded from the wheel. Engine integration
remains in the `vllm_plugin`, `local_delegate`, and `step_metadata` modules.

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

The `integration.vllm_smoke` module runs native and delegated backends in
isolated processes, requires exact generated token equality, checks sampled
logprobs within a fixed tolerance, and writes a hardware/version-qualified JSON
report. The July 20 Modal L4 run passed on vLLM 0.25.0 and Torch 2.11.0+cu130:
the maximum logprob delta was 0.0, and Loom telemetry recorded 30 layer
implementations, 1,050 forwards, zero failures, and metadata generation 34. The
complete report is
[modal-vllm-l4-2026-07-20.json](results/modal-vllm-l4-2026-07-20.json).

This proves real engine entry, metadata capture, native-kernel delegation, and
output equality. Its timings are not comparative performance evidence because
the native process paid cold download/initialization first. The adapter still
does not map vLLM physical block IDs to external `PoolObjectRef` values or
install the snapshot in the Rust runtime. Remote attention and split-KV
execution begin at M2.

## M2a Two-GPU Data-Path Gate

`integration.two_gpu_smoke` launches two exclusive CUDA processes with an NCCL
process group. Rank 0 acts as the model worker and owns Q plus the active tail.
Rank 1 owns the sealed prefix. The Route-Q path sends Q to rank 1, returns
an output tensor plus FP32 log-sum-exp values, and merges them with the
local-tail attention state. The result is compared with full attention over
the concatenated prefix and tail.

The same processes then run a Stage-KV baseline that sends prefix K/V from rank
1 to rank 0. The JSON report records p50/p99 latency, payload bytes, GPU/NCCL
versions, peer-access capability, workload shape, and correctness error.

The harness performs real CUDA computation and NCCL transfers. Its default
attention kernel is a PyTorch `einsum` output-plus-LSE reference. The
`flashinfer` mode runs `single_decode_with_kv_cache` over contiguous NHD KV. The
`flashinfer-paged` mode runs `BatchDecodeWithPagedKVCacheWrapper` over NHD pages
and uses `merge_states` for the local-tail merge. Both are checked against the
same independent full-attention reference.

## M2b Paged-KV Executor

`FlashInferPagedExecutor` consumes a device-resident `PagedKvView` containing a
logical table id, positive page-table generation, covering lease ids, page
indices, indptr, last-page lengths, layout, and page size. It rejects invalid
shape, dtype, layout, lease, generation, and cross-device contracts before
launch. The executor owns its zero-initialized workspace and reuses a planned
FlashInfer wrapper while table identity, generation, leases, shape, dtype,
device, and scale remain unchanged. Execution returns contiguous output and
FP32 LSE tensors without reading page-table values on the host.

The two-GPU gate now has a paged mode for Route-Q and Stage-KV. Its deterministic
fixture converts contiguous generated inputs into pages once before warmup;
measured iterations consume or receive into those pages directly. This proves
the acceptance-path shape but is not an external-pool zero-copy result.

The first Linux report was produced on Modal using two L4 GPUs, PyTorch
2.9.1+cu128, FlashInfer 0.6.15, and NCCL 2.27.5. For a 4K FP16 prefix, 16-token
tail, one decode row, 10 warmups, and 100 measured iterations, both Route-Q and
Stage-KV passed the independent reference at `atol=rtol=2e-3`. Route-Q measured
0.514 ms p50 and 0.753 ms p99 while transferring 16,512 bytes; Stage-KV measured
1.866 ms p50 and 1.906 ms p99 while transferring 16,777,216 bytes. The complete
report is [modal-l4-4k-2026-07-19.json](results/modal-l4-4k-2026-07-19.json).

This is one environment, not a general performance claim. That container
reported `device_peer_access=false`, and gVisor denied the NVML topology query,
so the result does not characterize NVLink, GPUDirect RDMA, or bare-metal PCIe.

The phase-instrumented July 20 sweep added CUDA-event measurements for remote
attention, local-tail attention, merge, and Stage-KV attention. It labels the
end-to-end remainder as a communication/queue/framework residual rather than
pure transfer time. Both ranks run the same fixed FP16 GEMM preconditioner before
each timed path. For 4K/8K/16K/32K prefixes, Route-Q p50 was
0.635/0.658/0.643/0.696 ms, while Stage-KV p50 grew nearly linearly at
1.780/3.407/6.697/13.432 ms. A reverse-order run reproduced the trend within
about 5% for Route-Q and 3% for Stage-KV. The complete reports are the
[forward sweep](results/modal-l4-prefix-sweep-2026-07-20.json) and
[reverse sweep](results/modal-l4-prefix-sweep-reverse-2026-07-20.json).

Fine-grained local-tail and merge timings remain sensitive to runtime state.
Mechanism-level attribution still requires a bare-metal Nsight trace. The
current residual also combines NCCL transfer, queueing, synchronization, and
framework overhead.

## Correctness Gate

For fixed Q/K/V tensors, compare:

1. one local reference attention over all KV;
2. local-tail state plus remote-prefix state;
3. exact output-plus-LSE merge.

The test must cover unequal shard lengths, extreme logits, multiple heads,
batched decode rows, GQA head mapping, empty tail, lease expiry, worker restart,
and layout mismatch.

## Performance Gate

Measure context lengths from 4K through the largest feasible configuration and
report p50/p99 TTFT, p50/p99 TPOT, tokens/s, SLO goodput, Q/O bytes, KV bytes
avoided, remote queue time, kernel time, merge time, and GPU utilization.

Baselines are local-only attention, fetch-KV-then-local, static route-Q, and the
dynamic planner under the same model, batch, prefix trace, and hardware.
