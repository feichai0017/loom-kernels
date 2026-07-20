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
| vLLM | local `CUSTOM` delegate, metadata observer, physical-block `KVConnector` bridge, and real-model Modal L4 reports | pool lookup/load and physical block to `PoolObjectRef` mapping |
| Attention state | Rust reference, contiguous PyTorch/FlashInfer paths, and generation-pinned FlashInfer paged executor | external-pool objects installed into the engine page table |
| Local-tail CUDA | opt-in Rust/C ABI/PyTorch FP16/BF16 tail, exact merge, and fused-tail executor; isolated H20 report | real two-GPU A/B, Nsight attribution, vLLM dispatch, and CUDA Graph coverage |
| One-node data path | NCCL Route-Q and Stage-KV harness with contiguous and paged modes; sequential/overlap/fused Route-Q strategies; phase-instrumented Modal L4 prefix sweep | overlap/fused two-GPU measurement and topology-comparable bare-metal sweep |
| Cross-node data path | contracts only | NIXL/UCX/GPUDirect RDMA implementation and measurements |

The installable Python package lives under `python/src/loom_attention`, while
tests live under `python/tests`. Reusable attention-state kernels stay in the
core package. CUDA smoke tests and benchmark orchestration live under
`python/tests/integration` and are excluded from the wheel. Engine integration
remains in the `vllm_plugin`, `vllm_connector`, `vllm_binding`,
`block_binding`, `local_delegate`, and `step_metadata` modules.

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

## M1b Physical-Block Bridge

vLLM does not retain a complete CPU mirror of its GPU attention block table.
Reading that table in `AttentionImpl.forward` would add a device-to-host
synchronization to every layer. Loom therefore uses vLLM's official
`KVConnector` lifecycle instead: scheduler output carries CPU physical block
allocations, and `register_kv_caches` provides the worker's actual per-layer
CUDA cache tensors.

`BlockBindingRegistry` mirrors new-request replacement, running-request append,
preemption/resume replacement, completion, and physical-slot reuse. Every step
advances a generation. A worker activates that generation while connector
metadata is bound; each CUSTOM attention forward validates it before entering
the native kernel. External bindings require an exact object generation,
matching layout digest, and unexpired read lease. Allocation updates invalidate
older bindings for reused physical slots without reading the GPU table.

The metadata-only connector returns zero external matches and performs no data
movement. CPU CI covers its state transitions, tensor registration, no-device-
readback rule, stale generation checks, lease checks, and same-step forward
validation.

The July 20 connector run passed on the same L4/vLLM 0.25.0/Torch 2.11.0+cu130
environment. It preserved exact token IDs and a 0.0 maximum logprob delta while
registering 30 real CUDA KV cache tensors, consuming 36 scheduler metadata
steps and eight request block updates, observing four physical block IDs, and
validating 960 attention forwards against an active binding generation. The
remaining initialization/profile forwards intentionally had no request binding.
The complete report is
[modal-vllm-l4-binding-2026-07-20.json](results/modal-vllm-l4-binding-2026-07-20.json).
Mooncake lookup, transfer, and `PoolObjectRef` installation remain M3 work.

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

The engine side now exposes three explicit Route-Q strategies. `sequential`
preserves the measured baseline. `overlap` launches local-tail attention on a
side stream after sending Q and waits on that stream only before merge.
`fused` waits for the remote state and then invokes the optional handwritten
CUDA tail-plus-merge operator. The report uses the matching critical-path
residual definition for each strategy. Only `sequential` has a real two-GPU
performance report so far.

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

## M2c Handwritten Local-Tail Fusion

The optional CUDA implementation shares one dependency-light C ABI between a
raw Rust binding package and a PyTorch custom operator. The checked Rust
executor validates device, owner, generation, address range, byte bounds, shape,
dtype, and non-aliasing outputs before submitting to a caller-owned CUDA stream.
The current kernels support FP16/BF16 GQA, `head_dim <= 256`, and
`tail_tokens <= 64`, with FP32 accumulation.

One baseline kernel computes and materializes the local-tail output/LSE; a
second merges it with the remote state. The fused kernel instead computes local
logits and the final output/LSE directly. On one NVIDIA H20 with CUDA 13.1, the
preallocated CUDA-event microbenchmark measured 1.114x-1.178x median speedup
across FP16/BF16 and 1/4/16 decode rows at 32 query heads, eight KV heads,
head-dim 128, and a 16-token tail.

The PyTorch gate compared against full reference attention over a 512-token
prefix plus the tail. At four rows, maximum output error was `7.63e-6` for FP16
and `6.10e-5` for BF16; maximum LSE error was `4.77e-7`. Single-GPU emulation
also validated sequential, side-stream overlap, and fused scheduling semantics.
The complete report is
[h20-fused-tail-2026-07-20.json](results/h20-fused-tail-2026-07-20.json).

This is isolated operator evidence, not an end-to-end Route-Q, TPOT, or model
result. `forge-gas1` exposes one GPU, so it cannot prove real NCCL/local-tail
overlap or compare the three strategies under two-GPU contention. Production
vLLM dispatch, CUDA Graph capture, masks, larger tails, and Nsight validation
remain open.

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
