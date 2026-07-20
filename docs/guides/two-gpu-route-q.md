# Two-GPU Route-Q Acceptance Gate

## Purpose

This gate validates the first physical Loom data path on one Linux host with
two NVIDIA GPUs. It answers two separate questions:

1. Does Q-only remote-prefix attention plus exact local-tail merge produce the
   same output as full attention?
2. At what prefix size does Route-Q beat transferring the historical KV?

It does not claim production attention-kernel performance.

## Topology

```mermaid
flowchart LR
    Engine["Rank 0 / GPU 0\nQ + active tail"]
    Worker["Rank 1 / GPU 1\nsealed prefix KV"]

    Engine -->|"Q via NCCL"| Worker
    Worker -->|"O + LSE state"| Engine
    Engine --> Merge["exact attention-state merge"]

    Worker -. "Stage-KV baseline: K and V" .-> Engine
```

Both ranks reconstruct identical deterministic tensors for the correctness
oracle. Only rank 1's prefix participates in the Route-Q execution. The extra
rank 0 copy is never included in the measured Route-Q path.

## Install

Use a CUDA-enabled PyTorch build with NCCL and two visible devices:

```bash
python3 -m pip install -e './python[cuda]'
```

To run the optimized contiguous or paged-KV kernel path, install the FlashInfer
extra:

```bash
python3 -m pip install -e './python[flashinfer]'
```

When running inside the vLLM environment, its existing compatible PyTorch
installation is sufficient:

```bash
python3 -m pip install -e ./python --no-deps
```

## Plan Without CUDA

```bash
PYTHONPATH=python/src:python/tests \
python3 -m integration.two_gpu_smoke plan \
  --prefix-tokens 4096 \
  --tail-tokens 16 \
  --rows 1 \
  --query-heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype float16
```

The command reports tensor payload bytes only. It excludes NCCL protocol,
launch, queueing, synchronization, and kernel costs.

The default correctness tolerance follows the attention-state wire dtype:
`2e-3` for FP16 and `2e-2` for BF16. Use `--atol` and `--rtol` to override it.

## Run

```bash
PYTHONPATH=python/src:python/tests \
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m integration.two_gpu_smoke run \
  --prefix-tokens 4096 \
  --tail-tokens 16 \
  --rows 1 \
  --query-heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --page-size 16 \
  --dtype float16 \
  --attention-backend flashinfer-paged \
  --route-strategy sequential \
  --warmup 10 \
  --iterations 100 \
  --report build/two-gpu-smoke/report-4k.json
```

Repeat at 4K, 8K, 16K, 32K, and the largest feasible prefix. Keep every other
argument fixed when comparing the two paths.

By default, both ranks execute a fixed 100-by-4096 FP16 GEMM preconditioner
before each timed path. This work is excluded from latency samples and controls
for accelerator clock state. Set `--precondition-iterations 0` to disable it,
and record that choice when comparing reports.

### Engine-Side Strategy

`--route-strategy` selects one of three engine-side schedules:

| Value | Behavior | Evidence |
| --- | --- | --- |
| `sequential` | receive remote state, compute local tail, merge | two-GPU Modal sweep |
| `overlap` | run local tail on a side stream while waiting for remote state, then merge | H20 stream/correctness gate; two-GPU latency open |
| `fused` | receive remote state, then run one handwritten CUDA tail-plus-merge kernel | H20 kernel/correctness gate; two-GPU latency open |

`fused` requires the optional PyTorch extension. Build it and review its shape
limits in the [Rust/CUDA fused-tail guide](rust-cuda-fused-tail.md). Keep the
backend, workload, warmup, and iteration count identical when comparing the
three strategies.

## Run On Modal

With a configured Modal profile, run the same gate on two co-located L4 GPUs:

```bash
uvx --from modal modal run python/tests/integration/modal_two_gpu.py \
  --prefix-tokens 4096 \
  --iterations 100 \
  --report build/modal/two-gpu-l4.json
```

Run a prefix sweep in one allocated container with:

```bash
uvx --from modal modal run python/tests/integration/modal_two_gpu.py \
  --prefix-sweep 4096,8192,16384,32768 \
  --warmup 10 \
  --iterations 100 \
  --report build/modal/two-gpu-l4-prefix-sweep.json
```

The function uses an ephemeral `L4:2` container and shuts down after writing the
report locally. The report includes the requested resource, GPU topology,
PyTorch/CUDA/FlashInfer versions, and the normal correctness and latency fields.
The Modal launcher remains under `python/tests/integration` and is excluded from
the installable wheel.

On a workstation that requires an HTTP or SOCKS proxy for outbound traffic,
install Modal's proxy extra and export the proxy variables before the same
command:

```bash
HTTPS_PROXY=http://127.0.0.1:7890 \
ALL_PROXY=socks5h://127.0.0.1:7890 \
uvx --from 'modal[api-proxy-support]' modal run \
  python/tests/integration/modal_two_gpu.py
```

## Route-Q Payload

Rank 0 sends Q. Rank 1 returns the attention output in the request dtype and
one FP32 log-sum-exp value for each row and query head:

```text
O_i = softmax(Q K_i^T) V_i
LSE_i = log(sum(exp(Q K_i^T)))
```

Rank 0 computes the same state over its active tail and merges each segment
with weights `exp(LSE_i - logsumexp(LSE))`. The returned payload is independent
of historical KV length and matches the contract exposed by optimized
attention kernels.

## Stage-KV Baseline

Rank 1 sends complete prefix K and V to preallocated buffers on rank 0. In
reference and contiguous FlashInfer modes, rank 0 concatenates the local tail
and computes full attention. In paged mode, prefix K/V are received directly
into preallocated page storage next to the tail pages and consumed without a
timed repack. This baseline transfers KV on every measured iteration; it does
not model amortization from retaining a staged copy across later decode tokens.

That limitation is intentional. Later experiments must add reuse horizon and
eviction probability to determine when one Stage-KV transfer amortizes over
multiple future tokens.

## Valid Report

A reviewable report must contain:

- `passed: true` under the configured `atol` and `rtol`;
- GPU names, compute capability, CUDA, NCCL, and PyTorch versions;
- peer-access capability;
- complete workload configuration;
- p50/p99 latency and payload bytes for both paths;
- CUDA-event kernel timings and the explicitly labelled non-kernel residual;
- explicit kernel, KV layout, paged-executor, and fixture-repack metadata.
- the selected Route-Q strategy and its matching phase fields.

The macOS development host cannot produce this report locally. The first Linux
report was produced through the Modal launcher on two L4 GPUs; broader topology
and workload coverage remains open.

`--attention-backend reference` uses the PyTorch oracle for every path.
`--attention-backend flashinfer` uses FlashInfer
`single_decode_with_kv_cache(..., return_lse=True)` and `merge_states` for the
measured paths, while full attention remains the independent PyTorch oracle.
This backend receives contiguous NHD KV.

`--attention-backend flashinfer-paged` uses
`BatchDecodeWithPagedKVCacheWrapper.plan/run` over NHD pages. Route-Q reads the
remote prefix pages, while Stage-KV receives directly into page storage. The
fixture is paged once before warmup because the deterministic input generator
starts from contiguous tensors; this is not evidence of an external-pool
zero-copy page-table bridge.

## Phase Timing Boundary

End-to-end latency uses a host monotonic clock bracketed by CUDA
synchronization. Remote attention, local-tail attention, merge, and Stage-KV
attention use CUDA events on their owning rank. Rank 1 sends its event durations
to rank 0 only after the measured Route-Q loop.

The report derives `communication_queue_framework_residual_ms` per sample from
the estimated critical path. Sequential subtracts remote attention, local-tail
attention, and merge in order. Overlap subtracts the larger of remote attention
and local-tail attention, then merge. Fused subtracts remote attention and the
combined tail-plus-merge kernel. The residual includes NCCL transport, queueing,
stream synchronization, Python/PyTorch overhead, and measurement overhead. It
is deliberately not named transfer time and must not be used as a pure
link-bandwidth result.

## Modal L4 Prefix Sweep

The July 20 sweep used two L4 GPUs, FP16, one decode row, a 16-token local tail,
10 workload warmups, 100 measured iterations, and the default GPU
preconditioner. All Route-Q and Stage-KV outputs passed the independent
full-attention oracle at `atol=rtol=2e-3`.

| Prefix | Route-Q p50 / p99 | Stage-KV p50 / p99 | Stage / Route p50 |
| ---: | ---: | ---: | ---: |
| 4K | 0.635 / 1.094 ms | 1.780 / 2.004 ms | 2.80x |
| 8K | 0.658 / 1.516 ms | 3.407 / 3.534 ms | 5.17x |
| 16K | 0.643 / 1.143 ms | 6.697 / 6.749 ms | 10.42x |
| 32K | 0.696 / 0.920 ms | 13.432 / 13.598 ms | 19.29x |

Stage-KV's payload divided by end-to-end p50 is about 9.4-10.0 GB/s across
the sweep, so its latency tracks KV bytes. This is effective application-level
payload throughput, not raw NCCL bandwidth. Route-Q always transfers 16,512
bytes. At 32K, its remote-attention kernel reached 0.552 ms p50 while the
communication/queue/framework residual was 0.125 ms p50, showing that the next
Route-Q optimization target has shifted toward attention execution rather than
moving the Q/output state.

A reverse-order sweep reproduced the endpoint trend: Stage-KV p50 differed by
0.2%-2.9% and Route-Q p50 by about 1%-5% at matching lengths. Fine-grained
local-tail and merge event timings varied with workload/runtime state even
after fixed preconditioning. They remain diagnostic observations until a
bare-metal Nsight trace can attribute streams, clocks, and allocator effects.

Raw reports:

- [forward sweep](../results/modal-l4-prefix-sweep-2026-07-20.json)
- [reverse sweep](../results/modal-l4-prefix-sweep-reverse-2026-07-20.json)

The Modal environment reported no CUDA peer access and blocked the NVML topology
query under gVisor. These results do not characterize NVLink, GPUDirect RDMA, or
bare-metal PCIe.
