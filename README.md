# Loom

Loom is a disaggregated core-attention runtime for externally managed KV.
vLLM or SGLang keeps model execution; Mooncake, LMCache, or another `KvPool`
owns sealed KV objects; Loom moves Q to a suitable attention worker,
executes near the historical KV, and returns the attention output. K_new/V_new
travel only to the worker that owns the mutable tail.

## Boundary

| Component | Owns |
| --- | --- |
| vLLM / SGLang | batching, weights, QKV projection, RoPE, FFN, sampling |
| Loom | generation-pinned KV views, attention plans, tensor transport, execution and exact merge |
| external `KvPool` | sealed KV allocation, placement, replication, eviction, durability |
| Holt catalog | persistent `prefix -> PoolObjectRef` metadata, revalidated after recovery |

The control service is a slow path. Per-layer execution uses node-local state and
never synchronously queries the global controller or Holt.

## Workspace

| Path | Responsibility |
| --- | --- |
| `crates/loom-attention` | one Rust package with runtime, pool, catalog, planner, transport, and attention modules |
| `loom-control` | slow-path catalog/scheduler binary from the `loom-attention` package |
| `loom-worker` | attention-worker control binary from the `loom-attention` package |
| `python/loom_attention` | out-of-tree vLLM attention backend adapters |

## Implemented

- one dependency-clean Rust package with separately deployable binaries;
- external `KvPool` API with object generations, ordered events, and read leases;
- Holt-backed persistent object catalog plus worker-epoch-aware hot directory;
- SLO cost comparison for local, route-query, KV-stage, and sharded execution;
- exact split-KV online-softmax merge with reference correctness tests;
- node-local single-writer step state, generation-pinned `KvView`, lease
  validation, and active-tail sealing;
- handle-based tensor transport API with registered-region bounds checks;
- control-only service endpoints for the controller and attention worker;
- vLLM `CUSTOM` backend that validates the local tensor contract and delegates
  unchanged execution to vLLM FlashAttention;
- vLLM metadata-builder wrapper that records generation-checked paged-KV tensor
  descriptors without reading device tensor contents.
- isolated CUDA A/B harness that compares native FlashAttention with the
  Loom delegate using exact token IDs and sampled-logprob tolerance.
- two-GPU NCCL harness for Q-only remote-prefix execution, exact local-tail
  merge, and a Stage-KV payload/latency baseline.

## Not Implemented Yet

- SGLang and remote-execution attention backend adapters;
- an executed two-GPU report and optimized CUDA partial-attention kernels;
- production CUDA IPC, NCCL, NIXL, or GPUDirect RDMA transports;
- a production Mooncake adapter;
- remote GPU end-to-end latency or throughput claims.

## Build

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

On a Linux CUDA host with vLLM installed, run the M1 acceptance gate:

```bash
loom-vllm-smoke compare --report build/vllm-smoke/report.json
```

Inspect the M2 payload asymmetry without CUDA, then run it on a Linux host with
two NVIDIA GPUs:

```bash
loom-two-gpu-smoke plan --prefix-tokens 4096
loom-two-gpu-smoke run \
  --prefix-tokens 4096 \
  --report build/two-gpu-smoke/report.json
```

Run the control endpoints:

```bash
cargo run -p loom-attention --bin loom-control -- --bind 127.0.0.1:8080
cargo run -p loom-attention --bin loom-worker -- --bind 127.0.0.1:8090
```

See [architecture](docs/architecture.md), [platform plan](docs/platform-plan.md),
[vLLM local backend](docs/vllm-local-backend.md),
[two-GPU Route-Q](docs/two-gpu-route-q.md), and
[protocol invariants](docs/protocols.md).

## License

MIT
