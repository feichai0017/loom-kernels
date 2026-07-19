# vLLM Local Attention Backend

## Purpose

M1 proves that Loom can enter the real vLLM V1 attention call path without
owning model execution or replacing its optimized kernel. The adapter registers
as an out-of-tree `CUSTOM` backend, validates the local tensor contract, and
delegates to vLLM `FlashAttentionImpl` with the same arguments and output tensor.

```text
vLLM model runner
  -> LoomFlashAttentionMetadataBuilder.build
       -> CPU query boundary validation
       -> opaque paged-KV tensor descriptors
       -> generation-checked step snapshot
  -> LoomFlashAttentionImpl.forward
       -> first-call Q/K/V/output layout and device validation
       -> process-local timing and failure accounting
       -> vLLM FlashAttentionImpl.forward
            -> GPU FlashAttention kernel
```

No Q/K/V payload enters the Rust control service. The only tensor values read by
the adapter are `query_start_loc_cpu`, which vLLM already maintains on CPU.
Device-side block tables, slot mappings, sequence lengths, and query offsets
remain opaque. The adapter adds the engine boundary that later route-Q and
split-KV executors will implement.

## Install And Run

The package currently targets vLLM 0.25.x and its V1 attention backend registry.
Install it in the same Python environment as vLLM:

```bash
python3 -m pip install -e './python[vllm]'
```

Load only the Loom plugin and select the registered backend:

```bash
VLLM_PLUGINS=loom \
  vllm serve MODEL \
  --attention-backend CUSTOM \
  --enforce-eager
```

`--enforce-eager` keeps the M1 validation path outside CUDA Graph capture. Graph
support is intentionally deferred until the remote execution contract is fixed.

## Contract

The first forward checks:

- positive MHA/GQA head counts and valid GQA divisibility;
- flattened or explicit-head Q/K/V/output layouts;
- one device for all non-empty attention tensors;
- a deterministic layout identity from attention type, head counts, head size,
  and KV-cache dtype.

Set `LOOM_VALIDATE_EVERY_FORWARD=1` for debugging dynamic layouts. The
default validates once per attention implementation to avoid repeated Python
shape walks in vLLM's per-layer critical path. Call count, failures, elapsed
time, layout identity, and last validated device are process-local telemetry;
they are not distributed scheduler evidence.

## Paged-KV Step Snapshot

The custom backend returns a subclass of vLLM's
`FlashAttentionMetadataBuilder`. After the native builder finishes, Loom
attaches one immutable `StepMetadataSnapshot` to the resulting metadata. It
contains:

- request count, actual/padded token bounds, maximum query and sequence lengths;
- the existing CPU query-start offsets;
- block size, layer group, head layout, KV dtype, and a layout digest;
- shape, dtype, device, item count, byte bounds, and process-local data pointer
  for the contiguous block table, slot mapping, sequence lengths, and device
  query offsets;
- a monotonically increasing generation, including `update_block_table` calls.

The snapshot deliberately does not contain physical block-table values. Reading
those values in Python would synchronize the GPU. Prefix identity and
`PoolObjectRef` mappings come from Loom's page table and pool events; a
later node-local bridge will join those control-plane identities with these
device tensor descriptors.

## Current Validation Boundary

CI tests registration, forwarding, metadata building, block-table replacement,
zero device readback, layout rejection, error propagation, and plugin
idempotence with fake tensor and vLLM modules.

## CUDA Acceptance Gate

The repository includes `loom-vllm-smoke`, but the result is valid only
when it runs on a Linux NVIDIA host. Install the adapter and pinned vLLM range in
the same Python 3.10-3.12 environment:

```bash
python3 -m pip install -e './python[vllm]'
loom-vllm-smoke compare \
  --report build/vllm-smoke/report.json
```

The default workload uses the public, revision-pinned
`HuggingFaceTB/SmolLM2-135M-Instruct` Llama model. The harness starts two clean
processes with identical model, dtype, prompts, seed, block size, eager mode,
and prefix-caching settings:

1. native vLLM `FLASH_ATTN`;
2. Loom `CUSTOM`, which delegates to the same implementation.

It disables V1 multiprocessing for deterministic same-process execution, warms
each engine, and checks every generated token ID and sampled token logprob.
Token IDs must match exactly; logprobs default to an absolute tolerance of
`1e-5`. Startup and median generation times are recorded, but timing does not
affect pass/fail because this small sequential smoke workload is not a
performance benchmark.

Exit status `0` means the correctness comparison passed, `1` means outputs
differed, and `2` means the environment or runtime contract failed. The JSON
report retains both raw runs and all differences. A report must include the GPU,
vLLM version, model revision, and workload settings to be reviewable.

The current macOS development host has no CUDA runtime, so the CUDA gate has not
yet run. M1 remains incomplete until a real report is produced. Even after it
passes, physical vLLM block IDs still need to be joined with external
`PoolObjectRef` values and installed in the Rust node runtime.
