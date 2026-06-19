# MegaScale-Infer Notes

Paper: `papers/megascale-infer-sigcomm25-2504.02263.pdf`

Context: ByteDance Seed / Peking University, SIGCOMM 2025. The paper targets
large-scale MoE inference. It is useful for QuillCache because it shows how a
production AI infra system turns resource imbalance into a disaggregation and
co-scheduling problem.

## Core Problem

MoE reduces activated FLOPs, but inference can still waste GPUs. During decode:

- attention is memory-intensive because every request reads its own KV cache;
- FFN/expert computation should be compute-intensive, but MoE sparsity sends only
  a fraction of tokens to each expert;
- the effective batch per expert becomes too small, so expert GPUs are underused;
- increasing batch size is limited by latency and KV memory.

The important lesson: lower FLOPs do not guarantee lower serving cost. Utilization
depends on whether the active work matches the GPU resource being consumed.

## Main Mechanism

MegaScale-Infer disaggregates attention and FFN modules within each layer.

- Attention nodes hold attention parameters and KV cache.
- Expert nodes hold expert parameters.
- Attention uses replication / data parallelism.
- Experts use expert parallelism.
- Requests from multiple attention replicas are aggregated before experts, raising
  expert-side batch size.
- Attention and experts can run on different GPU types: memory/bandwidth-rich GPUs
  for attention, compute-cost-efficient GPUs for experts.

This is deeper than prefill/decode separation. It splits a layer by resource
profile: memory-bound attention versus compute-bound FFN.

## Ping-Pong Pipeline

Splitting attention and FFN creates idle bubbles:

- attention waits while FFN computes;
- FFN waits while attention computes;
- both wait for token dispatch over the network.

MegaScale-Infer partitions a batch into micro-batches and shuttles them between
attention and expert nodes. With enough micro-batches, one side computes while
the other side transfers or computes a different micro-batch.

Key tradeoff:

- too few micro-batches leave idle bubbles;
- more micro-batches hide communication better;
- too many micro-batches add overhead and may hurt latency.

The paper's evaluation reports that moving from one micro-batch to two gives a
large throughput jump, and three micro-batches helps hide inter-node
communication further for larger models.

## M2N Communication

Disaggregated attention and experts replace normal All2All with M-to-N and N-to-M
traffic. Generic NCCL primitives have extra overhead for this pattern:

- unnecessary GPU-to-CPU copies;
- group initialization overhead;
- GPU synchronization;
- high tail latency as sender/receiver count grows.

MegaScale-Infer builds a custom M2N communication library. The paper reports
large wins over NCCL for this specific traffic pattern, including lower latency,
lower P99, and higher throughput.

The systems lesson is narrow but important: once the model execution boundary
changes, the communication primitive may also need to change. A generic
collective can be wrong for the new traffic shape.

## Evaluation Signals

The paper uses:

- per-GPU decoding throughput;
- time between tokens;
- end-to-end throughput including prefill;
- throughput per unit cost under heterogeneous GPUs;
- throughput per unit power;
- M2N median and P99 latency;
- M2N throughput;
- ablations for disaggregation, M2N optimization, micro-batch count, and
  deployment plan.

The strongest signal is not just speedup. It is the ablation chain:

1. colocated baseline underutilizes experts;
2. disaggregation improves effective expert batch size;
3. optimized M2N lowers communication below compute time;
4. ping-pong pipeline overlaps communication with compute;
5. deployment plan matters because attention and expert times must be balanced.

## Connection To QuillCache

QuillCache does not disaggregate attention and FFN today. The useful transfer is
the control-plane pattern:

- identify each module's resource profile;
- expose measurements for the slow boundary;
- split or route work only when the downstream batching and communication shape
  justify it;
- tune the split using measured idle time and SLO, not static policy.

Mapping:

| MegaScale-Infer | QuillCache |
| --- | --- |
| Attention nodes | Prefill/decode workers that own live KV and attention state |
| Expert nodes | Remote KV tiers / transfer backends / future specialized compute pools |
| M2N token dispatch | KV block/layer transfer among many producers and consumers |
| Ping-pong pipeline | Layer-wise KV transfer overlap |
| Deployment plan | Co-scheduler plan: P/D ratio, HBM split, tier placement, transfer depth |
| Expert imbalance | Hot prefix skew and uneven KV residency |

## Design Implications

1. Add overlap metrics, not just transfer time.
   QuillCache already reports `time_to_first_layer_us`, `full_transfer_us`, and
   `overlap_window_us`, and now derives `overlap_efficiency_pct` for the
   co-scheduler.

2. Treat transfer depth like micro-batch count.
   In-flight layer depth has the same shape as MegaScale-Infer's micro-batch
   count: too low leaves idle time; too high increases pressure and contention.

3. Balance phases by measured time.
   MegaScale-Infer balances attention and FFN time. QuillCache should balance
   prefill compute, decode compute, and KV fetch time in the co-scheduler.

4. Preserve generic fallback.
   MegaScale-Infer optimizes M2N because NCCL is a bad fit for that pattern.
   QuillCache should adopt NIXL/UCX for GPU transfer later, but TCP remains the
   correctness fallback and experiment baseline.

5. Use cost per GPU, not only latency.
   Heterogeneous deployment matters because attention and FFN have different
   resource profiles. For QuillCache, CXL/DRAM/SSD/HBM tiers should be compared
   by SLO-goodput per dollar or per GPU, not only raw fetch latency.

## Proposed QuillCache Tasks

P0: Document and expose overlap efficiency. Done.

- Add a derived value in state/docs:
  `overlap_efficiency = overlap_window_us / max(full_transfer_us, 1)`.
- Use it as a co-scheduler signal but keep actions dry-run.

P1: Build predicted-versus-actual experiment.

- Send repeated-prefix requests through gateway.
- Record estimated TTFT headers, actual streaming TTFT, and transfer telemetry.
- Plot predicted error and overlap efficiency.

P2: Add a transfer-depth dry-run action.

- If `time_to_first_layer` is good but `full_transfer` is long, keep overlap.
- If transfer queue depth is high and overlap is poor, reduce depth or prefer
  recompute.
- If repeated remote hits appear for a hot prefix, suggest replication.

## Interview Takeaway

MegaScale-Infer is a clean example of production AI infra reasoning:

- find the resource mismatch;
- split the system at that boundary;
- build the communication primitive that matches the new traffic pattern;
- use a pipeline to hide the new boundary cost;
- validate with throughput, latency tails, cost, and ablations.

For QuillCache, the analogous story is:

> KV cache movement should not be treated as a blob copy. It should be scheduled
> like a pipeline stage with measured time-to-first-layer, full-transfer time,
> queue depth, and overlap efficiency feeding back into routing and placement.
