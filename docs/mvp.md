# Distributed Attention MVP

## Supported Shape

- decode only;
- Llama-style MHA/GQA;
- FP16 or BF16 KV;
- one model process and one remote attention process;
- two GPUs in one host;
- CUDA Graph disabled;
- complete sealed prefix blocks plus one local active tail.

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
