# Documentation

## Read First

1. [Architecture](design/architecture.md) defines the system boundary and data
   path.
2. [Implementation status](status.md) records what the `main` branch actually
   implements and what has been validated.
3. [Protocol invariants](design/protocols.md) defines correctness and recovery
   requirements.
4. [Roadmap](roadmap.md) contains future milestones and exit criteria.

## Guides

- [vLLM local attention backend](guides/vllm-local-backend.md)
- [Two-GPU Route-Q acceptance gate](guides/two-gpu-route-q.md)
- [Rust/CUDA fused local-tail operator](guides/rust-cuda-fused-tail.md)

## Research Notes

- [ByteDance AML reading map](research/bytedance-aml-reading-map.md)
- [Flux and Comet overlap notes](research/flux-comet-overlap-notes.md)
- [MegaScale-Infer notes](research/megascale-infer-notes.md)

## Documentation Rules

- `design/` contains stable ownership, protocol, and correctness contracts.
- `guides/` contains commands and hardware-specific acceptance procedures.
- `status.md` is the only source of truth for implemented versus missing work.
- `roadmap.md` describes planned work and must not claim completion.
- `research/` records paper-derived ideas; it is not implementation evidence.
