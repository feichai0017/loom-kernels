"""Two-GPU NCCL benchmark runtime for the Route-Q data path.

The remote rank owns a sealed KV prefix. The engine rank sends only Q, receives
an output-plus-LSE attention state, merges it with its local active-tail state,
and compares the result with full attention. A Stage-KV baseline transfers the
remote prefix in the opposite direction under the same workload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Sequence

from loom_attention.attention_state import (
    ATTENTION_BACKENDS,
    compute_attention_state,
    merge_attention_states,
)
from loom_attention.paged_executor import FlashInferPagedExecutor, PagedKvView


DTYPE_BYTES = {"float16": 2, "bfloat16": 2}
PAGED_ATTENTION_BACKEND = "flashinfer-paged"
BENCHMARK_ATTENTION_BACKENDS = (*ATTENTION_BACKENDS, PAGED_ATTENTION_BACKEND)
ROUTE_STRATEGIES = ("sequential", "overlap", "fused")
DTYPE_TOLERANCES = {
    "float16": (2e-3, 2e-3),
    "bfloat16": (2e-2, 2e-2),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    prefix_tokens: int = 4096
    tail_tokens: int = 16
    rows: int = 1
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"
    attention_backend: str = "reference"
    route_strategy: str = "sequential"
    page_size: int = 16
    precondition_dimension: int = 4096
    precondition_iterations: int = 100
    warmup: int = 5
    iterations: int = 20
    seed: int = 7
    atol: float | None = None
    rtol: float | None = None
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if self.dtype in DTYPE_TOLERANCES:
            default_atol, default_rtol = DTYPE_TOLERANCES[self.dtype]
            if self.atol is None:
                object.__setattr__(self, "atol", default_atol)
            if self.rtol is None:
                object.__setattr__(self, "rtol", default_rtol)

    def validate(self) -> None:
        positive = {
            "prefix_tokens": self.prefix_tokens,
            "rows": self.rows,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "precondition_dimension": self.precondition_dimension,
            "iterations": self.iterations,
            "timeout_seconds": self.timeout_seconds,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"values must be positive: {', '.join(invalid)}")
        if (
            self.tail_tokens < 0
            or self.precondition_iterations < 0
            or self.warmup < 0
        ):
            raise ValueError(
                "tail_tokens, precondition_iterations, and warmup must be "
                "non-negative"
            )
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("kv_heads must divide query_heads")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.attention_backend not in BENCHMARK_ATTENTION_BACKENDS:
            raise ValueError(
                f"unsupported attention backend: {self.attention_backend}"
            )
        if self.route_strategy not in ROUTE_STRATEGIES:
            raise ValueError(
                f"unsupported Route-Q strategy: {self.route_strategy}"
            )
        if self.route_strategy == "fused":
            if not 0 < self.tail_tokens <= 64:
                raise ValueError(
                    "the fused Route-Q strategy requires 1..=64 tail tokens"
                )
            if self.head_dim > 256:
                raise ValueError(
                    "the fused Route-Q strategy requires head_dim <= 256"
                )
        if self.atol is None or self.rtol is None:
            raise ValueError("atol and rtol must be configured")
        if self.atol < 0.0 or self.rtol < 0.0:
            raise ValueError("atol and rtol must be non-negative")


def projected_transfer_bytes(config: BenchmarkConfig) -> dict[str, int]:
    config.validate()
    element_bytes = DTYPE_BYTES[config.dtype]
    query = config.rows * config.query_heads * config.head_dim * element_bytes
    output = config.rows * config.query_heads * config.head_dim * element_bytes
    logsumexp = config.rows * config.query_heads * 4
    attention_state = output + logsumexp
    staged_kv = (
        2
        * config.prefix_tokens
        * config.kv_heads
        * config.head_dim
        * element_bytes
    )
    return {
        "query": query,
        "output": output,
        "logsumexp": logsumexp,
        "attention_state": attention_state,
        "route_query_total": query + attention_state,
        "stage_kv_total": staged_kv,
    }


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("at least one sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": float(median(samples)),
        "p99_ms": percentile(samples, 0.99),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def _optional_latency_summary(
    samples: Sequence[float],
) -> dict[str, float] | None:
    return _latency_summary(samples) if samples else None


def _residual_samples(
    totals: Sequence[float], *components: Sequence[float]
) -> list[float]:
    """Return the non-kernel residual without claiming it is pure transport."""
    for component in components:
        if len(component) != len(totals):
            raise ValueError("phase sample counts must match end-to-end samples")
    return [
        max(0.0, total - sum(component[index] for component in components))
        for index, total in enumerate(totals)
    ]


def _route_residual_samples(
    totals: Sequence[float],
    remote_attention: Sequence[float],
    local_tail: Sequence[float],
    tail_merge: Sequence[float],
    *,
    strategy: str,
) -> list[float]:
    """Estimate the non-kernel Route-Q residual along the critical path."""
    required = (remote_attention, tail_merge)
    if any(len(component) != len(totals) for component in required):
        raise ValueError("phase sample counts must match end-to-end samples")
    if strategy == "fused":
        if local_tail:
            raise ValueError("fused Route-Q must not report a local-tail phase")
        critical_compute = [
            remote_attention[index] + tail_merge[index]
            for index in range(len(totals))
        ]
    else:
        if len(local_tail) != len(totals):
            raise ValueError("phase sample counts must match end-to-end samples")
        if strategy == "sequential":
            critical_compute = [
                remote_attention[index]
                + local_tail[index]
                + tail_merge[index]
                for index in range(len(totals))
            ]
        elif strategy == "overlap":
            critical_compute = [
                max(remote_attention[index], local_tail[index])
                + tail_merge[index]
                for index in range(len(totals))
            ]
        else:
            raise ValueError(f"unsupported Route-Q strategy: {strategy}")
    return [
        max(0.0, total - critical_compute[index])
        for index, total in enumerate(totals)
    ]


def _record_cuda_interval(
    torch: Any,
    operation: Any,
    intervals: list[tuple[Any, Any]] | None,
) -> Any:
    if intervals is None:
        return operation()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    intervals.append((start, end))
    return result


def _cuda_event_samples(
    torch: Any, intervals: Sequence[tuple[Any, Any]]
) -> list[float]:
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in intervals]


def _torch_dtype(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _make_inputs(torch: Any, config: BenchmarkConfig, device: Any) -> tuple[Any, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    dtype = _torch_dtype(torch, config.dtype)

    def sample(shape: tuple[int, ...]) -> Any:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .to(dtype=dtype)
            .to(device=device)
            .contiguous()
        )

    query = sample((config.rows, config.query_heads, config.head_dim))
    prefix_key = sample((config.prefix_tokens, config.kv_heads, config.head_dim))
    prefix_value = sample((config.prefix_tokens, config.kv_heads, config.head_dim))
    tail_key = sample((config.tail_tokens, config.kv_heads, config.head_dim))
    tail_value = sample((config.tail_tokens, config.kv_heads, config.head_dim))
    return query, prefix_key, prefix_value, tail_key, tail_value


@dataclass(frozen=True)
class _PagedFixture:
    executor: FlashInferPagedExecutor
    cache: tuple[Any, Any]
    view: PagedKvView


def _make_paged_fixture(
    torch: Any,
    key: Any,
    value: Any,
    config: BenchmarkConfig,
    *,
    table_id: str,
) -> _PagedFixture:
    """Build a benchmark fixture once; timed execution consumes pages directly."""
    token_count = key.shape[0]
    page_count = math.ceil(token_count / config.page_size)
    page_shape = (
        page_count,
        config.page_size,
        config.kv_heads,
        config.head_dim,
    )
    paged_key = torch.zeros(page_shape, dtype=key.dtype, device=key.device)
    paged_value = torch.zeros_like(paged_key)
    paged_key.view(-1, config.kv_heads, config.head_dim)[:token_count].copy_(key)
    paged_value.view(-1, config.kv_heads, config.head_dim)[:token_count].copy_(
        value
    )

    indices = torch.arange(
        page_count, dtype=torch.int32, device=key.device
    ).repeat(config.rows)
    indptr = torch.arange(
        0,
        (config.rows + 1) * page_count,
        page_count,
        dtype=torch.int32,
        device=key.device,
    )
    last_page_len = token_count % config.page_size or config.page_size
    last_page_lens = torch.full(
        (config.rows,),
        last_page_len,
        dtype=torch.int32,
        device=key.device,
    )
    view = PagedKvView(
        table_id=table_id,
        page_table_generation=1,
        lease_ids=(f"benchmark:{table_id}",),
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_lens,
        page_size=config.page_size,
        layout="NHD",
    )
    return _PagedFixture(
        executor=FlashInferPagedExecutor(),
        cache=(paged_key, paged_value),
        view=view,
    )


def _state_backend(config: BenchmarkConfig) -> str:
    if config.attention_backend == PAGED_ATTENTION_BACKEND:
        return "flashinfer"
    return config.attention_backend


def _batch_send(dist: Any, tensors: Sequence[Any], destination: int) -> None:
    operations = [dist.P2POp(dist.isend, tensor, destination) for tensor in tensors]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _batch_receive(dist: Any, tensors: Sequence[Any], source: int) -> None:
    operations = [dist.P2POp(dist.irecv, tensor, source) for tensor in tensors]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _route_engine(
    torch: Any,
    dist: Any,
    query: Any,
    remote_buffers: tuple[Any, Any],
    local_tail: tuple[Any, Any],
    config: BenchmarkConfig,
    local_tail_intervals: list[tuple[Any, Any]] | None = None,
    merge_intervals: list[tuple[Any, Any]] | None = None,
    local_tail_stream: Any | None = None,
) -> Any:
    state_backend = _state_backend(config)

    def compute_local_state() -> tuple[Any, Any]:
        return _record_cuda_interval(
            torch,
            lambda: compute_attention_state(
                torch,
                query,
                local_tail[0],
                local_tail[1],
                kv_heads=config.kv_heads,
                scale=config.head_dim**-0.5,
                backend=state_backend,
            ),
            local_tail_intervals,
        )

    dist.send(query, dst=1)
    local_state = None
    if config.tail_tokens and config.route_strategy == "overlap":
        if local_tail_stream is None:
            raise RuntimeError("overlap Route-Q requires a local-tail stream")
        local_tail_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(local_tail_stream):
            for tensor in (query, *local_tail):
                tensor.record_stream(local_tail_stream)
            local_state = compute_local_state()
    _batch_receive(dist, remote_buffers, source=1)

    if config.route_strategy == "fused":
        from loom_attention.cuda_ops import fused_tail_attention_merge

        return _record_cuda_interval(
            torch,
            lambda: fused_tail_attention_merge(
                query,
                local_tail[0],
                local_tail[1],
                remote_buffers[0],
                remote_buffers[1],
                scale=config.head_dim**-0.5,
            ),
            merge_intervals,
        )[0]

    states = [remote_buffers]
    if config.tail_tokens:
        if config.route_strategy == "overlap":
            if local_state is None:
                raise RuntimeError("overlap Route-Q produced no local state")
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(local_tail_stream)
            for tensor in local_state:
                tensor.record_stream(current_stream)
            states.append(local_state)
        else:
            states.append(compute_local_state())
    return _record_cuda_interval(
        torch,
        lambda: merge_attention_states(torch, states, backend=state_backend),
        merge_intervals,
    )[0]


def _route_worker(
    torch: Any,
    dist: Any,
    query_buffer: Any,
    prefix: tuple[Any, Any],
    config: BenchmarkConfig,
    paged: _PagedFixture | None,
    attention_intervals: list[tuple[Any, Any]] | None = None,
) -> None:
    dist.recv(query_buffer, src=0)

    def compute_remote_state() -> tuple[Any, Any]:
        if paged is not None:
            return paged.executor.execute(
                query_buffer,
                paged.cache,
                paged.view,
                kv_heads=config.kv_heads,
                scale=config.head_dim**-0.5,
            )
        return compute_attention_state(
            torch,
            query_buffer,
            prefix[0],
            prefix[1],
            kv_heads=config.kv_heads,
            scale=config.head_dim**-0.5,
            backend=config.attention_backend,
        )

    state = _record_cuda_interval(
        torch, compute_remote_state, attention_intervals
    )
    _batch_send(dist, state, destination=0)


def _stage_engine(
    torch: Any,
    dist: Any,
    query: Any,
    receive_buffers: tuple[Any, Any],
    full_buffers: tuple[Any, Any],
    config: BenchmarkConfig,
    paged: _PagedFixture | None,
    attention_intervals: list[tuple[Any, Any]] | None = None,
) -> Any:
    _batch_receive(dist, receive_buffers, source=1)

    def compute_full_state() -> tuple[Any, Any]:
        if paged is not None:
            return paged.executor.execute(
                query,
                paged.cache,
                paged.view,
                kv_heads=config.kv_heads,
                scale=config.head_dim**-0.5,
            )
        return compute_attention_state(
            torch,
            query,
            full_buffers[0],
            full_buffers[1],
            kv_heads=config.kv_heads,
            scale=config.head_dim**-0.5,
            backend=_state_backend(config),
        )

    return _record_cuda_interval(
        torch, compute_full_state, attention_intervals
    )[0]


def _stage_worker(dist: Any, prefix: tuple[Any, Any]) -> None:
    _batch_send(dist, prefix, destination=0)


def _full_attention(
    torch: Any,
    query: Any,
    prefix: tuple[Any, Any],
    local_tail: tuple[Any, Any],
    config: BenchmarkConfig,
) -> Any:
    if config.tail_tokens:
        key = torch.cat((prefix[0], local_tail[0]), dim=0)
        value = torch.cat((prefix[1], local_tail[1]), dim=0)
    else:
        key, value = prefix
    return compute_attention_state(
        torch,
        query,
        key,
        value,
        kv_heads=config.kv_heads,
        scale=config.head_dim**-0.5,
        backend="reference",
    )[0]


def _timed_cuda(torch: Any, operation: Any) -> float:
    torch.cuda.synchronize()
    started = perf_counter()
    operation()
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1_000.0


def _precondition_cuda(torch: Any, config: BenchmarkConfig, device: Any) -> None:
    """Bring both GPUs to an active clock state outside measured regions."""
    if config.precondition_iterations == 0:
        return
    dimension = config.precondition_dimension
    scale = dimension**-0.5
    left = torch.randn(
        (dimension, dimension),
        dtype=torch.float16,
        device=device,
    ).mul_(scale)
    right = torch.randn_like(left).mul_(scale)
    output = torch.empty_like(left)
    for _ in range(config.precondition_iterations):
        torch.mm(left, right, out=output)
        left, output = output, left
    torch.cuda.synchronize()


def _environment(torch: Any) -> dict[str, Any]:
    devices = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        nccl_version = ".".join(str(part) for part in nccl_version)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": nccl_version,
        "device_peer_access": torch.cuda.can_device_access_peer(0, 1),
        "devices": devices,
    }


def _write_report(
    torch: Any,
    config: BenchmarkConfig,
    route_output: Any,
    stage_output: Any,
    expected: Any,
    route_samples: Sequence[float],
    stage_samples: Sequence[float],
    route_remote_attention_samples: Sequence[float],
    route_local_tail_samples: Sequence[float],
    route_merge_samples: Sequence[float],
    stage_attention_samples: Sequence[float],
    report_path: str,
) -> None:
    paged = config.attention_backend == PAGED_ATTENTION_BACKEND

    def correctness(output: Any) -> dict[str, Any]:
        difference = (output - expected).abs()
        return {
            "passed": bool(
                torch.allclose(output, expected, atol=config.atol, rtol=config.rtol)
            ),
            "max_absolute_error": float(difference.max().item()),
            "max_relative_error": float(
                (difference / expected.abs().clamp_min(1e-6)).max().item()
            ),
        }

    route_correctness = correctness(route_output)
    stage_correctness = correctness(stage_output)
    passed = route_correctness["passed"] and stage_correctness["passed"]
    route = _latency_summary(route_samples)
    stage = _latency_summary(stage_samples)
    route["payload_bytes"] = projected_transfer_bytes(config)["route_query_total"]
    stage["payload_bytes"] = projected_transfer_bytes(config)["stage_kv_total"]
    local_tail_for_residual = (
        []
        if config.route_strategy == "fused"
        else route_local_tail_samples or [0.0 for _ in route_samples]
    )
    route_residual = _route_residual_samples(
        route_samples,
        route_remote_attention_samples,
        local_tail_for_residual,
        route_merge_samples,
        strategy=config.route_strategy,
    )
    stage_residual = _residual_samples(
        stage_samples, stage_attention_samples
    )
    report = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "implementation": {
            "transport": "torch.distributed NCCL point-to-point",
            "attention_backend": config.attention_backend,
            "route_strategy": config.route_strategy,
            "attention_kernel": (
                "FlashInfer BatchDecodeWithPagedKVCacheWrapper"
                if paged
                else "FlashInfer single_decode_with_kv_cache"
                if config.attention_backend == "flashinfer"
                else "PyTorch CUDA einsum output-plus-LSE reference"
            ),
            "merge_kernel": (
                "Loom handwritten CUDA fused local-tail attention plus exact "
                "output/LSE merge"
                if config.route_strategy == "fused"
                else "FlashInfer merge_states"
                if config.attention_backend
                in ("flashinfer", PAGED_ATTENTION_BACKEND)
                else "PyTorch output-plus-LSE merge"
            ),
            "kv_layout": "paged NHD" if paged else "contiguous NHD",
            "paged_executor": paged,
            "production_kernel": paged and config.route_strategy != "fused",
            "remote_attention_production_kernel": paged,
            "experimental_fused_tail_kernel": config.route_strategy == "fused",
            "fixture_repacked_before_timing": paged,
        },
        "environment": _environment(torch),
        "workload": asdict(config),
        "correctness": {
            "atol": config.atol,
            "rtol": config.rtol,
            "route_query": route_correctness,
            "stage_kv": stage_correctness,
        },
        "route_query": route,
        "stage_kv": stage,
        "phase_breakdown": {
            "measurement": {
                "end_to_end_clock": (
                    "host perf_counter bracketed by CUDA synchronization"
                ),
                "kernel_clock": "CUDA events on each rank's current stream",
                "residual_definition": (
                    "end-to-end minus the estimated critical-path CUDA "
                    "compute; overlap uses max(remote attention, local tail), "
                    "while sequential and fused use their ordered kernel "
                    "phases. The residual includes NCCL transport, queueing, "
                    "synchronization, and host/framework overhead, so it is "
                    "not a pure link-bandwidth measurement"
                ),
                "profiling_metadata_exchange": (
                    "rank-1 event durations sent after the timed Route-Q loop"
                ),
                "gpu_preconditioning": (
                    "fixed FP16 GEMM loop on both ranks before each timed path; "
                    "excluded from end-to-end samples"
                ),
            },
            "route_query": {
                "remote_attention_kernel_ms": _latency_summary(
                    route_remote_attention_samples
                ),
                "local_tail_kernel_ms": _optional_latency_summary(
                    route_local_tail_samples
                ),
                "merge_kernel_ms": (
                    None
                    if config.route_strategy == "fused"
                    else _latency_summary(route_merge_samples)
                ),
                "fused_tail_merge_kernel_ms": (
                    _latency_summary(route_merge_samples)
                    if config.route_strategy == "fused"
                    else None
                ),
                "communication_queue_framework_residual_ms": _latency_summary(
                    route_residual
                ),
            },
            "stage_kv": {
                "attention_kernel_ms": _latency_summary(
                    stage_attention_samples
                ),
                "communication_queue_framework_residual_ms": _latency_summary(
                    stage_residual
                ),
            },
        },
        "stage_over_route_p50": stage["p50_ms"] / route["p50_ms"],
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _run_rank(
    rank: int,
    config: BenchmarkConfig,
    init_method: str,
    report_path: str,
) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=config.timeout_seconds),
    )
    try:
        query, prefix_key, prefix_value, tail_key, tail_value = _make_inputs(
            torch, config, device
        )
        prefix = (prefix_key, prefix_value)
        local_tail = (tail_key, tail_value)
        paged_route = None
        paged_stage = None
        route_local_tail_stream = None

        if rank == 0:
            if config.route_strategy == "overlap" and config.tail_tokens:
                route_local_tail_stream = torch.cuda.Stream(device=device)
            remote_buffers = (
                torch.empty(
                    (config.rows, config.query_heads, config.head_dim),
                    dtype=query.dtype,
                    device=device,
                ),
                torch.empty(
                    (config.rows, config.query_heads),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            full_tokens = config.prefix_tokens + config.tail_tokens
            stage_key = torch.empty(
                (full_tokens, config.kv_heads, config.head_dim),
                dtype=prefix_key.dtype,
                device=device,
            )
            stage_value = torch.empty_like(stage_key)
            stage_receive_buffers = (
                stage_key[: config.prefix_tokens],
                stage_value[: config.prefix_tokens],
            )
            stage_full_buffers = (stage_key, stage_value)
            if config.tail_tokens:
                stage_key[config.prefix_tokens :].copy_(tail_key)
                stage_value[config.prefix_tokens :].copy_(tail_value)
            if config.attention_backend == PAGED_ATTENTION_BACKEND:
                paged_stage = _make_paged_fixture(
                    torch,
                    stage_key,
                    stage_value,
                    config,
                    table_id="stage/full/layer-0",
                )
                paged_stage_key, paged_stage_value = paged_stage.cache
                stage_receive_buffers = (
                    paged_stage_key.view(
                        -1, config.kv_heads, config.head_dim
                    )[: config.prefix_tokens],
                    paged_stage_value.view(
                        -1, config.kv_heads, config.head_dim
                    )[: config.prefix_tokens],
                )
        else:
            query_buffer = torch.empty_like(query)
            if config.attention_backend == PAGED_ATTENTION_BACKEND:
                paged_route = _make_paged_fixture(
                    torch,
                    prefix_key,
                    prefix_value,
                    config,
                    table_id="route/prefix/layer-0",
                )

        route_remote_attention_intervals: list[tuple[Any, Any]] = []
        route_local_tail_intervals: list[tuple[Any, Any]] = []
        route_merge_intervals: list[tuple[Any, Any]] = []
        stage_attention_intervals: list[tuple[Any, Any]] = []

        dist.barrier()
        if rank == 0:
            route_output = _route_engine(
                torch,
                dist,
                query,
                remote_buffers,
                local_tail,
                config,
                local_tail_stream=route_local_tail_stream,
            )
            expected = _full_attention(torch, query, prefix, local_tail, config)
        else:
            _route_worker(torch, dist, query_buffer, prefix, config, paged_route)

        dist.barrier()
        _precondition_cuda(torch, config, device)
        dist.barrier()
        route_samples = []
        for iteration in range(config.warmup + config.iterations):
            measured = iteration >= config.warmup
            if rank == 0:
                elapsed = _timed_cuda(
                    torch,
                    lambda: _route_engine(
                        torch,
                        dist,
                        query,
                        remote_buffers,
                        local_tail,
                        config,
                        (
                            route_local_tail_intervals
                            if measured
                            else None
                        ),
                        route_merge_intervals if measured else None,
                        route_local_tail_stream,
                    ),
                )
                if measured:
                    route_samples.append(elapsed)
            else:
                _route_worker(
                    torch,
                    dist,
                    query_buffer,
                    prefix,
                    config,
                    paged_route,
                    route_remote_attention_intervals if measured else None,
                )

        dist.barrier()
        if rank == 0:
            remote_attention_tensor = torch.empty(
                config.iterations, dtype=torch.float64, device=device
            )
            dist.recv(remote_attention_tensor, src=1)
            route_remote_attention_samples = (
                remote_attention_tensor.cpu().tolist()
            )
        else:
            route_remote_attention_samples = _cuda_event_samples(
                torch, route_remote_attention_intervals
            )
            dist.send(
                torch.tensor(
                    route_remote_attention_samples,
                    dtype=torch.float64,
                    device=device,
                ),
                dst=0,
            )
        dist.barrier()
        if rank == 0:
            stage_output = _stage_engine(
                torch,
                dist,
                query,
                stage_receive_buffers,
                stage_full_buffers,
                config,
                paged_stage,
            )
        else:
            _stage_worker(dist, prefix)

        dist.barrier()
        _precondition_cuda(torch, config, device)
        dist.barrier()
        stage_samples = []
        for iteration in range(config.warmup + config.iterations):
            measured = iteration >= config.warmup
            if rank == 0:
                elapsed = _timed_cuda(
                    torch,
                    lambda: _stage_engine(
                        torch,
                        dist,
                        query,
                        stage_receive_buffers,
                        stage_full_buffers,
                        config,
                        paged_stage,
                        stage_attention_intervals if measured else None,
                    ),
                )
                if measured:
                    stage_samples.append(elapsed)
            else:
                _stage_worker(dist, prefix)

        dist.barrier()
        if rank == 0:
            route_local_tail_samples = _cuda_event_samples(
                torch, route_local_tail_intervals
            )
            route_merge_samples = _cuda_event_samples(
                torch, route_merge_intervals
            )
            stage_attention_samples = _cuda_event_samples(
                torch, stage_attention_intervals
            )
            _write_report(
                torch,
                config,
                route_output,
                stage_output,
                expected,
                route_samples,
                stage_samples,
                route_remote_attention_samples,
                route_local_tail_samples,
                route_merge_samples,
                stage_attention_samples,
                report_path,
            )
    finally:
        dist.destroy_process_group()


def _require_cuda_environment() -> Any:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as error:
        raise RuntimeError("PyTorch is required; install './python[cuda]'") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA devices are required")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("a PyTorch build with torch.distributed NCCL is required")
    return torch


def run_benchmark(config: BenchmarkConfig, report_path: Path) -> dict[str, Any]:
    """Execute the two-rank benchmark and return its persisted report."""
    config.validate()
    torch = _require_cuda_environment()
    import torch.multiprocessing as multiprocessing

    with TemporaryDirectory(prefix="loom-two-gpu-") as directory:
        rendezvous = Path(directory) / "nccl-rendezvous"
        init_method = f"file://{rendezvous}"
        multiprocessing.spawn(
            _run_rank,
            args=(config, init_method, str(report_path)),
            nprocs=2,
            join=True,
        )
    return json.loads(report_path.read_text())
