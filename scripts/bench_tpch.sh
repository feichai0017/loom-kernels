#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sample_size="${QUILL_BENCH_SAMPLE_SIZE:-10}"
export QUILL_TPCH_SF="${QUILL_TPCH_SF:-0.01}"
export QUILL_TPCH_GEN_THREADS="${QUILL_TPCH_GEN_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 1)}"

cargo bench --bench tpch -- --sample-size "${sample_size}"
