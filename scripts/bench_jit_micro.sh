#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sample_size="${QUILL_BENCH_SAMPLE_SIZE:-10}"

cargo bench --bench jit_micro -- --sample-size "${sample_size}"
