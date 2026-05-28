# Testing & Documentation

The active tests focus on DataFusion execution, Parquet registration, and the
MLIR JIT boundary.

## Test Suite

| Location | Purpose |
| -------- | ------- |
| `tests/df_arrow_parquet.rs` | End-to-end SQL over DataFusion memory tables and registered Parquet datasets. |
| `crates/quill-plan/src/*` unit tests | Frontend-neutral expression and graph behavior. |
| `crates/quill-runtime/src/*` unit tests | Arrow runtime kernels and aggregate behavior. |
| `crates/quill-jit/src/*` unit tests | MLIR module generation, verification, and compiled invocation. |

Common commands:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo bench --no-run
```

The default build requires local MLIR/LLVM 22 libraries. It builds the
`quill-mlir` C++/TableGen package, verifies formal Quill dialect regions, and
runs compiled ExecutionEngine smoke tests. On a Homebrew LLVM 22 installation,
set:

```bash
MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
LLVM_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
cargo test
```

## Documentation

The `docs/` directory is an mdBook. It tracks the current frontend-adapter +
Arrow + MLIR architecture and intentionally omits the removed teaching
database storage stack.
