# JHCS-RS — Portable Heterogeneous Compute Broker (Scalar CPU, SIMD & `wgpu` GPU)

> **PT-BR:** [README.md](README.md) | **Ecosystem:** [Jinn_Project](https://github.com/GeovaneEstanislau/Jinn_Project) · [Aladdin](https://github.com/GeovaneEstanislau/Aladdin)

**`jhcs-rs`** (*Jinn Heterogeneous Compute Service*) is the lowest-friction standalone component in the Jinn ecosystem, ready for **direct adoption on host operating systems (Linux/Windows) as well as bare-metal `#![no_std]` microkernels (Jinn OS Ring 3)**. It models compute pipelines as Directed Acyclic Graphs (DAGs) and dispatches stages across **scalar CPU**, **SIMD vector kernels (`AVX2`/`FMA`)**, and **`wgpu` compute shaders** while explicitly accounting for dispatch overhead and empirical breakeven thresholds.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![no_std Core](https://img.shields.io/badge/core-no__std%20ready-success.svg)](#no_std-compatibility)

---

## Why Teams Evaluate `jhcs-rs` First

Adopting an entire operating system has high switching cost; adopting a clean Rust compute broker does not. `jhcs-rs` isolates backend mechanics behind crisp crate boundaries:

1. **`jhcs-core`** (`#![no_std]` + `alloc`): Task descriptors, aligned `BufferSet` metadata, capability flags, and cycle-checked DAG scheduling (`ComputeGraph`).
2. **`jhcs-broker`**: Adaptive and rule-based `ComputeBroker` evaluating batch size, transfer/serialization latency, and accelerator presence so small batches stay on fast scalar CPU rather than paying idle GPU dispatch penalties.
3. **`jhcs-backend-cpu`**: Deterministic scalar reference implementation.
4. **`jhcs-backend-simd`**: CPU vector kernels (`matmul`, `conv2d`, `box_blur_rgba_simd` using explicit 64-byte cache-line prefetching).
5. **`jhcs-backend-wgpu`**: Cross-platform WGSL GPU compute dispatch via `wgpu`, automatically falling back to CPU/SIMD when no GPU adapter is available.
6. **`jhcs-platform*`**: Portable host (`jhcs-platform-host`) and bare-metal Jinn OS Ring 3 (`jhcs-platform-jinn`) runtime adapters.

---

## Workspace Layout

```text
jhcs-rs/
├── crates/
│   ├── jhcs-core/             # Task model, DAG resolver, buffer metadata (#![no_std])
│   ├── jhcs-broker/           # ComputeBroker, cost model & breakeven selection
│   ├── jhcs-backend-cpu/      # Deterministic scalar reference backend
│   ├── jhcs-backend-simd/     # CPU SIMD kernels with 64B cache-line optimization
│   ├── jhcs-backend-wgpu/     # Cross-platform WGSL GPU backend
│   ├── jhcs-platform/         # Platform traits (timers, allocators, device discovery)
│   ├── jhcs-platform-host/    # Linux / Windows std adapter
│   └── jhcs-platform-jinn/    # Jinn OS Ring 3 bare-metal adapter
├── examples/
│   ├── matrix-multiplication/ # Basic DAG MatMul + ReLU pipeline example
│   └── benchmark-breakeven/   # Reproducible Scalar vs SIMD vs wgpu breakeven CLI
└── docs/                      # Architecture specs, backend contracts & safety audits
```

---

## Quick Verification (Host & Bare-Metal)

```bash
# 1. Run workspace unit tests on Linux or Windows
cargo test --workspace

# 2. Run reproducible Scalar vs SIMD vs wgpu breakeven benchmark
cargo run --release --example benchmark-breakeven

# 3. Verify #![no_std] bare-metal compilation for kernel/microkernel environments
cargo check --target x86_64-unknown-none --no-default-features \
  -p jhcs-core -p jhcs-backend-cpu -p jhcs-backend-simd -p jhcs-platform -p jhcs-platform-jinn
```

## Documentation & License

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/COMPUTE_GRAPH.md`](docs/COMPUTE_GRAPH.md) · [`docs/BACKEND_CONTRACT.md`](docs/BACKEND_CONTRACT.md) · [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) · [`docs/LICENSING_POLICY.md`](docs/LICENSING_POLICY.md)
- **License:** Repository root [`LICENSE`](LICENSE) is **`Apache-2.0`** (see individual crate manifests and [`docs/LICENSING_POLICY.md`](docs/LICENSING_POLICY.md) for historical dual-licensing context and commercial OEM options).
