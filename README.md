# JHCS-RS — Executor Heterogêneo Portátil (CPU Escalar, SIMD e GPU `wgpu`)

> **EN:** [README.en.md](README.en.md) | **Ecossistema:** [Jinn_Project](https://github.com/GeovaneEstanislau/Jinn_Project) · [Aladdin](https://github.com/GeovaneEstanislau/Aladdin)

O **`jhcs-rs`** (*Jinn Heterogeneous Compute Service*) é o componente independente de **menor atrito para adoção imediata em ambientes host (Linux/Windows) e microkernels (`#![no_std]` / Jinn OS Ring 3)**. Ele modela cargas numéricas e gráficas como grafos acíclicos dirigidos (DAG) e decide dinamicamente entre **CPU escalar**, **CPU vetorial SIMD (`AVX2`/`FMA`)** e **GPU (`wgpu` WGSL)** usando modelos explícitos de custo de transferência e ponto de equilíbrio (*breakeven*).

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![no_std Core](https://img.shields.io/badge/core-no__std%20ready-success.svg)](#compatibilidade-nostd)

---

## Por Que Começar pelo `jhcs-rs`

Ao contrário de adotar um sistema operacional completo, bibliotecas de infraestrutura e equipes de engenharia podem integrar e auditar o **`jhcs-rs`** diretamente em pipelines Rust no Linux ou Windows:

1. **`jhcs-core`** (`#![no_std]` + `alloc`): Definição de tarefas, `BufferSet` alinhado e resolução topológica de DAG (`ComputeGraph`).
2. **`jhcs-broker`**: *ComputeBroker* com políticas determinísticas/adaptativas que ponderam tamanho do lote, custo de despacho/serialização e disponibilidade real de aceleradores.
3. **`jhcs-backend-cpu`**: Baseline escalar determinístico (vence em micro-lotes onde o custo de transferência/despacho supera o ganho paralelo).
4. **`jhcs-backend-simd`**: Kernels vetoriais em CPU (`matmul`, `conv2d`, `box_blur_rgba_simd` com *prefetch* explícito de cache line de 64 bytes).
5. **`jhcs-backend-wgpu`**: Despacho GPU multi-plataforma via shaders WGSL (`wgpu`), com *fallback* garantido em hardware sem GPU.
6. **`jhcs-platform*`**: Adaptadores portáveis para host (`jhcs-platform-host`) e para o microkernel Jinn (`jhcs-platform-jinn`).

---

## Estrutura do Workspace

```text
jhcs-rs/
├── crates/
│   ├── jhcs-core/            # Contratos de tarefas, grafos DAG e capacidades (no_std)
│   ├── jhcs-broker/          # Orquestração, filas e política adaptativa de breakeven
│   ├── jhcs-backend-cpu/     # Implementação escalar de referência
│   ├── jhcs-backend-simd/    # Kernels SIMD (SSE/AVX2/FMA com alinhamento de 64B)
│   ├── jhcs-backend-wgpu/    # Compute shaders WGSL em GPU (host)
│   ├── jhcs-platform/        # Traits portáveis de tempo, memória e enumeração
│   ├── jhcs-platform-host/   # Implementação std para Linux / Windows
│   └── jhcs-platform-jinn/   # Integração bare-metal para Jinn OS Ring 3
├── examples/
│   ├── matrix-multiplication/ # Demonstração DAG MatMul + ReLU
│   └── benchmark-breakeven/   # Benchmark reproduzível CPU Escalar vs SIMD vs wgpu
└── docs/                     # Contratos de backend, modelo de custo e auditorias
```

---

## Validação Rápida (Host & Bare-Metal)

```bash
# 1. Executar testes unitários em todo o workspace host (Linux / Windows)
cargo test --workspace

# 2. Executar benchmark de ponto de equilíbrio (breakeven) Escalar vs SIMD vs wgpu
cargo run --release -p benchmark-breakeven

# 3. Validar compilação bare-metal (#![no_std]) das crates de núcleo e plataforma Jinn
cargo check --target x86_64-unknown-none --no-default-features \
  -p jhcs-core -p jhcs-backend-cpu -p jhcs-backend-simd -p jhcs-platform -p jhcs-platform-jinn
```

---

## Exemplo Mínimo em Rust

```rust
use jhcs_core::task::{ComputeTask, TaskId, OP_MATMUL, OP_RELU};
use jhcs_core::buffer::{BufferSet, BufferHandle};
use jhcs_core::graph::ComputeGraph;
use jhcs_broker::{ComputeBroker, DispatchPolicy};

fn main() {
    let mut broker = ComputeBroker::new().with_policy(DispatchPolicy::DeterministicRuleBased);
    let mut graph = ComputeGraph::new();

    let t1 = ComputeTask::new(
        TaskId::new(1),
        OP_MATMUL,
        BufferSet::from_handles(&[BufferHandle::new(10), BufferHandle::new(11)]),
        BufferSet::from_handles(&[BufferHandle::new(12)]),
    );
    let t2 = ComputeTask::new(
        TaskId::new(2),
        OP_RELU,
        BufferSet::from_handles(&[BufferHandle::new(12)]),
        BufferSet::from_handles(&[BufferHandle::new(13)]),
    ).with_dependency(TaskId::new(1));

    graph.add_task(t1).unwrap();
    graph.add_task(t2).unwrap();
    let batches = graph.execution_batches().unwrap();
    println!("Etapas independentes do DAG: {:?}", batches);
}
```

## Documentação e Licença

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/COMPUTE_GRAPH.md`](docs/COMPUTE_GRAPH.md) · [`docs/BACKEND_CONTRACT.md`](docs/BACKEND_CONTRACT.md) · [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) · [`docs/LICENSING_POLICY.md`](docs/LICENSING_POLICY.md)
- **Licença:** [`Apache-2.0`](LICENSE) no repositório raiz (consulte os manifests individuais e [`docs/LICENSING_POLICY.md`](docs/LICENSING_POLICY.md) para detalhes de governança e opção comercial/OEM).
