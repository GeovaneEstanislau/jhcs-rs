# ⚡ JHCS — Jinn Heterogeneous Compute Service (`jhcs-rs`)

> **Multi-Platform Cooperative & Heterogeneous Computing Library in Rust**  
> *Independent workspace for heterogeneous workload distribution across CPU, SIMD, and GPU.*

[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#no_std-compatibility)

---

## 1. O Princípio Oficial

# **Heterogeneous Computing by Design**
### *Computação cooperativa, adaptativa e independente de hardware.*

O **JHCS** parte da premissa fundamental de que nem todo processamento computacional deve recair sobre a CPU, e nem todo gargalo de CPU é resolvido transferindo trabalho cegamente para a GPU. 

O sistema analisa tarefas como grafos de dependência (DAG), avalia requisitos de precisão e tamanho de dados, e distribui as operações cooperativamente entre:
* **CPU Convencional:** Tarefas sequenciais, lógica de controle, pequenos lotes escalares.
* **FPU / SIMD:** Cálculo numérico e processamento vetorial (SSE, AVX, NEON).
* **GPU Compute:** Paralelismo massivo homogêneo, convoluções e multiplicação de matrizes grandes.

A ausência de acelerador secundário (GPU) **jamais compromete o funcionamento das aplicações**: o fallback para software / SIMD é garantido por construção.

---

## 2. Estrutura Modular do Workspace

O projeto é estruturado em crates desacopladas para permitir execução tanto em sistemas operacionais convencionais (**Windows**, **Linux**) quanto em ambientes embarcados ou microkernels bare-metal como o **Jinn OS (Ring 3)**:

```
jhcs-rs/
├── crates/
│   ├── jhcs-core/           # Contratos fundamentais, DAG, tarefas, buffers e capacidades (no_std)
│   ├── jhcs-broker/         # Compute Broker: filas, seleção de executor e políticas adaptativas
│   ├── jhcs-backend-cpu/    # Backend escalar de referência (baseline de precisão numérica)
│   ├── jhcs-backend-simd/   # Backend vetorial CPU (SSE/AVX com fallback automático)
│   ├── jhcs-backend-wgpu/   # Backend acelerado por GPU via WGSL compute shaders (opcional host)
│   ├── jhcs-platform/       # Traits e contratos de timers, alocadores e enumeração de hardware
│   ├── jhcs-platform-host/  # Adaptador de plataforma para Windows e Linux (std::time, threads)
│   └── jhcs-platform-jinn/  # Adaptador nativo para Jinn OS Ring 3 (KDM, JDA, IPC e shmem)
├── examples/                # Demonstrações práticas e benchmarks comparativos
├── docs/                    # Especificações arquiteturais e contratos de engenharia
└── benchmarks/              # Metodologia de medição de latência e throughput
```

---

## 3. Compatibilidade `no_std`

O núcleo `jhcs-core` e os adaptadores de baixo nível foram arquitetados para compilar sem a biblioteca padrão (`#![no_std]`), utilizando apenas a crate `alloc` para estruturas dinâmicas essenciais:

```bash
# Validação de compilação bare-metal para microkernels
cargo check --target x86_64-unknown-none --no-default-features -p jhcs-core -p jhcs-backend-cpu -p jhcs-backend-simd -p jhcs-platform -p jhcs-platform-jinn
```

---

## 4. Exemplo Rápido de Uso

```rust
use jhcs_core::task::{ComputeTask, TaskId, OP_MATMUL, OP_RELU};
use jhcs_core::buffer::{BufferSet, BufferHandle};
use jhcs_core::graph::ComputeGraph;
use jhcs_broker::{ComputeBroker, DispatchPolicy};

fn main() {
    let mut broker = ComputeBroker::new().with_policy(DispatchPolicy::DeterministicRuleBased);

    // 1. Definição declarativa do Grafo de Computação (DAG)
    let mut graph = ComputeGraph::new();

    let task1 = ComputeTask::new(
        TaskId::new(1),
        OP_MATMUL,
        BufferSet::from_handles(&[BufferHandle::new(10), BufferHandle::new(11)]),
        BufferSet::from_handles(&[BufferHandle::new(12)]),
    );

    let task2 = ComputeTask::new(
        TaskId::new(2),
        OP_RELU,
        BufferSet::from_handles(&[BufferHandle::new(12)]),
        BufferSet::from_handles(&[BufferHandle::new(13)]),
    ).with_dependency(TaskId::new(1)); // Encadeamento: ReLU depende de MatMul

    graph.add_task(task1).unwrap();
    graph.add_task(task2).unwrap();

    // 2. Validação de ausência de ciclos e determinação das etapas de execução
    let batches = graph.execution_batches().unwrap();
    println!("Etapas de execução independentes: {:?}", batches);

    // 3. Execução de multiplicação de matrizes 2x2
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [0.0f32; 4];

    let result = broker.execute_matmul(&graph.tasks()[0], &a, &b, &mut c, 2, 2, 2);
    println!("Resultado calculado pelo backend {:?}: {:?}", result.executed_by, c);
}
```

---

## 5. Documentação Técnica

* [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Modelo arquitetural completo e isolamento de anéis.
* [COMPUTE_GRAPH.md](docs/COMPUTE_GRAPH.md) — Especificação do modelo DAG e resolução de dependências.
* [BACKEND_CONTRACT.md](docs/BACKEND_CONTRACT.md) — Contratos e traits obrigatórias para implementar um backend.
* [PLATFORM_ADAPTERS.md](docs/PLATFORM_ADAPTERS.md) — Integração com Windows, Linux e Jinn OS.
* [BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) — Métricas de tempo de preparação, transferência e execução.
* [LICENSING_POLICY.md](docs/LICENSING_POLICY.md) — Política de licenciamento, direitos autorais e governança.
* [ROADMAP.md](docs/ROADMAP.md) — Roteiro de entregas em 8 fases.
