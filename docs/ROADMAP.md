# 🗺️ JHCS — Roteiro Estratégico de Desenvolvimento (8 Fases)

> **Documento Oficial de Engenharia — jhcs-rs**  
> **Status:** Ativo | **Execução:** Incremental e Verificável

---

## 1. As Oito Fases Evolutivas

```mermaid
flowchart LR
    F1["Fase 1: jhcs-core & DAG (Host)"] --> F2["Fase 2: Backend CPU & SIMD"]
    F2 --> F3["Fase 3: Backend GPU wgpu (Host)"]
    F3 --> F4["Fase 4: Broker Adaptativo & Benchmarks"]
    F4 --> F5["Fase 5: Platform Adapter Jinn Ring 3"]
    F5 --> F6["Fase 6: Integração JDA / KDM"]
    F6 --> F7["Fase 7: Backend GPU Nativo Jinn"]
    F7 --> F8["Fase 8: Aquecimento & Preempção"]
```

---

## 2. Detalhamento e Estado de Execução

| Fase | Escopo Técnico | Ambiente Alvo | Estado de Maturidade |
|:---:|---|:---:|:---:|
| **Fase 1** | **`jhcs-core` & DAG:** Contratos fundamentais, tarefas, buffers e detecção de ciclos por Kahn. | Windows / Linux / Bare-metal | `IMPLEMENTADO E VALIDADO` |
| **Fase 2** | **Backend CPU & SIMD:** Kernel escalar de referência (`matmul`, `relu`, `vector_add`) e SIMD otimizado com paridade $\epsilon \le 10^{-4}$. | Windows / Linux / Bare-metal | `IMPLEMENTADO E VALIDADO` |
| **Fase 3** | **Backend GPU (wgpu):** Compute shaders em WGSL para matmul em GPUs DirectX/Vulkan/Metal. | Windows / Linux Host | `PROJETADO` |
| **Fase 4** | **Compute Broker Adaptativo:** Despacho dinâmico por regras e medição de telemetria ($T_{\text{total}}$). | Windows / Linux Host | `IMPLEMENTADO PARCIALMENTE` |
| **Fase 5** | **Platform Adapter Jinn:** Bindings de Ring 3 com `#![no_std]`, timer TSC e gerenciador de memória. | Jinn OS Ring 3 | `PROJETADO / CONTRATO PRONTO` |
| **Fase 6** | **Integração JDA / KDM:** Conexão com anéis assíncronos (`JdaSqe`/`JdaCqe`) e DMA (Syscalls 60..68). | Jinn OS | `PLANEJADO` |
| **Fase 7** | **Backend GPU Nativo Jinn:** Emissão de comandos para VirtIO-GPU (`1AF4:1050`) e Bochs BGA sem wgpu. | Jinn OS | `PLANEJADO` |
| **Fase 8** | **Políticas de Aquecimento e Preempção:** Coordenação com preempção e limite estrito sob 64 MB. | Jinn OS | `PLANEJADO` |
