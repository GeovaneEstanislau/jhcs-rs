# 📜 JHCS — Contrato Formal de Backends de Execução

> **Documento Oficial de Engenharia — jhcs-rs**  
> **Status:** Ativo | **Módulos:** `jhcs-core::capabilities`, `jhcs-core::task`

---

## 1. O Contrato do Backend

Qualquer unidade executora (CPU escalar, CPU SIMD, GPU wgpu, ou acelerador nativo Jinn) deve satisfazer rigorosamente as seguintes cláusulas:

### Cláusula 1: Descrição e Descoberta de Capacidades
Cada backend deve fornecer estaticamente e dinamicamente:
* `descriptor(&self) -> &BackendDescriptor`: Identificador único, tipo de dispositivo (`BackendKind`), nome descritivo e memória total gerenciável.
* `capabilities(&self) -> &ComputeCapabilities`: Operações suportadas, grau máximo de paralelismo (`ParallelismClass`), capacidade de suporte a FP32/FP64/Int32 e tamanho máximo de buffer.

### Cláusula 2: Paridade Numérica Estrita (Ground-Truth Reference)
* O backend `jhcs-backend-cpu` é a **verdade fundamental matemática** do sistema.
* Qualquer resultado gerado por uma implementação SIMD (SSE/AVX) ou GPU (WGSL compute shader) deve convergir com a saída escalar com desvio máximo absoluto limitado por:
  $$|y_{\text{backend}}[i] - y_{\text{cpu}}[i]| \le \epsilon = 10^{-4}$$
* Divergências superiores a $\epsilon$ configuram bug de implementação do backend ou erro de precisão numérica inadmissível.

### Cláusula 3: Telemetria de Duração e Isolamento de Sobrecarga
Todo resultado retornado através de `ComputeResult` deve segregar com precisão os tempos de:
* `prep_duration_ns`: Tempo gasto alocando ou preparando estruturas.
* `transfer_duration_ns`: Tempo gasto copiando dados entre host e dispositivo (se aplicável).
* `exec_duration_ns`: Tempo puro gasto na computação.
* `sync_duration_ns`: Tempo gasto esperando conclusão e transferência de volta.

Isso permite que o Compute Broker avalie de forma realista se a aceleração compensou o custo de transferência.
