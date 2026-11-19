# 💾 JHCS — Auditoria de Memória, Gerenciamento de Buffers e Restrições de Sistema

> **Componente:** `jhcs-core::buffer`, `jhcs-platform`  
> **Escopo:** Alocações, Ownership, Alinhamento, Fragmentação e Operação sob o Perfil 64 MB do Jinn  
> **Classificação:** **PASS (Modelagem e Alinhamento Robustos; Alerta de Teto sob 64 MB)**

---

## 1. Auditoria de Alocações Intermediárias e Sobrecarga de Heap

A auditoria inspecionou todos os pontos de alocação dinâmica (`alloc`) nos módulos centrais:

### 1.1. Kernels de Execução Matemática
- **Arquivos:** `crates/jhcs-backend-cpu/src/ops.rs`, `matmul.rs` e `crates/jhcs-backend-simd/src/matmul.rs`.
- **Resultado:** **Zero alocações dinâmicas no heap.** Todos os laços de cálculo operam exclusivamente sobre fatias de memória pré-alocadas (`&[f32]` e `&mut [f32]`). Não há chamadas a `Vec::new()`, `alloc()` ou clonagens ocultas durante o cálculo numérico.
- **Status:** **PASS (Eficiência Máxima)**.

### 1.2. Resolução do Grafo DAG (`ComputeGraph`)
- **Arquivo:** `crates/jhcs-core/src/graph.rs`.
- **Resultado:** O método `topological_order()` aloca 3 vetores temporários (`in_degree`, `ready`, `order`) para executar o algoritmo de Kahn.
- **Avaliação:** Para grafos típicos de UI e pipelines de imagem (com 5 a 30 tarefas), a sobrecarga é desprezível ($< 1$ KiB de alocação temporária). No entanto, para sistemas embarcados *hard real-time* sem heap dinâmico, uma futura arena estática const-generic será desejável.

---

## 2. Auditoria do Ciclo de Vida e Ownership de Buffers

O modelo implementado em `crates/jhcs-core/src/buffer.rs` atende aos requisitos de isolamento:

1. **Tokens Opacos:** Aplicações de usuário recebem apenas `BufferHandle(u32)`, impedindo que processos em Ring 3 forjem ponteiros para endereços físicos do kernel.
2. **Semântica de Acesso:** Os modos `ReadOnly`, `WriteOnly` e `ReadWrite` permitem ao Broker impedir que múltiplos nós paralelos do DAG realizem escritas conflitantes sobre a mesma região de memória.
3. **Estados de Sincronização de Cache:** O enum `BufferSyncState` (`Clean`, `Modified`, `InFlight`, `Incoherent`) mapeia com precisão a coerência de cache necessária quando buffers transitam entre CPU e barramento PCI.
4. **Alinhamento de 64 Bytes:** Todos os descritores de buffer utilizam `alignment: 64`. Isso garante que qualquer buffer seja alinhado exatamente ao início de uma linha de cache x86_64, saturando os barramentos AVX-512 e DMA sem penalidades de divisão (*split-line penalty*).

---

## 3. Avaliação Estrita sob o Perfil de 64 MB de Memória do Jinn OS

No Jinn OS, computadores legados e ambientes embarcados de teste operam sob o perfil **JAHA 1.0 Legacy** com apenas **64 MB de RAM total**:

```text
Orçamento Total do Processo (Perfil Legacy): 14 MiB
```

### Análise de Consumo para Matrizes e Imagens:
- Uma matriz de floats de $1024 \times 1024$ consome:
  $$1024 \times 1024 \times 4 \text{ bytes} = 4.194.304 \text{ bytes} = 4 \text{ MiB}$$
- Uma operação simples de multiplicação $C = A \times B$ demanda 3 matrizes ($A$, $B$ e $C$), consumindo:
  $$3 \times 4 \text{ MiB} = 12 \text{ MiB}$$
- **Risco Crítico no Perfil Legacy:** Alocar 12 MiB de tensores em um processo restrito a 14 MiB consome **85,7% de toda a memória disponível do processo**, restando menos de 2 MiB para pilha (*stack*), binário ELF, IPC e buffers de mensagens.

### Diretrizes Obrigatórias de Gestão de Memória:
1. **Particionamento por Blocos (*Tiling*):** Para cargas com $N \ge 512$, utilizar obrigatoriamente a função `matmul_blocked_f32_scalar` com blocos de $64 \times 64$, reutilizando buffers temporários menores.
2. **Teto de Alocação do Broker:** O `ComputeBroker` deve rejeitar buffers individuais maiores que 4 MiB quando o perfil de hardware detectado for `Legacy`.
3. **Reutilização de Buffers (In-Place):** Promover o uso de `allow_in_place = true` para operações monádicas (como ReLU, Scaling e Clamping), reduzindo pela metade o consumo de memória transitória.
