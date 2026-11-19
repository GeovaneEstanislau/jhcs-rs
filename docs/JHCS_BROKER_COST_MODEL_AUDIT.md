# ⚖️ JHCS — Auditoria do Compute Broker e Modelo de Custo

> **Componente:** `jhcs-broker`  
> **Escopo:** Seleção de Executores, Políticas, Modelo de Custo, Fallback e Concorrência  
> **Classificação:** **PARTIAL (Aprovado em Regras e Fallback; Pendente em Concorrência de Despacho)**

---

## 1. Verificação das Políticas de Despacho (`policy.rs`)

A auditoria inspecionou as 6 políticas implementadas no Broker:

1. **`CpuOnly`:** `PASS`. Despacha invariavelmente para `BackendKind::CpuScalar`, ignorando disponibilidade de SIMD ou GPU. Ideal para validação de referência e recuperação de falhas.
2. **`PreferSimd`:** `PASS`. Seleciona `BackendKind::CpuSimd` para qualquer operação com $N \ge 8$ elementos se instruções vetoriais estiverem presentes.
3. **`LatencyFirst`:** `PASS`. Prioriza CPU/SIMD local para evitar latência de sincronização de barramento PCIe, bloqueando transferências para GPU.
4. **`ThroughputFirst`:** `PASS`. Descarrega tarefas com $N \ge 1024$ elementos na GPU para maximizar vazão em lote.
5. **`Balanced` (Padrão):** `PASS`. Avalia o nível de paralelismo declarado (`Sequential`, `Vectorized128/256/512`, `MassiveParallelGpu`) e o tamanho do problema.
6. **`CostOptimized`:** `PARTIAL`. Consulta `MetricsRegistry` buscando `avg_exec_ns()` e `total_overhead_ns()`. 

### Auditoria do Modelo de Custo: Heurística Estática vs. Aprendizado Dinâmico
- **Constatação da Auditoria:** As políticas `LatencyFirst`, `ThroughputFirst` e `Balanced` utilizam **limiares estáticos determinísticos** ($64$, $128$, $1024$ elementos). 
- O modo `CostOptimized` utiliza médias aritméticas acumuladas. Ele **não é ainda um modelo adaptativo preditivo contínuo**, mas sim uma tabela histórica de latências médias. Essa abordagem é prudente para a fase atual pois evita instabilidade nas decisões de agendamento.

---

## 2. Auditoria do Mecanismo de Fallback e Idempotência

O algoritmo de fallback implementado em `crates/jhcs-broker/src/dispatcher.rs` foi auditado contra os critérios da Seção 8.3 do Prompt Mestre:

```rust
let mut result = match chosen {
    BackendKind::CpuSimd => {
        let res = self.simd_backend.execute_matmul(task.id, a, b, c, m, k, n);
        if res.status != TaskStatus::Completed && task.requirements.allow_fallback {
            self.cpu_backend.execute_matmul(task.id, a, b, c, m, k, n)
        } else {
            res
        }
    }
    BackendKind::CpuScalar | _ => {
        self.cpu_backend.execute_matmul(task.id, a, b, c, m, k, n)
    }
};
```

### Análise de Idempotência e Integridade de Dados:
- **Operações Matemáticas Puras:** Multiplicação de matrizes, adição de vetores, convolução 2D e redução por soma são **funções matemáticas puras sem efeitos colaterais**.
- **Segurança pós-falha:** Caso o backend SIMD ou GPU falhe antes de completar o cálculo, o buffer de saída $C$ é reinicializado com zeros (`c.fill(0.0)`) pela rotina escalar de fallback, garantindo que nenhum estado parcialmente computado corrompa o resultado final.
- **Status do Fallback:** **PASS (100% Idempotente e Seguro)**.

---

## 3. Bloqueador de Concorrência Identificado para a Fase 5

Durante a auditoria de preparação para o paralelismo em grafos DAG (Fase 5), identificou-se um gargalo arquitetural em `crates/jhcs-broker/src/dispatcher.rs`:

```rust
// Assinatura atual:
pub fn execute_matmul(&mut self, task: &ComputeTask, ...) -> ComputeResult
```

- **Causa Raiz:** O método exige referência mutável exclusiva (`&mut self`) porque chama `self.metrics.record(...)`, que insere dados em um `BTreeMap<(u8, &'static str), OperationMetrics>`.
- **Impacto na Fase 5:** Um executor concorrente que processe múltiplos nós independentes de um DAG simultaneamente através de várias threads ou fibras **não conseguirá despachar operações concorrentes sobre uma mesma instância de `ComputeBroker`** sem travamento ou contenção severa.
- **Solução Mandatória para a Fase 5:**
  1. Converter os contadores do `OperationMetrics` para tipos atômicos (`core::sync::atomic::AtomicU64`).
  2. Prover mutabilidade interior para o `MetricsRegistry`, permitindo que os métodos de despacho do `ComputeBroker` recebam apenas `&self`.
