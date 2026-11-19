# 🚀 JHCS — Avaliação de Prontidão para a Fase 5 (Execução Concorrente em Grafos DAG)

> **Componente:** `jhcs-core::graph`, `jhcs-broker`  
> **Escopo:** Avaliação de Interfaces, Concorrência, Sincronização e Cancelamento Pré-Fase 5  
> **Classificação:** **APROVADO COM PENDÊNCIAS (Ready with Pre-requisites)**

---

## 1. Avaliação das Interfaces Existentes

A auditoria avaliou se os contratos atuais do `jhcs-core` e `jhcs-broker` suportam a transição para execução concorrente assíncrona sem ruptura de API pública:

| Requisito da Fase 5 | Interface Atual | Status | Análise Técnica |
|---|---|:---:|---|
| **Dependências em Grafo** | `ComputeGraph`, `topological_order()` | **PASS** | Algoritmo de Kahn detecta ciclos e ordena nós de forma determinística. |
| **Lotes Concorrentes** | `graph.execution_batches()` | **PASS** | Agrupa tarefas com dependências resolvidas em fatias paralelas disjuntas. |
| **Priorização de Tarefas** | `TaskPriority`, `deadline_ns` | **PASS** | Metadados prontos para fila de prioridades no despachante concorrente. |
| **Cancelamento de Tarefas** | `TaskStatus::Cancelled` | **PARTIAL** | Status existe, mas falta um `CancellationToken` atômico para interrupção cooperativa. |
| **Propagação de Erros** | `ComputeError::DependencyFailed` | **PASS** | Erros de nós anteriores invalidam nós dependentes à jusante. |
| **Mutabilidade do Broker** | `execute_*(&mut self)` | **FAIL** | Bloqueia múltiplas threads chamando o broker simultaneamente sem lock externo. |
| **Vinculação de Buffers (Dataflow)** | `BufferSet` | **PARTIAL** | Falta um registrador de memória para alimentar automaticamente saídas em entradas. |

---

## 2. As Três Pendências Obrigatórias Antes da Fase 5

Para que a implementação da Fase 5 seja sólida e não gere condições de corrida (*race conditions*) ou contenção de locks:

### 2.1. Pendência 1: Mutabilidade Interior no `MetricsRegistry`
- **Problema:** O método `MetricsRegistry::record` usa `&mut self`. Como consequência, o `ComputeBroker` precisa de `&mut self` para executar tarefas.
- **Ação:** Refatorar `OperationMetrics` para utilizar `AtomicU64` para contadores de execução e latência. Com isso, `MetricsRegistry::record` passará a receber `&self`, liberando o `ComputeBroker` para receber múltiplos despachos simultâneos de diferentes threads sem mutex global.

### 2.2. Pendência 2: Registro e Resolução de Buffers no Grafo (`BufferRegistry`)
- **Problema:** As tarefas possuem `inputs: BufferSet` e `outputs: BufferSet`, mas atualmente os buffers reais são passados como slices manuais nos métodos `execute_matmul(&task, &a, &b, &mut c, ...)`.
- **Ação:** Criar no `jhcs-core` ou `jhcs-broker` uma estrutura `BufferPool` / `BufferRegistry` que associe cada `BufferHandle` à sua memória alocada, permitindo que o executor do DAG execute nós encadeados automaticamente:
  $$\text{Nó 1 (Filtro)} \rightarrow \text{BufferHandle(10)} \rightarrow \text{Nó 2 (Composição)}$$

### 2.3. Pendência 3: Token Atômico de Cancelamento
- **Problema:** Se o Nó 1 de um batch falhar (ex.: erro de dimensão ou divisão por zero), os demais nós paralelos em execução no mesmo batch precisam ser notificados cooperativamente para cessar trabalho desnecessário.
- **Ação:** Implementar `AtomicBool` de cancelamento por execução de grafo (`ExecutionHandle::cancel()`).

---

## 3. Roteiro Passo a Passo de Execução da Fase 5

Assim que as três pendências forem ajustadas, a Fase 5 deverá ser implementada na seguinte ordem:

```text
Passo 5.1: Mutabilidade interior atômica em MetricsRegistry (desbloqueia &self no broker).
     │
     ▼
Passo 5.2: Implementação do BufferRegistry para resolução autônoma de dados no DAG.
     │
     ▼
Passo 5.3: Construção do DagExecutor (processamento paralelo por lotes de nós independentes).
     │
     ▼
Passo 5.4: Testes concorrentes de estresse (corrida de dados, cancelamento e falha induzida).
```

---

## 4. Decisão Conclusiva da Homologação

> **Veredito Oficial:** **APROVADO COM PENDÊNCIAS**
>
> As fundações das Fases 1 a 4 estão matematicamente corretas, determinísticas e 100% testadas. A arquitetura está pronta para receber a execução concorrente da Fase 5 assim que os três requisitos de sincronização e mutabilidade interior forem incorporados.
