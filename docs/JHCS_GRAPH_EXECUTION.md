# 🕸️ JHCS — Grafo Computacional e Execução em DAG

> **Componente:** `jhcs-core::graph`, `jhcs-broker`  
> **Status:** Implementado, Validado e Testado

---

## 1. O Grafo Acíclico Dirigido (`ComputeGraph`)

O **ComputeGraph** permite encadear múltiplas operações computacionais, eliminando interrupções intermediárias e viagens de ida e volta (*round-trips*) entre CPU e aceleradores.

### Vantagens do Grafo
- **Resolução de Dependências em Espaço de Usuário:** O grafo é montado e validado em Ring 3 pelo JHCS, sem consumir tempo de kernel.
- **Detecção de Ciclos:** Implementa o algoritmo de ordenação topológica de Kahn. Se houver dependência circular, o grafo rejeita a submissão com `ComputeError::CycleDetected`.
- **Lotes Concorrentes de Execução (`execution_batches`):** Tarefas independentes (grau de entrada 0) são agrupadas no mesmo lote e despachadas para execução concorrente em múltiplos núcleos de CPU ou unidades de execução.

---

## 2. Exemplo de Pipeline de Composição Visual

```text
Entrada: Textura Decodificada
   │
   ├── Filtro Bilinear (SIMD / GPU) ──┐
   │                                   │
   └── Matriz de Projeção (FPU / SIMD) ──┤
                                       ▼
                             Composição de Janela
                                       │
                                       ▼
                           Blit no Framebuffer (Display)
```

---

## 3. Cancelamento Seguro e Propagação de Erros

- Se uma tarefa predecessora falhar:
  - Todas as tarefas dependentes à jusante são imediatamente marcadas com `ComputeError::DependencyFailed(id)`.
  - Tarefas independentes em outros ramos do grafo continuam sua execução normalmente.
  - Buffers alocados em lote são reciclados ou liberados de forma determinística.
