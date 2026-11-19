# 🧠 JHCS — Compute Broker & Políticas de Despacho

> **Crate:** `jhcs-broker` | **Status:** Implementado & Validado  
> **Camada:** Ring 3 Daemon / Dispatcher

---

## 1. Responsabilidade do Compute Broker

O **Compute Broker** é o motor de decisão do JHCS. Ele atua como um intermediário inteligente entre as intenções de cálculo das aplicações e os executores de hardware.

A seleção do executor **jamais se baseia unicamente na frequência ou no número teórico de núcleos**, mas no tempo total estimado de ponta a ponta.

---

## 2. Modelo Formal de Custo

Antes de despachar uma tarefa para um acelerador (GPU ou unidade vetorial remota), o Broker avalia a equação de viabilidade econômica:

$$T_{\text{total}} = T_{\text{preparação}} + T_{\text{transferência}} + T_{\text{execução}} + T_{\text{sincronização}} + T_{\text{recuperação}}$$

- **Ponto de Equilíbrio (Break-Even):** Para matrizes pequenas ($N < 64$), o custo $T_{\text{transferência}} + T_{\text{sincronização}}$ da GPU supera amplamente qualquer ganho de processamento massivo. O Broker despacha essas cargas localmente para a CPU/SIMD.
- **Incerteza e Conservadorismo:** Na ausência de métricas prévias de telemetria, o Broker adota a postura conservadora (CPU/SIMD local), garantindo menor latência imediata.

---

## 3. Políticas Canônicas de Despacho (`DispatchPolicy`)

1. **`CpuOnly`:** Restringe 100% dos despachos ao backend escalar da CPU. Utilizado para auditorias de correção, depuração ou recuperação em caso de falha de hardware.
2. **`PreferSimd`:** Prioriza unidades vetoriais locais (SSE/AVX/NEON) sempre que disponíveis, eliminando tráfego de barramento PCI.
3. **`LatencyFirst`:** Otimizado para interfaces visuais interativas (Aladdin Desktop). Evita sincronizações lentas e mantém a taxa de quadros estável em 60 FPS.
4. **`ThroughputFirst`:** Otimizado para grandes volumes de dados (redimensionamento de imagens, inferência em lote). Descarrega tarefas em GPUs com paralelismo massivo.
5. **`Balanced` (Padrão):** Política adaptativa que chaveia automaticamente entre CPU escalar, SIMD e GPU com base no volume de elementos e histórico de latência.

---

## 4. Algoritmo de Degradação Graciosa e Fallback

Quando um executor secundário (como GPU ou acelerador dedicado) falha durante a execução ou encontra recursos esgotados:
1. A falha é capturada e registrada no `MetricsRegistry`.
2. O Broker verifica a integridade dos buffers de entrada (`BufferSyncState::Clean`).
3. Se `allow_fallback == true`, a operação é reenviada de forma idempotente para o executor CPU/SIMD.
4. O resultado final retorna com status de sucesso e anotação explícita da estratégia e do executor de fallback.
