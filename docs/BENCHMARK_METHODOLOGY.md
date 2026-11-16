# ⏱️ JHCS — Metodologia de Benchmarks e Avaliação de Desempenho

> **Documento Oficial de Engenharia — jhcs-rs**  
> **Status:** Ativo | **Foco:** Medições Reproduzíveis, Segregação de Fases e Verificação de Custo

---

## 1. Princípio Fundamental de Medição

> **Aviso de Engenharia:** Nunca declarar que uma GPU acelerou um sistema baseando-se apenas no tempo de execução do kernel de shader ($T_{\text{exec}}$). Se o envio de dados e a sincronização consumirem mais tempo do que o cálculo na CPU, a transferência representou um atraso líquido.

Toda avaliação de benchmark no ecossistema JHCS mede e relata obrigatoriamente a equação canônica:

$$T_{\text{total}} = T_{\text{prep}} + T_{\text{transfer}} + T_{\text{exec}} + T_{\text{sync}}$$

| Fase | Descrição da Atividade Medida |
|---|---|
| **$T_{\text{prep}}$** | Tempo gasto alocando buffers, validando dimensões e estruturando descritores de tarefa. |
| **$T_{\text{transfer}}$** | Tempo gasto copiando dados da memória host para o buffer de dispositivo (VRAM ou pool DMA). |
| **$T_{\text{exec}}$** | Tempo efetivo gasto pelas unidades de execução (ALUs escalares, registradores SIMD ou Compute Units da GPU). |
| **$T_{\text{sync}}$** | Tempo de espera por cercas (*fences*), interrupções de conclusão e cópia de volta para a aplicação. |

---

## 2. Matriz de Teste de Multiplicação de Matrizes ($N \times N$)

A primeira prova de conceito (PoC 01) adota matrizes quadradas com dimensões crescentes:

| Dimensão ($N$) | Elementos Totais ($N^2$) | Volume de Dados FP32 | Comportamento Esperado |
|:---:|:---:|:---:|---|
| **$16 \times 16$** | 256 | 1 KiB | **CPU Escalar ou SIMD vencem amplamente.** O custo de despacho de GPU é 100x superior ao cálculo. |
| **$64 \times 64$** | 4.096 | 16 KiB | **CPU SIMD domina.** Dados cabem confortavelmente no cache L1 de dados da CPU. |
| **$256 \times 256$** | 65.536 | 256 KiB | Ponto de inflexão: SIMD opera em alta performance; overhead de GPU começa a ser amortizado. |
| **$1024 \times 1024$** | 1.048.576 | 4 MiB | **GPU Compute domina de forma massiva.** O paralelismo massivo compensa integralmente o overhead de DMA. |
