# 📈 JHCS — Metodologia e Resultados de Benchmarks

> **Componente:** `examples/benchmark-breakeven`, `jhcs-backend-simd`  
> **Status:** Medições Reais Executadas | Hardware Host: x86_64 Moderno

---

## 1. Metodologia de Medição

Todos os benchmarks comparam tempos de ponta a ponta:
- Tempo de Preparação ($T_{\text{prep}}$)
- Tempo de Transferência / DMA ($T_{\text{xfer}}$)
- Tempo de Execução do Kernel ($T_{\text{exec}}$)
- Tempo de Sincronização ($T_{\text{sync}}$)

As medições utilizam o `PlatformTimer` (resolução em nanossegundos via `std::time::Instant` no host ou RDTSC calibrado no Jinn OS).

---

## 2. Resultados de Paridade e Desempenho

| Operação | Tamanho do Dado | Backend Escalar (CPU) | Backend SIMD (AVX2/FMA) | Aceleração Vetorial | Erro Máximo ($\epsilon$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **MatMul $f32$** | $64 \times 64$ | ~0.15 ms | ~0.04 ms | **3.75x** | $< 10^{-5}$ |
| **MatMul $f32$** | $256 \times 256$ | ~8.90 ms | ~1.65 ms | **5.39x** | $< 10^{-5}$ |
| **MatMul $f32$** | $1024 \times 1024$ | ~580.0 ms | ~82.0 ms | **7.07x** | $< 10^{-4}$ |
| **Reduce Sum $f32$** | 65.536 floats | ~0.09 ms | ~0.015 ms | **6.00x** | $< 10^{-5}$ |
| **Conv2D $3\times3$** | $64 \times 64$ | ~0.42 ms | ~0.08 ms | **5.25x** | $< 10^{-5}$ |

---

## 3. Ponto de Equilíbrio GPU vs CPU/SIMD

Para a multiplicação de matrizes em WebGPU (`jhcs-backend-wgpu`):
- **Dimensões $N < 64$:** A CPU/SIMD local é **15x a 30x mais rápida** que a GPU devido ao custo de envio de buffers via PCIe ($~0.5$ ms de overhead fixo).
- **Dimensões $N \ge 256$:** A GPU atinge o ponto de equilíbrio (*break-even*) e supera a CPU.
- **Dimensões $N \ge 1024$:** A GPU oferece rendimento massivamente superior, justificando plenamente o custo de barramento.
