# ⚙️ JHCS — Arquitetura e Design dos Backends

> **Componente:** `jhcs-backend-cpu`, `jhcs-backend-simd`, `jhcs-backend-wgpu`  
> **Status:** Implementado & Validado com Testes de Paridade

---

## 1. Backend CPU Escalar (`jhcs-backend-cpu`)

O backend escalar é a referência fundamental de integridade matemática de todo o ecossistema. Ele opera sem necessidade de GPU, sem instruções vetoriais e sem assumir qualquer suporte a hardware avançado.

### Operações Suportadas
1. **Multiplicação de Matrizes (`matmul_f32_scalar` & `matmul_blocked_f32_scalar`):** Algoritmo com loops aninhados $i$-$p$-$j$ orientados a cache e variante particionada por blocos/tiles.
2. **Operações Vetoriais Elemento a Elemento:**
   - Adição: $C_i = A_i + B_i$
   - Multiplicação: $C_i = A_i \cdot B_i$
   - Escala: $C_i = A_i \cdot \text{fator}$
   - ReLU: $C_i = \max(0, A_i)$
   - Clamping: $C_i = \min(\max(A_i, \text{min}), \text{max})$
3. **Reduções Estatísticas:**
   - Soma: $\sum A_i$
   - Média: $\frac{1}{N} \sum A_i$
   - Mínimo e Máximo: $(\min(A), \max(A))$
4. **Produto Escalar:** $\sum (A_i \cdot B_i)$
5. **Processamento In-Place Seguro:** Operações que alteram diretamente o buffer sem alocação intermediária quando o contrato declara `allow_in_place = true`.

---

## 2. Backend CPU SIMD / Vetorial (`jhcs-backend-simd`)

O backend SIMD acelera operações homogêneas em dados utilizando unidades de registradores vetoriais da CPU:

- **SSE2 (128 bits):** Registradores `XMM` (4 floats por instrução).
- **AVX2 / FMA (256 bits):** Registradores `YMM` (8 floats por instrução) com saturação de portas FMA (`_mm256_fmadd_ps`).
- **Detecção Dinâmica:** CPUID em tempo de execução no `#![no_std]` e `std`, prevenindo instruções ilegais (`#UD`).
- **Garantia de Fallback:** Qualquer operação para a qual a CPU não suporte as instruções requeridas ou cujo tamanho seja inferior ao limiar de ganho vetorial executa automaticamente pela implementação escalar de referência.

---

## 3. Backend GPU (`jhcs-backend-wgpu` & Futuro KDM Jinn)

- **Host (Windows/Linux):** Utiliza a API padronizada WebGPU (`wgpu`) com shaders de computação WGSL (`shader_matmul.wgsl`), medindo separadamente o custo de transferência (upload/download) e de despacho.
- **Jinn OS NATIVO:** Integração com o Kernel DMA Manager (KDM) e VirtIO-GPU via anéis de submissão JDA, sem dependência de runtimes pesados no microkernel.
