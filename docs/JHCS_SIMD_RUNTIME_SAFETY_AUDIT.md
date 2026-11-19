# 🛡️ JHCS — Auditoria de Segurança em Tempo de Execução SIMD & FPU

> **Componente:** `jhcs-backend-simd`  
> **Escopo:** Detecção de Instruções, Preservação de Registradores, Alinhamento e Integridade Numérica  
> **Classificação:** **PARTIAL (Aprovado no Host; SSE2 Aprovado no Jinn; AVX2 Bloqueado no Jinn)**

---

## 1. Mapeamento de Instruções e Detecção Dinâmica

O `jhcs-backend-simd` implementa dois caminhos de detecção na função `detect_avx2_fma()`:

### 1.1. Modo `std` (Host Windows / Linux)
- Utiliza as macros oficiais do compilador Rust:
  `is_x86_feature_detected!("avx2")` e `is_x86_feature_detected!("fma")`.
- **Comportamento:** O runtime da biblioteca padrão do Rust verifica não apenas as folhas CPUID `0x01` e `0x07`, mas também executa `_xgetbv(0)` para garantir que o sistema operacional tenha ativado o suporte a estados estendidos no registrador `XCR0`.
- **Status no Host:** **PASS (100% Seguro)**.

### 1.2. Modo `no_std` (Bare-Metal / Jinn OS)
- O código em `crates/jhcs-backend-simd/src/ops.rs` (linhas 19–34) executa:
  ```rust
  let res = core::arch::x86_64::__cpuid(1);
  let has_fma = (res.ecx & (1 << 12)) != 0;
  let has_osxsave = (res.ecx & (1 << 27)) != 0;
  let has_avx = (res.ecx & (1 << 28)) != 0;
  if has_osxsave && has_avx {
      let res7 = core::arch::x86_64::__cpuid_count(7, 0);
      let has_avx2 = (res7.ebx & (1 << 5)) != 0;
      (has_avx2, has_fma)
  } else {
      (false, false)
  }
  ```
- **Vulnerabilidade Identificada:** O código checa `OSXSAVE` (`CR4.OSXSAVE = 1`), mas **NÃO executa a instrução `XGETBV`** para ler `XCR0`. 
- **Risco:** Se o processador suportar AVX2 e o kernel tiver ativado `CR4.OSXSAVE`, mas o kernel tiver habilitado apenas o estado SSE (`XCR0 = 0x03`) sem habilitar o bit AVX (`XCR0[2] = 1`), a execução de qualquer mnemônico AVX (`_mm256_*`) disparará a interrupção de hardware `#UD` (Invalid Opcode), travando a aplicação em Ring 3.
- **Ação Corretiva Necessária:** Incluir a validação explícita de `_xgetbv(0)`:
  ```rust
  let xcr0 = core::arch::x86_64::_xgetbv(0);
  let avx_enabled = (xcr0 & 0x06) == 0x06; // Bits 1 (SSE) e 2 (AVX) devem ser 1
  ```

---

## 2. Preservação de Contexto e Troca de Tarefas no Jinn Microkernel

A auditoria inspecionou diretamente os arquivos de escalonamento e FPU do microkernel Jinn (`src/fpu.rs` e `src/scheduler.rs`):

```rust
// D:\Projetos\Jinn\src\fpu.rs
pub unsafe fn fxsave64(state: *mut FpuState) {
    core::arch::asm!("fxsave64 [{0}]", in(reg) state);
}
```

### Análise de Risco no Escalonador Preemptivo:
1. O Jinn OS utiliza atualmente a instrução legada de 64 bits **`fxsave64` / `fxrstor64`** para salvar e restaurar o estado numérico das tarefas preemptadas.
2. A especificação do Intel Software Developer's Manual (SDM Vol. 1, Seção 10.5.1) determina que `fxsave64` salva estritamente:
   - x87 FPU / MMX
   - Registradores `XMM0` a `XMM15` (128 bits cada).
3. **`fxsave64` NÃO SALVA a metade superior (bits 128..255) dos registradores `YMM0..YMM15`.**
4. **Consequência Crítica:** Se uma tarefa do JHCS no Jinn OS executar cálculos AVX2 (que utilizam registradores `YMM`) e sofrer preempção pelo timer PIT/APIC ou por troca de contexto IPC:
   - Apenas os primeiros 128 bits de cada acumulador serão salvos na struct `Task`.
   - A metade superior de 128 bits será **silenciosamente corrompida**, produzindo anomalias numéricas catastróficas ou corrupção de memória.

### Diretriz de Isolamento para o Jinn OS:
- **SSE2 (128 bits):** **TOTALMENTE SEGURO (PASS).** Os registradores `XMM0..XMM15` são rigorosamente preservados por `fxsave64` em todas as trocas de contexto em `src/scheduler.rs`.
- **AVX2 (256 bits):** **ESTRITAMENTE BLOQUEADO NO JINN (FAIL / BLOCKED)** até que a Etapa B do roadmap oficial do kernel implemente `XSAVE64`/`XRSTOR64` com buffer de 832+ bytes e ativação de `XCR0[2]`.

---

## 3. Segurança de Alinhamento e Manipulação de Memória

- Os kernels SIMD implementados em `crates/jhcs-backend-simd/src/matmul.rs` e `ops.rs` utilizam sistematicamente instruções de carga/descarga desalinhadas:
  - `_mm_loadu_ps` / `_mm_storeu_ps` (SSE2)
  - `_mm256_loadu_ps` / `_mm256_storeu_ps` (AVX2)
- **Vantagem de Robustez:** O uso de variantes `u` (unaligned) previne a ocorrência da exceção `#GP` (General Protection Fault), que ocorre em CPUs x86 quando instruções alinhadas (`_mm_load_ps`) recebem ponteiros fora do limite de 16 ou 32 bytes.
- **Alinhamento Recomendado:** Apesar da tolerância a ponteiros desalinhados, o `BufferDescriptor` do JHCS adota alinhamento canônico de **64 bytes** (tamanho da linha de cache L1/L2), evitando perdas de desempenho por travessia de borda de linha (*cache-line splits*).

---

## 4. Conclusão da Auditoria SIMD

| Recurso | Host (Win/Linux) | Jinn OS (Bare-Metal) | Decisão de Homologação |
|---|:---:|:---:|---|
| **SSE2 (128 bits)** | PASS | **PASS** | Homologado para produção e despacho imediato. |
| **AVX2 (256 bits)** | **PASS** | **BLOCKED** | Homologado no Host; desativado no Jinn até suporte a `XSAVE`. |
| **Detecção CPUID** | PASS | **PARTIAL** | Requer acréscimo de `_xgetbv(0)` para validação de `XCR0`. |
