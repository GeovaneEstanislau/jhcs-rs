# 🔌 JHCS — Especificação de Adaptadores de Plataforma

> **Documento Oficial de Engenharia — jhcs-rs**  
> **Status:** Ativo | **Módulos:** `jhcs-platform`, `jhcs-platform-host`, `jhcs-platform-jinn`

---

## 1. O Papel do Adaptador de Plataforma

Para garantir que o núcleo de computação (`jhcs-core`) e o corretor de tarefas (`jhcs-broker`) permaneçam 100% universais, toda interação com o sistema operacional hospedeiro é mediada pela trait `PlatformAdapter`:

```rust
pub trait PlatformAdapter: Send + Sync {
    fn timer(&self) -> &dyn PlatformTimer;
    fn memory(&self) -> &dyn PlatformMemory;
    fn devices(&self) -> &dyn PlatformDeviceEnumerator;
}
```

---

## 2. Adaptador Host (`jhcs-platform-host`)
* **Ambientes Suportados:** Windows 10/11, distribuições Linux x86_64 / ARM64, macOS.
* **Implementação de Timer:** `std::time::Instant` com resolução de sub-microssegundos.
* **Implementação de Memória:** Alocador nativo do sistema via `std::alloc::{alloc, dealloc}` com layout de alinhamento estrito a 16 e 64 bytes.
* **Propósito:** Permitir desenvolvimento rápido, suítes de testes automatizados e profiling de algoritmos em máquinas de desenvolvimento antes da integração com o microkernel.

---

## 3. Adaptador Jinn OS Ring 3 (`jhcs-platform-jinn`)
* **Ambiente Suportado:** Jinn OS executando em espaço de usuário (Ring 3).
* **Compilação:** Modo estrito `#![no_std]` com alocador de heap do processo.
* **Mapeamento de Syscalls do Microkernel:**
  * **Timer:** Leitura direta de ciclos TSC via instrução `RDTSC`, calibrada contra o temporizador do APIC/PIT (100 Hz).
  * **Memória Compartilhada Zero-Copy:** Syscalls `sys_jda_shm_alloc` (12) e `sys_jda_shm_share` (26).
  * **Buffers DMA Seguros:** Syscalls do Kernel DMA Manager `sys_dma_buffer_alloc` (62) e `sys_dma_transfer_prepare` (66).
  * **Descoberta de Dispositivos:** Enumeração de periféricos PCI classe `0x03` e controladores VirtIO (`1AF4:1050`).
