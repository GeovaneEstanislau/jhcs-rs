# 🛡️ JHCS — Modelo de Segurança, Isolamento e Contenção de Falhas

> **Componente:** `jhcs-core`, `jhcs-platform-jinn`, `KDM`  
> **Status:** Ativo & Em Conformidade com Zero-Trust Ring 3

---

## 1. Princípios de Segurança

1. **Ring 3 Exclusivo:**
   - O daemon JHCS, o Compute Broker e os executores de espaço de usuário não possuem privilégios de Ring 0.
   - Nenhuma instrução privilegiada (`in`/`out`, manipulação direta de `CR3`, controle de interrupções) é emitida pelo JHCS.

2. **Mediação por Capacidades e Syscalls Canônicas:**
   - O acesso a buffers DMA é mediado pelas chamadas de sistema do **Kernel DMA Manager (KDM)**:
     - `sys_dma_buffer_alloc` (Syscall 62)
     - `sys_dma_buffer_map` (Syscall 64)
     - `sys_dma_transfer_prepare` (Syscall 66)
   - Tokens de buffer possuem contadores de geração de 64 bits para prevenir vulnerabilidades de *use-after-free*.

3. **Quarentena de Dispositivos e Contenção de Pânico:**
   - Se um driver de GPU travar ou parar de responder:
     - O microkernel executa o isolamento do periférico (`sys_dma_device_reset_status`, Syscall 68).
     - O JHCS marca o dispositivo como `DeviceOperationalState::Quarantined`.
     - Todas as tarefas pendentes sofrem fallback transparente para CPU/SIMD.
     - **O sistema operacional jamais entra em kernel panic.**
