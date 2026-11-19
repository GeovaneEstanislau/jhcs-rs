# 🔌 JHCS — Integração com o Jinn OS Microkernel

> **Crate:** `jhcs-platform-jinn` | **Camada:** Ring 3 Daemon  
> **Status:** Especificado e em Conformidade com KDM / JDA

---

## 1. Mapeamento de Chamadas de Sistema (Syscalls)

O adaptador nativo do Jinn OS opera exclusivamente através das syscalls oficiais de espaço de usuário documentadas em `JINN_JHCS_FOUNDATIONS_AUDIT.md`:

### 1.1. Kernel DMA Manager (KDM)
- **Syscall 60 (`sys_dma_pool_create`):** Cria um pool de frames físicos contíguos de 4 KiB ou 2 MiB para o JHCS.
- **Syscall 62 (`sys_dma_buffer_alloc`):** Aloca um buffer físico e retorna um token opaco de segurança.
- **Syscall 64 (`sys_dma_buffer_map`):** Mapeia o buffer no espaço de endereçamento virtual da tarefa em Ring 3.
- **Syscall 66 (`sys_dma_transfer_prepare`):** Valida contadores de geração e enfileira a transferência no hardware.
- **Syscall 68 (`sys_dma_device_reset_status`):** Consulta a saúde do periférico e aciona quarentena se necessário.

### 1.2. Jinn Dataflow Architecture (JDA)
- **Syscall 12 (`sys_jda_shm_alloc`):** Alocação de memória física compartilhada Zero-Copy para comunicação entre o Aladdin Desktop e o JHCS.
- **Syscall 27 (`sys_jda_ring_submit`):** Submissão não-bloqueante de nós de computação para o anel SQE.
- **Syscall 28 (`sys_jda_ring_poll`):** Coleta orientada a eventos dos anéis de conclusão CQE.

---

## 2. Garantias Operacionais

1. **Boot Seguro:** O Jinn OS inicializa sem verificar a presença do daemon JHCS.
2. **Degradação Graciosa:** Se o processo do JHCS for finalizado, o Aladdin Desktop recorre automaticamente à sua renderização interna por software (`aladdin/src/primitives.rs`).
