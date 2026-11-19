# 💾 JHCS — Modelo de Memória, Buffers e Zero-Copy

> **Componente:** `jhcs-core::buffer`, `jhcs-platform`  
> **Status:** Especificado & Modelado com Traits Seguras

---

## 1. Princípios do Modelo de Memória

1. **Sem Promessas Falsas de Zero-Copy:**
   - O JHCS distingue rigorosamente quando existe **verdadeiro compartilhamento de memória física** (ex.: buffers DMA no Kernel DMA Manager ou memória unificada) e quando há **cópia necessária** (ex.: upload via barramento PCI para VRAM discreta).

2. **Opacidade e Não-Exposição de Ponteiros Físicos:**
   - Aplicações clientes manipulam apenas tokens fortemente tipados (`BufferHandle`).
   - Nenhum endereço de memória física de Ring 0 ou descritor de página é exposto a Ring 3.

3. **Alinhamento e Coerência de Cache:**
   - Buffers alocados pelo JHCS possuem alinhamento canônico de **64 bytes** (uma linha inteira de cache x86_64), garantindo transferências vetoriais AVX-512 e DMA sem penalidades de divisão de linha (*split-line penalty*).

---

## 2. Camadas de Localização (`BufferLocation`)

- **`HostMemory`:** Alocado na memória RAM principal acessível diretamente pela CPU.
- **`DeviceMemory`:** Reside na memória do acelerador (VRAM da placa gráfica).
- **`SharedUnifiedMemory`:** Mapeado simultaneamente para CPU e acelerador via KDM / JDA Shmem, permitindo leituras e escritas diretas sem cópias redundantes.

---

## 3. Estados de Sincronização (`BufferSyncState`)

| Estado | Significado Operacional | Ação Necessária Antes de Leitura |
|---|---|---|
| **`Clean`** | As representações de host e dispositivo estão idênticas. | Leitura imediata permitida. |
| **`Modified`** | O buffer foi alterado pela CPU ou acelerador. | Requer flush ou invalidação de cache antes da troca de executor. |
| **`InFlight`** | Uma operação assíncrona de DMA ou renderização está em andamento. | Leitura bloqueada até a conclusão do evento JDA CQE. |
| **`Incoherent`** | O acelerador foi reinicializado ou o driver falhou. | Re-fetch obrigatório a partir do armazenamento canônico. |
