# 📚 JHCS — Especificação da Core API e Contratos

> **Crate:** `jhcs-core` | **Compatibilidade:** `#![no_std]` + `alloc`  
> **Camada:** Ring 3 / Biblioteca Universal

---

## 1. Modelos Fundamentais

### 1.1. `ComputeDevice`
Representa um recurso de processamento físico ou virtualizado:
- `id: DeviceId` — Identificador único tipado.
- `kind: DeviceKind` — `ScalarCpu`, `SimdCpu`, `Fpu`, `Gpu`, `DedicatedAccelerator`, `Unknown`.
- `capabilities: DeviceCapabilities` — Capacidades de paralelismo (`Sequential`, `Vectorized128`, `Vectorized256`, `Vectorized512`, `MassiveParallelGpu`), suporte a precisões numéricas e limites de buffer.
- `state: DeviceOperationalState` — `Online`, `Busy`, `Quarantined`, `Offline`, `Unknown`.

### 1.2. `ComputeBuffer` (Trait & Descriptor)
Abstração desacoplada de armazenamento de dados:
- `handle(&self) -> BufferHandle` — Token opaco de referência.
- `size_bytes(&self) -> usize` — Dimensão total em bytes.
- `precision(&self) -> Precision` — `Float32`, `Float64`, `Int32`, `Uint8`, `RawBytes`.
- `location(&self) -> BufferLocation` — `HostMemory`, `DeviceMemory`, `SharedUnifiedMemory`.
- `alignment(&self) -> usize` — Alinhamento de memória (padrão 64 bytes para saturação de cache lines AVX).
- `ownership(&self) -> BufferOwnership` — `Owned`, `Borrowed`, `Shared`.
- `access_mode(&self) -> BufferAccessMode` — `ReadOnly`, `WriteOnly`, `ReadWrite`.
- `sync_state(&self) -> BufferSyncState` — `Clean`, `Modified`, `InFlight`, `Incoherent`.

### 1.3. `ComputeTask`
Descrição declarativa de intenção computacional:
- `id: TaskId` — Identificador da tarefa.
- `operation: OperationId` — `OP_MATMUL`, `OP_VECTOR_ADD`, `OP_CONV2D`, `OP_RELU`, `OP_SCALE`, `OP_REDUCE_SUM`, `OP_DOT_PRODUCT`.
- `inputs: BufferSet` & `outputs: BufferSet` — Conjuntos de buffers envolvidos.
- `requirements: ComputeRequirements` — Restrições de paralelismo, memória mínima e política de fallback.
- `dependencies: DependencySet` — Conjunto de predecessores obrigatórios.
- `priority: TaskPriority` — `Low`, `Normal`, `High`, `Critical`.
- `deadline_ns: Option<u64>` — Prazo máximo opcional para despacho sensível a frames (ex.: 16,6 ms para 60 FPS).
- `allow_in_place: bool` — Sinaliza permissão para processamento direto sobre o buffer de entrada.

### 1.4. `ComputeResult`
Resultado padronizado da execução:
- `task_id: TaskId`
- `status: TaskStatus` (`Completed`, `Failed`, `Cancelled`, `Running`, `Pending`)
- `executed_by: BackendKind`
- `prep_duration_ns`, `transfer_duration_ns`, `exec_duration_ns`, `sync_duration_ns`, `total_duration_ns`
- `strategy_used: Option<String>`
- `error: Option<ComputeError>`

---

## 2. Contratos Principais (Traits)

```rust
pub trait ComputeBackend: Send + Sync {
    fn id(&self) -> &'static str;
    fn kind(&self) -> BackendKind;
    fn is_available(&self) -> bool;
    fn capabilities(&self) -> &ComputeCapabilities;
}

pub trait ComputeScheduler: Send + Sync {
    fn schedule(&self, graph: &ComputeGraph) -> Result<Vec<TaskId>, ComputeError>;
}

pub trait ComputeTelemetry: Send + Sync {
    fn record_result(&mut self, result: &ComputeResult);
    fn total_executions(&self) -> u64;
}
```
