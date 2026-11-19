//! # ComputeDevice Model and Contracts
//!
//! Represents physical or virtual computational resources (CPU, FPU, SIMD, GPU,
//! dedicated accelerators) discovered across the host or native Jinn environment.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use crate::buffer::Precision;
use crate::capabilities::{BackendKind, ParallelismClass};
use crate::task::OperationId;

/// Strongly-typed identifier for a registered computational device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(pub u32);

impl DeviceId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn value(&self) -> u32 {
        self.0
    }
}

/// Operational state of a compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceOperationalState {
    /// Device is online, healthy, and ready for work.
    Online,
    /// Device is currently executing an asynchronous batch.
    Busy,
    /// Device has experienced unrecoverable errors and is quarantined (KDM isolation).
    Quarantined,
    /// Device is powered down, uninitialized, or driver offline.
    Offline,
    /// Device state could not be determined.
    Unknown,
}

/// Hardware classification of the execution unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// Pure scalar CPU instruction pipeline.
    ScalarCpu,
    /// Vectorized CPU unit (SSE/AVX/NEON).
    SimdCpu,
    /// Dedicated floating-point coprocessor (x87 or discrete FPU).
    Fpu,
    /// Graphics Processing Unit (VirtIO-GPU, Bochs BGA, or native host GPU).
    Gpu,
    /// Dedicated hardware acceleration engine (NPU, DSP, FPGA).
    DedicatedAccelerator,
    /// Unclassified hardware resource.
    Unknown,
}

impl DeviceKind {
    pub const fn to_backend_kind(&self) -> BackendKind {
        match self {
            Self::ScalarCpu => BackendKind::CpuScalar,
            Self::SimdCpu | Self::Fpu => BackendKind::CpuSimd,
            Self::Gpu => BackendKind::GpuCompute,
            Self::DedicatedAccelerator => BackendKind::AcceleratorDedicated,
            Self::Unknown => BackendKind::CpuScalar,
        }
    }
}

/// Detailed capabilities reported by a physical or virtual compute device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub kind: DeviceKind,
    pub execution_units: usize,
    pub max_parallelism: ParallelismClass,
    pub memory_capacity_bytes: Option<usize>,
    pub max_buffer_bytes: usize,
    pub supported_precisions: Vec<Precision>,
    pub supported_operations: Vec<OperationId>,
    pub supports_zero_copy: bool,
}

impl DeviceCapabilities {
    pub fn scalar_cpu_default() -> Self {
        Self {
            kind: DeviceKind::ScalarCpu,
            execution_units: 1,
            max_parallelism: ParallelismClass::Sequential,
            memory_capacity_bytes: None,
            max_buffer_bytes: usize::MAX,
            supported_precisions: alloc::vec![
                Precision::Float32,
                Precision::Float64,
                Precision::Int32,
                Precision::Uint8,
                Precision::RawBytes,
            ],
            supported_operations: alloc::vec![
                OperationId("matmul"),
                OperationId("vector_add"),
                OperationId("relu"),
                OperationId("scale"),
                OperationId("conv2d"),
                OperationId("reduce_sum"),
                OperationId("dot_product"),
            ],
            supports_zero_copy: true,
        }
    }
}

/// Universal trait representing an active computational device.
pub trait ComputeDevice: Send + Sync {
    /// Unique identifier for this device instance.
    fn id(&self) -> DeviceId;

    /// Specific device kind.
    fn kind(&self) -> DeviceKind;

    /// Human-readable device name.
    fn name(&self) -> &str;

    /// Discovered capabilities and constraints.
    fn capabilities(&self) -> &DeviceCapabilities;

    /// Current operational health state.
    fn operational_state(&self) -> DeviceOperationalState;

    /// Estimates the computational execution duration in nanoseconds for a given workload.
    fn estimate_compute_cost(&self, operation: OperationId, element_count: usize) -> u64;

    /// Estimates the data transfer overhead in nanoseconds for moving bytes to/from this device.
    fn estimate_transfer_cost(&self, byte_count: usize) -> u64;
}

/// Concrete descriptor of a discovered device, suitable for serialization and discovery tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDescriptor {
    pub id: DeviceId,
    pub kind: DeviceKind,
    pub name: String,
    pub state: DeviceOperationalState,
    pub capabilities: DeviceCapabilities,
}

impl DeviceDescriptor {
    pub fn new(
        id: DeviceId,
        kind: DeviceKind,
        name: String,
        state: DeviceOperationalState,
        capabilities: DeviceCapabilities,
    ) -> Self {
        Self {
            id,
            kind,
            name,
            state,
            capabilities,
        }
    }
}
