//! Hardware capability discovery and backend descriptor contracts.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use crate::buffer::Precision;

/// Classification of execution backend architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Pure scalar CPU execution (portable reference baseline).
    CpuScalar,
    /// Vectorized CPU execution (SSE, AVX, NEON).
    CpuSimd,
    /// Massive parallel GPU compute (wgpu WGSL or native driver).
    GpuCompute,
    /// Custom dedicated hardware accelerator (NPU, TPU, FPGA).
    AcceleratorDedicated,
}

/// Degree of data parallelism supported or requested by an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParallelismClass {
    /// Strictly sequential or control-heavy tasks.
    Sequential,
    /// 128-bit vectorization (x86 SSE, ARM NEON).
    Vectorized128,
    /// 256-bit vectorization (x86 AVX, AVX2).
    Vectorized256,
    /// 512-bit vectorization (x86 AVX-512).
    Vectorized512,
    /// Massive SIMT parallelism with thousands of concurrent execution threads.
    MassiveParallelGpu,
}

/// Requirements declared by an application or operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeRequirements {
    pub min_memory_bytes: usize,
    pub parallelism: ParallelismClass,
    pub precision: Precision,
    pub allow_fallback: bool,
}

impl ComputeRequirements {
    pub const fn default_scalar() -> Self {
        Self {
            min_memory_bytes: 0,
            parallelism: ParallelismClass::Sequential,
            precision: Precision::Float32,
            allow_fallback: true,
        }
    }

    pub const fn parallel_gpu(min_memory: usize) -> Self {
        Self {
            min_memory_bytes: min_memory,
            parallelism: ParallelismClass::MassiveParallelGpu,
            precision: Precision::Float32,
            allow_fallback: true,
        }
    }
}

/// Physical capabilities reported by an initialized backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeCapabilities {
    pub kind: BackendKind,
    pub max_buffer_size_bytes: usize,
    pub max_workgroup_size: usize,
    pub supports_fp64: bool,
    pub supports_fp32: bool,
    pub supports_int32: bool,
    pub max_parallelism: ParallelismClass,
    pub supported_operations: Vec<String>,
}

impl ComputeCapabilities {
    pub fn cpu_scalar_default() -> Self {
        Self {
            kind: BackendKind::CpuScalar,
            max_buffer_size_bytes: usize::MAX,
            max_workgroup_size: 1,
            supports_fp64: true,
            supports_fp32: true,
            supports_int32: true,
            max_parallelism: ParallelismClass::Sequential,
            supported_operations: alloc::vec!["matmul".into(), "vector_add".into(), "relu".into()],
        }
    }
}

/// Identification descriptor for a backend instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDescriptor {
    pub id: &'static str,
    pub kind: BackendKind,
    pub name: String,
    pub is_available: bool,
    pub memory_capacity_bytes: usize,
}
