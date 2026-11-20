//! # Dispatch Policies and Backend Selection Algorithms
//!
//! Provides deterministic, safe, and cost-aware scheduling policies for selecting
//! execution units across heterogeneous hardware resources.

use jhcs_core::capabilities::{BackendKind, ParallelismClass};
use jhcs_core::task::{ComputeBackend, ComputeTask};
use crate::metrics::MetricsRegistry;

/// Policy strategy used by the broker to assign tasks to backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPolicy {
    /// Strictly restricts all execution to scalar CPU (used for base validation or recovery).
    CpuOnly,
    /// Eagerly prefers vectorized SIMD/FPU whenever supported by the architecture.
    PreferSimd,
    /// Minimizes scheduling and transfer latency; avoids bus/DMA roundtrips for local processing.
    LatencyFirst,
    /// Maximizes aggregate parallel throughput by assigning large batches to GPU or wide SIMD.
    ThroughputFirst,
    /// Balanced standard policy: evaluates problem size, capabilities, and estimated total cost.
    Balanced,
    /// Strictly forces a specific backend across all eligible tasks.
    ForceBackend(BackendKind),
    /// Cost-optimized based on historical overhead and execution measurements.
    CostOptimized,
    /// Zero-GPU High Efficiency Policy:
    /// Treats CPU SIMD (AVX2/SSE2) as a first-class high-performance parallel compute engine.
    /// Eliminates all bus/DMA transfers and lock contention, dispatching directly to SIMD kernels.
    ZeroGpuOptimized,
    /// Legacy deterministic rule-based alias (equivalent to Balanced).
    DeterministicRuleBased,
}

impl Default for DispatchPolicy {
    fn default() -> Self {
        Self::Balanced
    }
}

impl DispatchPolicy {
    /// Selects the target backend for a given task according to the active policy.
    pub fn select_backend(
        &self,
        task: &ComputeTask,
        problem_elements: usize,
        has_simd: bool,
        has_gpu: bool,
        metrics: &MetricsRegistry,
    ) -> BackendKind {
        // 1. Strict override policies
        match self {
            Self::CpuOnly => return BackendKind::CpuScalar,
            Self::ForceBackend(kind) => return *kind,
            _ => {}
        }

        // 2. Explicit task request (if allowed by hardware)
        match task.preferred_backend {
            ComputeBackend::Cpu => return BackendKind::CpuScalar,
            ComputeBackend::Simd if has_simd => return BackendKind::CpuSimd,
            ComputeBackend::Gpu if has_gpu => return BackendKind::GpuCompute,
            _ => {}
        }

        // 3. Policy-driven selection
        match self {
            Self::CpuOnly | Self::ForceBackend(_) => unreachable!(),

            Self::PreferSimd => {
                if has_simd && problem_elements >= 8 {
                    BackendKind::CpuSimd
                } else {
                    BackendKind::CpuScalar
                }
            }

            Self::ZeroGpuOptimized => {
                // Modo Zero-GPU: SIMD vetorial de altíssima eficiência sem overhead de GPU/DMA
                if has_simd && problem_elements >= 4 {
                    BackendKind::CpuSimd
                } else {
                    BackendKind::CpuScalar
                }
            }

            Self::LatencyFirst => {
                // Latency-first avoids bus/GPU DMA transfer and driver handshake latency
                if has_simd && problem_elements >= 64 {
                    BackendKind::CpuSimd
                } else {
                    BackendKind::CpuScalar
                }
            }

            Self::ThroughputFirst => {
                // High throughput utilizes massive parallelism for large jobs
                if has_gpu && problem_elements >= 1024 {
                    BackendKind::GpuCompute
                } else if has_simd && problem_elements >= 128 {
                    BackendKind::CpuSimd
                } else {
                    BackendKind::CpuScalar
                }
            }

            Self::Balanced | Self::DeterministicRuleBased => {
                match task.requirements.parallelism {
                    ParallelismClass::Sequential => BackendKind::CpuScalar,
                    ParallelismClass::Vectorized128
                    | ParallelismClass::Vectorized256
                    | ParallelismClass::Vectorized512 => {
                        // Workloads below 64 elements have setup cost > vectorization gain
                        if has_simd && problem_elements >= 64 {
                            BackendKind::CpuSimd
                        } else {
                            BackendKind::CpuScalar
                        }
                    }
                    ParallelismClass::MassiveParallelGpu => {
                        // Small workloads (< 1024 elements) have negative GPU return due to transfer overhead
                        if has_gpu && problem_elements >= 1024 {
                            BackendKind::GpuCompute
                        } else if has_simd && problem_elements >= 64 {
                            BackendKind::CpuSimd
                        } else {
                            BackendKind::CpuScalar
                        }
                    }
                }
            }

            Self::CostOptimized => {
                let op = task.operation.0;
                let cpu_cost = metrics
                    .get(BackendKind::CpuScalar, op)
                    .map(|m| m.avg_exec_ns())
                    .unwrap_or(10_000);

                let simd_cost = if has_simd {
                    metrics
                        .get(BackendKind::CpuSimd, op)
                        .map(|m| m.avg_exec_ns())
                        .unwrap_or(3_000)
                } else {
                    u64::MAX
                };

                let gpu_cost = if has_gpu {
                    metrics
                        .get(BackendKind::GpuCompute, op)
                        .map(|m| m.avg_exec_ns() + m.total_overhead_ns())
                        .unwrap_or(50_000)
                } else {
                    u64::MAX
                };

                if gpu_cost < simd_cost && gpu_cost < cpu_cost && problem_elements >= 1024 {
                    BackendKind::GpuCompute
                } else if simd_cost < cpu_cost {
                    BackendKind::CpuSimd
                } else {
                    BackendKind::CpuScalar
                }
            }
        }
    }
}
