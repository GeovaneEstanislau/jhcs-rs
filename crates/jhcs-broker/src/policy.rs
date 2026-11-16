//! Dispatch policies and backend selection algorithms.

use jhcs_core::capabilities::{BackendKind, ParallelismClass};
use jhcs_core::task::{ComputeBackend, ComputeTask};
use crate::metrics::MetricsRegistry;

/// Policy strategy used by the broker to assign tasks to backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPolicy {
    /// Pure deterministic rules based on problem size, requirements, and hardware availability.
    DeterministicRuleBased,
    /// Strictly forces a specific backend across all eligible tasks.
    ForceBackend(BackendKind),
    /// Cost-optimized based on historical overhead and execution measurements.
    CostOptimized,
}

impl DispatchPolicy {
    /// Selects the target backend for a given task according to the policy rules.
    pub fn select_backend(
        &self,
        task: &ComputeTask,
        problem_elements: usize,
        has_simd: bool,
        has_gpu: bool,
        metrics: &MetricsRegistry,
    ) -> BackendKind {
        match self {
            Self::ForceBackend(kind) => *kind,

            Self::DeterministicRuleBased => {
                // Respect explicit task preferences first
                match task.preferred_backend {
                    ComputeBackend::Cpu => return BackendKind::CpuScalar,
                    ComputeBackend::Simd if has_simd => return BackendKind::CpuSimd,
                    ComputeBackend::Gpu if has_gpu => return BackendKind::GpuCompute,
                    _ => {}
                }

                // Problem size and parallelism heuristic
                match task.requirements.parallelism {
                    ParallelismClass::Sequential => BackendKind::CpuScalar,
                    ParallelismClass::Vectorized128 | ParallelismClass::Vectorized256 | ParallelismClass::Vectorized512 => {
                        // Workloads below 256 elements (< 16x16) suffer from broker dispatch overhead > SIMD gain
                        if has_simd && problem_elements >= 256 {
                            BackendKind::CpuSimd
                        } else {
                            BackendKind::CpuScalar
                        }
                    }
                    ParallelismClass::MassiveParallelGpu => {
                        // Small workloads (< 64 elements) have negative GPU return due to transfer overhead
                        if has_gpu && problem_elements >= 1024 {
                            BackendKind::GpuCompute
                        } else if has_simd {
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
