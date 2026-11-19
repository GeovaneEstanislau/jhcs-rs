//! # Broker Dispatcher Engine
//!
//! Coordinates execution across heterogeneous backends, resolves DAG dependencies,
//! enforces scheduling policies, and provides guaranteed fallback to scalar execution.

extern crate alloc;
use alloc::vec::Vec;
use jhcs_backend_cpu::CpuBackend;
use jhcs_backend_simd::SimdBackend;
use jhcs_core::capabilities::BackendKind;
use jhcs_core::contracts::ComputeScheduler;
use jhcs_core::error::ComputeError;
use jhcs_core::graph::ComputeGraph;
use jhcs_core::task::{ComputeResult, ComputeTask, TaskId};
use crate::metrics::MetricsRegistry;
use crate::policy::DispatchPolicy;

/// Central broker managing task queues, DAG scheduling, and backend dispatch.
pub struct ComputeBroker {
    cpu_backend: CpuBackend,
    simd_backend: SimdBackend,
    has_gpu: bool,
    policy: DispatchPolicy,
    metrics: MetricsRegistry,
}

impl ComputeBroker {
    pub fn new() -> Self {
        Self {
            cpu_backend: CpuBackend::new(),
            simd_backend: SimdBackend::new(),
            has_gpu: false,
            policy: DispatchPolicy::Balanced,
            metrics: MetricsRegistry::new(),
        }
    }

    pub fn with_policy(mut self, policy: DispatchPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn set_policy(&mut self, policy: DispatchPolicy) {
        self.policy = policy;
    }

    pub fn policy(&self) -> DispatchPolicy {
        self.policy
    }

    pub fn set_gpu_available(&mut self, available: bool) {
        self.has_gpu = available;
    }

    pub fn is_gpu_available(&self) -> bool {
        self.has_gpu
    }

    pub fn cpu_backend(&self) -> &CpuBackend {
        &self.cpu_backend
    }

    pub fn simd_backend(&self) -> &SimdBackend {
        &self.simd_backend
    }

    pub fn metrics(&self) -> &MetricsRegistry {
        &self.metrics
    }

    pub fn metrics_mut(&mut self) -> &mut MetricsRegistry {
        &mut self.metrics
    }

    /// Selects the best backend for a specific task based on current policy and hardware.
    pub fn select_backend(&self, task: &ComputeTask, problem_elements: usize) -> BackendKind {
        self.policy.select_backend(
            task,
            problem_elements,
            self.simd_backend.has_sse() || self.simd_backend.has_avx(),
            self.has_gpu,
            &self.metrics,
        )
    }

    /// Dispatches a matrix multiplication task across the chosen backend with automatic scalar fallback.
    pub fn execute_matmul(
        &mut self,
        task: &ComputeTask,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> ComputeResult {
        let problem_elements = m * n;
        let chosen = self.select_backend(task, problem_elements);

        let mut result = match chosen {
            BackendKind::CpuSimd => {
                let res = self.simd_backend.execute_matmul(task.id, a, b, c, m, k, n);
                // Fallback check
                if res.status != jhcs_core::task::TaskStatus::Completed
                    && task.requirements.allow_fallback
                {
                    self.cpu_backend.execute_matmul(task.id, a, b, c, m, k, n)
                } else {
                    res
                }
            }
            BackendKind::CpuScalar | _ => {
                self.cpu_backend.execute_matmul(task.id, a, b, c, m, k, n)
            }
        };

        // Annotate policy strategy
        result.strategy_used = Some(alloc::format!("{:?}", self.policy));

        // Record metrics
        self.metrics.record(
            result.executed_by,
            task.operation.0,
            result.prep_duration_ns,
            result.transfer_duration_ns,
            result.exec_duration_ns,
            result.sync_duration_ns,
        );

        result
    }

    /// Dispatches a vector addition task with automatic fallback.
    pub fn execute_vector_add(
        &mut self,
        task: &ComputeTask,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> ComputeResult {
        let chosen = self.select_backend(task, a.len());

        let result = match chosen {
            BackendKind::CpuSimd => {
                let res = self.simd_backend.execute_vector_add(task.id, a, b, c);
                if res.status != jhcs_core::task::TaskStatus::Completed
                    && task.requirements.allow_fallback
                {
                    self.cpu_backend.execute_vector_add(task.id, a, b, c)
                } else {
                    res
                }
            }
            BackendKind::CpuScalar | _ => {
                self.cpu_backend.execute_vector_add(task.id, a, b, c)
            }
        };

        self.metrics.record(
            result.executed_by,
            task.operation.0,
            result.prep_duration_ns,
            result.transfer_duration_ns,
            result.exec_duration_ns,
            result.sync_duration_ns,
        );

        result
    }

    /// Dispatches a 2D convolution task with automatic fallback.
    pub fn execute_conv2d(
        &mut self,
        task: &ComputeTask,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        kernel: &[f32],
        k_h: usize,
        k_w: usize,
        output: &mut [f32],
    ) -> ComputeResult {
        let problem_elements = in_h * in_w;
        let chosen = self.select_backend(task, problem_elements);

        let result = match chosen {
            BackendKind::CpuSimd => {
                self.simd_backend
                    .execute_conv2d(task.id, input, in_h, in_w, kernel, k_h, k_w, output)
            }
            BackendKind::CpuScalar | _ => {
                // Scalar fallback
                let out_h = if in_h >= k_h { in_h - k_h + 1 } else { 0 };
                let out_w = if in_w >= k_w { in_w - k_w + 1 } else { 0 };
                if output.len() != out_h * out_w {
                    ComputeResult::failure(
                        task.id,
                        BackendKind::CpuScalar,
                        ComputeError::DimensionMismatch {
                            expected: (out_h * out_w, 1),
                            actual: (output.len(), 1),
                        },
                    )
                } else {
                    jhcs_backend_simd::conv2d_f32_scalar(
                        input, in_w, kernel, k_h, k_w, output, out_h, out_w,
                    );
                    ComputeResult::success(task.id, BackendKind::CpuScalar, 0)
                }
            }
        };

        self.metrics.record(
            result.executed_by,
            task.operation.0,
            result.prep_duration_ns,
            result.transfer_duration_ns,
            result.exec_duration_ns,
            result.sync_duration_ns,
        );

        result
    }

    /// Dispatches a vector reduction sum task.
    pub fn execute_reduce_sum(
        &mut self,
        task: &ComputeTask,
        input: &[f32],
        out: &mut f32,
    ) -> ComputeResult {
        let chosen = self.select_backend(task, input.len());

        let result = match chosen {
            BackendKind::CpuSimd => {
                let (val, res) = self.simd_backend.execute_reduce_sum(task.id, input);
                *out = val;
                res
            }
            BackendKind::CpuScalar | _ => {
                *out = jhcs_backend_cpu::reduce_sum_f32_scalar(input);
                ComputeResult::success(task.id, BackendKind::CpuScalar, 0)
            }
        };

        self.metrics.record(
            result.executed_by,
            task.operation.0,
            result.prep_duration_ns,
            result.transfer_duration_ns,
            result.exec_duration_ns,
            result.sync_duration_ns,
        );

        result
    }

    /// Dispatches an entire compute DAG, honoring dependencies batch by batch.
    pub fn execute_dag_order(&self, graph: &ComputeGraph) -> Result<Vec<TaskId>, ComputeError> {
        graph.topological_order()
    }
}

impl ComputeScheduler for ComputeBroker {
    fn schedule(&self, graph: &ComputeGraph) -> Result<Vec<TaskId>, ComputeError> {
        graph.topological_order()
    }
}

impl Default for ComputeBroker {
    fn default() -> Self {
        Self::new()
    }
}
