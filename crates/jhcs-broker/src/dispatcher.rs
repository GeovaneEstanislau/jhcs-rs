//! Broker dispatcher engine coordinating execution across multiple heterogeneous backends.

extern crate alloc;
use alloc::vec::Vec;
use jhcs_backend_cpu::CpuBackend;
use jhcs_backend_simd::SimdBackend;
use jhcs_core::capabilities::BackendKind;
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
            policy: DispatchPolicy::DeterministicRuleBased,
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

    /// Dispatches a single matrix multiplication task across the chosen backend.
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

        let result = match chosen {
            BackendKind::CpuSimd => {
                self.simd_backend.execute_matmul(task.id, a, b, c, m, k, n)
            }
            BackendKind::CpuScalar | _ => {
                self.cpu_backend.execute_matmul(task.id, a, b, c, m, k, n)
            }
        };

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

    /// Dispatches an entire compute DAG, honoring dependencies batch by batch.
    pub fn execute_dag_order(&self, graph: &ComputeGraph) -> Result<Vec<TaskId>, ComputeError> {
        graph.topological_order()
    }
}

impl Default for ComputeBroker {
    fn default() -> Self {
        Self::new()
    }
}
