#![cfg_attr(not(feature = "std"), no_std)]

//! # JHCS Broker — Compute Broker & Dispatcher
//!
//! Orchestrates tasks, resolves dependency graphs, tracks execution metrics,
//! and deterministically selects the optimal backend for each workload.

extern crate alloc;

pub mod dispatcher;
pub mod metrics;
pub mod policy;

pub use dispatcher::ComputeBroker;
pub use metrics::{MetricsRegistry, OperationMetrics};
pub use policy::DispatchPolicy;

#[cfg(test)]
mod tests {
    use super::*;
    use jhcs_core::buffer::BufferSet;
    use jhcs_core::capabilities::{BackendKind, ComputeRequirements, ParallelismClass};
    use jhcs_core::contracts::{ComputeScheduler, ComputeTelemetry};
    use jhcs_core::graph::ComputeGraph;
    use jhcs_core::task::{ComputeTask, TaskId, OP_MATMUL, OP_VECTOR_ADD};

    #[test]
    fn test_broker_deterministic_selection() {
        let broker = ComputeBroker::new();

        // 1. Scalar task
        let task_scalar = ComputeTask::new(
            TaskId::new(1),
            OP_MATMUL,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_requirements(ComputeRequirements {
            min_memory_bytes: 0,
            parallelism: ParallelismClass::Sequential,
            precision: jhcs_core::buffer::Precision::Float32,
            allow_fallback: true,
        });

        assert_eq!(
            broker.select_backend(&task_scalar, 16),
            BackendKind::CpuScalar
        );

        // 2. Vectorized task
        let task_simd = ComputeTask::new(
            TaskId::new(2),
            OP_MATMUL,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_requirements(ComputeRequirements {
            min_memory_bytes: 0,
            parallelism: ParallelismClass::Vectorized256,
            precision: jhcs_core::buffer::Precision::Float32,
            allow_fallback: true,
        });

        assert_eq!(broker.select_backend(&task_simd, 1024), BackendKind::CpuSimd);
    }

    #[test]
    fn test_broker_policies() {
        let mut broker = ComputeBroker::new();

        let task = ComputeTask::new(
            TaskId::new(5),
            OP_VECTOR_ADD,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_requirements(ComputeRequirements {
            min_memory_bytes: 0,
            parallelism: ParallelismClass::Vectorized256,
            precision: jhcs_core::buffer::Precision::Float32,
            allow_fallback: true,
        });

        // CpuOnly policy strictly selects CpuScalar
        broker.set_policy(DispatchPolicy::CpuOnly);
        assert_eq!(broker.select_backend(&task, 1024), BackendKind::CpuScalar);

        // PreferSimd policy selects CpuSimd
        broker.set_policy(DispatchPolicy::PreferSimd);
        assert_eq!(broker.select_backend(&task, 1024), BackendKind::CpuSimd);
    }

    #[test]
    fn test_broker_execution_and_metrics_recording() {
        let mut broker = ComputeBroker::new();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        let task = ComputeTask::new(
            TaskId::new(100),
            OP_MATMUL,
            BufferSet::new(),
            BufferSet::new(),
        );

        let res = broker.execute_matmul(&task, &a, &b, &mut c, 2, 2, 2);
        assert_eq!(res.status, jhcs_core::task::TaskStatus::Completed);
        assert!(res.strategy_used.is_some());

        // Verify metrics
        let stats = broker.metrics().get(res.executed_by, "matmul");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().execution_count, 1);
        assert_eq!(broker.metrics().total_executions(), 1);
    }

    #[test]
    fn test_broker_scheduler_trait() {
        let broker = ComputeBroker::new();
        let mut graph = ComputeGraph::new();

        let t1 = ComputeTask::new(TaskId::new(1), OP_MATMUL, BufferSet::new(), BufferSet::new());
        let t2 = ComputeTask::new(TaskId::new(2), OP_MATMUL, BufferSet::new(), BufferSet::new())
            .with_dependency(TaskId::new(1));

        graph.add_task(t1).unwrap();
        graph.add_task(t2).unwrap();

        let order = broker.schedule(&graph).unwrap();
        assert_eq!(order, alloc::vec![TaskId::new(1), TaskId::new(2)]);
    }
}
