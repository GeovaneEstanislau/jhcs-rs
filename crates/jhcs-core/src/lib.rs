#![cfg_attr(not(feature = "std"), no_std)]

//! # JHCS Core — Jinn Heterogeneous Compute Service
//!
//! Foundational contracts, dataflow graph abstractions, capabilities,
//! and buffer models for cooperative, multi-platform heterogeneous computing.

extern crate alloc;

pub mod buffer;
pub mod capabilities;
pub mod error;
pub mod graph;
pub mod task;

pub use buffer::{BufferDescriptor, BufferHandle, BufferLocation, BufferSet, Precision};
pub use capabilities::{
    BackendDescriptor, BackendKind, ComputeCapabilities, ComputeRequirements, ParallelismClass,
};
pub use error::ComputeError;
pub use graph::ComputeGraph;
pub use task::{
    ComputeBackend, ComputeResult, ComputeTask, DependencySet, OperationId, TaskId, TaskStatus,
    OP_CONV2D, OP_MATMUL, OP_RELU, OP_SCALE, OP_VECTOR_ADD,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_graph_topological_sort() {
        let mut graph = ComputeGraph::new();

        let t1 = ComputeTask::new(
            TaskId::new(1),
            OP_MATMUL,
            BufferSet::new(),
            BufferSet::new(),
        );

        let t2 = ComputeTask::new(
            TaskId::new(2),
            OP_RELU,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_dependency(TaskId::new(1));

        let t3 = ComputeTask::new(
            TaskId::new(3),
            OP_SCALE,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_dependency(TaskId::new(2));

        graph.add_task(t3).unwrap();
        graph.add_task(t1).unwrap();
        graph.add_task(t2).unwrap();

        let order = graph.topological_order().unwrap();
        assert_eq!(order, alloc::vec![TaskId::new(1), TaskId::new(2), TaskId::new(3)]);
    }

    #[test]
    fn test_compute_graph_cycle_detection() {
        let mut graph = ComputeGraph::new();

        let t1 = ComputeTask::new(
            TaskId::new(1),
            OP_MATMUL,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_dependency(TaskId::new(2));

        let t2 = ComputeTask::new(
            TaskId::new(2),
            OP_MATMUL,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_dependency(TaskId::new(1));

        graph.add_task(t1).unwrap();
        graph.add_task(t2).unwrap();

        let res = graph.validate();
        assert_eq!(res, Err(ComputeError::CycleDetected));
    }

    #[test]
    fn test_execution_batches() {
        let mut graph = ComputeGraph::new();

        // T1 and T2 have no dependencies (Batch 0)
        let t1 = ComputeTask::new(TaskId::new(1), OP_MATMUL, BufferSet::new(), BufferSet::new());
        let t2 = ComputeTask::new(TaskId::new(2), OP_MATMUL, BufferSet::new(), BufferSet::new());
        // T3 depends on T1 and T2 (Batch 1)
        let t3 = ComputeTask::new(TaskId::new(3), OP_VECTOR_ADD, BufferSet::new(), BufferSet::new())
            .with_dependency(TaskId::new(1))
            .with_dependency(TaskId::new(2));

        graph.add_task(t1).unwrap();
        graph.add_task(t2).unwrap();
        graph.add_task(t3).unwrap();

        let batches = graph.execution_batches().unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1], alloc::vec![TaskId::new(3)]);
    }
}
