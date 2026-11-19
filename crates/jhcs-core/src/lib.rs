#![cfg_attr(not(feature = "std"), no_std)]

//! # JHCS Core — Jinn Heterogeneous Compute Service
//!
//! Foundational contracts, dataflow graph abstractions, capabilities,
//! devices, and buffer models for cooperative, multi-platform heterogeneous computing.

extern crate alloc;

pub mod buffer;
pub mod capabilities;
pub mod contracts;
pub mod device;
pub mod error;
pub mod graph;
pub mod task;

pub use buffer::{
    BufferAccessMode, BufferDescriptor, BufferHandle, BufferLocation, BufferOwnership, BufferSet,
    BufferSyncState, ComputeBuffer, Precision,
};
pub use capabilities::{
    BackendDescriptor, BackendKind, ComputeCapabilities, ComputeRequirements, ParallelismClass,
};
pub use contracts::{ComputeBackend as ComputeBackendContract, ComputeScheduler, ComputeTelemetry};
pub use device::{
    ComputeDevice, DeviceCapabilities, DeviceDescriptor, DeviceId, DeviceKind, DeviceOperationalState,
};
pub use error::ComputeError;
pub use graph::ComputeGraph;
pub use task::{
    ComputeBackend, ComputeResult, ComputeTask, DependencySet, OperationId, TaskId, TaskPriority,
    TaskStatus, OP_CONV2D, OP_DOT_PRODUCT, OP_MATMUL, OP_REDUCE_SUM, OP_RELU, OP_SCALE,
    OP_VECTOR_ADD,
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
        let t3 = ComputeTask::new(TaskId::new(3), OP_SCALE, BufferSet::new(), BufferSet::new())
            .with_dependency(TaskId::new(1))
            .with_dependency(TaskId::new(2));

        graph.add_task(t1).unwrap();
        graph.add_task(t2).unwrap();
        graph.add_task(t3).unwrap();

        let batches = graph.execution_batches().unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[1][0], TaskId::new(3));
    }

    #[test]
    fn test_device_model_and_capabilities() {
        let caps = DeviceCapabilities::scalar_cpu_default();
        assert_eq!(caps.kind, DeviceKind::ScalarCpu);
        assert_eq!(caps.max_parallelism, ParallelismClass::Sequential);
        assert!(caps.supports_zero_copy);

        let desc = DeviceDescriptor::new(
            DeviceId::new(42),
            DeviceKind::ScalarCpu,
            alloc::string::String::from("Core-i7-Scalar"),
            DeviceOperationalState::Online,
            caps,
        );
        assert_eq!(desc.id.value(), 42);
        assert_eq!(desc.state, DeviceOperationalState::Online);
    }

    #[test]
    fn test_buffer_model_and_traits() {
        let buf = BufferDescriptor::with_2d_shape(
            BufferHandle::new(7),
            128,
            64,
            BufferLocation::SharedUnifiedMemory,
            Precision::Float32,
        )
        .with_alignment(64)
        .with_ownership(BufferOwnership::Shared)
        .with_access_mode(BufferAccessMode::ReadWrite)
        .with_sync_state(BufferSyncState::Clean);

        assert_eq!(buf.handle().id(), 7);
        assert_eq!(buf.size_bytes(), 128 * 64 * 4);
        assert_eq!(buf.alignment(), 64);
        assert_eq!(buf.location(), BufferLocation::SharedUnifiedMemory);
        assert_eq!(buf.ownership(), BufferOwnership::Shared);
        assert_eq!(buf.access_mode(), BufferAccessMode::ReadWrite);
        assert!(buf.is_valid());
    }

    #[test]
    fn test_task_priority_and_in_place_declaration() {
        let task = ComputeTask::new(
            TaskId::new(99),
            OP_CONV2D,
            BufferSet::new(),
            BufferSet::new(),
        )
        .with_priority(TaskPriority::Critical)
        .with_in_place(true)
        .with_deadline_ns(16_666_667); // 60 FPS frame deadline

        assert_eq!(task.priority, TaskPriority::Critical);
        assert!(task.allow_in_place);
        assert_eq!(task.deadline_ns, Some(16_666_667));
    }
}
