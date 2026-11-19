//! # Foundational Traits and Contracts for JHCS
//!
//! Provides pluggable abstractions for backends, schedulers, buffers, and telemetry.

extern crate alloc;
use alloc::vec::Vec;
use crate::capabilities::{BackendKind, ComputeCapabilities};
use crate::error::ComputeError;
use crate::graph::ComputeGraph;
use crate::task::{ComputeResult, TaskId};

/// Trait implemented by any compute backend (Scalar CPU, SIMD CPU, GPU, Dedicated).
pub trait ComputeBackend: Send + Sync {
    /// Static identifier for this backend instance.
    fn id(&self) -> &'static str;

    /// Kind of hardware/execution architecture.
    fn kind(&self) -> BackendKind;

    /// Indicates whether the backend is online and ready to accept compute dispatches.
    fn is_available(&self) -> bool;

    /// Reported capabilities and limitations.
    fn capabilities(&self) -> &ComputeCapabilities;
}

/// Trait implemented by scheduling policy engines (topological, priority-driven, deadline-aware).
pub trait ComputeScheduler: Send + Sync {
    /// Computes the execution sequence or batching plan for a validated task graph.
    fn schedule(&self, graph: &ComputeGraph) -> Result<Vec<TaskId>, ComputeError>;
}

/// Trait for recording, aggregating, and observing compute telemetry.
pub trait ComputeTelemetry: Send + Sync {
    /// Records execution status and performance metrics for a completed or failed task.
    fn record_result(&mut self, result: &ComputeResult);

    /// Total number of task executions tracked.
    fn total_executions(&self) -> u64;
}
