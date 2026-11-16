//! Task definitions, dependency tracking and execution results.

extern crate alloc;
use alloc::vec::Vec;
use crate::buffer::BufferSet;
use crate::capabilities::{BackendKind, ComputeRequirements};
use crate::error::ComputeError;

/// Strongly-typed unique task identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub u64);

impl TaskId {
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    pub const fn value(&self) -> u64 {
        self.0
    }
}

/// Operation identifier representing the computation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationId(pub &'static str);

pub const OP_MATMUL: OperationId = OperationId("matmul");
pub const OP_VECTOR_ADD: OperationId = OperationId("vector_add");
pub const OP_CONV2D: OperationId = OperationId("conv2d");
pub const OP_RELU: OperationId = OperationId("relu");
pub const OP_SCALE: OperationId = OperationId("scale");

/// Target execution backend hint or policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeBackend {
    /// Broker autonomously picks the most suitable backend based on metrics and availability.
    Automatic,
    /// Explicitly requested CPU scalar execution.
    Cpu,
    /// Explicitly requested CPU SIMD vector execution.
    Simd,
    /// Explicitly requested GPU compute execution.
    Gpu,
}

/// Set of task IDs that must complete before this task can execute.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DependencySet {
    dependencies: Vec<TaskId>,
}

impl DependencySet {
    pub const fn new() -> Self {
        Self { dependencies: Vec::new() }
    }

    pub fn from_slice(deps: &[TaskId]) -> Self {
        Self {
            dependencies: deps.to_vec(),
        }
    }

    pub fn add(&mut self, dep: TaskId) {
        if !self.dependencies.contains(&dep) {
            self.dependencies.push(dep);
        }
    }

    pub fn contains(&self, dep: TaskId) -> bool {
        self.dependencies.contains(&dep)
    }

    pub fn is_empty(&self) -> bool {
        self.dependencies.is_empty()
    }

    pub fn len(&self) -> usize {
        self.dependencies.len()
    }

    pub fn as_slice(&self) -> &[TaskId] {
        &self.dependencies
    }
}

/// High-level declarative compute task specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeTask {
    pub id: TaskId,
    pub operation: OperationId,
    pub inputs: BufferSet,
    pub outputs: BufferSet,
    pub requirements: ComputeRequirements,
    pub dependencies: DependencySet,
    pub preferred_backend: ComputeBackend,
}

impl ComputeTask {
    pub fn new(
        id: TaskId,
        operation: OperationId,
        inputs: BufferSet,
        outputs: BufferSet,
    ) -> Self {
        Self {
            id,
            operation,
            inputs,
            outputs,
            requirements: ComputeRequirements::default_scalar(),
            dependencies: DependencySet::new(),
            preferred_backend: ComputeBackend::Automatic,
        }
    }

    pub fn with_requirements(mut self, requirements: ComputeRequirements) -> Self {
        self.requirements = requirements;
        self
    }

    pub fn with_dependency(mut self, dep: TaskId) -> Self {
        self.dependencies.add(dep);
        self
    }

    pub fn with_backend(mut self, backend: ComputeBackend) -> Self {
        self.preferred_backend = backend;
        self
    }
}

/// Lifecycle state of a task in the execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskStatus {
    Pending,
    Ready,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Execution telemetry and status result returned after task execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeResult {
    pub task_id: TaskId,
    pub status: TaskStatus,
    pub executed_by: BackendKind,
    pub prep_duration_ns: u64,
    pub transfer_duration_ns: u64,
    pub exec_duration_ns: u64,
    pub sync_duration_ns: u64,
    pub total_duration_ns: u64,
    pub error: Option<ComputeError>,
}

impl ComputeResult {
    pub fn success(task_id: TaskId, executed_by: BackendKind, exec_ns: u64) -> Self {
        Self {
            task_id,
            status: TaskStatus::Completed,
            executed_by,
            prep_duration_ns: 0,
            transfer_duration_ns: 0,
            exec_duration_ns: exec_ns,
            sync_duration_ns: 0,
            total_duration_ns: exec_ns,
            error: None,
        }
    }

    pub fn failure(task_id: TaskId, executed_by: BackendKind, error: ComputeError) -> Self {
        Self {
            task_id,
            status: TaskStatus::Failed,
            executed_by,
            prep_duration_ns: 0,
            transfer_duration_ns: 0,
            exec_duration_ns: 0,
            sync_duration_ns: 0,
            total_duration_ns: 0,
            error: Some(error),
        }
    }
}
