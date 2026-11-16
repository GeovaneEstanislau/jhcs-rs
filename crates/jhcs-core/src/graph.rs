//! Directed Acyclic Graph (DAG) for compute tasks with cycle detection and topological scheduling.

extern crate alloc;
use alloc::vec::Vec;
use crate::error::ComputeError;
use crate::task::{ComputeTask, TaskId};

/// Directed Acyclic Graph (DAG) representing tasks and their dependency topology.
#[derive(Debug, Clone, Default)]
pub struct ComputeGraph {
    tasks: Vec<ComputeTask>,
}

impl ComputeGraph {
    pub const fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    /// Appends a new task to the graph. Fails if the task ID already exists.
    pub fn add_task(&mut self, task: ComputeTask) -> Result<(), ComputeError> {
        if self.tasks.iter().any(|t| t.id == task.id) {
            return Err(ComputeError::ExecutionFailed(alloc::format!(
                "Duplicate task ID {}",
                task.id.value()
            )));
        }
        self.tasks.push(task);
        Ok(())
    }

    pub fn get_task(&self, id: TaskId) -> Option<&ComputeTask> {
        self.tasks.iter().find(|t| t.id == id)
    }

    pub fn get_task_mut(&mut self, id: TaskId) -> Option<&mut ComputeTask> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub fn tasks(&self) -> &[ComputeTask] {
        &self.tasks
    }

    /// Validates the graph:
    /// 1. Ensures all dependencies reference known tasks within the graph.
    /// 2. Ensures there are no cycles using topological sorting (Kahn's algorithm).
    pub fn validate(&self) -> Result<(), ComputeError> {
        // 1. Dependency target existence
        for task in &self.tasks {
            for dep in task.dependencies.as_slice() {
                if !self.tasks.iter().any(|t| t.id == *dep) {
                    return Err(ComputeError::DependencyFailed(dep.value()));
                }
            }
        }

        // 2. Cycle detection via topological sort
        self.topological_order().map(|_| ())
    }

    /// Computes a valid topological execution order of all tasks in the DAG.
    /// Returns `Err(ComputeError::CycleDetected)` if a dependency loop exists.
    pub fn topological_order(&self) -> Result<Vec<TaskId>, ComputeError> {
        let n = self.tasks.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // In-degree count (number of unresolved dependencies for each task)
        let mut in_degree: Vec<(TaskId, usize)> = self
            .tasks
            .iter()
            .map(|t| (t.id, t.dependencies.len()))
            .collect();

        // Queue of tasks with 0 unresolved dependencies
        let mut ready: Vec<TaskId> = in_degree
            .iter()
            .filter(|(_, deg)| *deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut order: Vec<TaskId> = Vec::with_capacity(n);

        while let Some(current_id) = ready.pop() {
            order.push(current_id);

            // Find all tasks that depend on `current_id` and decrement their in-degree
            for task in &self.tasks {
                if task.dependencies.contains(current_id) {
                    if let Some(entry) = in_degree.iter_mut().find(|(id, _)| *id == task.id) {
                        if entry.1 > 0 {
                            entry.1 -= 1;
                            if entry.1 == 0 {
                                ready.push(task.id);
                            }
                        }
                    }
                }
            }
        }

        if order.len() != n {
            Err(ComputeError::CycleDetected)
        } else {
            Ok(order)
        }
    }

    /// Returns a list of tasks that are ready to execute given a set of completed task IDs.
    pub fn ready_tasks(&self, completed: &[TaskId]) -> Vec<TaskId> {
        self.tasks
            .iter()
            .filter(|t| !completed.contains(&t.id))
            .filter(|t| t.dependencies.as_slice().iter().all(|d| completed.contains(d)))
            .map(|t| t.id)
            .collect()
    }

    /// Groups tasks into parallel execution stages (batches).
    /// All tasks within each batch are completely independent of one another
    /// and can be concurrently dispatched to different backends or cores.
    pub fn execution_batches(&self) -> Result<Vec<Vec<TaskId>>, ComputeError> {
        let _ = self.validate()?;
        let mut completed: Vec<TaskId> = Vec::new();
        let mut batches: Vec<Vec<TaskId>> = Vec::new();

        while completed.len() < self.tasks.len() {
            let ready = self.ready_tasks(&completed);
            if ready.is_empty() {
                return Err(ComputeError::CycleDetected);
            }
            completed.extend_from_slice(&ready);
            batches.push(ready);
        }

        Ok(batches)
    }
}
