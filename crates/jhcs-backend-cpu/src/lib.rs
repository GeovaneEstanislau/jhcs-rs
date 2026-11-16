#![cfg_attr(not(feature = "std"), no_std)]

//! # JHCS Backend CPU — Scalar Reference Backend
//!
//! Provides portable, pure scalar math implementations for all JHCS operations.
//! Serves as the ground-truth numerical reference to validate SIMD and GPU results.

extern crate alloc;

pub mod matmul;

pub use matmul::{matmul_f32_scalar, relu_f32_scalar, vector_add_f32_scalar};

use alloc::string::ToString;
use jhcs_core::capabilities::{
    BackendDescriptor, BackendKind, ComputeCapabilities, ParallelismClass,
};
use jhcs_core::task::{ComputeResult, TaskId};

/// Reference scalar execution backend.
pub struct CpuBackend {
    capabilities: ComputeCapabilities,
    descriptor: BackendDescriptor,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            capabilities: ComputeCapabilities {
                kind: BackendKind::CpuScalar,
                max_buffer_size_bytes: usize::MAX,
                max_workgroup_size: 1,
                supports_fp64: true,
                supports_fp32: true,
                supports_int32: true,
                max_parallelism: ParallelismClass::Sequential,
                supported_operations: alloc::vec![
                    "matmul".to_string(),
                    "vector_add".to_string(),
                    "relu".to_string(),
                ],
            },
            descriptor: BackendDescriptor {
                id: "cpu-scalar-ref",
                kind: BackendKind::CpuScalar,
                name: "JHCS CPU Scalar Reference".to_string(),
                is_available: true,
                memory_capacity_bytes: usize::MAX,
            },
        }
    }

    pub fn capabilities(&self) -> &ComputeCapabilities {
        &self.capabilities
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Executes a single task using raw buffer slices.
    pub fn execute_matmul(
        &self,
        task_id: TaskId,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> ComputeResult {
        match matmul_f32_scalar(a, b, c, m, k, n) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }

    pub fn execute_vector_add(
        &self,
        task_id: TaskId,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> ComputeResult {
        match vector_add_f32_scalar(a, b, c) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }

    pub fn execute_relu(&self, task_id: TaskId, a: &[f32], c: &mut [f32]) -> ComputeResult {
        match relu_f32_scalar(a, c) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul_correctness() {
        // A: 2x3
        // [ 1.0, 2.0, 3.0 ]
        // [ 4.0, 5.0, 6.0 ]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        // B: 3x2
        // [ 7.0,  8.0 ]
        // [ 9.0, 10.0 ]
        // [ 11.0, 12.0 ]
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

        // C: 2x2
        // Expected:
        // [ 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58,   1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64 ]
        // [ 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139, 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154 ]
        let mut c = [0.0f32; 4];

        let backend = CpuBackend::new();
        let res = backend.execute_matmul(TaskId::new(10), &a, &b, &mut c, 2, 3, 2);

        assert_eq!(res.status, jhcs_core::task::TaskStatus::Completed);
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_cpu_vector_add_and_relu() {
        let a = [1.0f32, -2.0, 3.5, -4.0];
        let b = [0.5f32, 1.0, -1.5, 5.0];
        let mut c = [0.0f32; 4];

        let backend = CpuBackend::new();
        backend.execute_vector_add(TaskId::new(1), &a, &b, &mut c);
        assert_eq!(c, [1.5, -1.0, 2.0, 1.0]);

        let mut relu_out = [0.0f32; 4];
        backend.execute_relu(TaskId::new(2), &c, &mut relu_out);
        assert_eq!(relu_out, [1.5, 0.0, 2.0, 1.0]);
    }
}
