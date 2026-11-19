#![cfg_attr(not(feature = "std"), no_std)]

//! # JHCS Backend CPU — Scalar Reference Backend
//!
//! Provides portable, pure scalar math implementations for all JHCS operations.
//! Serves as the ground-truth numerical reference to validate SIMD and GPU results.

extern crate alloc;

pub mod matmul;
pub mod ops;

pub use matmul::{matmul_f32_scalar, relu_f32_scalar, vector_add_f32_scalar};
pub use ops::{
    clamp_f32_scalar, dot_product_f32_scalar, in_place_relu_f32, in_place_scale_f32,
    in_place_vector_add_f32, matmul_blocked_f32_scalar, reduce_mean_f32_scalar,
    reduce_min_max_f32_scalar, reduce_sum_f32_scalar, scale_f32_scalar, vector_mul_f32_scalar,
};

use alloc::string::ToString;
use jhcs_core::capabilities::{
    BackendDescriptor, BackendKind, ComputeCapabilities, ParallelismClass,
};
use jhcs_core::contracts::ComputeBackend as ComputeBackendContract;
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
                    "matmul_blocked".to_string(),
                    "vector_add".to_string(),
                    "vector_mul".to_string(),
                    "dot_product".to_string(),
                    "relu".to_string(),
                    "scale".to_string(),
                    "clamp".to_string(),
                    "reduce_sum".to_string(),
                    "reduce_mean".to_string(),
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

    /// Executes matrix multiplication C = A * B.
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

    /// Executes tiled / blocked matrix multiplication.
    pub fn execute_matmul_blocked(
        &self,
        task_id: TaskId,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        block_size: usize,
    ) -> ComputeResult {
        match matmul_blocked_f32_scalar(a, b, c, m, k, n, block_size) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }

    /// Executes element-wise vector addition: C = A + B.
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

    /// Executes element-wise vector multiplication: C = A * B.
    pub fn execute_vector_mul(
        &self,
        task_id: TaskId,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> ComputeResult {
        match vector_mul_f32_scalar(a, b, c) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }

    /// Executes vector dot product: sum(A * B).
    pub fn execute_dot_product(
        &self,
        task_id: TaskId,
        a: &[f32],
        b: &[f32],
        out: &mut f32,
    ) -> ComputeResult {
        match dot_product_f32_scalar(a, b) {
            Ok(res) => {
                *out = res;
                ComputeResult::success(task_id, BackendKind::CpuScalar, 0)
            }
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }

    /// Executes element-wise scaling: C = A * factor.
    pub fn execute_scale(
        &self,
        task_id: TaskId,
        a: &[f32],
        factor: f32,
        c: &mut [f32],
    ) -> ComputeResult {
        match scale_f32_scalar(a, factor, c) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }

    /// Executes element-wise ReLU: C = max(0, A).
    pub fn execute_relu(&self, task_id: TaskId, a: &[f32], c: &mut [f32]) -> ComputeResult {
        match relu_f32_scalar(a, c) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }

    /// Executes element-wise clamping.
    pub fn execute_clamp(
        &self,
        task_id: TaskId,
        a: &[f32],
        min_val: f32,
        max_val: f32,
        c: &mut [f32],
    ) -> ComputeResult {
        match clamp_f32_scalar(a, min_val, max_val, c) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuScalar, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuScalar, e),
        }
    }
}

impl ComputeBackendContract for CpuBackend {
    fn id(&self) -> &'static str {
        "cpu-scalar-ref"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::CpuScalar
    }

    fn is_available(&self) -> bool {
        true
    }

    fn capabilities(&self) -> &ComputeCapabilities {
        &self.capabilities
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
        // A: 2x3, B: 3x2, C: 2x2
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4];

        let backend = CpuBackend::new();
        let res = backend.execute_matmul(TaskId::new(10), &a, &b, &mut c, 2, 3, 2);

        assert_eq!(res.status, jhcs_core::task::TaskStatus::Completed);
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);

        // Validate blocked matmul matches exactly
        let mut c_blocked = [0.0f32; 4];
        let res_blocked =
            backend.execute_matmul_blocked(TaskId::new(11), &a, &b, &mut c_blocked, 2, 3, 2, 2);
        assert_eq!(res_blocked.status, jhcs_core::task::TaskStatus::Completed);
        assert_eq!(c, c_blocked);
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

    #[test]
    fn test_cpu_vector_mul_and_dot_product() {
        let a = [2.0f32, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0];
        let mut c = [0.0f32; 3];

        let backend = CpuBackend::new();
        let res_mul = backend.execute_vector_mul(TaskId::new(3), &a, &b, &mut c);
        assert_eq!(res_mul.status, jhcs_core::task::TaskStatus::Completed);
        assert_eq!(c, [10.0, 18.0, 28.0]);

        let mut dot = 0.0f32;
        let res_dot = backend.execute_dot_product(TaskId::new(4), &a, &b, &mut dot);
        assert_eq!(res_dot.status, jhcs_core::task::TaskStatus::Completed);
        assert_eq!(dot, 10.0 + 18.0 + 28.0);
    }

    #[test]
    fn test_cpu_statistical_reductions() {
        let data = [10.0f32, -5.0, 20.0, 15.0];

        let sum = reduce_sum_f32_scalar(&data);
        assert_eq!(sum, 40.0);

        let mean = reduce_mean_f32_scalar(&data).unwrap();
        assert_eq!(mean, 10.0);

        let (min, max) = reduce_min_max_f32_scalar(&data).unwrap();
        assert_eq!(min, -5.0);
        assert_eq!(max, 20.0);
    }

    #[test]
    fn test_cpu_in_place_operations() {
        let mut data = [1.0f32, -2.0, 3.0];
        let delta = [0.5f32, 1.0, -1.0];

        in_place_vector_add_f32(&mut data, &delta).unwrap();
        assert_eq!(data, [1.5, -1.0, 2.0]);

        in_place_scale_f32(&mut data, 2.0);
        assert_eq!(data, [3.0, -2.0, 4.0]);

        in_place_relu_f32(&mut data);
        assert_eq!(data, [3.0, 0.0, 4.0]);
    }

    #[test]
    fn test_cpu_clamping_and_scaling() {
        let data = [-10.0f32, 0.0, 5.0, 15.0];
        let mut clamped = [0.0f32; 4];

        let backend = CpuBackend::new();
        backend.execute_clamp(TaskId::new(5), &data, -2.0, 10.0, &mut clamped);
        assert_eq!(clamped, [-2.0, 0.0, 5.0, 10.0]);

        let mut scaled = [0.0f32; 4];
        backend.execute_scale(TaskId::new(6), &clamped, 0.5, &mut scaled);
        assert_eq!(scaled, [-1.0, 0.0, 2.5, 5.0]);
    }
}
