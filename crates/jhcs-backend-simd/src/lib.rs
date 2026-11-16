#![cfg_attr(not(feature = "std"), no_std)]

//! # JHCS Backend SIMD — Vectorized CPU Backend
//!
//! Provides optimized vector SIMD math kernels with dynamic CPUID detection
//! and guaranteed fallback to scalar execution.

extern crate alloc;

pub mod matmul;

pub use matmul::{matmul_f32_simd, vector_add_f32_simd};

use alloc::string::ToString;
use jhcs_core::capabilities::{
    BackendDescriptor, BackendKind, ComputeCapabilities, ParallelismClass,
};
use jhcs_core::task::{ComputeResult, TaskId};

pub struct SimdBackend {
    capabilities: ComputeCapabilities,
    descriptor: BackendDescriptor,
    has_avx: bool,
    has_sse: bool,
}

impl SimdBackend {
    pub fn new() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        let (has_avx, has_sse) = (
            is_x86_feature_detected!("avx"),
            is_x86_feature_detected!("sse2"),
        );

        #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
        let (has_avx, has_sse) = (false, true);

        let max_parallelism = if has_avx {
            ParallelismClass::Vectorized256
        } else if has_sse {
            ParallelismClass::Vectorized128
        } else {
            ParallelismClass::Sequential
        };

        Self {
            capabilities: ComputeCapabilities {
                kind: BackendKind::CpuSimd,
                max_buffer_size_bytes: usize::MAX,
                max_workgroup_size: 8,
                supports_fp64: true,
                supports_fp32: true,
                supports_int32: true,
                max_parallelism,
                supported_operations: alloc::vec![
                    "matmul".to_string(),
                    "vector_add".to_string(),
                    "relu".to_string(),
                ],
            },
            descriptor: BackendDescriptor {
                id: "cpu-simd-opt",
                kind: BackendKind::CpuSimd,
                name: if has_avx {
                    "JHCS CPU SIMD (AVX2-enabled)".to_string()
                } else if has_sse {
                    "JHCS CPU SIMD (SSE2-enabled)".to_string()
                } else {
                    "JHCS CPU SIMD (Scalar Fallback)".to_string()
                },
                is_available: true,
                memory_capacity_bytes: usize::MAX,
            },
            has_avx,
            has_sse,
        }
    }

    pub fn capabilities(&self) -> &ComputeCapabilities {
        &self.capabilities
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    pub fn has_avx(&self) -> bool {
        self.has_avx
    }

    pub fn has_sse(&self) -> bool {
        self.has_sse
    }

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
        match matmul_f32_simd(a, b, c, m, k, n) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuSimd, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuSimd, e),
        }
    }

    pub fn execute_vector_add(
        &self,
        task_id: TaskId,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> ComputeResult {
        match vector_add_f32_simd(a, b, c) {
            Ok(()) => ComputeResult::success(task_id, BackendKind::CpuSimd, 0),
            Err(e) => ComputeResult::failure(task_id, BackendKind::CpuSimd, e),
        }
    }
}

impl Default for SimdBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jhcs_backend_cpu::CpuBackend;

    #[test]
    fn test_simd_vs_scalar_numerical_equivalence() {
        let m = 16;
        let k = 32;
        let n = 24;

        // Initialize pseudo-random data
        let mut a = alloc::vec![0.0f32; m * k];
        let mut b = alloc::vec![0.0f32; k * n];

        for i in 0..a.len() {
            a[i] = ((i * 17 + 3) % 100) as f32 * 0.1;
        }
        for i in 0..b.len() {
            b[i] = ((i * 31 + 7) % 100) as f32 * 0.1;
        }

        let mut c_scalar = alloc::vec![0.0f32; m * n];
        let mut c_simd = alloc::vec![0.0f32; m * n];

        let cpu = CpuBackend::new();
        let simd = SimdBackend::new();

        cpu.execute_matmul(TaskId::new(1), &a, &b, &mut c_scalar, m, k, n);
        simd.execute_matmul(TaskId::new(2), &a, &b, &mut c_simd, m, k, n);

        for idx in 0..(m * n) {
            let diff = (c_scalar[idx] - c_simd[idx]).abs();
            assert!(
                diff < 1e-4,
                "Divergence at index {}: scalar={}, simd={}, diff={}",
                idx,
                c_scalar[idx],
                c_simd[idx],
                diff
            );
        }
    }
}
