#![cfg_attr(not(feature = "std"), no_std)]
#![allow(x86_softfloat_sse)]

//! # JHCS Backend SIMD — Vectorized CPU Backend
//!
//! Provides optimized vector SIMD math kernels with dynamic CPUID detection
//! and guaranteed fallback to scalar execution.

extern crate alloc;

pub mod matmul;
pub mod ops;

pub use matmul::{matmul_f32_simd, vector_add_f32_simd};
pub use ops::{conv2d_f32, conv2d_f32_scalar, reduce_sum_f32, reduce_sum_f32_scalar};

use alloc::string::ToString;
use jhcs_core::capabilities::{
    BackendDescriptor, BackendKind, ComputeCapabilities, ParallelismClass,
};
use jhcs_core::contracts::ComputeBackend as ComputeBackendContract;
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
                    "conv2d".to_string(),
                    "reduce_sum".to_string(),
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

    pub fn execute_conv2d(
        &self,
        task_id: TaskId,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        kernel: &[f32],
        k_h: usize,
        k_w: usize,
        output: &mut [f32],
    ) -> ComputeResult {
        conv2d_f32(input, in_h, in_w, kernel, k_h, k_w, output);
        ComputeResult::success(task_id, BackendKind::CpuSimd, 0)
    }

    pub fn execute_reduce_sum(
        &self,
        task_id: TaskId,
        input: &[f32],
    ) -> (f32, ComputeResult) {
        let sum = reduce_sum_f32(input);
        (sum, ComputeResult::success(task_id, BackendKind::CpuSimd, 0))
    }
}

impl ComputeBackendContract for SimdBackend {
    fn id(&self) -> &'static str {
        "cpu-simd-opt"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::CpuSimd
    }

    fn is_available(&self) -> bool {
        self.has_sse || self.has_avx
    }

    fn capabilities(&self) -> &ComputeCapabilities {
        &self.capabilities
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

    fn verify_matmul_parity(m: usize, k: usize, n: usize) {
        let mut a = alloc::vec![0.0f32; m * k];
        let mut b = alloc::vec![0.0f32; k * n];

        for i in 0..a.len() {
            a[i] = ((i * 17 + 3) % 100) as f32 * 0.05;
        }
        for i in 0..b.len() {
            b[i] = ((i * 31 + 7) % 100) as f32 * 0.05;
        }

        let mut c_scalar = alloc::vec![0.0f32; m * n];
        let mut c_simd = alloc::vec![0.0f32; m * n];

        let cpu = CpuBackend::new();
        cpu.execute_matmul(jhcs_core::task::TaskId::new(1), &a, &b, &mut c_scalar, m, k, n);

        matmul_f32_simd(&a, &b, &mut c_simd, m, k, n).expect("SIMD matmul should succeed");

        for i in 0..c_scalar.len() {
            let diff = (c_scalar[i] - c_simd[i]).abs();
            let max_val = c_scalar[i].abs().max(1.0);
            let rel_diff = diff / max_val;
            assert!(
                diff < 1e-3 || rel_diff < 1e-4,
                "Parity failure at {}: scalar={}, simd={}, diff={}, rel_diff={}",
                i,
                c_scalar[i],
                c_simd[i],
                diff,
                rel_diff
            );
        }
    }

    #[test]
    fn test_simd_vs_scalar_numerical_equivalence_arbitrary() {
        verify_matmul_parity(16, 32, 24);
    }

    #[test]
    fn test_simd_vs_scalar_numerical_equivalence_64x64() {
        verify_matmul_parity(64, 64, 64);
    }

    #[test]
    fn test_simd_vs_scalar_numerical_equivalence_256x256() {
        verify_matmul_parity(256, 256, 256);
    }

    #[test]
    fn test_simd_vs_scalar_numerical_equivalence_1024x1024() {
        verify_matmul_parity(1024, 1024, 1024);
    }

    #[test]
    fn test_reduce_sum_parity() {
        let sizes = [0, 1, 7, 8, 15, 31, 32, 64, 127, 256, 1024, 65536];
        for &sz in &sizes {
            let mut data = alloc::vec![0.0f32; sz];
            for i in 0..sz {
                data[i] = ((i * 13 + 5) % 100) as f32 * 0.02 - 1.0;
            }

            let sum_scalar = reduce_sum_f32_scalar(&data);
            let sum_simd = reduce_sum_f32(&data);

            let diff = (sum_scalar - sum_simd).abs();
            let tolerance = 1e-3 * (sz as f32).max(1.0);
            assert!(
                diff <= tolerance,
                "Reduce sum mismatch for size {}: scalar={}, simd={}, diff={}, tol={}",
                sz,
                sum_scalar,
                sum_simd,
                diff,
                tolerance
            );
        }
    }

    #[test]
    fn test_conv2d_parity() {
        // Test case 1: 3x3 kernel on 16x16
        {
            let in_h = 16;
            let in_w = 16;
            let k_h = 3;
            let k_w = 3;
            let out_h = in_h - k_h + 1;
            let out_w = in_w - k_w + 1;

            let mut input = alloc::vec![0.0f32; in_h * in_w];
            for i in 0..input.len() {
                input[i] = (i % 20) as f32 * 0.1;
            }

            let kernel = [
                0.0625, 0.125, 0.0625,
                0.125,  0.25,  0.125,
                0.0625, 0.125, 0.0625,
            ];

            let mut out_scalar = alloc::vec![0.0f32; out_h * out_w];
            let mut out_simd = alloc::vec![0.0f32; out_h * out_w];

            conv2d_f32_scalar(&input, in_w, &kernel, k_h, k_w, &mut out_scalar, out_h, out_w);
            conv2d_f32(&input, in_h, in_w, &kernel, k_h, k_w, &mut out_simd);

            for i in 0..out_scalar.len() {
                let diff = (out_scalar[i] - out_simd[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Conv2D mismatch at {}: scalar={}, simd={}, diff={}",
                    i,
                    out_scalar[i],
                    out_simd[i],
                    diff
                );
            }
        }

        // Test case 2: 5x5 kernel on 32x32
        {
            let in_h = 32;
            let in_w = 32;
            let k_h = 5;
            let k_w = 5;
            let out_h = in_h - k_h + 1;
            let out_w = in_w - k_w + 1;

            let mut input = alloc::vec![0.0f32; in_h * in_w];
            for i in 0..input.len() {
                input[i] = ((i * 7 + 11) % 50) as f32 * 0.05;
            }

            let mut kernel = [0.0f32; 25];
            for i in 0..25 {
                kernel[i] = ((i * 3 + 1) % 10) as f32 * 0.01;
            }

            let mut out_scalar = alloc::vec![0.0f32; out_h * out_w];
            let mut out_simd = alloc::vec![0.0f32; out_h * out_w];

            conv2d_f32_scalar(&input, in_w, &kernel, k_h, k_w, &mut out_scalar, out_h, out_w);
            conv2d_f32(&input, in_h, in_w, &kernel, k_h, k_w, &mut out_simd);

            for i in 0..out_scalar.len() {
                let diff = (out_scalar[i] - out_simd[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Conv2D mismatch at {}: scalar={}, simd={}, diff={}",
                    i,
                    out_scalar[i],
                    out_simd[i],
                    diff
                );
            }
        }
    }
}
