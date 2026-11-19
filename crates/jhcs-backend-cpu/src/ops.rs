//! # Scalar Reference Mathematical Kernels
//!
//! Provides deterministic, portable, pure scalar implementations for all JHCS operations.
//! Serves as the ground-truth numerical reference for SIMD, GPU, and accelerator backends.

use jhcs_core::error::ComputeError;

/// Element-wise vector multiplication: C[i] = A[i] * B[i]
pub fn vector_mul_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), ComputeError> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (b.len(), 1),
        });
    }

    for i in 0..a.len() {
        c[i] = a[i] * b[i];
    }

    Ok(())
}

/// Vector dot product (scalar inner product): sum(A[i] * B[i])
pub fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> Result<f32, ComputeError> {
    if a.len() != b.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (b.len(), 1),
        });
    }

    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    Ok(sum)
}

/// Scalar multiplication / scaling: C[i] = A[i] * factor
pub fn scale_f32_scalar(a: &[f32], factor: f32, c: &mut [f32]) -> Result<(), ComputeError> {
    if a.len() != c.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (c.len(), 1),
        });
    }

    for i in 0..a.len() {
        c[i] = a[i] * factor;
    }

    Ok(())
}

/// Element-wise clamping: C[i] = min(max(A[i], min_val), max_val)
pub fn clamp_f32_scalar(
    a: &[f32],
    min_val: f32,
    max_val: f32,
    c: &mut [f32],
) -> Result<(), ComputeError> {
    if a.len() != c.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (c.len(), 1),
        });
    }

    for i in 0..a.len() {
        let val = a[i];
        c[i] = if val < min_val {
            min_val
        } else if val > max_val {
            max_val
        } else {
            val
        };
    }

    Ok(())
}

/// Reduction sum: sum(A)
pub fn reduce_sum_f32_scalar(a: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &val in a {
        sum += val;
    }
    sum
}

/// Reduction mean / average: sum(A) / len(A)
pub fn reduce_mean_f32_scalar(a: &[f32]) -> Result<f32, ComputeError> {
    if a.is_empty() {
        return Err(ComputeError::DimensionMismatch {
            expected: (1, 1),
            actual: (0, 0),
        });
    }
    Ok(reduce_sum_f32_scalar(a) / (a.len() as f32))
}

/// Reduction min and max: (min(A), max(A))
pub fn reduce_min_max_f32_scalar(a: &[f32]) -> Result<(f32, f32), ComputeError> {
    if a.is_empty() {
        return Err(ComputeError::DimensionMismatch {
            expected: (1, 1),
            actual: (0, 0),
        });
    }

    let mut min = a[0];
    let mut max = a[0];

    for &val in &a[1..] {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    Ok((min, max))
}

/// In-place vector addition: A[i] += B[i]
pub fn in_place_vector_add_f32(a: &mut [f32], b: &[f32]) -> Result<(), ComputeError> {
    if a.len() != b.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (b.len(), 1),
        });
    }

    for i in 0..a.len() {
        a[i] += b[i];
    }

    Ok(())
}

/// In-place scaling: A[i] *= factor
pub fn in_place_scale_f32(a: &mut [f32], factor: f32) {
    for val in a.iter_mut() {
        *val *= factor;
    }
}

/// In-place ReLU: A[i] = max(0, A[i])
pub fn in_place_relu_f32(a: &mut [f32]) {
    for val in a.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }
}

/// Blocked / tiled cache-friendly matrix multiplication for large matrices:
/// C = A * B in tiles of `block_size` x `block_size`.
pub fn matmul_blocked_f32_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
) -> Result<(), ComputeError> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(ComputeError::DimensionMismatch {
            expected: (m, n),
            actual: (c.len() / n.max(1), n),
        });
    }

    c.fill(0.0);
    let bs = block_size.max(1);

    for i0 in (0..m).step_by(bs) {
        let imax = (i0 + bs).min(m);
        for p0 in (0..k).step_by(bs) {
            let pmax = (p0 + bs).min(k);
            for j0 in (0..n).step_by(bs) {
                let jmax = (j0 + bs).min(n);

                for i in i0..imax {
                    let a_row = i * k;
                    let c_row = i * n;
                    for p in p0..pmax {
                        let a_val = a[a_row + p];
                        let b_row = p * n;
                        for j in j0..jmax {
                            c[c_row + j] += a_val * b[b_row + j];
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
