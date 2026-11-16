//! Pure scalar reference matrix multiplication and elementary math kernels.

use jhcs_core::error::ComputeError;

/// Computes standard matrix multiplication C = A * B in row-major order:
/// - A has dimension (m, k)
/// - B has dimension (k, n)
/// - C has dimension (m, n)
pub fn matmul_f32_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), ComputeError> {
    if a.len() != m * k {
        return Err(ComputeError::DimensionMismatch {
            expected: (m, k),
            actual: (a.len() / k.max(1), k),
        });
    }
    if b.len() != k * n {
        return Err(ComputeError::DimensionMismatch {
            expected: (k, n),
            actual: (b.len() / n.max(1), n),
        });
    }
    if c.len() != m * n {
        return Err(ComputeError::DimensionMismatch {
            expected: (m, n),
            actual: (c.len() / n.max(1), n),
        });
    }

    // Zero-out destination matrix
    c.fill(0.0);

    // Standard cache-friendly i-p-j nested loops
    for i in 0..m {
        let a_row_offset = i * k;
        let c_row_offset = i * n;
        for p in 0..k {
            let a_val = a[a_row_offset + p];
            let b_row_offset = p * n;
            for j in 0..n {
                c[c_row_offset + j] += a_val * b[b_row_offset + j];
            }
        }
    }

    Ok(())
}

/// Element-wise vector addition: C = A + B
pub fn vector_add_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), ComputeError> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (b.len(), 1),
        });
    }

    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }

    Ok(())
}

/// Element-wise Rectified Linear Unit: C = max(0, A)
pub fn relu_f32_scalar(a: &[f32], c: &mut [f32]) -> Result<(), ComputeError> {
    if a.len() != c.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (c.len(), 1),
        });
    }

    for i in 0..a.len() {
        c[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
    }

    Ok(())
}
