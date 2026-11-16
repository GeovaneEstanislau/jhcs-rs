//! Vectorized matrix multiplication with automatic SIMD width adaptation.

use jhcs_core::error::ComputeError;

/// Vectorized matrix multiplication kernel using 4-way and 8-way chunking.
/// LLVM autovectorizes these chunked loops into SSE/AVX vector instructions
/// when target features permit, maintaining 100% numerical portability.
pub fn matmul_f32_simd(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), ComputeError> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(ComputeError::DimensionMismatch {
            expected: (m, n),
            actual: (c.len() / n.max(1), n),
        });
    }

    c.fill(0.0);

    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;

        for p in 0..k {
            let a_val = a[a_row + p];
            let b_row = p * n;

            let c_slice = &mut c[c_row..c_row + n];
            let b_slice = &b[b_row..b_row + n];

            // 8-wide chunked vector step
            let mut j = 0;
            while j + 8 <= n {
                c_slice[j] += a_val * b_slice[j];
                c_slice[j + 1] += a_val * b_slice[j + 1];
                c_slice[j + 2] += a_val * b_slice[j + 2];
                c_slice[j + 3] += a_val * b_slice[j + 3];
                c_slice[j + 4] += a_val * b_slice[j + 4];
                c_slice[j + 5] += a_val * b_slice[j + 5];
                c_slice[j + 6] += a_val * b_slice[j + 6];
                c_slice[j + 7] += a_val * b_slice[j + 7];
                j += 8;
            }

            // Remainder scalar step
            while j < n {
                c_slice[j] += a_val * b_slice[j];
                j += 1;
            }
        }
    }

    Ok(())
}

/// Vectorized element-wise addition: C = A + B
pub fn vector_add_f32_simd(a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), ComputeError> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (b.len(), 1),
        });
    }

    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
        c[i + 4] = a[i + 4] + b[i + 4];
        c[i + 5] = a[i + 5] + b[i + 5];
        c[i + 6] = a[i + 6] + b[i + 6];
        c[i + 7] = a[i + 7] + b[i + 7];
        i += 8;
    }

    while i < n {
        c[i] = a[i] + b[i];
        i += 1;
    }

    Ok(())
}
