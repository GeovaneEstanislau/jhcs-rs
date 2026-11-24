//! Vectorized matrix multiplication with automatic SIMD width adaptation.

use jhcs_core::error::ComputeError;

/// Vectorized matrix multiplication kernel using explicit SIMD intrinsics.
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

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let (has_avx2, has_fma) = crate::ops::detect_avx2_fma();
        if has_avx2 && has_fma && n >= 8 {
            matmul_f32_simd_x86_avx2_fma(a, b, c, m, k, n);
        } else {
            matmul_f32_simd_x86_sse(a, b, c, m, k, n);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        matmul_f32_simd_fallback(a, b, c, m, k, n);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_f32_simd_x86_avx2_fma(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use core::arch::x86_64::*;

    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;

        for p in 0..k {
            let a_val = *a.get_unchecked(a_row + p);
            let a_vec = _mm256_set1_ps(a_val);
            let b_row = p * n;

            let mut j = 0;
            // 32-wide unrolled AVX2+FMA pipeline (4 x 256-bit vector registers)
            while j + 32 <= n {
                let c_ptr = c.as_mut_ptr().add(c_row + j);
                let b_ptr = b.as_ptr().add(b_row + j);

                // Pré-aquecimento de cache L1/L2 uma janela à frente (64 bytes / 16 floats à frente)
                if j + 48 <= n {
                    _mm_prefetch(b_ptr.add(32) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(c_ptr.add(32) as *const i8, _MM_HINT_T0);
                }

                let c0 = _mm256_loadu_ps(c_ptr);
                let b0 = _mm256_loadu_ps(b_ptr);
                _mm256_storeu_ps(c_ptr, _mm256_fmadd_ps(a_vec, b0, c0));

                let c1 = _mm256_loadu_ps(c_ptr.add(8));
                let b1 = _mm256_loadu_ps(b_ptr.add(8));
                _mm256_storeu_ps(c_ptr.add(8), _mm256_fmadd_ps(a_vec, b1, c1));

                let c2 = _mm256_loadu_ps(c_ptr.add(16));
                let b2 = _mm256_loadu_ps(b_ptr.add(16));
                _mm256_storeu_ps(c_ptr.add(16), _mm256_fmadd_ps(a_vec, b2, c2));

                let c3 = _mm256_loadu_ps(c_ptr.add(24));
                let b3 = _mm256_loadu_ps(b_ptr.add(24));
                _mm256_storeu_ps(c_ptr.add(24), _mm256_fmadd_ps(a_vec, b3, c3));

                j += 32;
            }

            while j + 8 <= n {
                let c_ptr = c.as_mut_ptr().add(c_row + j);
                let b_ptr = b.as_ptr().add(b_row + j);

                let c0 = _mm256_loadu_ps(c_ptr);
                let b0 = _mm256_loadu_ps(b_ptr);
                _mm256_storeu_ps(c_ptr, _mm256_fmadd_ps(a_vec, b0, c0));

                j += 8;
            }

            while j < n {
                *c.get_unchecked_mut(c_row + j) += a_val * *b.get_unchecked(b_row + j);
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse,sse2")]
unsafe fn matmul_f32_simd_x86_sse(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use core::arch::x86_64::*;

    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;

        for p in 0..k {
            let a_val = *a.get_unchecked(a_row + p);
            let a_vec = _mm_set1_ps(a_val);
            let b_row = p * n;

            let mut j = 0;
            // 16-wide unrolled SSE pipeline (4 x 128-bit vector registers)
            while j + 16 <= n {
                let c_ptr = c.as_mut_ptr().add(c_row + j);
                let b_ptr = b.as_ptr().add(b_row + j);

                // Pré-aquecimento de cache L1 uma janela à frente (64 bytes / 16 floats à frente)
                if j + 32 <= n {
                    _mm_prefetch(b_ptr.add(16) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(c_ptr.add(16) as *const i8, _MM_HINT_T0);
                }

                let c0 = _mm_loadu_ps(c_ptr);
                let b0 = _mm_loadu_ps(b_ptr);
                _mm_storeu_ps(c_ptr, _mm_add_ps(c0, _mm_mul_ps(a_vec, b0)));

                let c1 = _mm_loadu_ps(c_ptr.add(4));
                let b1 = _mm_loadu_ps(b_ptr.add(4));
                _mm_storeu_ps(c_ptr.add(4), _mm_add_ps(c1, _mm_mul_ps(a_vec, b1)));

                let c2 = _mm_loadu_ps(c_ptr.add(8));
                let b2 = _mm_loadu_ps(b_ptr.add(8));
                _mm_storeu_ps(c_ptr.add(8), _mm_add_ps(c2, _mm_mul_ps(a_vec, b2)));

                let c3 = _mm_loadu_ps(c_ptr.add(12));
                let b3 = _mm_loadu_ps(b_ptr.add(12));
                _mm_storeu_ps(c_ptr.add(12), _mm_add_ps(c3, _mm_mul_ps(a_vec, b3)));

                j += 16;
            }

            while j + 4 <= n {
                let c_ptr = c.as_mut_ptr().add(c_row + j);
                let b_ptr = b.as_ptr().add(b_row + j);

                let c0 = _mm_loadu_ps(c_ptr);
                let b0 = _mm_loadu_ps(b_ptr);
                _mm_storeu_ps(c_ptr, _mm_add_ps(c0, _mm_mul_ps(a_vec, b0)));

                j += 4;
            }

            while j < n {
                *c.get_unchecked_mut(c_row + j) += a_val * *b.get_unchecked(b_row + j);
                j += 1;
            }
        }
    }
}

#[allow(dead_code)]
fn matmul_f32_simd_fallback(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;

        for p in 0..k {
            let a_val = a[a_row + p];
            let b_row = p * n;

            let c_slice = &mut c[c_row..c_row + n];
            let b_slice = &b[b_row..b_row + n];

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

            while j < n {
                c_slice[j] += a_val * b_slice[j];
                j += 1;
            }
        }
    }
}

/// Vectorized element-wise addition: C = A + B
pub fn vector_add_f32_simd(a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), ComputeError> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: (a.len(), 1),
            actual: (b.len(), 1),
        });
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let (has_avx2, _) = crate::ops::detect_avx2_fma();
        if has_avx2 && a.len() >= 32 {
            vector_add_f32_simd_x86_avx2(a, b, c);
        } else {
            vector_add_f32_simd_x86_sse(a, b, c);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        vector_add_f32_simd_fallback(a, b, c);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_add_f32_simd_x86_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    while i + 32 <= n {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);
        let c_ptr = c.as_mut_ptr().add(i);

        _mm256_storeu_ps(c_ptr, _mm256_add_ps(_mm256_loadu_ps(a_ptr), _mm256_loadu_ps(b_ptr)));
        _mm256_storeu_ps(c_ptr.add(8), _mm256_add_ps(_mm256_loadu_ps(a_ptr.add(8)), _mm256_loadu_ps(b_ptr.add(8))));
        _mm256_storeu_ps(c_ptr.add(16), _mm256_add_ps(_mm256_loadu_ps(a_ptr.add(16)), _mm256_loadu_ps(b_ptr.add(16))));
        _mm256_storeu_ps(c_ptr.add(24), _mm256_add_ps(_mm256_loadu_ps(a_ptr.add(24)), _mm256_loadu_ps(b_ptr.add(24))));
        i += 32;
    }

    while i + 8 <= n {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);
        let c_ptr = c.as_mut_ptr().add(i);
        _mm256_storeu_ps(c_ptr, _mm256_add_ps(_mm256_loadu_ps(a_ptr), _mm256_loadu_ps(b_ptr)));
        i += 8;
    }

    while i < n {
        *c.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse,sse2")]
unsafe fn vector_add_f32_simd_x86_sse(a: &[f32], b: &[f32], c: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    while i + 16 <= n {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);
        let c_ptr = c.as_mut_ptr().add(i);

        _mm_storeu_ps(c_ptr, _mm_add_ps(_mm_loadu_ps(a_ptr), _mm_loadu_ps(b_ptr)));
        _mm_storeu_ps(c_ptr.add(4), _mm_add_ps(_mm_loadu_ps(a_ptr.add(4)), _mm_loadu_ps(b_ptr.add(4))));
        _mm_storeu_ps(c_ptr.add(8), _mm_add_ps(_mm_loadu_ps(a_ptr.add(8)), _mm_loadu_ps(b_ptr.add(8))));
        _mm_storeu_ps(c_ptr.add(12), _mm_add_ps(_mm_loadu_ps(a_ptr.add(12)), _mm_loadu_ps(b_ptr.add(12))));
        i += 16;
    }

    while i + 4 <= n {
        let a_ptr = a.as_ptr().add(i);
        let b_ptr = b.as_ptr().add(i);
        let c_ptr = c.as_mut_ptr().add(i);
        _mm_storeu_ps(c_ptr, _mm_add_ps(_mm_loadu_ps(a_ptr), _mm_loadu_ps(b_ptr)));
        i += 4;
    }

    while i < n {
        *c.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
        i += 1;
    }
}

#[allow(dead_code)]
fn vector_add_f32_simd_fallback(a: &[f32], b: &[f32], c: &mut [f32]) {
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
}
