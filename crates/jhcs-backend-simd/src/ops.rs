//! Optimized SIMD Math Operations (Conv2D and Vector Reductions)
//! Supporting runtime feature detection with automatic fallback to scalar paths.

extern crate alloc;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Detects AVX2 and FMA support at runtime dynamically across `std` and `no_std`.
#[inline]
pub fn detect_avx2_fma() -> (bool, bool) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        (
            is_x86_feature_detected!("avx2"),
            is_x86_feature_detected!("fma"),
        )
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
    {
        let res = core::arch::x86_64::__cpuid(1);
        let has_fma = (res.ecx & (1 << 12)) != 0;
        let has_osxsave = (res.ecx & (1 << 27)) != 0;
        let has_avx = (res.ecx & (1 << 28)) != 0;
        if has_osxsave && has_avx {
            let res7 = core::arch::x86_64::__cpuid_count(7, 0);
            let has_avx2 = (res7.ebx & (1 << 5)) != 0;
            (has_avx2, has_fma)
        } else {
            (false, false)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (false, false)
    }
}

/// Redução de soma vetorial (`sum(input)`) com precisão numérica e aceleração AVX2.
///
/// Utiliza acumuladores vetoriais de 8 pistas (`__m256`) para evitar dependências
/// de dados sequenciais no pipeline da FPU, finalizando com extração de 128 bits.
pub fn reduce_sum_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let (has_avx2, _) = detect_avx2_fma();
        if has_avx2 && input.len() >= 32 {
            // SAFETY: `input` é um slice válido e todos os acessos são estritamente delimitados por `len`.
            unsafe {
                return reduce_sum_f32_avx2(input);
            }
        }
    }

    reduce_sum_f32_scalar(input)
}

#[inline]
pub fn reduce_sum_f32_scalar(input: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &x in input {
        sum += x;
    }
    sum
}

/// # Safety
/// Requer suporte de CPU para instruções AVX2 (`target_feature = "avx2"`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn reduce_sum_f32_avx2(input: &[f32]) -> f32 {
    let len = input.len();
    let ptr = input.as_ptr();
    let mut i = 0;

    // 4 acumuladores independentes (32 floats por iteração) para saturar as portas FMA/ADD da CPU
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= len {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(ptr.add(i)));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(ptr.add(i + 8)));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(ptr.add(i + 16)));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(ptr.add(i + 24)));
        i += 32;
    }

    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    while i + 8 <= len {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }

    // Redução horizontal rápida de 256 -> 128 -> escalar
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_vec = _mm_add_ss(sums, shuf2);

    let mut total = _mm_cvtss_f32(final_vec);

    // Cauda escalar para elementos restantes
    while i < len {
        total += *ptr.add(i);
        i += 1;
    }

    total
}

/// Convolução 2D válida (sem padding, stride 1) para tensores monocanal (`[H x W]`).
///
/// Vetoriza ao longo do eixo horizontal (`out_x`), transmitindo (`broadcast`) cada peso
/// do kernel para 8 pixels de saída simultaneamente.
///
/// # Panics
/// Entra em pânico se as dimensões dos slices de entrada, kernel ou saída forem inconsistentes.
pub fn conv2d_f32(
    input: &[f32],
    in_h: usize,
    in_w: usize,
    kernel: &[f32],
    k_h: usize,
    k_w: usize,
    output: &mut [f32],
) {
    assert!(in_h >= k_h && in_w >= k_w, "Kernel maior que a entrada");
    let out_h = in_h - k_h + 1;
    let out_w = in_w - k_w + 1;

    assert!(input.len() >= in_h * in_w, "Buffer de entrada insuficiente");
    assert!(kernel.len() >= k_h * k_w, "Buffer de kernel insuficiente");
    assert!(output.len() >= out_h * out_w, "Buffer de saída insuficiente");

    #[cfg(target_arch = "x86_64")]
    {
        let (has_avx2, has_fma) = detect_avx2_fma();
        if has_avx2 && has_fma && out_w >= 8 {
            // SAFETY: Todos os limites de dimensões foram validados pelos asserts acima.
            unsafe {
                conv2d_f32_avx2_fma(input, in_w, kernel, k_h, k_w, output, out_h, out_w);
            }
            return;
        }
    }

    conv2d_f32_scalar(input, in_w, kernel, k_h, k_w, output, out_h, out_w);
}

pub fn conv2d_f32_scalar(
    input: &[f32],
    in_w: usize,
    kernel: &[f32],
    k_h: usize,
    k_w: usize,
    output: &mut [f32],
    out_h: usize,
    out_w: usize,
) {
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = 0.0_f32;
            for ky in 0..k_h {
                let in_row = (oy + ky) * in_w;
                let k_row = ky * k_w;
                for kx in 0..k_w {
                    acc += input[in_row + ox + kx] * kernel[k_row + kx];
                }
            }
            output[oy * out_w + ox] = acc;
        }
    }
}

/// # Safety
/// Requer AVX2 + FMA e buffers dimensionados conforme validado em `conv2d_f32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv2d_f32_avx2_fma(
    input: &[f32],
    in_w: usize,
    kernel: &[f32],
    k_h: usize,
    k_w: usize,
    output: &mut [f32],
    out_h: usize,
    out_w: usize,
) {
    let in_ptr = input.as_ptr();
    let k_ptr = kernel.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for oy in 0..out_h {
        let out_row_ptr = out_ptr.add(oy * out_w);
        let mut ox = 0;

        // Processa 8 pixels horizontais de saída por ciclo
        while ox + 8 <= out_w {
            let mut acc = _mm256_setzero_ps();

            for ky in 0..k_h {
                let in_row_base = (oy + ky) * in_w + ox;
                let k_row_base = ky * k_w;

                for kx in 0..k_w {
                    let w_val = *k_ptr.add(k_row_base + kx);
                    let vw = _mm256_set1_ps(w_val);
                    let vin = _mm256_loadu_ps(in_ptr.add(in_row_base + kx));
                    acc = _mm256_fmadd_ps(vin, vw, acc);
                }
            }

            _mm256_storeu_ps(out_row_ptr.add(ox), acc);
            ox += 8;
        }

        // Cauda horizontal escalar
        while ox < out_w {
            let mut acc = 0.0_f32;
            for ky in 0..k_h {
                let in_row = (oy + ky) * in_w;
                let k_row = ky * k_w;
                for kx in 0..k_w {
                    acc += *in_ptr.add(in_row + ox + kx) * *k_ptr.add(k_row + kx);
                }
            }
            *out_row_ptr.add(ox) = acc;
            ox += 1;
        }
    }
}
