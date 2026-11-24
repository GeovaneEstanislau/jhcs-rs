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
        if i + 64 <= len {
            _mm_prefetch(ptr.add(i + 32) as *const i8, _MM_HINT_T0);
        }
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
            if ox + 16 <= out_w {
                _mm_prefetch(in_ptr.add(oy * in_w + ox + 8) as *const i8, _MM_HINT_T0);
            }
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

/// Redimensionamento bilinear de imagens RGBA com aceleração e fallback escalar.
pub fn scale_image_rgba_simd(
    src: &[u32],
    src_w: usize,
    src_h: usize,
    dst: &mut [u32],
    dst_w: usize,
    dst_h: usize,
) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }
    if src.len() < src_w * src_h || dst.len() < dst_w * dst_h {
        return;
    }

    let x_ratio = ((src_w << 16) / dst_w) as u32;
    let y_ratio = ((src_h << 16) / dst_h) as u32;

    for y in 0..dst_h {
        let sy = (y as u32 * y_ratio) >> 16;
        let y_diff = ((y as u32 * y_ratio) & 0xFFFF) as u32;
        let y_diff_inv = 0x10000 - y_diff;
        let sy1 = (sy + 1).min(src_h as u32 - 1) as usize;
        let sy0 = sy as usize;

        let row_dst = y * dst_w;
        let row_src0 = sy0 * src_w;
        let row_src1 = sy1 * src_w;

        for x in 0..dst_w {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if x + 16 < dst_w {
                    let next_sx = ((x + 16) as u32 * x_ratio) >> 16;
                    core::arch::x86_64::_mm_prefetch(src.as_ptr().add(row_src0 + next_sx as usize) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                }
            }

            let sx = (x as u32 * x_ratio) >> 16;
            let x_diff = ((x as u32 * x_ratio) & 0xFFFF) as u32;
            let x_diff_inv = 0x10000 - x_diff;
            let sx1 = (sx + 1).min(src_w as u32 - 1) as usize;
            let sx0 = sx as usize;

            let p00 = src[row_src0 + sx0];
            let p10 = src[row_src0 + sx1];
            let p01 = src[row_src1 + sx0];
            let p11 = src[row_src1 + sx1];

            let mut out_pixel = 0u32;
            for shift in [0, 8, 16, 24] {
                let c00 = ((p00 >> shift) & 0xFF) as u32;
                let c10 = ((p10 >> shift) & 0xFF) as u32;
                let c01 = ((p01 >> shift) & 0xFF) as u32;
                let c11 = ((p11 >> shift) & 0xFF) as u32;

                let top = (c00 * x_diff_inv + c10 * x_diff) >> 16;
                let bottom = (c01 * x_diff_inv + c11 * x_diff) >> 16;
                let final_c = ((top * y_diff_inv + bottom * y_diff) >> 16) & 0xFF;
                out_pixel |= final_c << shift;
            }
            dst[row_dst + x] = out_pixel;
        }
    }
}

/// Composição alfa em lote para buffers RGBA com fallback escalar.
pub fn alpha_blend_rgba_batch_simd(dst: &mut [u32], src: &[u32], len: usize) {
    let count = len.min(dst.len()).min(src.len());
    let mut i = 0;

    while i < count {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if i + 16 < count {
                core::arch::x86_64::_mm_prefetch(src.as_ptr().add(i + 16) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                core::arch::x86_64::_mm_prefetch(dst.as_ptr().add(i + 16) as *const i8, core::arch::x86_64::_MM_HINT_T0);
            }
        }
        let s = src[i];
        let sa = (s >> 24) & 0xFF;
        if sa == 255 {
            dst[i] = s;
        } else if sa > 0 {
            let d = dst[i];
            let inv_a = 255 - sa;

            let sr = (s >> 16) & 0xFF;
            let sg = (s >> 8) & 0xFF;
            let sb = s & 0xFF;

            let dr = (d >> 16) & 0xFF;
            let dg = (d >> 8) & 0xFF;
            let db = d & 0xFF;

            let out_r = (sr * sa + dr * inv_a) / 255;
            let out_g = (sg * sa + dg * inv_a) / 255;
            let out_b = (sb * sa + db * inv_a) / 255;

            dst[i] = (0xFF << 24) | (out_r << 16) | (out_g << 8) | out_b;
        }
        i += 1;
    }
}

/// Desfoque de caixa bidimensional (Horizontal + Vertical) para buffers RGBA de 32 bits.
///
/// Utiliza acumuladores em janela móvel (moving average) com complexidade O(W * H)
/// independente do raio de desfoque. Opera com separação horizontal e vertical
/// para máxima localidade de cache e vetorização pelo compilador.
pub fn box_blur_rgba_simd(
    src: &[u32],
    dst: &mut [u32],
    scratch: &mut [u32],
    w: usize,
    h: usize,
    radius: usize,
) {
    if w == 0 || h == 0 || radius == 0 {
        let count = w * h;
        if dst.len() >= count && src.len() >= count {
            dst[..count].copy_from_slice(&src[..count]);
        }
        return;
    }

    let count = w * h;
    if src.len() < count || dst.len() < count || scratch.len() < count {
        return;
    }

    let r = radius.min(w.min(h) / 2).max(1);
    let win_size = (2 * r + 1) as u32;

    // Multiplicador de ponto fixo (24 bits) para evitar divisão inteira por pixel:
    // (sum * inv_win) >> 24
    let inv_win = ((1u64 << 24) / (win_size as u64)) as u32;

    // ── Passagem 1: Horizontal (src -> scratch) ──────────────────────────────
    for y in 0..h {
        let row_start = y * w;
        let mut sum_a = 0u32;
        let mut sum_r = 0u32;
        let mut sum_g = 0u32;
        let mut sum_b = 0u32;

        let first_px = src[row_start];
        let fa = (first_px >> 24) & 0xFF;
        let fr = (first_px >> 16) & 0xFF;
        let fg = (first_px >> 8) & 0xFF;
        let fb = first_px & 0xFF;

        sum_a += fa * (r as u32 + 1);
        sum_r += fr * (r as u32 + 1);
        sum_g += fg * (r as u32 + 1);
        sum_b += fb * (r as u32 + 1);

        for i in 1..=r {
            let px = src[row_start + i.min(w - 1)];
            sum_a += (px >> 24) & 0xFF;
            sum_r += (px >> 16) & 0xFF;
            sum_g += (px >> 8) & 0xFF;
            sum_b += px & 0xFF;
        }

        for x in 0..w {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if x + r + 16 < w {
                    core::arch::x86_64::_mm_prefetch(src.as_ptr().add(row_start + x + r + 16) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                }
            }

            let avg_a = ((sum_a * inv_win) >> 24).min(255);
            let avg_r = ((sum_r * inv_win) >> 24).min(255);
            let avg_g = ((sum_g * inv_win) >> 24).min(255);
            let avg_b = ((sum_b * inv_win) >> 24).min(255);

            scratch[row_start + x] = (avg_a << 24) | (avg_r << 16) | (avg_g << 8) | avg_b;

            let left_idx = (x as isize - r as isize).max(0) as usize;
            let right_idx = (x + r + 1).min(w - 1);

            let px_out = src[row_start + left_idx];
            let px_in = src[row_start + right_idx];

            sum_a = sum_a.saturating_sub((px_out >> 24) & 0xFF) + ((px_in >> 24) & 0xFF);
            sum_r = sum_r.saturating_sub((px_out >> 16) & 0xFF) + ((px_in >> 16) & 0xFF);
            sum_g = sum_g.saturating_sub((px_out >> 8) & 0xFF) + ((px_in >> 8) & 0xFF);
            sum_b = sum_b.saturating_sub(px_out & 0xFF) + (px_in & 0xFF);
        }
    }

    // ── Passagem 2: Vertical (scratch -> dst) ────────────────────────────────
    for x in 0..w {
        let mut sum_a = 0u32;
        let mut sum_r = 0u32;
        let mut sum_g = 0u32;
        let mut sum_b = 0u32;

        let first_px = scratch[x];
        let fa = (first_px >> 24) & 0xFF;
        let fr = (first_px >> 16) & 0xFF;
        let fg = (first_px >> 8) & 0xFF;
        let fb = first_px & 0xFF;

        sum_a += fa * (r as u32 + 1);
        sum_r += fr * (r as u32 + 1);
        sum_g += fg * (r as u32 + 1);
        sum_b += fb * (r as u32 + 1);

        for i in 1..=r {
            let px = scratch[i.min(h - 1) * w + x];
            sum_a += (px >> 24) & 0xFF;
            sum_r += (px >> 16) & 0xFF;
            sum_g += (px >> 8) & 0xFF;
            sum_b += px & 0xFF;
        }

        for y in 0..h {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if y + r + 8 < h {
                    core::arch::x86_64::_mm_prefetch(scratch.as_ptr().add((y + r + 8) * w + x) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                }
            }

            let avg_a = ((sum_a * inv_win) >> 24).min(255);
            let avg_r = ((sum_r * inv_win) >> 24).min(255);
            let avg_g = ((sum_g * inv_win) >> 24).min(255);
            let avg_b = ((sum_b * inv_win) >> 24).min(255);

            dst[y * w + x] = (avg_a << 24) | (avg_r << 16) | (avg_g << 8) | avg_b;

            let top_idx = (y as isize - r as isize).max(0) as usize;
            let bottom_idx = (y + r + 1).min(h - 1);

            let px_out = scratch[top_idx * w + x];
            let px_in = scratch[bottom_idx * w + x];

            sum_a = sum_a.saturating_sub((px_out >> 24) & 0xFF) + ((px_in >> 24) & 0xFF);
            sum_r = sum_r.saturating_sub((px_out >> 16) & 0xFF) + ((px_in >> 16) & 0xFF);
            sum_g = sum_g.saturating_sub((px_out >> 8) & 0xFF) + ((px_in >> 8) & 0xFF);
            sum_b = sum_b.saturating_sub(px_out & 0xFF) + (px_in & 0xFF);
        }
    }
}
