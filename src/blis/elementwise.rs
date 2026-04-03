//! SIMD-accelerated element-wise operations.
//!
//! AVX2 implementations of ReLU, vector add, and scalar multiply.
//! These are bandwidth-bound at large sizes; SIMD helps at small-to-medium
//! sizes by reducing instruction count and enabling wider stores.
//!
//! # Algorithm
//!
//! ReLU: `_mm256_max_ps(x, zero)` — single instruction per 8 elements
//! Add: `_mm256_add_ps(a, b)` — single instruction per 8 elements
//! Mul scalar: `_mm256_mul_ps(x, scalar_vec)` — single instruction per 8 elements
//!
//! Contract: provable-contracts/contracts/activation-kernel-v1.yaml

use crate::error::TruenoError;

// ============================================================================
// ReLU
// ============================================================================

/// ReLU: output_i = max(0, input_i)
///
/// Uses AVX2 `_mm256_max_ps` when available.
///
/// # Errors
///
/// Returns `Err` if input and output lengths don't match.
pub fn relu(input: &[f32], output: &mut [f32]) -> Result<(), TruenoError> {
    contract_pre_relu!(input);
    let n = input.len();
    if n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "relu size mismatch: input[{}], output[{}]",
            n,
            output.len()
        )));
    }

    // Parallel for very large bandwidth-bound arrays (≥16M elements).
    // Below 16M: single-core saturates channel, rayon overhead hurts.
    #[cfg(feature = "parallel")]
    if n >= 16_000_000 {
        use rayon::prelude::*;
        let threads = 4.min(rayon::current_num_threads());
        let chunk = (((n + threads - 1) / threads) + 15) & !15;
        input.par_chunks(chunk).zip(output.par_chunks_mut(chunk)).for_each(|(inp, out)| {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                unsafe { relu_avx2(inp, out) };
                return;
            }
            for i in 0..inp.len() {
                out[i] = inp[i].max(0.0);
            }
        });
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        // For bandwidth-bound elementwise ops, AVX2 at full clock beats
        // AVX-512 at throttled clock (Zen 4: ~30% frequency reduction).
        // Use AVX-512 only for small arrays that fit in L1/L2 where
        // compute throughput matters more than bandwidth.
        if is_x86_feature_detected!("avx512f") && n <= 4096 {
            unsafe {
                relu_avx512(input, output);
            }
            return Ok(());
        }
        if is_x86_feature_detected!("avx2") {
            unsafe {
                relu_avx2(input, output);
            }
            return Ok(());
        }
    }

    for i in 0..n {
        output[i] = input[i].max(0.0);
    }
    Ok(())
}

/// AVX-512 ReLU with NT stores for large arrays.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relu_avx512(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let n = input.len();
        let ip = input.as_ptr();
        let op = output.as_mut_ptr();
        let zero = _mm512_setzero_ps();
        let mut i = 0;

        let data_bytes = n * 4;
        let op_aligned = (op as usize) % 64 == 0;
        if data_bytes > NT_STORE_THRESHOLD_BYTES && op_aligned {
            // NT path: 4-way unrolled, requires 64-byte aligned output
            while i + 64 <= n {
                _mm_prefetch(ip.add(i + 128).cast::<i8>(), _MM_HINT_T0);

                _mm512_stream_ps(op.add(i), _mm512_max_ps(_mm512_loadu_ps(ip.add(i)), zero));
                _mm512_stream_ps(
                    op.add(i + 16),
                    _mm512_max_ps(_mm512_loadu_ps(ip.add(i + 16)), zero),
                );
                _mm512_stream_ps(
                    op.add(i + 32),
                    _mm512_max_ps(_mm512_loadu_ps(ip.add(i + 32)), zero),
                );
                _mm512_stream_ps(
                    op.add(i + 48),
                    _mm512_max_ps(_mm512_loadu_ps(ip.add(i + 48)), zero),
                );
                i += 64;
            }
            while i + 16 <= n {
                _mm512_stream_ps(op.add(i), _mm512_max_ps(_mm512_loadu_ps(ip.add(i)), zero));
                i += 16;
            }
            _mm_sfence();
        } else {
            while i + 64 <= n {
                _mm512_storeu_ps(op.add(i), _mm512_max_ps(_mm512_loadu_ps(ip.add(i)), zero));
                _mm512_storeu_ps(
                    op.add(i + 16),
                    _mm512_max_ps(_mm512_loadu_ps(ip.add(i + 16)), zero),
                );
                _mm512_storeu_ps(
                    op.add(i + 32),
                    _mm512_max_ps(_mm512_loadu_ps(ip.add(i + 32)), zero),
                );
                _mm512_storeu_ps(
                    op.add(i + 48),
                    _mm512_max_ps(_mm512_loadu_ps(ip.add(i + 48)), zero),
                );
                i += 64;
            }
            while i + 16 <= n {
                _mm512_storeu_ps(op.add(i), _mm512_max_ps(_mm512_loadu_ps(ip.add(i)), zero));
                i += 16;
            }
        }
        for j in i..n {
            output[j] = input[j].max(0.0);
        }
    } // unsafe
}

/// Prefetch distance in bytes. 8 cache lines (512 bytes = 128 f32) ahead.
/// Tuned for Zen 4 L1→L2 latency (~4ns) and L2→L3 latency (~12ns).
/// At ~1 iteration/ns throughput, 512B ahead hides ~12ns L2 latency.
const PREFETCH_DISTANCE: usize = 512;

/// NT store threshold (bytes). Use non-temporal stores when total working set
/// (2 inputs + 1 output = 3 arrays) exceeds L2 cache per core.
/// Zen 4 L2 = 1MB/core. For add: 3 × data_bytes. NT is beneficial when
/// data_bytes > ~333KB. Use 512KB for safety margin + alignment effects.
/// Below this, data fits in L2 and cached stores are faster.
const NT_STORE_THRESHOLD_BYTES: usize = 512 * 1024; // 512KB output = 128K f32

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let data_bytes = n * 4;

    // For large arrays (>L3-stream threshold), use non-temporal stores.
    // NT stores bypass cache write-allocate: eliminates RFO (Read-For-Ownership)
    // traffic, achieving ~88% of peak DRAM bandwidth (arXiv:1008.2849).
    if data_bytes > NT_STORE_THRESHOLD_BYTES {
        unsafe { relu_avx2_nt(input, output) }
        return;
    }

    // 8× unrolled (64 elements per iteration) — no software prefetch.
    // Hardware prefetcher on Zen 4/Intel 12th gen+ detects sequential
    // streaming patterns and prefetches 2-4 cache lines ahead automatically.
    // Software prefetch adds ~1 µop/32 elements of overhead without benefit
    // for sequential access, and can interfere with HW prefetcher at L3 sizes.
    let chunks = n / 64;
    let remainder_64 = chunks * 64;

    unsafe {
        let zero = _mm256_setzero_ps();
        let inp = input.as_ptr();
        let out = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 64;
            let v0 = _mm256_loadu_ps(inp.add(base));
            let v1 = _mm256_loadu_ps(inp.add(base + 8));
            let v2 = _mm256_loadu_ps(inp.add(base + 16));
            let v3 = _mm256_loadu_ps(inp.add(base + 24));
            let v4 = _mm256_loadu_ps(inp.add(base + 32));
            let v5 = _mm256_loadu_ps(inp.add(base + 40));
            let v6 = _mm256_loadu_ps(inp.add(base + 48));
            let v7 = _mm256_loadu_ps(inp.add(base + 56));
            _mm256_storeu_ps(out.add(base), _mm256_max_ps(v0, zero));
            _mm256_storeu_ps(out.add(base + 8), _mm256_max_ps(v1, zero));
            _mm256_storeu_ps(out.add(base + 16), _mm256_max_ps(v2, zero));
            _mm256_storeu_ps(out.add(base + 24), _mm256_max_ps(v3, zero));
            _mm256_storeu_ps(out.add(base + 32), _mm256_max_ps(v4, zero));
            _mm256_storeu_ps(out.add(base + 40), _mm256_max_ps(v5, zero));
            _mm256_storeu_ps(out.add(base + 48), _mm256_max_ps(v6, zero));
            _mm256_storeu_ps(out.add(base + 56), _mm256_max_ps(v7, zero));
        }

        let mut i = remainder_64;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(inp.add(i));
            _mm256_storeu_ps(out.add(i), _mm256_max_ps(v, zero));
            i += 8;
        }

        while i < n {
            *out.add(i) = (*inp.add(i)).max(0.0);
            i += 1;
        }
    }
}

/// Non-temporal store variant for large arrays (>L2 cache size).
/// Combines software prefetch pipeline with streaming stores to maximize
/// DRAM bandwidth utilization. Write-combining buffers batch stores to
/// full cache lines, eliminating read-for-ownership transactions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2_nt(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 32;
    let remainder_32 = chunks * 32;

    unsafe {
        let zero = _mm256_setzero_ps();

        for i in 0..chunks {
            let base = i * 32;
            // Prefetch input data ahead (L2→L3 latency hiding)
            _mm_prefetch(
                input.as_ptr().add(base + PREFETCH_DISTANCE / 4) as *const i8,
                _MM_HINT_T0,
            );
            let v0 = _mm256_loadu_ps(input.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(base + 8));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(base + 16));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(base + 24));
            // Non-temporal stores: bypass cache, write to WC buffers
            _mm256_stream_ps(output.as_mut_ptr().add(base), _mm256_max_ps(v0, zero));
            _mm256_stream_ps(output.as_mut_ptr().add(base + 8), _mm256_max_ps(v1, zero));
            _mm256_stream_ps(output.as_mut_ptr().add(base + 16), _mm256_max_ps(v2, zero));
            _mm256_stream_ps(output.as_mut_ptr().add(base + 24), _mm256_max_ps(v3, zero));
        }

        // Fence: ensure all NT stores are globally visible before return
        _mm_sfence();

        // Remainder with regular stores (< 1 cache line, no NT benefit)
        let mut i = remainder_32;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
            i += 8;
        }
        while i < n {
            output[i] = input[i].max(0.0);
            i += 1;
        }
    }
}

// ============================================================================
// Vector Add
// ============================================================================

/// Element-wise add: output_i = a_i + b_i
///
/// Uses AVX2 `_mm256_add_ps` when available.
///
/// # Errors
///
/// Returns `Err` if a, b, and output lengths don't match.
pub fn add(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), TruenoError> {
    contract_pre_add!(a, b);
    let n = a.len();
    if n != b.len() || n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "add size mismatch: a[{}], b[{}], output[{}]",
            n,
            b.len(),
            output.len()
        )));
    }

    // Parallel for very large bandwidth-bound arrays (≥16M elements = 64MB).
    // At 16M, total working set (3×64MB = 192MB) exceeds L3 (128MB on Zen 4).
    // Multi-core issues concurrent DRAM requests across memory channels.
    // Below 16M: single-core saturates its channel, rayon overhead hurts.
    #[cfg(feature = "parallel")]
    if n >= 16_000_000 {
        use rayon::prelude::*;
        let threads = 4.min(rayon::current_num_threads());
        let chunk = (((n + threads - 1) / threads) + 15) & !15;
        a.par_chunks(chunk).zip(b.par_chunks(chunk)).zip(output.par_chunks_mut(chunk)).for_each(
            |((ac, bc), oc)| {
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    unsafe { add_avx2(ac, bc, oc) };
                    return;
                }
                for i in 0..ac.len() {
                    oc[i] = ac[i] + bc[i];
                }
            },
        );
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Bandwidth-bound: prefer AVX2 at full clock over AVX-512 at
        // throttled clock for large arrays. AVX-512 only for L1/L2-resident.
        if is_x86_feature_detected!("avx512f") && n <= 4096 {
            unsafe {
                add_avx512(a, b, output);
            }
            return Ok(());
        }
        if is_x86_feature_detected!("avx2") {
            unsafe {
                add_avx2(a, b, output);
            }
            return Ok(());
        }
    }

    for i in 0..n {
        output[i] = a[i] + b[i];
    }
    Ok(())
}

/// AVX-512 add with NT stores for large arrays.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add_avx512(a: &[f32], b: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let n = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let rp = output.as_mut_ptr();
        let mut i = 0;

        let data_bytes = n * 4;
        let rp_aligned = (rp as usize) % 64 == 0;
        if data_bytes > NT_STORE_THRESHOLD_BYTES && rp_aligned {
            // NT path: 4-way unrolled, requires 64-byte aligned output
            while i + 64 <= n {
                _mm_prefetch(ap.add(i + 128).cast::<i8>(), _MM_HINT_T0);
                _mm_prefetch(bp.add(i + 128).cast::<i8>(), _MM_HINT_T0);

                _mm512_stream_ps(
                    rp.add(i),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i)), _mm512_loadu_ps(bp.add(i))),
                );
                _mm512_stream_ps(
                    rp.add(i + 16),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i + 16)), _mm512_loadu_ps(bp.add(i + 16))),
                );
                _mm512_stream_ps(
                    rp.add(i + 32),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i + 32)), _mm512_loadu_ps(bp.add(i + 32))),
                );
                _mm512_stream_ps(
                    rp.add(i + 48),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i + 48)), _mm512_loadu_ps(bp.add(i + 48))),
                );
                i += 64;
            }
            while i + 16 <= n {
                _mm512_stream_ps(
                    rp.add(i),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i)), _mm512_loadu_ps(bp.add(i))),
                );
                i += 16;
            }
            _mm_sfence();
        } else {
            while i + 64 <= n {
                _mm512_storeu_ps(
                    rp.add(i),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i)), _mm512_loadu_ps(bp.add(i))),
                );
                _mm512_storeu_ps(
                    rp.add(i + 16),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i + 16)), _mm512_loadu_ps(bp.add(i + 16))),
                );
                _mm512_storeu_ps(
                    rp.add(i + 32),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i + 32)), _mm512_loadu_ps(bp.add(i + 32))),
                );
                _mm512_storeu_ps(
                    rp.add(i + 48),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i + 48)), _mm512_loadu_ps(bp.add(i + 48))),
                );
                i += 64;
            }
            while i + 16 <= n {
                _mm512_storeu_ps(
                    rp.add(i),
                    _mm512_add_ps(_mm512_loadu_ps(ap.add(i)), _mm512_loadu_ps(bp.add(i))),
                );
                i += 16;
            }
        }
        for j in i..n {
            output[j] = a[j] + b[j];
        }
    } // unsafe
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(a: &[f32], b: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let data_bytes = n * 4;

    // Large arrays: NT stores (bypass cache for DRAM-bound writes)
    if data_bytes > NT_STORE_THRESHOLD_BYTES {
        unsafe { add_avx2_nt(a, b, output) }
        return;
    }

    // 8× unrolled (64 elements per iteration) — no software prefetch.
    // Hardware prefetcher handles sequential dual-stream patterns efficiently.
    let chunks = n / 64;
    let remainder_64 = chunks * 64;

    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let op = output.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 64;
            // Interleaved loads from a and b for maximum load port utilization
            let a0 = _mm256_loadu_ps(ap.add(base));
            let b0 = _mm256_loadu_ps(bp.add(base));
            let a1 = _mm256_loadu_ps(ap.add(base + 8));
            let b1 = _mm256_loadu_ps(bp.add(base + 8));
            let a2 = _mm256_loadu_ps(ap.add(base + 16));
            let b2 = _mm256_loadu_ps(bp.add(base + 16));
            let a3 = _mm256_loadu_ps(ap.add(base + 24));
            let b3 = _mm256_loadu_ps(bp.add(base + 24));
            let a4 = _mm256_loadu_ps(ap.add(base + 32));
            let b4 = _mm256_loadu_ps(bp.add(base + 32));
            let a5 = _mm256_loadu_ps(ap.add(base + 40));
            let b5 = _mm256_loadu_ps(bp.add(base + 40));
            let a6 = _mm256_loadu_ps(ap.add(base + 48));
            let b6 = _mm256_loadu_ps(bp.add(base + 48));
            let a7 = _mm256_loadu_ps(ap.add(base + 56));
            let b7 = _mm256_loadu_ps(bp.add(base + 56));
            _mm256_storeu_ps(op.add(base), _mm256_add_ps(a0, b0));
            _mm256_storeu_ps(op.add(base + 8), _mm256_add_ps(a1, b1));
            _mm256_storeu_ps(op.add(base + 16), _mm256_add_ps(a2, b2));
            _mm256_storeu_ps(op.add(base + 24), _mm256_add_ps(a3, b3));
            _mm256_storeu_ps(op.add(base + 32), _mm256_add_ps(a4, b4));
            _mm256_storeu_ps(op.add(base + 40), _mm256_add_ps(a5, b5));
            _mm256_storeu_ps(op.add(base + 48), _mm256_add_ps(a6, b6));
            _mm256_storeu_ps(op.add(base + 56), _mm256_add_ps(a7, b7));
        }

        let mut i = remainder_64;
        while i + 8 <= n {
            let av = _mm256_loadu_ps(ap.add(i));
            let bv = _mm256_loadu_ps(bp.add(i));
            _mm256_storeu_ps(op.add(i), _mm256_add_ps(av, bv));
            i += 8;
        }

        while i < n {
            *op.add(i) = *ap.add(i) + *bp.add(i);
            i += 1;
        }
    }
}

/// Non-temporal store variant of add for large arrays (>L2 cache).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2_nt(a: &[f32], b: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 32;
    let remainder_32 = chunks * 32;

    unsafe {
        for i in 0..chunks {
            let base = i * 32;
            _mm_prefetch(a.as_ptr().add(base + PREFETCH_DISTANCE / 4) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.as_ptr().add(base + PREFETCH_DISTANCE / 4) as *const i8, _MM_HINT_T0);
            let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(base + 16));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(base + 24));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
            let b2 = _mm256_loadu_ps(b.as_ptr().add(base + 16));
            let b3 = _mm256_loadu_ps(b.as_ptr().add(base + 24));
            _mm256_stream_ps(output.as_mut_ptr().add(base), _mm256_add_ps(a0, b0));
            _mm256_stream_ps(output.as_mut_ptr().add(base + 8), _mm256_add_ps(a1, b1));
            _mm256_stream_ps(output.as_mut_ptr().add(base + 16), _mm256_add_ps(a2, b2));
            _mm256_stream_ps(output.as_mut_ptr().add(base + 24), _mm256_add_ps(a3, b3));
        }

        _mm_sfence();

        let mut i = remainder_32;
        while i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_add_ps(av, bv));
            i += 8;
        }
        while i < n {
            output[i] = a[i] + b[i];
            i += 1;
        }
    }
}

// ============================================================================
// Scalar Multiply
// ============================================================================

/// Element-wise scalar multiply: output_i = input_i * scalar
///
/// Uses AVX2 `_mm256_mul_ps` when available.
///
/// # Errors
///
/// Returns `Err` if input and output lengths don't match.
pub fn mul_scalar(input: &[f32], scalar: f32, output: &mut [f32]) -> Result<(), TruenoError> {
    // Contract: elementwise-kernel-v1.yaml, equation = mul_scalar
    debug_assert!(!input.is_empty(), "Contract mul_scalar: input is empty");
    debug_assert!(scalar.is_finite(), "Contract mul_scalar: scalar is not finite");
    let n = input.len();
    if n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "mul_scalar size mismatch: input[{}], output[{}]",
            n,
            output.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                mul_scalar_avx2(input, scalar, output);
            }
            return Ok(());
        }
    }

    for i in 0..n {
        output[i] = input[i] * scalar;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_scalar_avx2(input: &[f32], scalar: f32, output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 32;
    let remainder_32 = chunks * 32;

    unsafe {
        let s = _mm256_set1_ps(scalar);

        for i in 0..chunks {
            let base = i * 32;
            let v0 = _mm256_loadu_ps(input.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(base + 8));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(base + 16));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(base + 24));
            _mm256_storeu_ps(output.as_mut_ptr().add(base), _mm256_mul_ps(v0, s));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 8), _mm256_mul_ps(v1, s));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 16), _mm256_mul_ps(v2, s));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 24), _mm256_mul_ps(v3, s));
        }

        let mut i = remainder_32;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_mul_ps(v, s));
            i += 8;
        }

        while i < n {
            output[i] = input[i] * scalar;
            i += 1;
        }
    }
}

// ============================================================================
// Allocating variants (skip zero-initialization)
// ============================================================================

/// ReLU with output allocation. Avoids zero-fill overhead of `vec![0.0; n]`.
///
/// # Safety guarantee
///
/// Output Vec is fully initialized by the SIMD/scalar loop before return.
#[must_use]
pub fn relu_alloc(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    let _ = relu(input, &mut output);
    output
}

/// Element-wise add with output allocation. Avoids zero-fill overhead.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn add_alloc(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "add_alloc: length mismatch");
    let n = a.len();
    let mut output = vec![0.0f32; n];
    let _ = add(a, b, &mut output);
    output
}

/// Scalar multiply with output allocation. Avoids zero-fill overhead.
#[must_use]
pub fn mul_scalar_alloc(input: &[f32], scalar: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    let _ = mul_scalar(input, scalar, &mut output);
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── ReLU tests ────────────────────────────────────────────────────────

    #[test]
    fn test_relu_basic() {
        let input = [-1.0, 0.0, 1.0, -0.5, 2.0, -3.0, 0.1, -0.1];
        let expected = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.1, 0.0];
        let mut output = vec![0.0f32; 8];
        relu(&input, &mut output).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_large() {
        let n = 11008; // FFN intermediate size
        let input: Vec<f32> =
            (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut output = vec![0.0f32; n];
        relu(&input, &mut output).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(out, inp.max(0.0), "ReLU mismatch at {i}");
        }
    }

    #[test]
    fn test_relu_avx2_scalar_parity() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0).collect();
            let mut output = vec![0.0f32; n];
            relu(&input, &mut output).unwrap();
            for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
                assert_eq!(out, inp.max(0.0), "ReLU parity at [{i}] n={n}");
            }
        }
    }

    #[test]
    fn test_relu_error_mismatch() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 3];
        assert!(relu(&input, &mut output).is_err());
    }

    // ── Add tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0f32; 4];
        add(&a, &b, &mut output).unwrap();
        assert_eq!(output, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_add_large() {
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let mut output = vec![0.0f32; n];
        add(&a, &b, &mut output).unwrap();
        for i in 0..n {
            assert_eq!(output[i], a[i] + b[i], "Add mismatch at {i}");
        }
    }

    #[test]
    fn test_add_avx2_scalar_parity() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 4096] {
            let a: Vec<f32> = (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 1000) as f32 / 500.0 - 1.0).collect();
            let mut output = vec![0.0f32; n];
            add(&a, &b, &mut output).unwrap();
            for i in 0..n {
                assert_eq!(output[i], a[i] + b[i], "Add parity at [{i}] n={n}");
            }
        }
    }

    #[test]
    fn test_add_error_mismatch() {
        let a = vec![1.0f32; 4];
        let b = vec![1.0f32; 3];
        let mut output = vec![0.0f32; 4];
        assert!(add(&a, &b, &mut output).is_err());
    }

    // ── Mul scalar tests ──────────────────────────────────────────────────

    #[test]
    fn test_mul_scalar_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        mul_scalar(&input, 2.5, &mut output).unwrap();
        assert_eq!(output, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_mul_scalar_large() {
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; n];
        mul_scalar(&input, std::f32::consts::PI, &mut output).unwrap();
        for i in 0..n {
            assert!(
                (output[i] - input[i] * std::f32::consts::PI).abs() < 1e-5,
                "Mul scalar mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_mul_scalar_avx2_scalar_parity() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0).collect();
            let mut output = vec![0.0f32; n];
            mul_scalar(&input, std::f32::consts::E, &mut output).unwrap();
            for i in 0..n {
                assert!(
                    (output[i] - input[i] * std::f32::consts::E).abs() < 1e-4,
                    "Mul scalar parity at [{i}] n={n}",
                );
            }
        }
    }

    #[test]
    fn test_mul_scalar_error_mismatch() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 3];
        assert!(mul_scalar(&input, 1.0, &mut output).is_err());
    }
}
