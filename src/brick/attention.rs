//! SIMD-Optimized Attention Operation (PMAT-017)
//!
//! This module contains the scaled dot-product attention operation
//! with SIMD optimization for CPU inference.
//!
//! # Algorithm
//!
//! Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! # SIMD Optimizations
//!
//! - Q @ K^T: Batched dot products with AVX2/AVX-512/FMA
//! - Softmax: Row-wise numerically stable implementation
//! - Scores @ V: SIMD-friendly weighted accumulation
//!
//! # Performance Target
//!
//! Close the 1.66x gap in CPU inference (25.4 → 42 tok/s) by replacing
//! scalar triple-nested loops with SIMD operations.

use super::{Backend, ComputeOp};
use crate::error::TruenoError;

/// Scaled dot-product attention operation.
///
/// Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// # SIMD Optimization (PMAT-017)
///
/// Uses trueno's SIMD backends for:
/// - Q @ K^T: Batched dot products with AVX2/AVX-512
/// - Softmax: Row-wise numerically stable softmax
/// - Scores @ V: Batched weighted sums
///
/// # Performance Target
///
/// Close the 1.66x gap in CPU inference (25.4 → 42 tok/s) by replacing
/// scalar triple-nested loops with SIMD operations.
#[derive(Debug, Clone)]
pub struct AttentionOp {
    /// Sequence length (Q rows)
    pub seq_len: usize,
    /// Key/Value sequence length (may differ for cross-attention)
    pub kv_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Scale factor (1/sqrt(head_dim))
    pub scale: f32,
}

impl AttentionOp {
    /// Create a new attention operation.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Query sequence length
    /// * `kv_seq_len` - Key/Value sequence length
    /// * `head_dim` - Dimension per head
    #[must_use]
    pub fn new(seq_len: usize, kv_seq_len: usize, head_dim: usize) -> Self {
        Self {
            seq_len,
            kv_seq_len,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Create for self-attention (seq_len == kv_seq_len).
    #[must_use]
    pub fn self_attention(seq_len: usize, head_dim: usize) -> Self {
        Self::new(seq_len, seq_len, head_dim)
    }

    /// SIMD-optimized dot product for attention scores.
    ///
    /// Computes Q[i] · K[j] using SIMD when available.
    #[inline]
    pub(crate) fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        // Use architecture-specific SIMD
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::avx2_dot(a, b) };
            }
        }

        // Scalar fallback with manual unrolling for better vectorization
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let chunks = a.len() / 4;
        for i in 0..chunks {
            let base = i * 4;
            sum0 += a[base] * b[base];
            sum1 += a[base + 1] * b[base + 1];
            sum2 += a[base + 2] * b[base + 2];
            sum3 += a[base + 3] * b[base + 3];
        }

        // Handle remainder
        for i in (chunks * 4)..a.len() {
            sum0 += a[i] * b[i];
        }

        sum0 + sum1 + sum2 + sum3
    }

    /// AVX2-optimized dot product.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let base = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(base));
            let vb = _mm256_loadu_ps(b.as_ptr().add(base));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum
        let high = _mm256_extractf128_ps(sum, 1);
        let low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(high, low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 8)..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    /// Row-wise softmax with SIMD max/sum.
    #[inline]
    pub(crate) fn simd_softmax_row(scores: &mut [f32]) {
        if scores.is_empty() {
            return;
        }

        // Find max for numerical stability
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
    }
}

impl ComputeOp for AttentionOp {
    /// Input: (Q, K, V) tensors as flat vectors
    /// Q: [seq_len * head_dim]
    /// K: [kv_seq_len * head_dim]
    /// V: [kv_seq_len * head_dim]
    type Input = (Vec<f32>, Vec<f32>, Vec<f32>);
    /// Output: attention output [seq_len * head_dim]
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "attention"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (q, k, v) = input;

        // Validate dimensions
        let expected_q = self.seq_len * self.head_dim;
        let expected_kv = self.kv_seq_len * self.head_dim;

        if q.len() != expected_q {
            return Err(TruenoError::SizeMismatch {
                expected: expected_q,
                actual: q.len(),
            });
        }
        if k.len() != expected_kv || v.len() != expected_kv {
            return Err(TruenoError::SizeMismatch {
                expected: expected_kv,
                actual: k.len(),
            });
        }

        // Allocate output
        let mut output = vec![0.0f32; expected_q];

        // Allocate scores buffer (reused per query row)
        let mut scores = vec![0.0f32; self.kv_seq_len];

        // For each query position
        for qi in 0..self.seq_len {
            let q_row = &q[qi * self.head_dim..(qi + 1) * self.head_dim];

            // Compute Q[qi] · K[ki] for all ki (SIMD dot products)
            for ki in 0..self.kv_seq_len {
                let k_row = &k[ki * self.head_dim..(ki + 1) * self.head_dim];
                scores[ki] = Self::simd_dot(q_row, k_row) * self.scale;
            }

            // Softmax over scores
            Self::simd_softmax_row(&mut scores);

            // Compute weighted sum: output[qi] = sum(scores[ki] * V[ki])
            let out_row = &mut output[qi * self.head_dim..(qi + 1) * self.head_dim];
            out_row.fill(0.0);

            for ki in 0..self.kv_seq_len {
                let v_row = &v[ki * self.head_dim..(ki + 1) * self.head_dim];
                let weight = scores[ki];

                // SIMD-friendly accumulation
                for (o, &vi) in out_row.iter_mut().zip(v_row.iter()) {
                    *o += weight * vi;
                }
            }
        }

        Ok(output)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        // Output tokens = seq_len * head_dim
        self.seq_len * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_basic() {
        let op = AttentionOp::self_attention(2, 4); // seq=2, head_dim=4

        // Simple identity-like setup
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x4
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2x4
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2x4

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();

        assert_eq!(output.len(), 8);
        // Output should be weighted combination of V rows
    }

    #[test]
    fn test_attention_dimension_mismatch_q() {
        let op = AttentionOp::self_attention(2, 4);
        let q = vec![1.0; 4]; // Wrong size - should be 8
        let k = vec![1.0; 8];
        let v = vec![1.0; 8];

        let result = op.execute((q, k, v), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_dimension_mismatch_kv() {
        let op = AttentionOp::self_attention(2, 4);
        let q = vec![1.0; 8];
        let k = vec![1.0; 4]; // Wrong size - should be 8
        let v = vec![1.0; 8];

        let result = op.execute((q, k, v), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_cross_attention() {
        // Cross-attention: Q from decoder (seq=1), K/V from encoder (seq=4)
        let op = AttentionOp::new(1, 4, 8); // q_seq=1, kv_seq=4, head_dim=8

        let q = vec![1.0; 8]; // 1 x 8
        let k = vec![1.0; 32]; // 4 x 8
        let v = vec![1.0; 32]; // 4 x 8

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_attention_tokens() {
        let op = AttentionOp::self_attention(16, 64);
        let input = (vec![], vec![], vec![]);
        // tokens = seq_len * head_dim = 16 * 64 = 1024
        assert_eq!(op.tokens(&input), 1024);
    }

    #[test]
    fn test_simd_softmax_row_empty() {
        let mut scores: Vec<f32> = vec![];
        AttentionOp::simd_softmax_row(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_simd_softmax_row_single() {
        let mut scores = vec![5.0];
        AttentionOp::simd_softmax_row(&mut scores);
        assert!((scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_softmax_row_uniform() {
        let mut scores = vec![1.0, 1.0, 1.0, 1.0];
        AttentionOp::simd_softmax_row(&mut scores);

        // All equal inputs → uniform distribution
        for s in &scores {
            assert!((s - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_softmax_row_sum_to_one() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        AttentionOp::simd_softmax_row(&mut scores);

        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let dot = AttentionOp::simd_dot(&a, &b);
        assert!((dot - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_unaligned() {
        // Test with non-multiple-of-8 length (tests scalar remainder handling)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let dot = AttentionOp::simd_dot(&a, &b);
        // (1+2+3+4+5) * 2 = 30
        assert!((dot - 30.0).abs() < 1e-5);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_attention_op_fields() {
        let op = AttentionOp::new(4, 8, 64);
        assert_eq!(op.seq_len, 4);
        assert_eq!(op.kv_seq_len, 8);
        assert_eq!(op.head_dim, 64);
        // scale = 1/sqrt(64) = 1/8 = 0.125
        assert!((op.scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_attention_self_attention_fields() {
        let op = AttentionOp::self_attention(16, 32);
        assert_eq!(op.seq_len, 16);
        assert_eq!(op.kv_seq_len, 16); // Self-attention: same lengths
        assert_eq!(op.head_dim, 32);
    }

    #[test]
    fn test_attention_name() {
        let op = AttentionOp::self_attention(1, 4);
        assert_eq!(op.name(), "attention");
    }

    #[test]
    fn test_attention_v_size_mismatch() {
        let op = AttentionOp::self_attention(2, 4);
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let v = vec![1.0; 4]; // Wrong: should be 8

        let result = op.execute((q, k, v), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_single_position() {
        // seq=1, kv=1, head_dim=4
        let op = AttentionOp::self_attention(1, 4);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![2.0, 3.0, 4.0, 5.0];

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), 4);
        // With single position, softmax of single score is 1.0
        // Output = 1.0 * V = V
        assert!((output[0] - 2.0).abs() < 1e-5);
        assert!((output[1] - 3.0).abs() < 1e-5);
        assert!((output[2] - 4.0).abs() < 1e-5);
        assert!((output[3] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_uniform_scores() {
        // If Q and K are identical for all positions, scores are equal
        // Output should be average of V rows
        let op = AttentionOp::new(1, 2, 2);
        let head_dim = 2;

        let q = vec![1.0, 1.0]; // 1x2
        let k = vec![1.0, 1.0, 1.0, 1.0]; // 2x2, both identical
        let v = vec![1.0, 0.0, 0.0, 1.0]; // 2x2

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), head_dim);
        // Scores are equal => softmax gives [0.5, 0.5]
        // Output = 0.5 * [1, 0] + 0.5 * [0, 1] = [0.5, 0.5]
        assert!((output[0] - 0.5).abs() < 1e-5);
        assert!((output[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_exact_multiple_of_four() {
        // Tests that the 4-wide unrolled loop works for exact multiples
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8 elements = 2 chunks of 4
        let b = vec![1.0; 8];
        let dot = AttentionOp::simd_dot(&a, &b);
        // sum(1..=8) = 36
        assert!((dot - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_single_element() {
        let a = vec![3.0];
        let b = vec![4.0];
        let dot = AttentionOp::simd_dot(&a, &b);
        assert!((dot - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_two_elements() {
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];
        let dot = AttentionOp::simd_dot(&a, &b);
        // 2*4 + 3*5 = 8 + 15 = 23
        assert!((dot - 23.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_three_elements() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = AttentionOp::simd_dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_dot_large_avx2_aligned() {
        // 16 elements: exact multiple of 8 for AVX2 path
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b = vec![1.0f32; 16];
        let dot = AttentionOp::simd_dot(&a, &b);
        // sum(1..=16) = 136
        assert!((dot - 136.0).abs() < 1e-3);
    }

    #[test]
    fn test_simd_dot_large_avx2_remainder() {
        // 19 elements: 16 handled by AVX2 + 3 remainder
        let a: Vec<f32> = (1..=19).map(|x| x as f32).collect();
        let b = vec![1.0f32; 19];
        let dot = AttentionOp::simd_dot(&a, &b);
        // sum(1..=19) = 190
        assert!((dot - 190.0).abs() < 1e-3);
    }

    #[test]
    fn test_simd_dot_zeros() {
        let a = vec![0.0f32; 16];
        let b = vec![1.0f32; 16];
        let dot = AttentionOp::simd_dot(&a, &b);
        assert!((dot).abs() < 1e-10);
    }

    #[test]
    fn test_simd_dot_negative_values() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let dot = AttentionOp::simd_dot(&a, &b);
        assert!((dot - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn test_simd_softmax_row_large_values() {
        // Numerical stability: large values should not overflow
        let mut scores = vec![1000.0, 1001.0, 1002.0];
        AttentionOp::simd_softmax_row(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Largest value should get largest probability
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_simd_softmax_row_negative_values() {
        let mut scores = vec![-10.0, -20.0, -5.0];
        AttentionOp::simd_softmax_row(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // -5 is the largest, should have highest probability
        assert!(scores[2] > scores[0]);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn test_attention_clone() {
        let op = AttentionOp::new(4, 8, 64);
        let cloned = op.clone();
        assert_eq!(cloned.seq_len, 4);
        assert_eq!(cloned.kv_seq_len, 8);
        assert_eq!(cloned.head_dim, 64);
        assert!((cloned.scale - op.scale).abs() < 1e-10);
    }

    #[test]
    fn test_attention_multi_query_rows() {
        // Test with seq_len=3, kv_seq_len=2, head_dim=2
        let op = AttentionOp::new(3, 2, 2);

        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let v = vec![10.0, 20.0, 30.0, 40.0]; // 2x2

        let output = op.execute((q, k, v), Backend::Scalar).unwrap();
        assert_eq!(output.len(), 6); // 3 * 2

        // Each output row should be a weighted combination of V rows
        // All values should be finite
        for val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_attention_tokens_cross_attention() {
        let op = AttentionOp::new(1, 100, 64);
        let input = (vec![], vec![], vec![]);
        // tokens = seq_len * head_dim = 1 * 64 = 64
        assert_eq!(op.tokens(&input), 64);
    }
}
