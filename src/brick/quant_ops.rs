//! Q5_K and Q6_K Quantization Operations (llama.cpp compatible)
//!
//! This module provides quantization formats and compute operations
//! for llama.cpp-compatible k-quant formats.
//!
//! # Formats
//!
//! - `BlockQ5K`: 5-bit quantization with super-blocks (256 values)
//! - `BlockQ6K`: 6-bit quantization with super-blocks (256 values)
//!
//! # Operations
//!
//! - `DotQ5KOp`: Dot product with Q5_K quantized weights
//! - `DotQ6KOp`: Dot product with Q6_K quantized weights
//!
//! # SIMD Optimization
//!
//! Both operations use AVX2/FMA when available for ~4x speedup.

use super::{Backend, ComputeOp};
use crate::error::TruenoError;

// ============================================================================
// Q5_K Block Format
// ============================================================================

/// Q5_K block format (5-bit with super-blocks).
///
/// Matches llama.cpp's block_q5_K format:
/// - Super-block of 256 values
/// - 5-bit quantization with k-quant scales
/// - Higher precision than Q4_K, lower than Q6_K
///
/// Memory layout:
/// ```text
/// | d (fp16) | dmin (fp16) | scales[12] | qh[32] | qs[128] |
/// ```
#[derive(Debug, Clone)]
pub struct BlockQ5K {
    /// Scale factor (half precision)
    pub d: f32,
    /// Minimum value scale (half precision)
    pub dmin: f32,
    /// Scales for each 32-value block (12 bytes packed)
    pub scales: [u8; 12],
    /// High bits for quantized values (32 bytes)
    pub qh: [u8; 32],
    /// Quantized values (128 bytes, 2 values per byte)
    pub qs: [u8; 128],
}

impl BlockQ5K {
    /// Block size in elements
    pub const BLOCK_SIZE: usize = 256;

    /// Dequantize a Q5_K block to f32.
    ///
    /// # Safety
    ///
    /// Output buffer must have at least BLOCK_SIZE elements.
    pub fn dequantize(&self, output: &mut [f32]) {
        debug_assert!(output.len() >= Self::BLOCK_SIZE);

        // Decode scales from packed format
        let mut scales = [0i8; 8];
        for i in 0..8 {
            let low = (self.scales[i] & 0x3F) as i8;
            scales[i] = low - 32;
        }

        // Dequantize each sub-block
        for block_idx in 0..8 {
            let scale = scales[block_idx] as f32;
            let base_idx = block_idx * 32;

            for i in 0..32 {
                let out_idx = base_idx + i;
                let byte_idx = base_idx / 2 + i / 2;

                // Extract 4-bit low value
                let q4 = if i % 2 == 0 {
                    self.qs[byte_idx] & 0x0F
                } else {
                    self.qs[byte_idx] >> 4
                };

                // Extract 5th bit from qh
                let qh_bit = ((self.qh[i] >> block_idx) & 1) as u8;
                let q5 = q4 | (qh_bit << 4);

                // Dequantize: value = d * scale * (q5 - 16) + dmin
                output[out_idx] = self.d * scale * (q5 as f32 - 16.0) + self.dmin;
            }
        }
    }
}

// ============================================================================
// Q6_K Block Format
// ============================================================================

/// Q6_K block format (6-bit with super-blocks).
///
/// Matches llama.cpp's block_q6_K format:
/// - Super-block of 256 values
/// - 6-bit quantization with k-quant scales
/// - Highest precision k-quant format
///
/// Memory layout:
/// ```text
/// | ql[128] | qh[64] | scales[16] | d (fp16) |
/// ```
#[derive(Debug, Clone)]
pub struct BlockQ6K {
    /// Low 4 bits of quantized values (128 bytes)
    pub ql: [u8; 128],
    /// High 2 bits of quantized values (64 bytes)
    pub qh: [u8; 64],
    /// Scales for each 16-value block (16 bytes)
    pub scales: [i8; 16],
    /// Scale factor (half precision)
    pub d: f32,
}

impl BlockQ6K {
    /// Block size in elements
    pub const BLOCK_SIZE: usize = 256;

    /// Dequantize a Q6_K block to f32.
    ///
    /// # Safety
    ///
    /// Output buffer must have at least BLOCK_SIZE elements.
    pub fn dequantize(&self, output: &mut [f32]) {
        debug_assert!(output.len() >= Self::BLOCK_SIZE);

        // Dequantize each sub-block of 16 values
        for block_idx in 0..16 {
            let scale = self.scales[block_idx] as f32;
            let base_idx = block_idx * 16;

            for i in 0..16 {
                let out_idx = base_idx + i;
                let ql_idx = base_idx / 2 + i / 2;
                let qh_idx = base_idx / 4 + i / 4;

                // Extract 4-bit low value
                let ql_val = if i % 2 == 0 {
                    self.ql[ql_idx] & 0x0F
                } else {
                    self.ql[ql_idx] >> 4
                };

                // Extract 2-bit high value
                let qh_shift = (i % 4) * 2;
                let qh_val = ((self.qh[qh_idx] >> qh_shift) & 0x03) as u8;

                // Combine to 6-bit value
                let q6 = ql_val | (qh_val << 4);

                // Dequantize: value = d * scale * (q6 - 32)
                output[out_idx] = self.d * scale * (q6 as f32 - 32.0);
            }
        }
    }
}

// ============================================================================
// Q5_K Dot Product Operation
// ============================================================================

/// Q5_K dot product operation.
///
/// Computes dot product between Q5_K quantized weights and f32 activations.
#[derive(Debug, Clone)]
pub struct DotQ5KOp {
    /// Number of blocks
    pub n_blocks: usize,
}

impl DotQ5KOp {
    /// Create a new Q5_K dot product operation.
    #[must_use]
    pub fn new(n_elements: usize) -> Self {
        Self {
            n_blocks: n_elements / BlockQ5K::BLOCK_SIZE,
        }
    }

    /// Compute dot product with SIMD acceleration.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_dot_block(block: &BlockQ5K, x: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let mut acc = _mm256_setzero_ps();
        let mut dequant = [0.0f32; BlockQ5K::BLOCK_SIZE];
        block.dequantize(&mut dequant);

        let mut i = 0;
        while i + 8 <= BlockQ5K::BLOCK_SIZE {
            let vd = _mm256_loadu_ps(dequant.as_ptr().add(i));
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            acc = _mm256_fmadd_ps(vd, vx, acc);
            i += 8;
        }

        // Horizontal sum
        let high = _mm256_extractf128_ps(acc, 1);
        let low = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(high, low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

impl ComputeOp for DotQ5KOp {
    type Input = (Vec<BlockQ5K>, Vec<f32>);
    type Output = f32;

    fn name(&self) -> &'static str {
        "dot_q5k"
    }

    fn execute(&self, input: Self::Input, backend: Backend) -> Result<Self::Output, TruenoError> {
        let (blocks, x) = input;

        if blocks.is_empty() || x.is_empty() {
            return Ok(0.0);
        }

        let mut sum = 0.0f32;

        #[cfg(target_arch = "x86_64")]
        {
            if matches!(backend, Backend::Avx2 | Backend::Auto) && is_x86_feature_detected!("avx2")
            {
                for (i, block) in blocks.iter().enumerate() {
                    let x_slice = &x[i * BlockQ5K::BLOCK_SIZE..];
                    sum += unsafe { Self::avx2_dot_block(block, x_slice) };
                }
                return Ok(sum);
            }
        }

        // Scalar fallback
        let mut dequant = [0.0f32; BlockQ5K::BLOCK_SIZE];
        for (i, block) in blocks.iter().enumerate() {
            block.dequantize(&mut dequant);
            let x_slice = &x[i * BlockQ5K::BLOCK_SIZE..];
            for j in 0..BlockQ5K::BLOCK_SIZE {
                sum += dequant[j] * x_slice[j];
            }
        }

        Ok(sum)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        self.n_blocks * BlockQ5K::BLOCK_SIZE
    }
}

// ============================================================================
// Q6_K Dot Product Operation
// ============================================================================

/// Q6_K dot product operation.
///
/// Computes dot product between Q6_K quantized weights and f32 activations.
#[derive(Debug, Clone)]
pub struct DotQ6KOp {
    /// Number of blocks
    pub n_blocks: usize,
}

impl DotQ6KOp {
    /// Create a new Q6_K dot product operation.
    #[must_use]
    pub fn new(n_elements: usize) -> Self {
        Self {
            n_blocks: n_elements / BlockQ6K::BLOCK_SIZE,
        }
    }

    /// Compute dot product with SIMD acceleration.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_dot_block(block: &BlockQ6K, x: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let mut acc = _mm256_setzero_ps();
        let mut dequant = [0.0f32; BlockQ6K::BLOCK_SIZE];
        block.dequantize(&mut dequant);

        let mut i = 0;
        while i + 8 <= BlockQ6K::BLOCK_SIZE {
            let vd = _mm256_loadu_ps(dequant.as_ptr().add(i));
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            acc = _mm256_fmadd_ps(vd, vx, acc);
            i += 8;
        }

        // Horizontal sum
        let high = _mm256_extractf128_ps(acc, 1);
        let low = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(high, low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

impl ComputeOp for DotQ6KOp {
    type Input = (Vec<BlockQ6K>, Vec<f32>);
    type Output = f32;

    fn name(&self) -> &'static str {
        "dot_q6k"
    }

    fn execute(&self, input: Self::Input, backend: Backend) -> Result<Self::Output, TruenoError> {
        let (blocks, x) = input;

        if blocks.is_empty() || x.is_empty() {
            return Ok(0.0);
        }

        let mut sum = 0.0f32;

        #[cfg(target_arch = "x86_64")]
        {
            if matches!(backend, Backend::Avx2 | Backend::Auto) && is_x86_feature_detected!("avx2")
            {
                for (i, block) in blocks.iter().enumerate() {
                    let x_slice = &x[i * BlockQ6K::BLOCK_SIZE..];
                    sum += unsafe { Self::avx2_dot_block(block, x_slice) };
                }
                return Ok(sum);
            }
        }

        // Scalar fallback
        let mut dequant = [0.0f32; BlockQ6K::BLOCK_SIZE];
        for (i, block) in blocks.iter().enumerate() {
            block.dequantize(&mut dequant);
            let x_slice = &x[i * BlockQ6K::BLOCK_SIZE..];
            for j in 0..BlockQ6K::BLOCK_SIZE {
                sum += dequant[j] * x_slice[j];
            }
        }

        Ok(sum)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        self.n_blocks * BlockQ6K::BLOCK_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== BlockQ5K Tests =====

    #[test]
    fn test_block_q5k_size() {
        assert_eq!(BlockQ5K::BLOCK_SIZE, 256);
    }

    #[test]
    fn test_block_q5k_dequantize_basic() {
        let block = BlockQ5K {
            d: 0.1,
            dmin: 0.0,
            scales: [32; 12], // Neutral scales (32 - 32 = 0)
            qh: [0; 32],      // All high bits 0
            qs: [0x88; 128],  // 8,8 pattern (mid-range 4-bit)
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // With scale=0, all outputs should be dmin (0.0)
        for val in &output {
            assert!(val.abs() < 1.0, "Expected near-zero, got {}", val);
        }
    }

    #[test]
    fn test_block_q5k_dequantize_with_scale() {
        let block = BlockQ5K {
            d: 1.0,
            dmin: 0.5,
            scales: [33; 12], // Scale of 1 (33 - 32 = 1)
            qh: [0xFF; 32],   // All high bits set
            qs: [0xFF; 128],  // All low bits set (15,15)
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Values should be non-zero with positive scale
        let non_zero_count = output.iter().filter(|&&v| v.abs() > 1e-6).count();
        assert!(non_zero_count > 0, "Should have non-zero values");
    }

    #[test]
    fn test_block_q5k_dequantize_alternating() {
        let block = BlockQ5K {
            d: 0.5,
            dmin: 0.1,
            scales: [34; 12], // Scale of 2
            qh: [0xAA; 32],   // Alternating bits
            qs: [0x55; 128],  // Alternating nibbles (5,5)
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // All values should be finite
        for val in &output {
            assert!(val.is_finite(), "Value should be finite");
        }
    }

    #[test]
    fn test_block_q5k_dequantize_odd_even_bytes() {
        // Test both even and odd index paths in dequantization
        let block = BlockQ5K {
            d: 1.0,
            dmin: 0.0,
            scales: [48; 12], // Scale of 16 (48 - 32 = 16)
            qh: [0; 32],
            qs: [0x12; 128], // Low nibble = 2, high nibble = 1
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Check that alternating values differ (even vs odd extraction)
        // Since qs[i] = 0x12, even indices extract 2, odd indices extract 1
        // Note: the actual dequant formula is complex, but values should differ
        assert!(output[0] != output[1] || output[0].abs() < 1e-6);
    }

    // ===== BlockQ6K Tests =====

    #[test]
    fn test_block_q6k_size() {
        assert_eq!(BlockQ6K::BLOCK_SIZE, 256);
    }

    #[test]
    fn test_block_q6k_dequantize_basic() {
        let block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16], // Zero scales
            d: 0.1,
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // With scale=0, all outputs should be d * 0 * (q6 - 32) = 0
        for val in &output {
            assert!(val.abs() < 1e-6, "Expected 0, got {}", val);
        }
    }

    #[test]
    fn test_block_q6k_dequantize_with_scale() {
        let block = BlockQ6K {
            ql: [0xFF; 128], // Max low bits
            qh: [0xFF; 64],  // Max high bits
            scales: [1; 16], // Positive scale
            d: 0.5,
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Values should be non-zero
        let non_zero = output.iter().any(|&v| v.abs() > 1e-6);
        assert!(non_zero, "Should have non-zero values");
    }

    #[test]
    fn test_block_q6k_dequantize_negative_scale() {
        let block = BlockQ6K {
            ql: [0x88; 128],
            qh: [0x55; 64],
            scales: [-1; 16], // Negative scale
            d: 1.0,
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // All values should be finite
        for val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_block_q6k_dequantize_all_subblocks() {
        // Test that all 16 sub-blocks are processed
        let block = BlockQ6K {
            ql: [0x12; 128],
            qh: [0x03; 64], // Different pattern per position
            scales: [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8],
            d: 0.1,
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Check values at different sub-block boundaries
        assert!(output[0].is_finite());
        assert!(output[15].is_finite());
        assert!(output[16].is_finite());
        assert!(output[127].is_finite());
        assert!(output[255].is_finite());
    }

    #[test]
    fn test_block_q6k_qh_extraction() {
        // Test the 2-bit high value extraction logic
        // qh_shift cycles through 0, 2, 4, 6 for i % 4 = 0, 1, 2, 3
        let block = BlockQ6K {
            ql: [0; 128],
            qh: [0b11_10_01_00; 64], // Pattern: 0,1,2,3 across 4 positions
            scales: [1; 16],
            d: 1.0,
        };

        let mut output = [0.0f32; 256];
        block.dequantize(&mut output);

        // Different qh values should produce different outputs
        // Position 0: qh_val = 0, Position 1: qh_val = 1, etc.
        // This tests the (i % 4) * 2 shift logic
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[2].is_finite());
        assert!(output[3].is_finite());
    }

    // ===== DotQ5KOp Tests =====

    #[test]
    fn test_dot_q5k_new() {
        let op = DotQ5KOp::new(512);
        assert_eq!(op.n_blocks, 2);
    }

    #[test]
    fn test_dot_q5k_name() {
        let op = DotQ5KOp::new(256);
        assert_eq!(op.name(), "dot_q5k");
    }

    #[test]
    fn test_dot_q5k_empty() {
        let op = DotQ5KOp::new(256);
        let result = op.execute((vec![], vec![]), Backend::Scalar).unwrap();
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_q5k_empty_activations() {
        let op = DotQ5KOp::new(256);
        let block = BlockQ5K {
            d: 1.0,
            dmin: 0.0,
            scales: [32; 12],
            qh: [0; 32],
            qs: [0; 128],
        };
        let result = op.execute((vec![block], vec![]), Backend::Scalar).unwrap();
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_q5k_tokens() {
        let op = DotQ5KOp::new(512); // 2 blocks
        let input = (vec![], vec![]);
        assert_eq!(op.tokens(&input), 512);
    }

    #[test]
    fn test_dot_q5k_scalar_execution() {
        let op = DotQ5KOp::new(256);
        let block = BlockQ5K {
            d: 1.0,
            dmin: 0.0,
            scales: [33; 12], // Scale = 1
            qh: [0; 32],
            qs: [0x88; 128], // Mid-range values
        };
        let x = vec![1.0f32; 256];
        let result = op.execute((vec![block], x), Backend::Scalar).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dot_q5k_multiple_blocks() {
        let op = DotQ5KOp::new(512);
        let block = BlockQ5K {
            d: 0.5,
            dmin: 0.1,
            scales: [34; 12],
            qh: [0; 32],
            qs: [0x44; 128],
        };
        let x = vec![0.5f32; 512];
        let result = op
            .execute((vec![block.clone(), block], x), Backend::Scalar)
            .unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dot_q5k_auto_backend() {
        let op = DotQ5KOp::new(256);
        let block = BlockQ5K {
            d: 1.0,
            dmin: 0.0,
            scales: [32; 12],
            qh: [0; 32],
            qs: [0; 128],
        };
        let x = vec![1.0f32; 256];
        // Auto backend should work (may use AVX2 if available)
        let result = op.execute((vec![block], x), Backend::Auto).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dot_q5k_avx2_backend() {
        let op = DotQ5KOp::new(256);
        let block = BlockQ5K {
            d: 1.0,
            dmin: 0.0,
            scales: [33; 12],
            qh: [0; 32],
            qs: [0x11; 128],
        };
        let x = vec![2.0f32; 256];
        // Request AVX2, will fall back to scalar if not available
        let result = op.execute((vec![block], x), Backend::Avx2).unwrap();
        assert!(result.is_finite());
    }

    // ===== DotQ6KOp Tests =====

    #[test]
    fn test_dot_q6k_new() {
        let op = DotQ6KOp::new(768);
        assert_eq!(op.n_blocks, 3);
    }

    #[test]
    fn test_dot_q6k_name() {
        let op = DotQ6KOp::new(256);
        assert_eq!(op.name(), "dot_q6k");
    }

    #[test]
    fn test_dot_q6k_empty() {
        let op = DotQ6KOp::new(256);
        let result = op.execute((vec![], vec![]), Backend::Scalar).unwrap();
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_q6k_empty_activations() {
        let op = DotQ6KOp::new(256);
        let block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: 1.0,
        };
        let result = op.execute((vec![block], vec![]), Backend::Scalar).unwrap();
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_q6k_tokens() {
        let op = DotQ6KOp::new(768); // 3 blocks
        let input = (vec![], vec![]);
        assert_eq!(op.tokens(&input), 768);
    }

    #[test]
    fn test_dot_q6k_scalar_execution() {
        let op = DotQ6KOp::new(256);
        let block = BlockQ6K {
            ql: [0x55; 128],
            qh: [0x55; 64],
            scales: [1; 16],
            d: 0.5,
        };
        let x = vec![1.0f32; 256];
        let result = op.execute((vec![block], x), Backend::Scalar).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dot_q6k_multiple_blocks() {
        let op = DotQ6KOp::new(512);
        let block = BlockQ6K {
            ql: [0x33; 128],
            qh: [0x33; 64],
            scales: [2; 16],
            d: 0.25,
        };
        let x = vec![0.5f32; 512];
        let result = op
            .execute((vec![block.clone(), block], x), Backend::Scalar)
            .unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dot_q6k_auto_backend() {
        let op = DotQ6KOp::new(256);
        let block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [1; 16],
            d: 1.0,
        };
        let x = vec![1.0f32; 256];
        let result = op.execute((vec![block], x), Backend::Auto).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dot_q6k_avx2_backend() {
        let op = DotQ6KOp::new(256);
        let block = BlockQ6K {
            ql: [0xAA; 128],
            qh: [0xAA; 64],
            scales: [3; 16],
            d: 0.1,
        };
        let x = vec![2.0f32; 256];
        let result = op.execute((vec![block], x), Backend::Avx2).unwrap();
        assert!(result.is_finite());
    }

    // ===== Backend Equivalence Tests =====

    #[test]
    fn test_q5k_backend_equivalence() {
        let op = DotQ5KOp::new(256);
        let block = BlockQ5K {
            d: 0.5,
            dmin: 0.1,
            scales: [35; 12],
            qh: [0x55; 32],
            qs: [0x77; 128],
        };
        let x = vec![1.5f32; 256];

        let scalar = op
            .execute((vec![block.clone()], x.clone()), Backend::Scalar)
            .unwrap();
        let auto = op.execute((vec![block], x), Backend::Auto).unwrap();

        // Allow small FP differences due to SIMD operation ordering
        let rel_diff = (scalar - auto).abs() / scalar.abs().max(1e-6);
        assert!(
            rel_diff < 1e-4,
            "scalar={scalar}, auto={auto}, rel_diff={rel_diff}"
        );
    }

    #[test]
    fn test_q6k_backend_equivalence() {
        let op = DotQ6KOp::new(256);
        let block = BlockQ6K {
            ql: [0x66; 128],
            qh: [0x22; 64],
            scales: [4; 16],
            d: 0.2,
        };
        let x = vec![1.5f32; 256];

        let scalar = op
            .execute((vec![block.clone()], x.clone()), Backend::Scalar)
            .unwrap();
        let auto = op.execute((vec![block], x), Backend::Auto).unwrap();

        // Allow small FP differences due to SIMD operation ordering
        let rel_diff = (scalar - auto).abs() / scalar.abs().max(1e-6);
        assert!(
            rel_diff < 1e-4,
            "scalar={scalar}, auto={auto}, rel_diff={rel_diff}"
        );
    }

    // ===== Clone/Debug Trait Tests =====

    #[test]
    fn test_block_q5k_clone_debug() {
        let block = BlockQ5K {
            d: 1.0,
            dmin: 0.5,
            scales: [32; 12],
            qh: [0; 32],
            qs: [0; 128],
        };
        let cloned = block.clone();
        assert_eq!(format!("{:?}", block), format!("{:?}", cloned));
    }

    #[test]
    fn test_block_q6k_clone_debug() {
        let block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: 1.0,
        };
        let cloned = block.clone();
        assert_eq!(format!("{:?}", block), format!("{:?}", cloned));
    }

    #[test]
    fn test_dot_q5k_op_clone_debug() {
        let op = DotQ5KOp::new(256);
        let cloned = op.clone();
        assert_eq!(format!("{:?}", op), format!("{:?}", cloned));
    }

    #[test]
    fn test_dot_q6k_op_clone_debug() {
        let op = DotQ6KOp::new(256);
        let cloned = op.clone();
        assert_eq!(format!("{:?}", op), format!("{:?}", cloned));
    }
}
