//! BLIS matrix packing routines.
//!
//! Packing transforms row-major matrices into micro-panel layouts optimized
//! for sequential access in the microkernel, ensuring optimal cache line
//! utilization and aligned loads for SIMD.
//!
//! # References
//!
//! - Van Zee, F. G., & Van de Geijn, R. A. (2015). BLIS: A Framework for Rapidly Instantiating
//!   BLAS Functionality. ACM TOMS, 41(3), Fig. 4.

use super::{MR, NR};

/// Pack A into MC x KC panel with MR-aligned micro-panels
///
/// Memory layout (Van Zee & Van de Geijn, 2015, Fig. 4):
/// Original A (row-major):     Packed A (column-major micro-panels):
/// [a00 a01 a02 ...]           [a00 a10 a20 ... a(MR-1)0 | a01 a11 ...]
/// [a10 a11 a12 ...]            \____ MR elements ____/
///
/// This layout ensures:
/// 1. Sequential access in the microkernel
/// 2. Optimal cache line utilization
/// 3. Aligned loads for SIMD
pub fn pack_a(
    a: &[f32],
    lda: usize, // Leading dimension of A (number of columns in original)
    mc: usize,  // Number of rows to pack
    kc: usize,  // Number of columns to pack
    packed: &mut [f32],
) {
    let mut pack_idx = 0;

    // Process MR rows at a time
    let full_panels = mc / MR;
    let remainder = mc % MR;

    for panel in 0..full_panels {
        let row_start = panel * MR;

        for col in 0..kc {
            for row in 0..MR {
                packed[pack_idx] = a[(row_start + row) * lda + col];
                pack_idx += 1;
            }
        }
    }

    // Handle remainder rows (pad with zeros)
    if remainder > 0 {
        let row_start = full_panels * MR;

        for col in 0..kc {
            for row in 0..MR {
                if row < remainder {
                    packed[pack_idx] = a[(row_start + row) * lda + col];
                } else {
                    packed[pack_idx] = 0.0; // Zero padding
                }
                pack_idx += 1;
            }
        }
    }
}

/// Pack B into KC x NC panel with NR-aligned micro-panels
///
/// Memory layout:
/// Original B (row-major):     Packed B (row-major micro-panels):
/// [b00 b01 b02 ...]           [b00 b01 ... b(NR-1) | b10 b11 ...]
/// [b10 b11 b12 ...]            \____ NR elements ____/
pub fn pack_b(
    b: &[f32],
    ldb: usize, // Leading dimension of B (number of columns in original)
    kc: usize,  // Number of rows to pack
    nc: usize,  // Number of columns to pack
    packed: &mut [f32],
) {
    let mut pack_idx = 0;

    let full_panels = nc / NR;
    let remainder = nc % NR;

    for panel in 0..full_panels {
        let col_start = panel * NR;

        for row in 0..kc {
            for col in 0..NR {
                packed[pack_idx] = b[row * ldb + col_start + col];
                pack_idx += 1;
            }
        }
    }

    // Handle remainder columns (pad with zeros)
    if remainder > 0 {
        let col_start = full_panels * NR;

        for row in 0..kc {
            for col in 0..NR {
                if col < remainder {
                    packed[pack_idx] = b[row * ldb + col_start + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

/// Compute required packed A buffer size
#[inline]
pub fn packed_a_size(mc: usize, kc: usize) -> usize {
    let panels = (mc + MR - 1) / MR;
    panels * MR * kc
}

/// Compute required packed B buffer size
#[inline]
pub fn packed_b_size(kc: usize, nc: usize) -> usize {
    let panels = (nc + NR - 1) / NR;
    panels * NR * kc
}

/// Pack A block from row-major source
pub(super) fn pack_a_block(
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut pack_idx = 0;
    let panels = (rows + MR - 1) / MR;

    for panel in 0..panels {
        let ir = panel * MR;
        let mr_actual = MR.min(rows - ir);

        for col in 0..cols {
            for row in 0..MR {
                if row < mr_actual {
                    packed[pack_idx] = a[(row_start + ir + row) * lda + col_start + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

/// Pack B block from row-major source
pub(super) fn pack_b_block(
    b: &[f32],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut pack_idx = 0;
    let panels = (cols + NR - 1) / NR;

    for panel in 0..panels {
        let jr = panel * NR;
        let nr_actual = NR.min(cols - jr);

        for row in 0..rows {
            for col in 0..NR {
                if col < nr_actual {
                    packed[pack_idx] = b[(row_start + row) * ldb + col_start + jr + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}
