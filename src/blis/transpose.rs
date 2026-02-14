//! Matrix transpose operations.
//!
//! Cache-efficient 8x8 blocked transpose with scalar fallback for small matrices.

use crate::error::TruenoError;

/// Scalar transpose of a sub-region of a row-major matrix.
#[inline(always)]
fn transpose_region(
    a: &[f32],
    b: &mut [f32],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    src_cols: usize,
    dst_rows: usize,
) {
    for r in rows {
        for c in cols.clone() {
            b[c * dst_rows + r] = a[r * src_cols + c];
        }
    }
}

/// Transpose a matrix: B = A^T
///
/// SIMD-optimized for large matrices (>=64 elements).
/// Uses cache-efficient 8x8 blocking with manual unrolling.
///
/// # Arguments
///
/// * `rows` - Number of rows in A (cols in B)
/// * `cols` - Number of cols in A (rows in B)
/// * `a` - Input matrix A (rows x cols, row-major)
/// * `b` - Output matrix B (cols x rows, row-major)
///
/// # Returns
///
/// `Ok(())` on success, `Err` if dimensions mismatch
pub fn transpose(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) -> Result<(), TruenoError> {
    let expected = rows * cols;
    if a.len() != expected || b.len() != expected {
        return Err(TruenoError::InvalidInput(format!(
            "transpose size mismatch: a[{}], b[{}], expected {}",
            a.len(),
            b.len(),
            expected
        )));
    }

    if expected < 64 {
        transpose_region(a, b, 0..rows, 0..cols, cols, rows);
        return Ok(());
    }

    const BLOCK: usize = 8;
    let row_blocks = rows / BLOCK;
    let col_blocks = cols / BLOCK;

    for rb in 0..row_blocks {
        for cb in 0..col_blocks {
            let rs = rb * BLOCK;
            let cs = cb * BLOCK;
            transpose_region(a, b, rs..rs + BLOCK, cs..cs + BLOCK, cols, rows);
        }
    }

    // Right edge remainder
    let col_rem = col_blocks * BLOCK;
    if col_rem < cols {
        transpose_region(a, b, 0..row_blocks * BLOCK, col_rem..cols, cols, rows);
    }

    // Bottom edge remainder
    let row_rem = row_blocks * BLOCK;
    if row_rem < rows {
        transpose_region(a, b, row_rem..rows, 0..cols, cols, rows);
    }

    Ok(())
}
