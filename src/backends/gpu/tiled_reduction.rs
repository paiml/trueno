//! CPU fallback implementation of tiled reduction algorithms
//!
//! This module provides CPU implementations that mirror the GPU tiled reduction
//! algorithms. These are useful for:
//! - Testing and validation (compare GPU results against CPU reference)
//! - Fallback when GPU is unavailable
//! - Understanding the algorithm without GPU complexity
//!
//! The algorithms use the same 16×16 tile structure as the GPU shaders.

use super::partition_view::PartitionView;
use super::tensor_view::TensorView;

/// Default tile size for 2D reductions (matches GPU workgroup size)
pub const TILE_SIZE: usize = 16;

/// Reduction operation trait for generic tile reduction
pub trait ReduceOp {
    /// Identity element for the reduction (0 for sum, -inf for max, inf for min)
    fn identity() -> f32;
    /// Combine two values
    fn combine(a: f32, b: f32) -> f32;
}

/// Sum reduction operation
pub struct SumOp;

impl ReduceOp for SumOp {
    #[inline]
    fn identity() -> f32 {
        0.0
    }

    #[inline]
    fn combine(a: f32, b: f32) -> f32 {
        a + b
    }
}

/// Max reduction operation
pub struct MaxOp;

impl ReduceOp for MaxOp {
    #[inline]
    fn identity() -> f32 {
        f32::NEG_INFINITY
    }

    #[inline]
    fn combine(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

/// Min reduction operation
pub struct MinOp;

impl ReduceOp for MinOp {
    #[inline]
    fn identity() -> f32 {
        f32::INFINITY
    }

    #[inline]
    fn combine(a: f32, b: f32) -> f32 {
        a.min(b)
    }
}

/// Perform tiled reduction on 2D data (CPU fallback)
///
/// This simulates the GPU algorithm:
/// 1. Partition input into 16×16 tiles
/// 2. Reduce each tile to a single value
/// 3. Combine partial results
///
/// # Arguments
/// * `data` - Input data in row-major order
/// * `width` - Number of columns
/// * `height` - Number of rows
///
/// # Returns
/// The reduction result
pub fn tiled_reduce_2d<Op: ReduceOp>(data: &[f32], width: usize, height: usize) -> f32 {
    if data.is_empty() || width == 0 || height == 0 {
        return Op::identity();
    }

    // Create TensorView for the input data
    let view: TensorView<f32> = TensorView::new([height, width, 1, 1]);

    // Partition into 16×16 tiles
    let partition: PartitionView<f32> = PartitionView::new(view, [TILE_SIZE, TILE_SIZE, 1, 1]);

    // Compute number of tiles
    let tiles_y = partition.tile_count()[0];
    let tiles_x = partition.tile_count()[1];

    // Reduce each tile and collect partial results
    let mut partial_results = Vec::with_capacity(tiles_y * tiles_x);

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let tile_sum = reduce_tile::<Op>(data, width, height, tile_x, tile_y);
            partial_results.push(tile_sum);
        }
    }

    // Combine partial results
    partial_results
        .iter()
        .copied()
        .fold(Op::identity(), Op::combine)
}

/// Reduce a single 16×16 tile using tree reduction pattern
///
/// This mirrors the GPU shared memory reduction:
/// 1. Load tile to "shared memory" (local array)
/// 2. Row reduction: 16 -> 8 -> 4 -> 2 -> 1
/// 3. Column reduction: 16 -> 8 -> 4 -> 2 -> 1
fn reduce_tile<Op: ReduceOp>(
    data: &[f32],
    width: usize,
    height: usize,
    tile_x: usize,
    tile_y: usize,
) -> f32 {
    // Simulated shared memory tile (16×16)
    let mut tile = [[Op::identity(); TILE_SIZE]; TILE_SIZE];

    // Load data into tile (bounds-checked)
    let start_y = tile_y * TILE_SIZE;
    let start_x = tile_x * TILE_SIZE;

    // Index-based loops are intentional here - we need indices for:
    // - Calculating global positions (gy, gx)
    // - Early exit on bounds check
    // - Accessing both data array and tile array
    #[allow(clippy::needless_range_loop)]
    for ly in 0..TILE_SIZE {
        let gy = start_y + ly;
        if gy >= height {
            break;
        }
        #[allow(clippy::needless_range_loop)]
        for lx in 0..TILE_SIZE {
            let gx = start_x + lx;
            if gx >= width {
                break;
            }
            let idx = gy * width + gx;
            tile[ly][lx] = data[idx];
        }
    }

    // Row reduction (horizontal): 16 -> 8 -> 4 -> 2 -> 1
    // Index-based loops mirror GPU shader structure for validation
    #[allow(clippy::needless_range_loop)]
    for ly in 0..TILE_SIZE {
        // Step 1: 16 -> 8
        for lx in 0..8 {
            tile[ly][lx] = Op::combine(tile[ly][lx], tile[ly][lx + 8]);
        }
        // Step 2: 8 -> 4
        for lx in 0..4 {
            tile[ly][lx] = Op::combine(tile[ly][lx], tile[ly][lx + 4]);
        }
        // Step 3: 4 -> 2
        for lx in 0..2 {
            tile[ly][lx] = Op::combine(tile[ly][lx], tile[ly][lx + 2]);
        }
        // Step 4: 2 -> 1
        tile[ly][0] = Op::combine(tile[ly][0], tile[ly][1]);
    }

    // Column reduction (vertical): 16 -> 8 -> 4 -> 2 -> 1
    // Step 1: 16 -> 8
    for ly in 0..8 {
        tile[ly][0] = Op::combine(tile[ly][0], tile[ly + 8][0]);
    }
    // Step 2: 8 -> 4
    for ly in 0..4 {
        tile[ly][0] = Op::combine(tile[ly][0], tile[ly + 4][0]);
    }
    // Step 3: 4 -> 2
    for ly in 0..2 {
        tile[ly][0] = Op::combine(tile[ly][0], tile[ly + 2][0]);
    }
    // Step 4: 2 -> 1
    tile[0][0] = Op::combine(tile[0][0], tile[1][0]);

    tile[0][0]
}

/// Convenience function for tiled sum reduction
#[inline]
pub fn tiled_sum_2d(data: &[f32], width: usize, height: usize) -> f32 {
    tiled_reduce_2d::<SumOp>(data, width, height)
}

/// Convenience function for tiled max reduction
#[inline]
pub fn tiled_max_2d(data: &[f32], width: usize, height: usize) -> f32 {
    tiled_reduce_2d::<MaxOp>(data, width, height)
}

/// Convenience function for tiled min reduction
#[inline]
pub fn tiled_min_2d(data: &[f32], width: usize, height: usize) -> f32 {
    tiled_reduce_2d::<MinOp>(data, width, height)
}

/// Compute partial tile results for verification
///
/// Returns the partial reduction result for each tile, which can be
/// compared against GPU partial results buffer for validation.
pub fn tiled_reduce_partial<Op: ReduceOp>(
    data: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    if data.is_empty() || width == 0 || height == 0 {
        return vec![];
    }

    let tiles_y = height.div_ceil(TILE_SIZE);
    let tiles_x = width.div_ceil(TILE_SIZE);

    let mut partial_results = Vec::with_capacity(tiles_y * tiles_x);

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let tile_result = reduce_tile::<Op>(data, width, height, tile_x, tile_y);
            partial_results.push(tile_result);
        }
    }

    partial_results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiled_sum_small() {
        // 4×4 data (single tile, partial)
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let sum = tiled_sum_2d(&data, 4, 4);
        let expected: f32 = (1..=16).sum::<i32>() as f32;
        assert!((sum - expected).abs() < 1e-5, "sum={sum}, expected={expected}");
    }

    #[test]
    fn test_tiled_sum_exact_tile() {
        // Exactly 16×16 = 256 elements
        let data: Vec<f32> = (1..=256).map(|x| x as f32).collect();
        let sum = tiled_sum_2d(&data, 16, 16);
        let expected: f32 = (1..=256).sum::<i32>() as f32;
        assert!((sum - expected).abs() < 1e-3, "sum={sum}, expected={expected}");
    }

    #[test]
    fn test_tiled_sum_multiple_tiles() {
        // 32×32 = 1024 elements (4 tiles: 2×2)
        let data: Vec<f32> = (1..=1024).map(|x| x as f32).collect();
        let sum = tiled_sum_2d(&data, 32, 32);
        let expected: f32 = (1..=1024).sum::<i32>() as f32;
        assert!(
            (sum - expected).abs() < 1e-2,
            "sum={sum}, expected={expected}"
        );
    }

    #[test]
    fn test_tiled_sum_non_aligned() {
        // 20×20 = 400 elements (partial tiles)
        let data: Vec<f32> = (1..=400).map(|x| x as f32).collect();
        let sum = tiled_sum_2d(&data, 20, 20);
        let expected: f32 = (1..=400).sum::<i32>() as f32;
        assert!(
            (sum - expected).abs() < 1e-2,
            "sum={sum}, expected={expected}"
        );
    }

    #[test]
    fn test_tiled_max() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 8.0, 4.0, 6.0];
        let max = tiled_max_2d(&data, 3, 3);
        assert!((max - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_tiled_max_large() {
        let data: Vec<f32> = (1..=256).map(|x| x as f32).collect();
        let max = tiled_max_2d(&data, 16, 16);
        assert!((max - 256.0).abs() < 1e-5);
    }

    #[test]
    fn test_tiled_min() {
        let data: Vec<f32> = vec![5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0, 6.0];
        let min = tiled_min_2d(&data, 3, 3);
        assert!((min - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tiled_min_negative() {
        let data: Vec<f32> = vec![-5.0, 3.0, -7.0, 1.0, -9.0, 2.0, 8.0, -4.0, 6.0];
        let min = tiled_min_2d(&data, 3, 3);
        assert!((min - (-9.0)).abs() < 1e-5);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<f32> = vec![];
        assert!((tiled_sum_2d(&data, 0, 0) - 0.0).abs() < 1e-10);
        assert!(tiled_max_2d(&data, 0, 0) == f32::NEG_INFINITY);
        assert!(tiled_min_2d(&data, 0, 0) == f32::INFINITY);
    }

    #[test]
    fn test_partial_results() {
        // 32×32 data should produce 4 partial results (2×2 tiles)
        let data: Vec<f32> = vec![1.0; 32 * 32];
        let partial = tiled_reduce_partial::<SumOp>(&data, 32, 32);
        assert_eq!(partial.len(), 4);
        // Each 16×16 tile has 256 ones
        for &p in &partial {
            assert!((p - 256.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_partial_results_non_aligned() {
        // 20×20 data should produce 4 partial results (2×2 tiles)
        // but edge tiles have fewer elements
        let data: Vec<f32> = vec![1.0; 20 * 20];
        let partial = tiled_reduce_partial::<SumOp>(&data, 20, 20);
        assert_eq!(partial.len(), 4);
        // Tile (0,0): 16×16 = 256
        // Tile (1,0): 4×16 = 64
        // Tile (0,1): 16×4 = 64
        // Tile (1,1): 4×4 = 16
        // Total: 256 + 64 + 64 + 16 = 400
        let total: f32 = partial.iter().sum();
        assert!((total - 400.0).abs() < 1e-5);
    }

    #[test]
    fn test_single_element() {
        let data = vec![42.0];
        assert!((tiled_sum_2d(&data, 1, 1) - 42.0).abs() < 1e-5);
        assert!((tiled_max_2d(&data, 1, 1) - 42.0).abs() < 1e-5);
        assert!((tiled_min_2d(&data, 1, 1) - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_equivalence_with_simple_sum() {
        // Verify tiled sum matches simple iteration
        let data: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
        let tiled = tiled_sum_2d(&data, 50, 20);
        let simple: f32 = data.iter().sum();
        let rel_err = (tiled - simple).abs() / simple;
        assert!(rel_err < 1e-5, "rel_err={rel_err}");
    }

    #[test]
    fn test_tile_boundaries() {
        // Test that tile boundaries are handled correctly
        // 17×17 = 289 elements (needs 2×2 tiles)
        let data: Vec<f32> = (1..=289).map(|x| x as f32).collect();
        let sum = tiled_sum_2d(&data, 17, 17);
        let expected: f32 = (1..=289).sum::<i32>() as f32;
        assert!(
            (sum - expected).abs() < 1e-2,
            "sum={sum}, expected={expected}"
        );
    }

    #[test]
    fn test_wide_matrix() {
        // 100×5 matrix (many columns, few rows)
        let data: Vec<f32> = (1..=500).map(|x| x as f32).collect();
        let sum = tiled_sum_2d(&data, 100, 5);
        let expected: f32 = (1..=500).sum::<i32>() as f32;
        assert!(
            (sum - expected).abs() < 1e-2,
            "sum={sum}, expected={expected}"
        );
    }

    #[test]
    fn test_tall_matrix() {
        // 5×100 matrix (few columns, many rows)
        let data: Vec<f32> = (1..=500).map(|x| x as f32).collect();
        let sum = tiled_sum_2d(&data, 5, 100);
        let expected: f32 = (1..=500).sum::<i32>() as f32;
        assert!(
            (sum - expected).abs() < 1e-2,
            "sum={sum}, expected={expected}"
        );
    }
}
