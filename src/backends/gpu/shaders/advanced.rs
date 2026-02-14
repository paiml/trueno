//! Advanced WGSL shaders: Jacobi eigenvalue, tiled 2D reductions.

/// Jacobi rotation shader (WGSL) - Apply Givens rotation to matrix columns
///
/// Applies rotation to columns p and q of matrix A and eigenvector matrix V:
/// - A[:,p] = c * A[:,p] - s * A[:,q]
/// - A[:,q] = s * old_A[:,p] + c * A[:,q]
///
/// Same transformation is applied to the V matrix (eigenvectors).
///
/// This is a single rotation step in the Jacobi eigenvalue algorithm.
/// Parallelizes over rows (each thread handles one row).
pub const JACOBI_ROTATION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> eigenvectors: array<f32>;

struct JacobiParams {
    n: u32,      // Matrix dimension
    p: u32,      // First column index
    q: u32,      // Second column index
    c: f32,      // cos(theta)
    s: f32,      // sin(theta)
}

@group(0) @binding(2) var<uniform> params: JacobiParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let n = params.n;
    let p = params.p;
    let q = params.q;
    let c = params.c;
    let s = params.s;

    if (k >= n) {
        return;
    }

    // Update matrix row k, columns p and q
    let idx_kp = k * n + p;
    let idx_kq = k * n + q;

    let akp = matrix[idx_kp];
    let akq = matrix[idx_kq];

    matrix[idx_kp] = c * akp - s * akq;
    matrix[idx_kq] = s * akp + c * akq;

    // Update eigenvector matrix row k, columns p and q
    let vkp = eigenvectors[idx_kp];
    let vkq = eigenvectors[idx_kq];

    eigenvectors[idx_kp] = c * vkp - s * vkq;
    eigenvectors[idx_kq] = s * vkp + c * vkq;
}
"#;

/// 2D Tiled Sum Reduction compute shader (WGSL)
///
/// Computes sum reduction using 16×16 workgroups for optimal memory coalescing.
/// Phase 1: Each workgroup reduces a tile to partial sums
/// Phase 2: Combine partial sums (can be done on CPU for small number of workgroups)
///
/// This is more efficient than 1D reduction for 2D data (images, matrices)
/// as it exploits 2D spatial locality in GPU memory hierarchies.
///
pub const TILED_SUM_REDUCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_results: array<f32>;

struct Dimensions {
    width: u32,   // Input width (columns)
    height: u32,  // Input height (rows)
}

@group(0) @binding(2) var<uniform> dims: Dimensions;

// 16×16 workgroup shared memory tile
var<workgroup> tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let lx = local_id.x;
    let ly = local_id.y;
    let gx = global_id.x;
    let gy = global_id.y;

    // Load to shared memory (bounds-checked)
    var val: f32 = 0.0;
    if (gx < dims.width && gy < dims.height) {
        let idx = gy * dims.width + gx;
        val = input[idx];
    }
    tile[ly][lx] = val;

    workgroupBarrier();

    // Row reduction (horizontal): 16 -> 8 -> 4 -> 2 -> 1
    if (lx < 8u) { tile[ly][lx] = tile[ly][lx] + tile[ly][lx + 8u]; }
    workgroupBarrier();
    if (lx < 4u) { tile[ly][lx] = tile[ly][lx] + tile[ly][lx + 4u]; }
    workgroupBarrier();
    if (lx < 2u) { tile[ly][lx] = tile[ly][lx] + tile[ly][lx + 2u]; }
    workgroupBarrier();
    if (lx < 1u) { tile[ly][lx] = tile[ly][lx] + tile[ly][lx + 1u]; }
    workgroupBarrier();

    // Column reduction (vertical): first column only, 16 -> 8 -> 4 -> 2 -> 1
    if (lx == 0u) {
        if (ly < 8u) { tile[ly][0] = tile[ly][0] + tile[ly + 8u][0]; }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 4u) { tile[ly][0] = tile[ly][0] + tile[ly + 4u][0]; }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 2u) { tile[ly][0] = tile[ly][0] + tile[ly + 2u][0]; }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 1u) { tile[ly][0] = tile[ly][0] + tile[ly + 1u][0]; }
    }

    // First thread writes workgroup result
    if (lx == 0u && ly == 0u) {
        let wg_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
        partial_results[wg_idx] = tile[0][0];
    }
}
"#;

/// 2D Tiled Max Reduction compute shader (WGSL)
///
/// Computes max reduction using 16×16 workgroups for optimal memory coalescing.
/// Same algorithm as tiled sum reduction but with max operation.
pub const TILED_MAX_REDUCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_results: array<f32>;

struct Dimensions {
    width: u32,
    height: u32,
}

@group(0) @binding(2) var<uniform> dims: Dimensions;

var<workgroup> tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let lx = local_id.x;
    let ly = local_id.y;
    let gx = global_id.x;
    let gy = global_id.y;

    // Load to shared memory (use -inf for out-of-bounds)
    var val: f32 = -3.402823466e+38; // -FLT_MAX
    if (gx < dims.width && gy < dims.height) {
        let idx = gy * dims.width + gx;
        val = input[idx];
    }
    tile[ly][lx] = val;

    workgroupBarrier();

    // Row reduction with max
    if (lx < 8u) { tile[ly][lx] = max(tile[ly][lx], tile[ly][lx + 8u]); }
    workgroupBarrier();
    if (lx < 4u) { tile[ly][lx] = max(tile[ly][lx], tile[ly][lx + 4u]); }
    workgroupBarrier();
    if (lx < 2u) { tile[ly][lx] = max(tile[ly][lx], tile[ly][lx + 2u]); }
    workgroupBarrier();
    if (lx < 1u) { tile[ly][lx] = max(tile[ly][lx], tile[ly][lx + 1u]); }
    workgroupBarrier();

    // Column reduction with max
    if (lx == 0u) {
        if (ly < 8u) { tile[ly][0] = max(tile[ly][0], tile[ly + 8u][0]); }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 4u) { tile[ly][0] = max(tile[ly][0], tile[ly + 4u][0]); }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 2u) { tile[ly][0] = max(tile[ly][0], tile[ly + 2u][0]); }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 1u) { tile[ly][0] = max(tile[ly][0], tile[ly + 1u][0]); }
    }

    // First thread writes workgroup result
    if (lx == 0u && ly == 0u) {
        let wg_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
        partial_results[wg_idx] = tile[0][0];
    }
}
"#;

/// 2D Tiled Min Reduction compute shader (WGSL)
///
/// Computes min reduction using 16×16 workgroups for optimal memory coalescing.
/// Same algorithm as tiled sum reduction but with min operation.
pub const TILED_MIN_REDUCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_results: array<f32>;

struct Dimensions {
    width: u32,
    height: u32,
}

@group(0) @binding(2) var<uniform> dims: Dimensions;

var<workgroup> tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let lx = local_id.x;
    let ly = local_id.y;
    let gx = global_id.x;
    let gy = global_id.y;

    // Load to shared memory (use +inf for out-of-bounds)
    var val: f32 = 3.402823466e+38; // +FLT_MAX
    if (gx < dims.width && gy < dims.height) {
        let idx = gy * dims.width + gx;
        val = input[idx];
    }
    tile[ly][lx] = val;

    workgroupBarrier();

    // Row reduction with min
    if (lx < 8u) { tile[ly][lx] = min(tile[ly][lx], tile[ly][lx + 8u]); }
    workgroupBarrier();
    if (lx < 4u) { tile[ly][lx] = min(tile[ly][lx], tile[ly][lx + 4u]); }
    workgroupBarrier();
    if (lx < 2u) { tile[ly][lx] = min(tile[ly][lx], tile[ly][lx + 2u]); }
    workgroupBarrier();
    if (lx < 1u) { tile[ly][lx] = min(tile[ly][lx], tile[ly][lx + 1u]); }
    workgroupBarrier();

    // Column reduction with min
    if (lx == 0u) {
        if (ly < 8u) { tile[ly][0] = min(tile[ly][0], tile[ly + 8u][0]); }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 4u) { tile[ly][0] = min(tile[ly][0], tile[ly + 4u][0]); }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 2u) { tile[ly][0] = min(tile[ly][0], tile[ly + 2u][0]); }
    }
    workgroupBarrier();
    if (lx == 0u) {
        if (ly < 1u) { tile[ly][0] = min(tile[ly][0], tile[ly + 1u][0]); }
    }

    // First thread writes workgroup result
    if (lx == 0u && ly == 0u) {
        let wg_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
        partial_results[wg_idx] = tile[0][0];
    }
}
"#;

/// Find max off-diagonal element shader (WGSL) - parallel reduction
///
/// Finds the largest absolute off-diagonal element for Jacobi pivot selection.
/// Returns (max_value, row_index, col_index) packed in result buffer.
///
/// Note: Currently unused - pivot selection done on CPU for simplicity.
/// Future optimization: use this shader for fully GPU-based pivot selection.
#[allow(dead_code)]
pub const JACOBI_MAX_OFFDIAG_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct MatrixParams {
    n: u32,
}

@group(0) @binding(2) var<uniform> params: MatrixParams;

// Workgroup shared memory for reduction
var<workgroup> partial_max: array<f32, 256>;
var<workgroup> partial_row: array<u32, 256>;
var<workgroup> partial_col: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let n = params.n;

    // Total off-diagonal elements: n*(n-1)/2
    let total_pairs = n * (n - 1u) / 2u;

    // Convert linear index to (i, j) where i < j
    var max_val: f32 = 0.0;
    var max_row: u32 = 0u;
    var max_col: u32 = 1u;

    if (idx < total_pairs) {
        // Map linear index to upper triangular (i, j) where i < j
        // Using quadratic formula inversion
        var i: u32 = 0u;
        var j: u32 = 0u;
        var count: u32 = 0u;

        for (var row: u32 = 0u; row < n - 1u; row = row + 1u) {
            let pairs_in_row = n - 1u - row;
            if (count + pairs_in_row > idx) {
                i = row;
                j = row + 1u + (idx - count);
                break;
            }
            count = count + pairs_in_row;
        }

        let aij = matrix[i * n + j];
        max_val = abs(aij);
        max_row = i;
        max_col = j;
    }

    partial_max[local_idx] = max_val;
    partial_row[local_idx] = max_row;
    partial_col[local_idx] = max_col;

    workgroupBarrier();

    // Parallel reduction to find max within workgroup
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            if (partial_max[local_idx + stride] > partial_max[local_idx]) {
                partial_max[local_idx] = partial_max[local_idx + stride];
                partial_row[local_idx] = partial_row[local_idx + stride];
                partial_col[local_idx] = partial_col[local_idx + stride];
            }
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_idx == 0u) {
        let wg_idx = workgroup_id.x * 3u;
        result[wg_idx] = partial_max[0];
        result[wg_idx + 1u] = f32(partial_row[0]);
        result[wg_idx + 2u] = f32(partial_col[0]);
    }
}
"#;
