//! Advanced WGSL shaders: Jacobi eigenvalue, tiled 2D reductions, causal attention.

/// Causal multi-head attention (WGSL) — scaled dot-product with GQA
///
/// Computes: attn_out = softmax(Q @ K^T / sqrt(d) + causal_mask) @ V
///
/// Supports grouped-query attention (GQA): num_heads >= num_kv_heads.
/// Each KV head serves `num_heads / num_kv_heads` query heads.
///
/// Grid: (num_heads, seq_len, 1) — one workgroup per (head, query position)
/// Each workgroup computes one row of the attention output.
///
/// # Contract (causal-attention-v1)
///
/// - Causal mask: position i can only attend to positions [0..i]
/// - Softmax sums to 1.0 per row (within floating-point tolerance)
/// - Output shape matches Q shape: [seq_len, num_heads * head_dim]
pub const CAUSAL_ATTENTION_SHADER: &str = r#"
// Causal multi-head attention — 2-pass (not 3), no recomputation.
// Pass 1: compute all QK^T scores, find max, compute softmax weights.
// Pass 2: weighted sum of V using stored weights.
// Scores stored in workgroup shared memory (max 2048 positions).
//
// Each workgroup: one (head, query_position) pair.
// @workgroup_size(1) — sequential per position but NO recomputation of QK^T.
// The old shader computed QK^T 3 times (max, softmax, weighted_sum × head_dim).
// This shader computes QK^T ONCE, stores scores, then reuses.
// For head_dim=128: 3× → 1× = 3x fewer dot products.
// For the weighted sum: eliminates recomputing QK^T per output dimension.

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct AttnParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
}

@group(0) @binding(4) var<uniform> cfg: AttnParams;

// Store softmax weights in workgroup shared memory (max 2048 seq positions)
var<workgroup> weights: array<f32, 2048>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let pos = gid.y;
    let seq = cfg.seq_len;
    let hd = cfg.head_dim;
    let kv_group = cfg.num_heads / cfg.num_kv_heads;
    let kv_head = head / kv_group;

    if (head >= cfg.num_heads || pos >= seq) { return; }

    let q_offset = pos * cfg.num_heads * hd + head * hd;
    let scale = 1.0 / sqrt(f32(hd));

    // Pass 1: compute QK^T scores ONCE, find max, compute softmax weights
    var max_score: f32 = -1e30;

    // 1a: compute all scores and find max
    for (var s = 0u; s <= pos; s++) {
        let k_offset = s * cfg.num_kv_heads * hd + kv_head * hd;
        var dot: f32 = 0.0;
        for (var d = 0u; d < hd; d++) {
            dot += q[q_offset + d] * k[k_offset + d];
        }
        let score = dot * scale;
        weights[s] = score;
        max_score = max(max_score, score);
    }

    // 1b: compute exp(score - max) and sum
    var sum_exp: f32 = 0.0;
    for (var s = 0u; s <= pos; s++) {
        let w = exp(weights[s] - max_score);
        weights[s] = w;
        sum_exp += w;
    }

    // 1c: normalize weights
    let inv_sum = 1.0 / sum_exp;
    for (var s = 0u; s <= pos; s++) {
        weights[s] = weights[s] * inv_sum;
    }

    // Pass 2: weighted sum of V using stored weights (NO QK^T recomputation)
    let out_offset = pos * cfg.num_heads * hd + head * hd;
    for (var d = 0u; d < hd; d++) {
        var val: f32 = 0.0;
        for (var s = 0u; s <= pos; s++) {
            let v_offset = s * cfg.num_kv_heads * hd + kv_head * hd;
            val += weights[s] * v[v_offset + d];
        }
        out[out_offset + d] = val;
    }
}
"#;

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
pub(crate) const JACOBI_ROTATION_SHADER: &str = r#"
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
pub(crate) const TILED_SUM_REDUCTION_SHADER: &str = r#"
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
pub(crate) const TILED_MAX_REDUCTION_SHADER: &str = r#"
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
pub(crate) const TILED_MIN_REDUCTION_SHADER: &str = r#"
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
pub(crate) const _JACOBI_MAX_OFFDIAG_SHADER: &str = r#"
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
