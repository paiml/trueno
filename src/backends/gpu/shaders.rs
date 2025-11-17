//! WGSL compute shaders for GPU operations

/// Matrix multiplication compute shader (WGSL)
///
/// Computes C = A × B where:
/// - A is M×K
/// - B is K×N
/// - C is M×N
///
/// Uses workgroups of 16×16 threads for optimal GPU utilization
pub const MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,  // rows of A and C
    K: u32,  // cols of A, rows of B
    N: u32,  // cols of B and C
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Workgroup size: 16×16 = 256 threads
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    // Bounds check
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;

    // Compute dot product: C[row,col] = sum(A[row,k] * B[k,col])
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        let a_idx = row * dims.K + k;        // A is row-major
        let b_idx = k * dims.N + col;        // B is row-major
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * dims.N + col;          // C is row-major
    c[c_idx] = sum;
}
"#;

/// Vector addition compute shader (WGSL)
///
/// Computes c = a + b element-wise
#[allow(dead_code)] // Reserved for future vector operations on GPU
pub const VEC_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

/// Dot product reduction shader (WGSL)
///
/// Computes sum(a[i] * b[i]) using parallel reduction
#[allow(dead_code)] // Reserved for future vector operations on GPU
pub const DOT_PRODUCT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let len = arrayLength(&a);

    // Load and multiply
    var sum: f32 = 0.0;
    if (idx < len) {
        sum = a[idx] * b[idx];
    }
    partial_sums[local_idx] = sum;

    workgroupBarrier();

    // Parallel reduction within workgroup
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            partial_sums[local_idx] = partial_sums[local_idx] + partial_sums[local_idx + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_idx == 0u) {
        result[global_id.x / 256u] = partial_sums[0];
    }
}
"#;
