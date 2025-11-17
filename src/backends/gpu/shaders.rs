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

/// ReLU activation compute shader (WGSL)
///
/// Computes element-wise ReLU: max(0, x)
///
/// This is one of the simplest GPU operations - a single comparison and selection per element.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // ReLU: max(0, x)
        output[idx] = max(0.0, input[idx]);
    }
}
"#;

/// Sigmoid activation compute shader (WGSL)
///
/// Computes element-wise sigmoid: σ(x) = 1 / (1 + e^(-x))
///
/// Classic logistic function used in binary classification and attention mechanisms.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const SIGMOID_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Sigmoid: σ(x) = 1 / (1 + exp(-x))
        // Numerically stable implementation:
        // For x >= 0: σ(x) = 1 / (1 + exp(-x))
        // For x < 0: σ(x) = exp(x) / (1 + exp(x))
        var result: f32;
        if (x >= 0.0) {
            result = 1.0 / (1.0 + exp(-x));
        } else {
            let exp_x = exp(x);
            result = exp_x / (1.0 + exp_x);
        }

        output[idx] = result;
    }
}
"#;

/// Clip (clamp) compute shader (WGSL)
///
/// Computes element-wise clip: clamp(x, min_val, max_val)
///
/// Constrains values to the range [min_val, max_val].
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const CLIP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ClipParams {
    min_val: f32,
    max_val: f32,
}

@group(0) @binding(2) var<uniform> params: ClipParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // Clip: clamp(x, min_val, max_val) = max(min_val, min(max_val, x))
        output[idx] = clamp(input[idx], params.min_val, params.max_val);
    }
}
"#;

/// 2D Convolution compute shader (WGSL)
///
/// Computes 2D convolution: output = input ⊗ kernel
/// Uses "valid" padding (no padding, output smaller than input)
///
/// Output dimensions:
/// - output_rows = input_rows - kernel_rows + 1
/// - output_cols = input_cols - kernel_cols + 1
///
/// Uses workgroups of 16×16 threads for optimal GPU utilization
pub const CONVOLVE2D_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct ConvDimensions {
    input_rows: u32,
    input_cols: u32,
    kernel_rows: u32,
    kernel_cols: u32,
    output_rows: u32,
    output_cols: u32,
}

@group(0) @binding(3) var<uniform> dims: ConvDimensions;

// Workgroup size: 16×16 = 256 threads
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_row = global_id.x;
    let out_col = global_id.y;

    // Bounds check
    if (out_row >= dims.output_rows || out_col >= dims.output_cols) {
        return;
    }

    var sum: f32 = 0.0;

    // Apply kernel: iterate over kernel dimensions
    for (var k_row: u32 = 0u; k_row < dims.kernel_rows; k_row = k_row + 1u) {
        for (var k_col: u32 = 0u; k_col < dims.kernel_cols; k_col = k_col + 1u) {
            // Input pixel coordinates
            let in_row = out_row + k_row;
            let in_col = out_col + k_col;

            // Input and kernel are row-major
            let input_idx = in_row * dims.input_cols + in_col;
            let kernel_idx = k_row * dims.kernel_cols + k_col;

            sum = sum + input[input_idx] * kernel[kernel_idx];
        }
    }

    // Write output (row-major)
    let output_idx = out_row * dims.output_cols + out_col;
    output[output_idx] = sum;
}
"#;
