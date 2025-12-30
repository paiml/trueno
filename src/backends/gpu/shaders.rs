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

/// Element-wise multiplication shader (WGSL)
///
/// Computes c = a * b element-wise
pub const VEC_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] * b[idx];
    }
}
"#;

/// Element-wise subtraction shader (WGSL)
///
/// Computes c = a - b element-wise
pub const VEC_SUB_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] - b[idx];
    }
}
"#;

/// Scalar multiplication shader (WGSL)
///
/// Computes output = input * scalar element-wise
pub const SCALE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ScaleParams {
    scalar: f32,
}

@group(0) @binding(2) var<uniform> params: ScaleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        output[idx] = input[idx] * params.scalar;
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

/// Leaky ReLU activation compute shader (WGSL)
///
/// Computes element-wise Leaky ReLU: leaky_relu(x, α) = max(αx, x) = x if x > 0, else αx
///
/// Leaky ReLU addresses the "dying ReLU" problem by allowing small negative activations.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const LEAKY_RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct LeakyReluParams {
    negative_slope: f32,
}

@group(0) @binding(2) var<uniform> params: LeakyReluParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Leaky ReLU: leaky_relu(x, α) = x if x > 0, else αx
        if (x > 0.0) {
            output[idx] = x;
        } else {
            output[idx] = params.negative_slope * x;
        }
    }
}
"#;

/// ELU (Exponential Linear Unit) activation compute shader (WGSL)
///
/// Computes element-wise ELU: elu(x, α) = x if x > 0, else α(e^x - 1)
///
/// ELU has smooth gradients everywhere and pushes mean activations closer to zero,
/// improving learning in deep networks.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const ELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct EluParams {
    alpha: f32,
}

@group(0) @binding(2) var<uniform> params: EluParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // ELU: elu(x, α) = x if x > 0, else α(e^x - 1)
        if (x > 0.0) {
            output[idx] = x;
        } else {
            output[idx] = params.alpha * (exp(x) - 1.0);
        }
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

/// Tanh (hyperbolic tangent) activation compute shader (WGSL)
///
/// Computes element-wise tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
///
/// Classic activation function used in LSTM, GRU, and traditional neural networks.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const TANH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        //                = (e^(2x) - 1) / (e^(2x) + 1)
        // Numerically stable implementation:
        // For |x| > 20: tanh(x) ≈ sign(x) (saturates at ±1)
        // Otherwise: use standard formula
        var result: f32;
        if (x > 20.0) {
            result = 1.0;
        } else if (x < -20.0) {
            result = -1.0;
        } else {
            let exp_2x = exp(2.0 * x);
            result = (exp_2x - 1.0) / (exp_2x + 1.0);
        }

        output[idx] = result;
    }
}
"#;

/// Swish activation compute shader (WGSL)
///
/// Computes element-wise swish: swish(x) = x * σ(x) = x / (1 + e^(-x))
///
/// Modern activation function (SiLU) used in transformers and modern architectures.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const SWISH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Swish: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // Numerically stable implementation:
        // For x >= 0: swish(x) = x / (1 + exp(-x))
        // For x < 0: swish(x) = x * exp(x) / (1 + exp(x))
        var result: f32;
        if (x >= 0.0) {
            result = x / (1.0 + exp(-x));
        } else {
            let exp_x = exp(x);
            result = x * exp_x / (1.0 + exp_x);
        }

        output[idx] = result;
    }
}
"#;

/// GELU activation compute shader (WGSL)
///
/// Computes element-wise GELU using tanh approximation:
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// Standard activation in BERT, GPT-2, GPT-3, and modern transformers.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub const GELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // GELU approximation (tanh-based):
        // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        let SQRT_2_OVER_PI: f32 = 0.7978846; // √(2/π)
        let COEFF: f32 = 0.044715;

        let x_cubed = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        let result = 0.5 * x * (1.0 + tanh(inner));

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

/// Max reduction compute shader (WGSL)
///
/// Computes max(input) using parallel reduction
/// Used as first pass in softmax to ensure numerical stability
pub const MAX_REDUCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> partial_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let len = arrayLength(&input);

    // Load value or negative infinity
    var max_val: f32 = -3.402823466e+38; // -FLT_MAX
    if (idx < len) {
        max_val = input[idx];
    }
    partial_max[local_idx] = max_val;

    workgroupBarrier();

    // Parallel reduction within workgroup (find max)
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            partial_max[local_idx] = max(partial_max[local_idx], partial_max[local_idx + stride]);
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_idx == 0u) {
        result[global_id.x / 256u] = partial_max[0];
    }
}
"#;

/// Sum reduction compute shader (WGSL)
///
/// Computes sum(input) using parallel reduction
/// Used in softmax to sum exp values
pub const SUM_REDUCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let len = arrayLength(&input);

    // Load value
    var sum: f32 = 0.0;
    if (idx < len) {
        sum = input[idx];
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

/// Softmax exp-subtract compute shader (WGSL)
///
/// Computes exp(input[i] - max_val) for each element
/// Second pass in softmax: numerically stable exp computation
pub const SOFTMAX_EXP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct MaxValue {
    max_val: f32,
}

@group(0) @binding(2) var<uniform> params: MaxValue;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // exp(x - max) for numerical stability
        output[idx] = exp(input[idx] - params.max_val);
    }
}
"#;

/// Softmax normalize compute shader (WGSL)
///
/// Computes output[i] = input[i] / sum_val for each element
/// Fourth pass in softmax: normalize by sum of exp values
pub const SOFTMAX_NORMALIZE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct SumValue {
    sum_val: f32,
}

@group(0) @binding(2) var<uniform> params: SumValue;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // Normalize by sum
        output[idx] = input[idx] / params.sum_val;
    }
}
"#;

/// Log-softmax compute shader (WGSL)
///
/// Computes log_softmax[i] = input[i] - max_val - log(sum_val) for each element
/// Numerically stable log-softmax in single pass after reductions
pub const LOG_SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct LogSoftmaxParams {
    max_val: f32,
    log_sum_exp: f32,
}

@group(0) @binding(2) var<uniform> params: LogSoftmaxParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // log_softmax(x)[i] = x[i] - max - log(sum(exp(x - max)))
        output[idx] = input[idx] - params.max_val - params.log_sum_exp;
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
/// Note: Not yet integrated into GpuDevice - see GitHub issue #76 for GPU validation.
#[allow(dead_code)]
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
///
/// Note: Not yet integrated into GpuDevice - see GitHub issue #76 for GPU validation.
#[allow(dead_code)]
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
