//! WGSL shader source strings for transformer forward pass operations.
//! Extracted from wgsl_forward.rs for file health (line count) management.

pub(super) const RMSNORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // (dim, 0, 0, 0)

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let dim = params.x;
    let tid = lid.x;

    // Compute sum of squares (reduction)
    var local_sum: f32 = 0.0;
    var i = tid;
    while (i < dim) {
        let val = input[i];
        local_sum += val * val;
        i += 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let rms = sqrt(shared_sum[0] / f32(dim) + 1e-6);

    // Normalize and scale
    i = tid;
    while (i < dim) {
        output[i] = (input[i] / rms) * weight[i];
        i += 256u;
    }
}
"#;

pub(super) const SILU_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // (dim, 0, 0, 0)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    let g = gate[idx];
    let silu_g = g / (1.0 + exp(-g));
    output[idx] = silu_g * up[idx];
}
"#;

pub(super) const RESIDUAL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // (dim, 0, 0, 0)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    output[idx] = a[idx] + b[idx];
}
"#;

pub(super) const ROPE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> qk: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // (dim, position, num_heads, head_dim)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = params.x;
    let position = params.y;
    let head_dim = params.w;

    if (idx >= dim) { return; }

    let half_hd = head_dim / 2u;
    let head_idx = idx / head_dim;
    let pos_in_head = idx % head_dim;

    if (pos_in_head >= half_hd) { return; }

    let theta = pow(1000000.0, -f32(pos_in_head * 2u) / f32(head_dim));
    let angle = f32(position) * theta;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let i0 = head_idx * head_dim + pos_in_head;
    let i1 = i0 + half_hd;

    let x0 = qk[i0];
    let x1 = qk[i1];
    qk[i0] = x0 * cos_a - x1 * sin_a;
    qk[i1] = x0 * sin_a + x1 * cos_a;
}
"#;
