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

/// PMAT-361: Single-head M=1 attention — one workgroup per Q head.
/// Computes Q·K^T scores, softmax, weighted V sum for one head.
/// GQA: kv_head = head / kv_group.
pub(super) const ATTENTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params { num_heads: u32, kv_group: u32, head_dim: u32, seq_len: u32,
                kv_dim: u32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scores: array<f32, 2048>;
var<workgroup> smax: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wg.x;
    let tid = lid.x;
    let hd = params.head_dim;
    let sl = params.seq_len;
    let kvd = params.kv_dim;
    let kvh = head / params.kv_group;
    let scale = 1.0 / sqrt(f32(hd));

    // Phase 1: Q·K scores
    var lmax: f32 = -1e30;
    for (var s = tid; s < sl; s += 256u) {
        var dot: f32 = 0.0;
        let qo = head * hd; let ko = s * kvd + kvh * hd;
        for (var d = 0u; d < hd; d++) { dot += q[qo + d] * k_cache[ko + d]; }
        let sc = dot * scale;
        scores[s] = sc;
        lmax = max(lmax, sc);
    }
    smax[tid] = lmax;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) { smax[tid] = max(smax[tid], smax[tid + stride]); }
        workgroupBarrier(); stride >>= 1u;
    }
    let mx = smax[0];
    workgroupBarrier();

    // Phase 2: exp + sum
    var lsum: f32 = 0.0;
    for (var s = tid; s < sl; s += 256u) {
        scores[s] = exp(scores[s] - mx);
        lsum += scores[s];
    }
    smax[tid] = lsum;
    workgroupBarrier();
    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) { smax[tid] += smax[tid + stride]; }
        workgroupBarrier(); stride >>= 1u;
    }
    let sm = smax[0];
    workgroupBarrier();
    for (var s = tid; s < sl; s += 256u) { scores[s] /= sm; }
    workgroupBarrier();

    // Phase 3: weighted V sum
    for (var d = tid; d < hd; d += 256u) {
        var val: f32 = 0.0;
        for (var s = 0u; s < sl; s++) {
            val += scores[s] * v_cache[s * kvd + kvh * hd + d];
        }
        output[head * hd + d] = val;
    }
}
"#;

/// PMAT-363: Fused Q4K dequant+GEMV — dequantize on-the-fly, 4× bandwidth reduction.
/// Each workgroup computes one output row: y[row] = sum_k W_q4k[row,k] * x[k]
/// Q4K superblock: 144 bytes = 2B d(f16) + 2B dmin(f16) + 12B scales + 128B nibbles = 256 elements
pub(super) const Q4K_GEMV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;        // input [K]
@group(0) @binding(1) var<storage, read> w_q4k: array<u32>;    // Q4K raw bytes as u32
@group(0) @binding(2) var<storage, read_write> y: array<f32>;  // output [N]
struct Params { n: u32, k: u32, _p1: u32, _p2: u32, }
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

// Read f16 from 2 bytes packed in u32 word
fn read_f16_bits(word: u32, byte_offset: u32) -> f32 {
    let shifted = (word >> (byte_offset * 8u)) & 0xFFFFu;
    return unpack2x16float(shifted).x;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x;
    let tid = lid.x;
    if (row >= params.n) { return; }

    let k = params.k;
    let sb_count = k / 256u;  // superblocks per row
    let row_bytes = sb_count * 144u;  // bytes per row in Q4K
    let row_u32_offset = row * (row_bytes / 4u);  // u32 offset for this row

    var partial_sum: f32 = 0.0;

    // Each thread handles some superblocks
    for (var sb = tid; sb < sb_count; sb += 256u) {
        let sb_u32 = row_u32_offset + sb * 36u;  // 144 bytes = 36 u32s per superblock
        let x_base = sb * 256u;

        // Read d and dmin (f16 packed in first u32)
        let hdr0 = w_q4k[sb_u32];
        let d = unpack2x16float(hdr0 & 0xFFFFu).x;
        let dmin = unpack2x16float((hdr0 >> 16u) & 0xFFFFu).x;

        // Read 12 scale bytes (3 u32s: sb_u32+1, sb_u32+2, sb_u32+3)
        let sc0 = w_q4k[sb_u32 + 1u];
        let sc1 = w_q4k[sb_u32 + 2u];
        let sc2 = w_q4k[sb_u32 + 3u];

        // Process 4 chunks of 64 values each
        for (var chunk = 0u; chunk < 4u; chunk++) {
            let is = chunk * 2u;
            // Extract 6-bit scale and min for this chunk's two 32-value halves
            let s0 = extractBits(sc0, (is % 4u) * 8u, 6u);
            let m0 = extractBits(sc0, ((is % 4u) + 4u) * 8u % 32u, 6u);

            let d1 = d * f32(s0);
            let dm1 = dmin * f32(m0);

            // Read 32 bytes of nibbles for this chunk (8 u32s)
            let q_u32 = sb_u32 + 4u + chunk * 8u;

            // Low nibbles (first 32 values)
            for (var i = 0u; i < 8u; i++) {
                let word = w_q4k[q_u32 + i];
                for (var b = 0u; b < 4u; b++) {
                    let nibble = (word >> (b * 8u)) & 0xFu;
                    let xi = x_base + chunk * 64u + i * 4u + b;
                    if (xi < k) { partial_sum += (d1 * f32(nibble) - dm1) * x[xi]; }
                }
            }
            // High nibbles (next 32 values)
            for (var i = 0u; i < 8u; i++) {
                let word = w_q4k[q_u32 + i];
                for (var b = 0u; b < 4u; b++) {
                    let nibble = (word >> (b * 8u + 4u)) & 0xFu;
                    let xi = x_base + chunk * 64u + 32u + i * 4u + b;
                    if (xi < k) { partial_sum += (d1 * f32(nibble) - dm1) * x[xi]; }
                }
            }
        }
    }

    // Tree reduction
    sdata[tid] = partial_sum;
    workgroupBarrier();
    if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; } workgroupBarrier();
    if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; } workgroupBarrier();
    if (tid < 32u) { sdata[tid] += sdata[tid + 32u]; } workgroupBarrier();
    if (tid < 16u) { sdata[tid] += sdata[tid + 16u]; } workgroupBarrier();
    if (tid < 8u) { sdata[tid] += sdata[tid + 8u]; } workgroupBarrier();
    if (tid < 4u) { sdata[tid] += sdata[tid + 4u]; } workgroupBarrier();
    if (tid < 2u) { sdata[tid] += sdata[tid + 2u]; } workgroupBarrier();
    if (tid == 0u) { y[row] = sdata[0u] + sdata[1u]; }
}
"#;

/// PMAT-356: In-place bias addition for GPU-side QKV bias application.
/// data[i] += bias[i]. Bindings: data(rw), bias(r), params(uniform).
pub(super) const BIAS_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    data[idx] = data[idx] + bias[idx];
}
"#;
