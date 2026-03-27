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

/// PMAT-365: Fused Q4K dequant+GEMV — fixed scale extraction.
/// Q4K superblock: 144B = 2B d(f16) + 2B dmin(f16) + 12B scales + 128B nibbles = 256 elements.
/// Scale table: bytes[0..4] = 6-bit scales for blocks 0-3, bytes[4..8] = 6-bit mins for blocks 0-3,
/// bytes[8..12] = packed 4-bit scale/min for blocks 4-7 (high 2 bits from bytes[0..4]/[4..8]).
pub(super) const Q4K_GEMV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_q4k: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
struct Params { n: u32, k: u32, _p1: u32, _p2: u32, }
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> sdata: array<f32, 256>;

// Read byte `b` from the 12-byte scale table (stored in 3 u32s starting at sc_base)
fn sc_byte(sb_u32: u32, b: u32) -> u32 {
    // Scale bytes are at sb_u32+1 (bytes 0-3), sb_u32+2 (bytes 4-7), sb_u32+3 (bytes 8-11)
    let word_idx = b / 4u;
    let byte_idx = b % 4u;
    return (w_q4k[sb_u32 + 1u + word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

// Extract 6-bit scale and min for block j (0..7) per CPU extract_scale_min()
fn get_scale_min(sb_u32: u32, j: u32) -> vec2<f32> {
    var sc: u32; var mn: u32;
    if (j < 4u) {
        sc = sc_byte(sb_u32, j) & 63u;
        mn = sc_byte(sb_u32, j + 4u) & 63u;
    } else {
        sc = (sc_byte(sb_u32, j + 4u) & 0x0Fu) | ((sc_byte(sb_u32, j - 4u) >> 6u) << 4u);
        mn = (sc_byte(sb_u32, j + 4u) >> 4u) | ((sc_byte(sb_u32, j) >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(mn));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg.x; let tid = lid.x;
    if (row >= params.n) { return; }
    let k = params.k;
    let sb_count = k / 256u;
    let row_u32_off = row * sb_count * 36u;
    var psum: f32 = 0.0;

    for (var sb = tid; sb < sb_count; sb += 256u) {
        let sbu = row_u32_off + sb * 36u;
        let x_base = sb * 256u;
        let hdr = w_q4k[sbu];
        let d = unpack2x16float(hdr & 0xFFFFu).x;
        let dmin = unpack2x16float((hdr >> 16u) & 0xFFFFu).x;

        // PMAT-383: Precompute all 8 scale/min pairs from 3 u32 reads
        let s0 = w_q4k[sbu + 1u]; let s1 = w_q4k[sbu + 2u]; let s2 = w_q4k[sbu + 3u];
        var sc: array<f32, 8>; var mn: array<f32, 8>;
        // Blocks 0-3: simple 6-bit
        sc[0]=f32(s0&63u); sc[1]=f32((s0>>8u)&63u); sc[2]=f32((s0>>16u)&63u); sc[3]=f32((s0>>24u)&63u);
        mn[0]=f32(s1&63u); mn[1]=f32((s1>>8u)&63u); mn[2]=f32((s1>>16u)&63u); mn[3]=f32((s1>>24u)&63u);
        // Blocks 4-7: packed (low 4 from s2, high 2 from s0/s1 bit 6-7)
        sc[4]=f32((s2&0xFu)|((s0>>6u)&3u)<<4u); sc[5]=f32(((s2>>8u)&0xFu)|((s0>>14u)&3u)<<4u);
        sc[6]=f32(((s2>>16u)&0xFu)|((s0>>22u)&3u)<<4u); sc[7]=f32(((s2>>24u)&0xFu)|((s0>>30u)&3u)<<4u);
        mn[4]=f32(((s2>>4u)&0xFu)|((s1>>6u)&3u)<<4u); mn[5]=f32(((s2>>12u)&0xFu)|((s1>>14u)&3u)<<4u);
        mn[6]=f32(((s2>>20u)&0xFu)|((s1>>22u)&3u)<<4u); mn[7]=f32(((s2>>28u)&0xFu)|((s1>>30u)&3u)<<4u);

        for (var chunk = 0u; chunk < 4u; chunk++) {
            let is = chunk * 2u;
            let d1 = d * sc[is]; let dm1 = dmin * mn[is];
            let d2 = d * sc[is+1u]; let dm2 = dmin * mn[is+1u];
            // PMAT-381: Vec4 nibble extraction — process 4 values per iteration
            let qu = sbu + 4u + chunk * 8u;
            let lo_base = x_base + chunk * 64u;
            let hi_base = lo_base + 32u;
            // PMAT-381: Vec4 nibble extraction — process 4 values per iteration
            for (var i = 0u; i < 8u; i++) {
                let w = w_q4k[qu + i];
                let xi = lo_base + i * 4u;
                let nib = vec4<f32>(f32(w & 0xFu), f32((w >> 8u) & 0xFu),
                                    f32((w >> 16u) & 0xFu), f32((w >> 24u) & 0xFu));
                let xv = vec4<f32>(x[xi], x[xi+1u], x[xi+2u], x[xi+3u]);
                psum += dot(nib * d1 - vec4(dm1), xv);
                let hxi = hi_base + i * 4u;
                let hnib = vec4<f32>(f32((w >> 4u) & 0xFu), f32((w >> 12u) & 0xFu),
                                     f32((w >> 20u) & 0xFu), f32((w >> 28u) & 0xFu));
                let hxv = vec4<f32>(x[hxi], x[hxi+1u], x[hxi+2u], x[hxi+3u]);
                psum += dot(hnib * d2 - vec4(dm2), hxv);
            }
        }
    }
    sdata[tid] = psum; workgroupBarrier();
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
