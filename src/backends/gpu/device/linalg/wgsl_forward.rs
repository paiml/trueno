//! PMAT-324: WGSL transformer forward pass — multi-pass single submission.
//!
//! Instead of one matmul per CPU call (2ms roundtrip each), this encodes
//! ALL operations for one transformer layer into a single command encoder.
//! Only one submit + one readback per layer (or per full forward pass).
//!
//! Architecture: separate WGSL kernels per operation type, dispatched
//! sequentially within one command encoder. All intermediate data stays
//! GPU-resident in persistent buffers.

use std::collections::HashMap;

/// GPU-resident transformer layer state.
/// All buffers persist across tokens — only input/output change per step.
pub struct WgslForwardPass {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Kernels (compiled once)
    matmul_pipeline: wgpu::ComputePipeline,
    /// PMAT-327: GEMV pipeline for M=1 decode (cooperative K-reduction)
    gemv_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    silu_mul_pipeline: wgpu::ComputePipeline,
    rope_pipeline: wgpu::ComputePipeline,
    residual_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    matmul_bgl: wgpu::BindGroupLayout,
    elementwise_bgl: wgpu::BindGroupLayout,

    // Weight buffers (persistent, uploaded once)
    weight_buffers: HashMap<String, wgpu::Buffer>,

    // Intermediate buffers (persistent, reused across calls)
    // For 1.5B: hidden=1536, kv=256, intermediate=8960
    hidden_buf: wgpu::Buffer,      // [hidden_dim] working state
    q_buf: wgpu::Buffer,           // [q_dim]
    k_buf: wgpu::Buffer,           // [kv_dim]
    v_buf: wgpu::Buffer,           // [kv_dim]
    attn_out_buf: wgpu::Buffer,    // [hidden_dim]
    ffn_gate_buf: wgpu::Buffer,    // [intermediate_dim]
    ffn_up_buf: wgpu::Buffer,      // [intermediate_dim]
    ffn_out_buf: wgpu::Buffer,     // [hidden_dim]
    norm_buf: wgpu::Buffer,        // [hidden_dim] for RMSNorm output
    staging_buf: wgpu::Buffer,     // readback

    // Config
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    intermediate_dim: u32,
}

// WGSL shader source for RMSNorm
const RMSNORM_SHADER: &str = r#"
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

// WGSL shader for SiLU(gate) * up
const SILU_MUL_SHADER: &str = r#"
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

// WGSL shader for residual add: output = a + b
const RESIDUAL_SHADER: &str = r#"
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

// RoPE shader (NeoX-style interleaved)
const ROPE_SHADER: &str = r#"
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

impl WgslForwardPass {
    /// Get the shader sources for external inspection/testing
    pub fn rmsnorm_shader() -> &'static str { RMSNORM_SHADER }
    pub fn silu_mul_shader() -> &'static str { SILU_MUL_SHADER }
    pub fn residual_shader() -> &'static str { RESIDUAL_SHADER }
    pub fn rope_shader() -> &'static str { ROPE_SHADER }

    /// PMAT-325: Create a new WGSL forward pass context.
    ///
    /// Compiles all shader pipelines and allocates persistent intermediate buffers.
    /// Call once at model init. All GPU resources persist until dropped.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Compile shaders
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"), source: wgpu::ShaderSource::Wgsl(
                crate::backends::gpu::shaders::MATMUL_SHADER.into()),
        });
        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm"), source: wgpu::ShaderSource::Wgsl(RMSNORM_SHADER.into()),
        });
        let silu_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silu_mul"), source: wgpu::ShaderSource::Wgsl(SILU_MUL_SHADER.into()),
        });
        let rope_shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope"), source: wgpu::ShaderSource::Wgsl(ROPE_SHADER.into()),
        });
        let residual_shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("residual"), source: wgpu::ShaderSource::Wgsl(RESIDUAL_SHADER.into()),
        });

        // Bind group layouts
        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bgl"),
            entries: &[
                bgl_storage(0, true), bgl_storage(1, true),
                bgl_storage(2, false), bgl_uniform(3),
            ],
        });
        let elementwise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ew_bgl"),
            entries: &[
                bgl_storage(0, true), bgl_storage(1, true),
                bgl_storage(2, false), bgl_uniform(3),
            ],
        });

        // Pipelines
        let make_pipeline = |shader: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout, label: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label), bind_group_layouts: &[bgl], push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&pl), module: shader,
                entry_point: Some("main"), compilation_options: Default::default(), cache: None,
            })
        };

        let matmul_pipeline = make_pipeline(&matmul_shader, &matmul_bgl, "matmul_pipe");

        // PMAT-327: GEMV pipeline — same bind group layout as matmul but cooperative reduction
        let gemv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv"), source: wgpu::ShaderSource::Wgsl(
                crate::backends::gpu::shaders::GEMV_SHADER.into()),
        });
        let gemv_pipeline = make_pipeline(&gemv_shader, &matmul_bgl, "gemv_pipe");

        let rmsnorm_pipeline = make_pipeline(&rmsnorm_shader, &elementwise_bgl, "rmsnorm_pipe");
        let silu_mul_pipeline = make_pipeline(&silu_mul_shader, &elementwise_bgl, "silu_pipe");
        let residual_pipeline = make_pipeline(&residual_shader_mod, &elementwise_bgl, "res_pipe");

        // RoPE has a 2-binding layout (in-place + uniform)
        let rope_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rope_bgl"),
            entries: &[bgl_storage(0, false), bgl_uniform(1)],
        });
        let rope_pipeline = {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rope_pl"), bind_group_layouts: &[&rope_bgl], push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rope_pipe"), layout: Some(&pl), module: &rope_shader_mod,
                entry_point: Some("main"), compilation_options: Default::default(), cache: None,
            })
        };

        // Allocate persistent intermediate buffers
        let buf = |size: usize, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label), size: (size * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let hidden_buf = buf(hidden_dim, "hidden");
        let q_buf = buf(q_dim, "q");
        let k_buf = buf(kv_dim, "k");
        let v_buf = buf(kv_dim, "v");
        let attn_out_buf = buf(hidden_dim, "attn_out");
        let ffn_gate_buf = buf(intermediate_dim, "ffn_gate");
        let ffn_up_buf = buf(intermediate_dim, "ffn_up");
        let ffn_out_buf = buf(hidden_dim, "ffn_out");
        let norm_buf = buf(hidden_dim, "norm");

        let max_out = hidden_dim.max(intermediate_dim);
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"), size: (max_out * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device, queue,
            matmul_pipeline, gemv_pipeline, rmsnorm_pipeline, silu_mul_pipeline,
            rope_pipeline, residual_pipeline,
            matmul_bgl, elementwise_bgl,
            weight_buffers: HashMap::new(),
            hidden_buf, q_buf, k_buf, v_buf, attn_out_buf,
            ffn_gate_buf, ffn_up_buf, ffn_out_buf, norm_buf,
            staging_buf,
            hidden_dim: hidden_dim as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            intermediate_dim: intermediate_dim as u32,
        }
    }

    /// Upload a weight matrix (call once per layer at init).
    pub fn upload_weight(&mut self, name: &str, data: &[f32]) {
        use wgpu::util::DeviceExt;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.weight_buffers.insert(name.to_string(), buffer);
    }

    /// Total VRAM used by all buffers (bytes).
    pub fn total_vram_bytes(&self) -> usize {
        let weight_bytes: usize = self.weight_buffers.values()
            .map(|b| b.size() as usize)
            .sum();
        let intermediate_bytes = (self.hidden_dim as usize * 4) * 4  // hidden, attn_out, ffn_out, norm
            + (self.num_heads as usize * self.head_dim as usize * 4) // q
            + (self.num_kv_heads as usize * self.head_dim as usize * 4) * 2 // k, v
            + (self.intermediate_dim as usize * 4) * 2; // gate, up
        weight_bytes + intermediate_bytes
    }

    /// PMAT-336: Full model forward — embedding + all layers + output norm + LM head.
    ///
    /// Returns logits [vocab_size] for the given token at the given position.
    /// Embedding lookup and final LM head are CPU-side (not yet GPU-accelerated).
    #[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "rmsnorm_correctness")]
    pub fn forward_model(
        &self,
        token_id: u32,
        position: usize,
        num_layers: usize,
        token_embedding: &[f32],    // [vocab_size, hidden_dim]
        output_norm_weight: &[f32], // [hidden_dim]
        lm_head_weight: &[f32],     // [vocab_size, hidden_dim]
        vocab_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let hd = self.hidden_dim as usize;

        // 1. Embedding lookup (CPU)
        let embed_start = token_id as usize * hd;
        if embed_start + hd > token_embedding.len() {
            return Err(format!("Token {} out of range (embedding size {})", token_id, token_embedding.len() / hd));
        }
        let mut hidden: Vec<f32> = token_embedding[embed_start..embed_start + hd].to_vec();

        // 2. Transformer layers (GPU via forward_layer)
        for layer_idx in 0..num_layers {
            let prefix = format!("layer.{layer_idx}");
            self.forward_layer(&mut hidden, &prefix, position)?;
        }

        // 3. Output RMSNorm (CPU — small, not worth GPU dispatch)
        let rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hd as f32 + eps).sqrt();
        for i in 0..hd {
            hidden[i] = (hidden[i] / rms) * output_norm_weight[i];
        }

        // 4. LM head matmul (CPU — vocab_size × hidden_dim, large but one-shot)
        // TODO: Move to GPU for large vocab_size
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            let row_start = v * hd;
            for j in 0..hd {
                sum += lm_head_weight[row_start + j] * hidden[j];
            }
            logits[v] = sum;
        }

        Ok(logits)
    }

    /// PMAT-325: Execute one transformer layer — 14 passes, 1 submit, 1 readback.
    ///
    /// Input: hidden state [hidden_dim] on CPU.
    /// Output: updated hidden state [hidden_dim] on CPU.
    /// All intermediate computation stays GPU-resident.
    pub fn forward_layer(
        &self,
        hidden: &mut [f32],
        layer_prefix: &str,  // e.g., "layer.0"
        _position: usize,
    ) -> Result<(), String> {
        let hd = self.hidden_dim;

        // Upload hidden state
        self.queue.write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(hidden));

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 1: RMSNorm(hidden → norm_buf)
        let norm_w = self.weight_buffers.get(&format!("{layer_prefix}.attn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.attn_norm"))?;
        self.encode_rmsnorm(&mut encoder, &self.hidden_buf, norm_w, &self.norm_buf, hd);

        // Passes 2-4: Q/K/V projections (norm_buf × W → q/k/v_buf)
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "q_proj", &self.q_buf, 1, hd, q_dim);
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "k_proj", &self.k_buf, 1, hd, kv_dim);
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "v_proj", &self.v_buf, 1, hd, kv_dim);

        // Pass 5-6: RoPE on Q and K (skipped in this MVP — would need position tracking)
        // TODO: encode_rope passes

        // Pass 7: O projection (attn_out × W_o → ffn_out_buf) — using q_buf as attn placeholder
        // NOTE: Full attention not yet implemented in WGSL; this MVP tests the dispatch pipeline
        self.encode_matmul(&mut encoder, &self.q_buf, layer_prefix, "o_proj", &self.attn_out_buf, 1, q_dim, hd);

        // Pass 8: Residual(hidden + attn_out → hidden)
        self.encode_residual(&mut encoder, &self.hidden_buf, &self.attn_out_buf, &self.ffn_out_buf, hd);

        // Pass 9: FFN RMSNorm(ffn_out → norm_buf)
        let ffn_norm_w = self.weight_buffers.get(&format!("{layer_prefix}.ffn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.ffn_norm"))?;
        self.encode_rmsnorm(&mut encoder, &self.ffn_out_buf, ffn_norm_w, &self.norm_buf, hd);

        // Passes 10-11: Gate + Up projections
        let inter = self.intermediate_dim;
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "gate_proj", &self.ffn_gate_buf, 1, hd, inter);
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "up_proj", &self.ffn_up_buf, 1, hd, inter);

        // Pass 12: SiLU(gate) × up → ffn_out_buf (reused as intermediate)
        self.encode_silu_mul(&mut encoder, &self.ffn_gate_buf, &self.ffn_up_buf, &self.attn_out_buf, inter);

        // Pass 13: Down projection
        self.encode_matmul(&mut encoder, &self.attn_out_buf, layer_prefix, "down_proj", &self.norm_buf, 1, inter, hd);

        // Pass 14: Residual(ffn_out + down → hidden)
        self.encode_residual(&mut encoder, &self.ffn_out_buf, &self.norm_buf, &self.hidden_buf, hd);

        // Single readback
        encoder.copy_buffer_to_buffer(&self.hidden_buf, 0, &self.staging_buf, 0, (hd * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        // Readback
        let slice = self.staging_buf.slice(..(hd as u64 * 4));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().map_err(|e| format!("recv: {e}"))?.map_err(|e| format!("map: {e:?}"))?;
        {
            let data = slice.get_mapped_range();
            hidden.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..self.hidden_dim as usize]);
        }
        self.staging_buf.unmap();

        Ok(())
    }

    // --- Encode helpers (add compute passes to an existing encoder) ---

    fn encode_rmsnorm(&self, encoder: &mut wgpu::CommandEncoder,
                      input: &wgpu::Buffer, weight: &wgpu::Buffer,
                      output: &wgpu::Buffer, dim: u32) {
        let params = [dim, 0u32, 0, 0];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.elementwise_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.rmsnorm_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single workgroup for reduction
    }

    fn encode_matmul(&self, encoder: &mut wgpu::CommandEncoder,
                     input: &wgpu::Buffer, layer_prefix: &str, proj_name: &str,
                     output: &wgpu::Buffer, m: u32, k: u32, n: u32) {
        let weight_key = format!("{layer_prefix}.{proj_name}");
        let weight = match self.weight_buffers.get(&weight_key) {
            Some(w) => w,
            None => return, // Skip missing weights silently
        };
        let params = [m, k, n, 0u32];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        if m == 1 {
            // PMAT-327: GEMV for M=1 — cooperative K-reduction, N workgroups
            pass.set_pipeline(&self.gemv_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n, 1, 1);
        } else {
            // Tiled GEMM for M>1 (batch/prefill)
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(16), n.div_ceil(16), 1);
        }
    }

    fn encode_silu_mul(&self, encoder: &mut wgpu::CommandEncoder,
                       gate: &wgpu::Buffer, up: &wgpu::Buffer,
                       output: &wgpu::Buffer, dim: u32) {
        let params = [dim, 0u32, 0, 0];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.elementwise_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: up.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.silu_mul_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dim.div_ceil(256), 1, 1);
    }

    fn encode_residual(&self, encoder: &mut wgpu::CommandEncoder,
                       a: &wgpu::Buffer, b: &wgpu::Buffer,
                       output: &wgpu::Buffer, dim: u32) {
        let params = [dim, 0u32, 0, 0];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.elementwise_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.residual_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dim.div_ceil(256), 1, 1);
    }

    fn make_uniform(&self, data: &[u32; 4]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
}

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false, min_binding_size: None,
        }, count: None,
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false, min_binding_size: None,
        }, count: None,
    }
}
