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
    /// PMAT-342: CPU-side bias data (small, not worth GPU dispatch)
    cpu_biases: HashMap<String, Vec<f32>>,

    // Intermediate buffers (persistent, reused across calls)
    // For 1.5B: hidden=1536, kv=256, intermediate=8960
    hidden_buf: wgpu::Buffer,   // [hidden_dim] working state
    q_buf: wgpu::Buffer,        // [q_dim]
    k_buf: wgpu::Buffer,        // [kv_dim]
    v_buf: wgpu::Buffer,        // [kv_dim]
    attn_out_buf: wgpu::Buffer, // [hidden_dim]
    ffn_gate_buf: wgpu::Buffer, // [intermediate_dim]
    ffn_up_buf: wgpu::Buffer,   // [intermediate_dim]
    ffn_out_buf: wgpu::Buffer,  // [hidden_dim]
    norm_buf: wgpu::Buffer,     // [hidden_dim] for RMSNorm output
    staging_buf: wgpu::Buffer,  // readback

    // Config
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    intermediate_dim: u32,
}
use super::wgsl_shaders::{RESIDUAL_SHADER, RMSNORM_SHADER, ROPE_SHADER, SILU_MUL_SHADER};

#[rustfmt::skip] // Compact style required — file health limit
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
        let attn_out_buf = buf(hidden_dim.max(intermediate_dim), "attn_out"); // SiLU writes inter elements
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
            cpu_biases: HashMap::new(),
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
    /// PMAT-342: Bias weights (name contains "bias") are stored CPU-side.
    pub fn upload_weight(&mut self, name: &str, data: &[f32]) {
        if name.contains("bias") {
            self.cpu_biases.insert(name.to_string(), data.to_vec());
            return;
        }
        use wgpu::util::DeviceExt;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.weight_buffers.insert(name.to_string(), buffer);
    }
    /// PMAT-347: Upload weight transposed from [rows,cols] to [cols,rows].
    /// Required for matmul shader which expects B in [K,N] layout.
    pub fn upload_weight_transposed(&mut self, name: &str, data: &[f32], rows: usize, cols: usize) {
        let mut transposed = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        use wgpu::util::DeviceExt;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(&transposed),
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
    /// PMAT-344: Added kv_caches for multi-token context
    #[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "rmsnorm_correctness")]
    pub fn forward_model(
        &self,
        token_id: u32,
        position: usize,
        num_layers: usize,
        token_embedding: &[f32],
        output_norm_weight: &[f32],
        lm_head_weight: &[f32],
        vocab_size: usize,
        eps: f32,
        kv_caches: &mut Vec<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Vec<f32>, String> {
        let hd = self.hidden_dim as usize;

        // 1. Embedding lookup (CPU)
        let embed_start = token_id as usize * hd;
        if embed_start + hd > token_embedding.len() {
            return Err(format!("Token {} out of range (embedding size {})", token_id, token_embedding.len() / hd));
        }
        let mut hidden: Vec<f32> = token_embedding[embed_start..embed_start + hd].to_vec();

        // 2. Transformer layers (GPU via forward_layer with KV cache)
        // Initialize KV caches if empty
        while kv_caches.len() < num_layers {
            kv_caches.push((Vec::new(), Vec::new()));
        }
        for layer_idx in 0..num_layers {
            let prefix = format!("layer.{layer_idx}");
            let (ref mut k_cache, ref mut v_cache) = kv_caches[layer_idx];
            self.forward_layer(&mut hidden, &prefix, position, k_cache, v_cache)?;
        }

        // 3. Output RMSNorm (CPU — small, not worth GPU dispatch)
        let rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hd as f32 + eps).sqrt();
        for i in 0..hd {
            hidden[i] = (hidden[i] / rms) * output_norm_weight[i];
        }

        // 4. LM head — GPU tiled GEMM if transposed weight uploaded, else CPU
        // PMAT-347: upload_weight_transposed("lm_head") stores [K,N] layout for matmul
        if self.weight_buffers.contains_key("lm_head") {
            let input_bytes = (hd * 4) as u64;
            let output_bytes = (vocab_size * 4) as u64;
            let input_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lm_in"), size: input_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&input_buf, 0, bytemuck::cast_slice(&hidden));
            let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lm_out"), size: output_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lm_stg"), size: output_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dims = [1u32, hd as u32, vocab_size as u32, 0u32];
            let dims_buf = self.make_uniform(&dims);
            let weight_buf = self.weight_buffers.get("lm_head").unwrap();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.matmul_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: weight_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
                ],
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.matmul_pipeline);
              pass.set_bind_group(0, &bg, &[]);
              pass.dispatch_workgroups(1, (vocab_size as u32).div_ceil(16), 1); }
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_bytes);
            self.queue.submit(Some(encoder.finish()));
            let slice = staging.slice(..output_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().map_err(|e| format!("recv: {e}"))?.map_err(|e| format!("map: {e:?}"))?;
            let mut logits = vec![0.0f32; vocab_size];
            { let data = slice.get_mapped_range();
              logits.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..vocab_size]); }
            staging.unmap();
            Ok(logits)
        } else {
            // CPU fallback (no transposed weight uploaded)
            let mut logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let mut sum = 0.0f32;
                for j in 0..hd { sum += lm_head_weight[v * hd + j] * hidden[j]; }
                logits[v] = sum;
            }
            Ok(logits)
        }
    }

    /// PMAT-325: Execute one transformer layer — 14 passes, 1 submit, 1 readback.
    ///
    /// Input: hidden state [hidden_dim] on CPU.
    /// Output: updated hidden state [hidden_dim] on CPU.
    /// All intermediate computation stays GPU-resident.
    /// PMAT-344: KV cache parameters for multi-token context
    pub fn forward_layer(
        &self,
        hidden: &mut [f32],
        layer_prefix: &str,
        _position: usize,
        kv_cache_k: &mut Vec<f32>,  // accumulated K: [seq_len * kv_dim]
        kv_cache_v: &mut Vec<f32>,  // accumulated V: [seq_len * kv_dim]
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

        // PMAT-342: Submit Q/K/V projections, readback, do attention on CPU
        // GPU handles the heavy matmuls; CPU handles attention (small at M=1)
        let q_bytes = (q_dim * 4) as u64;
        let kv_bytes = (kv_dim * 4) as u64;

        // Readback Q/K/V from GPU
        let q_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_stg"), size: q_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k_stg"), size: kv_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_stg"), size: kv_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.q_buf, 0, &q_staging, 0, q_bytes);
        encoder.copy_buffer_to_buffer(&self.k_buf, 0, &k_staging, 0, kv_bytes);
        encoder.copy_buffer_to_buffer(&self.v_buf, 0, &v_staging, 0, kv_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Readback Q
        let mut q_data = vec![0.0f32; q_dim as usize];
        {
            let slice = q_staging.slice(..q_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().map_err(|e| format!("q recv: {e}"))?.map_err(|e| format!("q map: {e:?}"))?;
            let data = slice.get_mapped_range();
            q_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..q_dim as usize]);
        }
        q_staging.unmap();

        // Readback K
        let mut k_data = vec![0.0f32; kv_dim as usize];
        {
            let slice = k_staging.slice(..kv_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().map_err(|e| format!("k recv: {e}"))?.map_err(|e| format!("k map: {e:?}"))?;
            let data = slice.get_mapped_range();
            k_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..kv_dim as usize]);
        }
        k_staging.unmap();

        // Readback V
        let mut v_data = vec![0.0f32; kv_dim as usize];
        {
            let slice = v_staging.slice(..kv_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().map_err(|e| format!("v recv: {e}"))?.map_err(|e| format!("v map: {e:?}"))?;
            let data = slice.get_mapped_range();
            v_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..kv_dim as usize]);
        }
        v_staging.unmap();

        // PMAT-342: Add QKV biases (required for Qwen2)
        if let Some(q_bias) = self.cpu_biases.get(&format!("{layer_prefix}.q_bias")) {
            for (q, b) in q_data.iter_mut().zip(q_bias.iter()) { *q += *b; }
        }
        if let Some(k_bias) = self.cpu_biases.get(&format!("{layer_prefix}.k_bias")) {
            for (k, b) in k_data.iter_mut().zip(k_bias.iter()) { *k += *b; }
        }
        if let Some(v_bias) = self.cpu_biases.get(&format!("{layer_prefix}.v_bias")) {
            for (v, b) in v_data.iter_mut().zip(v_bias.iter()) { *v += *b; }
        }
        // PMAT-343: Apply RoPE (NeoX-style interleaved) to Q and K
        let head_dim = self.head_dim as usize;
        let position = _position; // Use the position parameter
        let rope_theta = 1_000_000.0f64; // Qwen2 rope_theta

        // RoPE on Q (num_heads × head_dim)
        for h in 0..(self.num_heads as usize) {
            let offset = h * head_dim;
            let half = head_dim / 2;
            for i in 0..half {
                let theta = rope_theta.powf(-((2 * i) as f64) / head_dim as f64);
                let angle = position as f64 * theta;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;
                let x0 = q_data[offset + i];
                let x1 = q_data[offset + i + half];
                q_data[offset + i] = x0 * cos_a - x1 * sin_a;
                q_data[offset + i + half] = x0 * sin_a + x1 * cos_a;
            }
        }

        // RoPE on K (num_kv_heads × head_dim)
        for h in 0..(self.num_kv_heads as usize) {
            let offset = h * head_dim;
            let half = head_dim / 2;
            for i in 0..half {
                let theta = rope_theta.powf(-((2 * i) as f64) / head_dim as f64);
                let angle = position as f64 * theta;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;
                let x0 = k_data[offset + i];
                let x1 = k_data[offset + i + half];
                k_data[offset + i] = x0 * cos_a - x1 * sin_a;
                k_data[offset + i + half] = x0 * sin_a + x1 * cos_a;
            }
        }

        // PMAT-344: Append K,V to cache and compute full attention
        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;
        let num_kv_heads = self.num_kv_heads as usize;
        let kv_dim_usize = kv_dim as usize;

        kv_cache_k.extend_from_slice(&k_data);
        kv_cache_v.extend_from_slice(&v_data);
        let seq_len = kv_cache_k.len() / kv_dim_usize;

        // Scaled dot-product attention with GQA
        let kv_group = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; q_dim as usize];


        for h in 0..num_heads {
            let kv_h = h / kv_group;
            let q_offset = h * head_dim;

            // Compute attention scores: Q[h] · K[kv_h, :seq_len]^T / sqrt(d)
            let mut scores = vec![0.0f32; seq_len];
            for s in 0..seq_len {
                let k_offset = s * kv_dim_usize + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_offset + d] * kv_cache_k[k_offset + d];
                }
                scores[s] = dot * scale;
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores.iter_mut() { *s /= sum; }
            }

            // Weighted sum of V
            let out_offset = h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for s in 0..seq_len {
                    let v_offset = s * kv_dim_usize + kv_h * head_dim;
                    val += scores[s] * kv_cache_v[v_offset + d];
                }
                attn_out[out_offset + d] = val;
            }
        }

        // Upload attention output back to GPU for O projection
        self.queue.write_buffer(&self.q_buf, 0, bytemuck::cast_slice(&attn_out));

        // New encoder for remaining passes
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 7: O projection (attn_out × W_o → attn_out_buf)
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
        // PMAT-346: GEMV and matmul have different uniform struct layouts.
        // GEMV: Params { n (output dim), k (input dim), _, _ }
        // Matmul: Dimensions { M, K, N, _ }
        let params = if m == 1 {
            [n, k, 0u32, 0u32]
        } else {
            [m, k, n, 0u32]
        };
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

#[rustfmt::skip]
fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false, min_binding_size: None,
        }, count: None,
    }
}
#[rustfmt::skip]
fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false, min_binding_size: None,
        }, count: None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_weight_transpose_layout() {
        // PMAT-347: Verify transpose [2,3] → [3,2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2 rows, 3 cols]
        let (rows, cols) = (2, 3);
        let mut transposed = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        // Expected: column-major read gives [1,4,2,5,3,6]
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_shader_sources_non_empty() {
        assert!(!super::RMSNORM_SHADER.is_empty());
        assert!(!super::SILU_MUL_SHADER.is_empty());
        assert!(!super::RESIDUAL_SHADER.is_empty());
        assert!(!super::ROPE_SHADER.is_empty());
    }
}
