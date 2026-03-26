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
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) matmul_pipeline: wgpu::ComputePipeline,
    pub(super) gemv_pipeline: wgpu::ComputePipeline,
    pub(super) rmsnorm_pipeline: wgpu::ComputePipeline,
    pub(super) silu_mul_pipeline: wgpu::ComputePipeline,
    pub(super) rope_pipeline: wgpu::ComputePipeline,
    pub(super) residual_pipeline: wgpu::ComputePipeline,
    pub(super) matmul_bgl: wgpu::BindGroupLayout,
    pub(super) elementwise_bgl: wgpu::BindGroupLayout,

    // Weight buffers (persistent, uploaded once)
    pub(super) weight_buffers: HashMap<String, wgpu::Buffer>,
    pub(super) cpu_biases: HashMap<String, Vec<f32>>,
    pub(super) hidden_buf: wgpu::Buffer,
    pub(super) q_buf: wgpu::Buffer,
    pub(super) k_buf: wgpu::Buffer,
    pub(super) v_buf: wgpu::Buffer,
    pub(super) attn_out_buf: wgpu::Buffer,
    pub(super) ffn_gate_buf: wgpu::Buffer,
    pub(super) ffn_up_buf: wgpu::Buffer,
    pub(super) ffn_out_buf: wgpu::Buffer,
    pub(super) norm_buf: wgpu::Buffer,
    pub(super) staging_buf: wgpu::Buffer,
    pub(super) hidden_dim: u32,
    pub(super) num_heads: u32,
    pub(super) num_kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) intermediate_dim: u32,
    pub(super) bias_add_pipeline: wgpu::ComputePipeline,
    pub(super) bias_add_bgl: wgpu::BindGroupLayout,
    pub(super) rope_bgl: wgpu::BindGroupLayout,
    /// PMAT-361: GPU attention pipeline + per-layer KV cache
    pub(super) attention_pipeline: wgpu::ComputePipeline,
    pub(super) attention_bgl: wgpu::BindGroupLayout,
    pub(super) kv_cache_k: Vec<wgpu::Buffer>, // [num_layers][max_seq * kv_dim]
    pub(super) kv_cache_v: Vec<wgpu::Buffer>,
    pub(super) max_seq_len: u32,
    /// PMAT-363: Fused Q4K dequant+GEMV
    pub(super) q4k_gemv_pipeline: wgpu::ComputePipeline,
    pub(super) q4k_weights: HashMap<String, wgpu::Buffer>,
}
use super::wgsl_shaders::{
    ATTENTION_SHADER, BIAS_ADD_SHADER, Q4K_GEMV_SHADER, RESIDUAL_SHADER, RMSNORM_SHADER,
    ROPE_SHADER, SILU_MUL_SHADER,
};
#[rustfmt::skip]
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
    #[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "buffer_size_safety")]
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

        // PMAT-356: Bias-add pipeline
        let bias_add_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bias_add_bgl"),
            entries: &[bgl_storage(0, false), bgl_storage(1, true), bgl_uniform(2)],
        });
        let bias_add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bias_add"), source: wgpu::ShaderSource::Wgsl(BIAS_ADD_SHADER.into()),
        });
        let bias_add_pipeline = make_pipeline(&bias_add_shader, &bias_add_bgl, "bias_add_pipe");

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

        // PMAT-361: Attention pipeline (5 bindings: q, k_cache, v_cache, output, params)
        let attention_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn_bgl"),
            entries: &[
                bgl_storage(0, true), bgl_storage(1, true), bgl_storage(2, true),
                bgl_storage(3, false), bgl_uniform(4),
            ],
        });
        let attn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention"), source: wgpu::ShaderSource::Wgsl(ATTENTION_SHADER.into()),
        });
        let attention_pipeline = make_pipeline(&attn_shader, &attention_bgl, "attn_pipe");

        // PMAT-363: Q4K fused dequant+GEMV (same BGL as matmul: x, w, y, params)
        let q4k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q4k_gemv"), source: wgpu::ShaderSource::Wgsl(Q4K_GEMV_SHADER.into()),
        });
        let q4k_gemv_pipeline = make_pipeline(&q4k_shader, &matmul_bgl, "q4k_gemv_pipe");

        Self {
            device, queue,
            matmul_pipeline, gemv_pipeline, rmsnorm_pipeline, silu_mul_pipeline,
            rope_pipeline, residual_pipeline, bias_add_pipeline, attention_pipeline,
            q4k_gemv_pipeline, q4k_weights: HashMap::new(),
            matmul_bgl, elementwise_bgl, bias_add_bgl, rope_bgl, attention_bgl,
            kv_cache_k: Vec::new(), kv_cache_v: Vec::new(), max_seq_len: 2048,
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

    /// PMAT-356: Biases stored on BOTH CPU and GPU for GPU-side bias+RoPE.
    /// PMAT-377: Skip GPU upload for buffers > 2 GB (WGPU limit). CPU fallback used.
    pub fn upload_weight(&mut self, name: &str, data: &[f32]) {
        let size_bytes = data.len() * 4;
        if size_bytes > 2_000_000_000 {
            eprintln!("[PMAT-377] Skipping GPU upload for {} ({:.1} GB > 2 GB limit)", name, size_bytes as f64 / 1e9);
            return;
        }
        if name.contains("bias") {
            self.cpu_biases.insert(name.to_string(), data.to_vec());
            // Fall through to ALSO store on GPU for encode_bias_add
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
    #[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "weight_transpose")]
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

    /// PMAT-363: Upload raw Q4K weight bytes (144 bytes per 256-element superblock).
    /// Stored as u32 array for shader access. Used instead of F32 dequant for 4× BW reduction.
    pub fn upload_q4k_weight(&mut self, name: &str, raw_data: &[u8]) {
        use wgpu::util::DeviceExt;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name), contents: raw_data,
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.q4k_weights.insert(name.to_string(), buffer);
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

    /// PMAT-361: Allocate GPU KV cache buffers for all layers.
    pub fn init_kv_cache(&mut self, num_layers: usize) {
        let kv_dim = (self.num_kv_heads * self.head_dim) as usize;
        let max_seq = self.max_seq_len as usize;
        let size = (max_seq * kv_dim * 4) as u64;
        for i in 0..num_layers {
            let k = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("kv_k_{i}")), size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let v = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("kv_v_{i}")), size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.kv_cache_k.push(k);
            self.kv_cache_v.push(v);
        }
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

        // 2. Transformer layers (GPU via forward_layer)
        // Track seq_len per layer from CPU-side cache
        while kv_caches.len() < num_layers { kv_caches.push((Vec::new(), Vec::new())); }
        for layer_idx in 0..num_layers {
            let prefix = format!("layer.{layer_idx}");
            let kv_dim = (self.num_kv_heads * self.head_dim) as usize;
            let seq_len = kv_caches[layer_idx].0.len() / kv_dim.max(1);
            let (ref mut k_cache, ref mut v_cache) = kv_caches[layer_idx];
            self.forward_layer(&mut hidden, &prefix, position, layer_idx, seq_len, k_cache, v_cache)?;
        }

        // 3. Output RMSNorm (CPU — small, not worth GPU dispatch)
        let rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hd as f32 + eps).sqrt();
        for i in 0..hd {
            hidden[i] = (hidden[i] / rms) * output_norm_weight[i];
        }

        // 4. LM head — CPU (weight is [vocab,hidden], matmul shader needs [hidden,vocab])
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            for j in 0..hd { sum += lm_head_weight[v * hd + j] * hidden[j]; }
            logits[v] = sum;
        }
        Ok(logits)
    }

    // forward_layer + encode helpers moved to wgsl_forward_layer.rs (PMAT-357)
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
