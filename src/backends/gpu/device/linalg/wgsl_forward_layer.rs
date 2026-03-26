//! PMAT-357: Transformer layer execution for WGSL forward pass.
//!
//! Extracted from wgsl_forward.rs for file health management.
//! Contains forward_layer() and GPU encode helpers.

use super::wgsl_forward::WgslForwardPass;

#[rustfmt::skip]
impl WgslForwardPass {
    /// PMAT-361: Execute one transformer layer — single submit when GPU KV cache available.
    /// GPU: RMSNorm + QKV GEMV + bias + RoPE + KV append + attention + O proj + FFN.
    #[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "gpu_bias_rope_order")]
    pub fn forward_layer(
        &self, hidden: &mut [f32], layer_prefix: &str, _position: usize,
        layer_idx: usize, seq_len_before: usize,
        kv_cache_k: &mut Vec<f32>, kv_cache_v: &mut Vec<f32>,
    ) -> Result<(), String> {
        let hd = self.hidden_dim;
        self.queue.write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(hidden));
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 1: RMSNorm(hidden → norm_buf)
        let norm_w = self.weight_buffers.get(&format!("{layer_prefix}.attn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.attn_norm"))?;
        self.encode_rmsnorm(&mut encoder, &self.hidden_buf, norm_w, &self.norm_buf, hd);

        // Passes 2-4: Q/K/V projections
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "q_proj", &self.q_buf, 1, hd, q_dim);
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "k_proj", &self.k_buf, 1, hd, kv_dim);
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "v_proj", &self.v_buf, 1, hd, kv_dim);

        // PMAT-358: GPU bias + RoPE (same encoder, before readback)
        self.encode_bias_add(&mut encoder, &self.q_buf, layer_prefix, "q_bias", q_dim);
        self.encode_bias_add(&mut encoder, &self.k_buf, layer_prefix, "k_bias", kv_dim);
        self.encode_bias_add(&mut encoder, &self.v_buf, layer_prefix, "v_bias", kv_dim);
        self.encode_rope(&mut encoder, &self.q_buf, q_dim, _position as u32, self.head_dim);
        self.encode_rope(&mut encoder, &self.k_buf, kv_dim, _position as u32, self.head_dim);

        // PMAT-361: Append K/V to GPU cache, dispatch attention — ALL in same encoder
        let kv_bytes = (kv_dim * 4) as u64;
        let kv_dim_usize = kv_dim as usize;
        let kv_offset = (seq_len_before * kv_dim_usize * 4) as u64;
        // Copy current K/V from intermediate bufs to GPU cache at seq_len_before position
        encoder.copy_buffer_to_buffer(&self.k_buf, 0, &self.kv_cache_k[layer_idx], kv_offset, kv_bytes);
        encoder.copy_buffer_to_buffer(&self.v_buf, 0, &self.kv_cache_v[layer_idx], kv_offset, kv_bytes);
        // Also update CPU-side cache for seq_len tracking
        // (readback K/V for CPU cache happens here — small cost, needed for seq_len bookkeeping)
        let k_stg = self.make_staging(kv_bytes);
        let v_stg = self.make_staging(kv_bytes);
        encoder.copy_buffer_to_buffer(&self.k_buf, 0, &k_stg, 0, kv_bytes);
        encoder.copy_buffer_to_buffer(&self.v_buf, 0, &v_stg, 0, kv_bytes);

        // GPU attention: Q (from q_buf) × K_cache × V_cache → attn_out_buf
        let new_seq_len = (seq_len_before + 1) as u32;
        let kv_group = self.num_heads / self.num_kv_heads;
        let attn_params = [self.num_heads, kv_group, self.head_dim, new_seq_len,
                           kv_dim, 0u32, 0u32, 0u32];
        let attn_params_buf = {
            use wgpu::util::DeviceExt;
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(&attn_params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let attn_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.attention_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.kv_cache_k[layer_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.kv_cache_v[layer_idx].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.attn_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: attn_params_buf.as_entire_binding() },
            ],
        });
        { let mut pass = encoder.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.attention_pipeline);
          pass.set_bind_group(0, &attn_bg, &[]);
          pass.dispatch_workgroups(self.num_heads, 1, 1); }

        // O proj: attn_out → norm_buf (reuse norm_buf as scratch)
        self.encode_matmul(&mut encoder, &self.attn_out_buf, layer_prefix, "o_proj", &self.norm_buf, 1, q_dim, hd);
        // Residual: hidden + o_proj → ffn_out
        self.encode_residual(&mut encoder, &self.hidden_buf, &self.norm_buf, &self.ffn_out_buf, hd);

        let ffn_norm_w = self.weight_buffers.get(&format!("{layer_prefix}.ffn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.ffn_norm"))?;
        self.encode_rmsnorm(&mut encoder, &self.ffn_out_buf, ffn_norm_w, &self.norm_buf, hd);

        let inter = self.intermediate_dim;
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "gate_proj", &self.ffn_gate_buf, 1, hd, inter);
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "up_proj", &self.ffn_up_buf, 1, hd, inter);
        self.encode_silu_mul(&mut encoder, &self.ffn_gate_buf, &self.ffn_up_buf, &self.attn_out_buf, inter);
        self.encode_matmul(&mut encoder, &self.attn_out_buf, layer_prefix, "down_proj", &self.norm_buf, 1, inter, hd);
        self.encode_residual(&mut encoder, &self.ffn_out_buf, &self.norm_buf, &self.hidden_buf, hd);

        // PMAT-361: Single submit — readback hidden + K/V for CPU cache tracking
        encoder.copy_buffer_to_buffer(&self.hidden_buf, 0, &self.staging_buf, 0, (hd * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        // Readback hidden
        let slice = self.staging_buf.slice(..(hd as u64 * 4));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().map_err(|e| format!("recv: {e}"))?.map_err(|e| format!("map: {e:?}"))?;
        { let data = slice.get_mapped_range();
          hidden.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..self.hidden_dim as usize]); }
        self.staging_buf.unmap();
        // Readback K/V for CPU-side seq_len tracking
        let k_data = self.readback_map(&k_stg, kv_dim_usize)?;
        let v_data = self.readback_map(&v_stg, kv_dim_usize)?;
        kv_cache_k.extend_from_slice(&k_data);
        kv_cache_v.extend_from_slice(&v_data);
        Ok(())
    }

    // --- Encode helpers ---

    pub(super) fn encode_rmsnorm(&self, encoder: &mut wgpu::CommandEncoder,
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
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub(super) fn encode_matmul(&self, encoder: &mut wgpu::CommandEncoder,
                     input: &wgpu::Buffer, layer_prefix: &str, proj_name: &str,
                     output: &wgpu::Buffer, m: u32, k: u32, n: u32) {
        let weight_key = format!("{layer_prefix}.{proj_name}");
        // PMAT-365: Q4K fused GEMV — scale extraction fixed
        if m == 1 {
            if let Some(q4k_buf) = self.q4k_weights.get(&weight_key) {
                let params = [n, k, 0u32, 0u32];
                let params_buf = self.make_uniform(&params);
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &self.matmul_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: q4k_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.q4k_gemv_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(n, 1, 1);
                return;
            }
        }
        // Fallback: F32 GEMV/GEMM
        let weight = match self.weight_buffers.get(&weight_key) {
            Some(w) => w, None => return,
        };
        let params = if m == 1 { [n, k, 0u32, 0u32] } else { [m, k, n, 0u32] };
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
            pass.set_pipeline(&self.gemv_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n, 1, 1);
        } else {
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(16), n.div_ceil(16), 1);
        }
    }

    fn encode_bias_add(&self, encoder: &mut wgpu::CommandEncoder,
                       data_buf: &wgpu::Buffer, layer_prefix: &str, bias_name: &str, dim: u32) {
        let key = format!("{layer_prefix}.{bias_name}");
        let bias_buf = match self.weight_buffers.get(&key) { Some(b) => b, None => return };
        let params = [dim, 0u32, 0, 0];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bias_add_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bias_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.bias_add_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dim.div_ceil(256), 1, 1);
    }

    fn encode_rope(&self, encoder: &mut wgpu::CommandEncoder,
                   data: &wgpu::Buffer, dim: u32, position: u32, head_dim: u32) {
        let params = [dim, position, 0u32, head_dim];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.rope_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.rope_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dim.div_ceil(256), 1, 1);
    }

    pub(super) fn encode_silu_mul(&self, encoder: &mut wgpu::CommandEncoder,
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

    pub(super) fn encode_residual(&self, encoder: &mut wgpu::CommandEncoder,
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

    pub(super) fn make_staging(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stg"), size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(super) fn readback_map(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        let bytes = (count * 4) as u64;
        let slice = buf.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().map_err(|e| format!("recv: {e}"))?.map_err(|e| format!("map: {e:?}"))?;
        let mut out = vec![0.0f32; count];
        { let data = slice.get_mapped_range();
          out.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..count]); }
        buf.unmap();
        Ok(out)
    }

    pub(super) fn make_uniform(&self, data: &[u32; 4]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
}
