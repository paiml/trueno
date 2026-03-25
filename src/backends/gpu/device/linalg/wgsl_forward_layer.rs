//! PMAT-357: Transformer layer execution for WGSL forward pass.
//!
//! Extracted from wgsl_forward.rs for file health management.
//! Contains forward_layer() and GPU encode helpers.

use super::wgsl_forward::WgslForwardPass;

#[rustfmt::skip]
impl WgslForwardPass {
    /// PMAT-325: Execute one transformer layer.
    /// GPU: RMSNorm + QKV GEMV + bias + RoPE → readback → CPU attention → GPU: O proj + FFN.
    /// PMAT-356: Bias applied on GPU before RoPE (they don't commute).
    pub fn forward_layer(
        &self, hidden: &mut [f32], layer_prefix: &str, _position: usize,
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

        // Submit + readback Q/K/V for CPU bias+RoPE+attention
        let q_bytes = (q_dim * 4) as u64;
        let kv_bytes = (kv_dim * 4) as u64;
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
        let mut q_data = vec![0.0f32; q_dim as usize];
        { let slice = q_staging.slice(..q_bytes);
          let (tx, rx) = std::sync::mpsc::channel();
          slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
          self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
          rx.recv().map_err(|e| format!("q recv: {e}"))?.map_err(|e| format!("q map: {e:?}"))?;
          let data = slice.get_mapped_range();
          q_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..q_dim as usize]); }
        q_staging.unmap();
        let mut k_data = vec![0.0f32; kv_dim as usize];
        { let slice = k_staging.slice(..kv_bytes);
          let (tx, rx) = std::sync::mpsc::channel();
          slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
          self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
          rx.recv().map_err(|e| format!("k recv: {e}"))?.map_err(|e| format!("k map: {e:?}"))?;
          let data = slice.get_mapped_range();
          k_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..kv_dim as usize]); }
        k_staging.unmap();
        let mut v_data = vec![0.0f32; kv_dim as usize];
        { let slice = v_staging.slice(..kv_bytes);
          let (tx, rx) = std::sync::mpsc::channel();
          slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
          self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
          rx.recv().map_err(|e| format!("v recv: {e}"))?.map_err(|e| format!("v map: {e:?}"))?;
          let data = slice.get_mapped_range();
          v_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..kv_dim as usize]); }
        v_staging.unmap();
        let head_dim = self.head_dim as usize;

        // CPU bias (PMAT-342)
        if let Some(qb) = self.cpu_biases.get(&format!("{layer_prefix}.q_bias")) {
            for (q, b) in q_data.iter_mut().zip(qb.iter()) { *q += *b; }
        }
        if let Some(kb) = self.cpu_biases.get(&format!("{layer_prefix}.k_bias")) {
            for (k, b) in k_data.iter_mut().zip(kb.iter()) { *k += *b; }
        }
        if let Some(vb) = self.cpu_biases.get(&format!("{layer_prefix}.v_bias")) {
            for (v, b) in v_data.iter_mut().zip(vb.iter()) { *v += *b; }
        }
        // CPU RoPE (PMAT-343, NeoX-style)
        let rope_theta = 1_000_000.0f64;
        let position = _position;
        for h in 0..(self.num_heads as usize) {
            let off = h * head_dim; let half = head_dim / 2;
            for i in 0..half {
                let theta = rope_theta.powf(-((2 * i) as f64) / head_dim as f64);
                let angle = position as f64 * theta;
                let (cos_a, sin_a) = (angle.cos() as f32, angle.sin() as f32);
                let (x0, x1) = (q_data[off + i], q_data[off + i + half]);
                q_data[off + i] = x0 * cos_a - x1 * sin_a;
                q_data[off + i + half] = x0 * sin_a + x1 * cos_a;
            }
        }
        for h in 0..(self.num_kv_heads as usize) {
            let off = h * head_dim; let half = head_dim / 2;
            for i in 0..half {
                let theta = rope_theta.powf(-((2 * i) as f64) / head_dim as f64);
                let angle = position as f64 * theta;
                let (cos_a, sin_a) = (angle.cos() as f32, angle.sin() as f32);
                let (x0, x1) = (k_data[off + i], k_data[off + i + half]);
                k_data[off + i] = x0 * cos_a - x1 * sin_a;
                k_data[off + i + half] = x0 * sin_a + x1 * cos_a;
            }
        }

        // CPU attention (small at M=1, cheaper than GPU dispatch overhead)
        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;
        let num_kv_heads = self.num_kv_heads as usize;
        let kv_dim_usize = kv_dim as usize;

        kv_cache_k.extend_from_slice(&k_data);
        kv_cache_v.extend_from_slice(&v_data);
        let seq_len = kv_cache_k.len() / kv_dim_usize;

        let kv_group = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; q_dim as usize];

        for h in 0..num_heads {
            let kv_h = h / kv_group;
            let q_offset = h * head_dim;
            let mut scores = vec![0.0f32; seq_len];
            for s in 0..seq_len {
                let k_offset = s * kv_dim_usize + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += q_data[q_offset + d] * kv_cache_k[k_offset + d]; }
                scores[s] = dot * scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - max_score).exp(); sum += *s; }
            if sum > 0.0 { for s in scores.iter_mut() { *s /= sum; } }
            let out_offset = h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for s in 0..seq_len {
                    val += scores[s] * kv_cache_v[s * kv_dim_usize + kv_h * head_dim + d];
                }
                attn_out[out_offset + d] = val;
            }
        }

        // Upload attention output, continue with O proj + FFN on GPU
        self.queue.write_buffer(&self.q_buf, 0, bytemuck::cast_slice(&attn_out));
        let mut encoder = self.device.create_command_encoder(&Default::default());

        self.encode_matmul(&mut encoder, &self.q_buf, layer_prefix, "o_proj", &self.attn_out_buf, 1, q_dim, hd);
        self.encode_residual(&mut encoder, &self.hidden_buf, &self.attn_out_buf, &self.ffn_out_buf, hd);

        let ffn_norm_w = self.weight_buffers.get(&format!("{layer_prefix}.ffn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.ffn_norm"))?;
        self.encode_rmsnorm(&mut encoder, &self.ffn_out_buf, ffn_norm_w, &self.norm_buf, hd);

        let inter = self.intermediate_dim;
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "gate_proj", &self.ffn_gate_buf, 1, hd, inter);
        self.encode_matmul(&mut encoder, &self.norm_buf, layer_prefix, "up_proj", &self.ffn_up_buf, 1, hd, inter);
        self.encode_silu_mul(&mut encoder, &self.ffn_gate_buf, &self.ffn_up_buf, &self.attn_out_buf, inter);
        self.encode_matmul(&mut encoder, &self.attn_out_buf, layer_prefix, "down_proj", &self.norm_buf, 1, inter, hd);
        self.encode_residual(&mut encoder, &self.ffn_out_buf, &self.norm_buf, &self.hidden_buf, hd);

        // Readback hidden state
        encoder.copy_buffer_to_buffer(&self.hidden_buf, 0, &self.staging_buf, 0, (hd * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.staging_buf.slice(..(hd as u64 * 4));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().map_err(|e| format!("recv: {e}"))?.map_err(|e| format!("map: {e:?}"))?;
        { let data = slice.get_mapped_range();
          hidden.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..self.hidden_dim as usize]); }
        self.staging_buf.unmap();
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
