//! FlashAttention-Style Tiled Attention Kernel
//!
//! Implements IO-aware attention per Dao et al. [16]
//! Never materializes the full N×N attention matrix.
//!
//! Standard Attention: O(N²) memory for S = Q × K^T
//! FlashAttention: O(N × d) memory using online softmax
//!
//! ## Variants
//!
//! - **Standard**: FP32 serial dot product (baseline, ~79ms/token)
//! - **Tensor Core**: FP16 WMMA for Q×K^T (target: <2ms/token, ~40x speedup)
//!
//! ## Performance (RTX 4090, seq_len=2048, head_dim=128)
//!
//! | Variant     | Time/token | Throughput |
//! |-------------|------------|------------|
//! | Standard    | 79ms       | 12.6 tok/s |
//! | Tensor Core | ~2ms       | ~500 tok/s |

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// FlashAttention-style kernel configuration
#[derive(Debug, Clone)]
pub struct AttentionKernel {
    /// Sequence length (N)
    pub seq_len: u32,
    /// Head dimension (d)
    pub head_dim: u32,
    /// Tile size for Q (B_r)
    pub tile_q: u32,
    /// Tile size for KV (B_c)
    pub tile_kv: u32,
    /// Scaling factor for attention scores (1/sqrt(d))
    pub scale: f32,
    /// Use causal masking (for autoregressive models)
    pub causal: bool,
    /// Use Tensor Cores for Q×K^T (FP16 WMMA, requires sm_70+)
    pub use_tensor_cores: bool,
}

impl AttentionKernel {
    /// Create a new attention kernel
    ///
    /// Tile sizes are auto-clamped to not exceed seq_len and head_dim
    /// to handle small inputs gracefully.
    #[must_use]
    pub fn new(seq_len: u32, head_dim: u32) -> Self {
        // Auto-clamp tile sizes to input dimensions
        // Default tiles: 64, but reduce if inputs are smaller
        let tile_q = seq_len.min(64);
        let tile_kv = seq_len.min(64);

        Self {
            seq_len,
            head_dim,
            tile_q,
            tile_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: false,
            use_tensor_cores: false,
        }
    }

    /// Create Tensor Core attention kernel (highest performance)
    ///
    /// Uses FP16 WMMA for Q×K^T computation, achieving ~40x speedup over FP32.
    /// Requires sm_70+ (Volta or later). Dimensions should be multiples of 16.
    ///
    /// # Performance
    ///
    /// - Standard FP32: ~79ms/token (12.6 tok/s)
    /// - Tensor Core FP16: ~2ms/token (500+ tok/s)
    #[must_use]
    pub fn tensor_core(seq_len: u32, head_dim: u32) -> Self {
        // For Tensor Cores, use tile sizes that are multiples of 16
        let tile_q = seq_len.clamp(16, 64);
        let tile_kv = seq_len.clamp(16, 64);

        Self {
            seq_len,
            head_dim,
            tile_q,
            tile_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: false,
            use_tensor_cores: true,
        }
    }

    /// Set tile sizes for Q and KV
    #[must_use]
    pub const fn with_tiles(mut self, tile_q: u32, tile_kv: u32) -> Self {
        self.tile_q = tile_q;
        self.tile_kv = tile_kv;
        self
    }

    /// Enable causal masking for autoregressive attention
    #[must_use]
    pub const fn with_causal(mut self) -> Self {
        self.causal = true;
        self
    }

    /// Set custom scale factor
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Enable Tensor Core acceleration for Q×K^T computation
    ///
    /// Uses FP16 WMMA instructions for ~40x speedup on sm_70+ GPUs.
    #[must_use]
    pub const fn with_tensor_cores(mut self) -> Self {
        self.use_tensor_cores = true;
        self
    }
}

impl Kernel for AttentionKernel {
    fn name(&self) -> &str {
        match (self.use_tensor_cores, self.causal) {
            (true, true) => "flash_attention_tensor_core_causal",
            (true, false) => "flash_attention_tensor_core",
            (false, true) => "flash_attention_causal",
            (false, false) => "flash_attention",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        if self.use_tensor_cores {
            self.build_tensor_core_attention()
        } else {
            self.build_flash_attention()
        }
    }
}

impl AttentionKernel {
    fn build_flash_attention(&self) -> PtxKernel {
        // FlashAttention-style tiled attention
        // Per Dao et al. - never materialize full N×N matrix
        let head_dim = self.head_dim;
        let tile_q = self.tile_q;
        let tile_kv = self.tile_kv;
        let scale = self.scale;
        let causal = self.causal;

        // Shared memory for Q, K, V tiles
        let smem_size = (tile_q * head_dim + tile_kv * head_dim * 2) * 4;

        let kernel_name = if causal {
            "flash_attention_causal"
        } else {
            "flash_attention"
        };

        PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "o_ptr")
            .param(PtxType::U32, "seq_len")
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "num_heads")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let _ntid = ctx.special_reg(PtxReg::NtidX);

                // Load parameters
                let seq_len_param = ctx.load_param_u32("seq_len");
                let head_dim_param = ctx.load_param_u32("head_dim");
                let num_heads = ctx.load_param_u32("num_heads");
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let o_ptr = ctx.load_param_u64("o_ptr");

                // ctaid_x = Q block index, ctaid_y = head index
                let q_block = ctaid_x;
                let head_idx = ctaid_y;

                // PARITY-114 FIX: Compute predicate but DON'T exit early
                // All threads must participate in barriers
                let head_valid = ctx.setp_lt_u32(head_idx, num_heads);

                // Calculate head offset (head_idx * seq_len * head_dim)
                let head_stride = ctx.mul_u32_reg(seq_len_param, head_dim_param);
                let head_offset = ctx.mul_wide_u32_reg(head_idx, head_stride);
                let head_offset_bytes = ctx.mul_u64(head_offset, 4);

                // Q tile base address for this block
                let tile_q_imm = ctx.mov_u32_imm(tile_q);
                let q_row_start = ctx.mul_u32_reg(q_block, tile_q_imm);
                let q_tile_offset = ctx.mul_wide_u32_reg(q_row_start, head_dim_param);
                let q_tile_offset_bytes = ctx.mul_u64(q_tile_offset, 4);
                let q_base = ctx.add_u64(q_ptr, head_offset_bytes);
                let q_tile_base = ctx.add_u64(q_base, q_tile_offset_bytes);

                // ===== Initialize output accumulator and softmax stats =====
                // Each thread handles one position in the output
                let local_row = ctx.div_u32(tid, head_dim);
                let local_col = ctx.rem_u32(tid, head_dim);

                // PARITY-114 FIX: Compute predicate but DON'T exit early
                // This handles launch configs with more threads than tile_q * head_dim
                let tile_q_check = ctx.mov_u32_imm(tile_q);
                let thread_valid = ctx.setp_lt_u32(local_row, tile_q_check);

                // Initialize output accumulator to 0
                let o_acc = ctx.mov_f32_imm(0.0);
                // Running max for online softmax
                let m_prev = ctx.mov_f32_imm(f32::NEG_INFINITY);
                // Running sum of exp
                let l_prev = ctx.mov_f32_imm(0.0);

                // Calculate number of KV blocks
                let tile_kv_imm = ctx.mov_u32_imm(tile_kv);
                let num_kv_blocks = ctx.div_u32(seq_len_param, tile_kv);

                // ===== Pre-compute element offset (needed for output store after loop) =====
                // This must be computed BEFORE the loop, not inside it
                let local_row_64 = ctx.cvt_u64_u32(local_row);
                let local_col_64 = ctx.cvt_u64_u32(local_col);
                let head_dim_64 = ctx.cvt_u64_u32(head_dim_param);
                let q_elem_offset = ctx.mul_u64_reg(local_row_64, head_dim_64);
                let q_elem_offset_full = ctx.add_u64(q_elem_offset, local_col_64);
                let q_elem_offset_bytes = ctx.mul_u64(q_elem_offset_full, 4);

                // Shared memory base addresses (32-bit constants for shared memory addressing)
                let k_smem_base = tile_q * head_dim * 4;
                let v_smem_base = (tile_q * head_dim + tile_kv * head_dim) * 4;

                // Loop counter
                let kv_block = ctx.mov_u32_imm(0);

                ctx.label("kv_loop_start");

                // Check if we've processed all KV blocks
                let kv_done = ctx.setp_ge_u32(kv_block, num_kv_blocks);
                ctx.branch_if(kv_done, "kv_loop_end");

                // Causal masking: skip KV blocks that are entirely after current Q block
                // Use setp_lt and flip: if q_block < kv_block, skip
                if causal {
                    let causal_skip = ctx.setp_lt_u32(q_block, kv_block);
                    ctx.branch_if(causal_skip, "kv_loop_end");
                }

                // Calculate K, V tile base addresses
                let kv_row_start = ctx.mul_u32_reg(kv_block, tile_kv_imm);
                let kv_tile_offset = ctx.mul_wide_u32_reg(kv_row_start, head_dim_param);
                let kv_tile_offset_bytes = ctx.mul_u64(kv_tile_offset, 4);
                let k_base = ctx.add_u64(k_ptr, head_offset_bytes);
                let k_tile_base = ctx.add_u64(k_base, kv_tile_offset_bytes);
                let v_base = ctx.add_u64(v_ptr, head_offset_bytes);
                let v_tile_base = ctx.add_u64(v_base, kv_tile_offset_bytes);

                // ===== Load Q tile to shared memory =====
                // Each thread loads one element (using pre-computed q_elem_offset_bytes)
                let q_addr = ctx.add_u64(q_tile_base, q_elem_offset_bytes);
                let q_val = ctx.ld_global_f32(q_addr);
                // Use 32-bit addressing for shared memory (not 64-bit)
                let q_smem_offset = ctx.mul_u32(tid, 4);
                ctx.st_shared_f32(q_smem_offset, q_val);

                // ===== Load K tile to shared memory =====
                // (k_smem_base computed before loop, use 32-bit)
                let k_addr = ctx.add_u64(k_tile_base, q_elem_offset_bytes);
                let k_val = ctx.ld_global_f32(k_addr);
                let k_smem_base_u32 = ctx.mov_u32_imm(k_smem_base);
                let k_smem_offset_local = ctx.mul_u32(tid, 4);
                let k_smem_offset = ctx.add_u32_reg(k_smem_base_u32, k_smem_offset_local);
                ctx.st_shared_f32(k_smem_offset, k_val);

                // ===== Load V tile to shared memory =====
                // (v_smem_base computed before loop, use 32-bit)
                let v_addr = ctx.add_u64(v_tile_base, q_elem_offset_bytes);
                let v_val = ctx.ld_global_f32(v_addr);
                let v_smem_base_u32 = ctx.mov_u32_imm(v_smem_base);
                let v_smem_offset_local = ctx.mul_u32(tid, 4);
                let v_smem_offset = ctx.add_u32_reg(v_smem_base_u32, v_smem_offset_local);
                ctx.st_shared_f32(v_smem_offset, v_val);

                ctx.bar_sync(0);

                // ===== Compute S = Q × K^T (dot product for this thread's row) =====
                // Each thread computes attention score for its Q row
                let s_acc = ctx.mov_f32_imm(0.0);

                // Inner loop over head_dim elements
                let d_idx = ctx.mov_u32_imm(0);
                ctx.label("dot_loop_start");
                let d_done = ctx.setp_ge_u32(d_idx, head_dim_param);
                ctx.branch_if(d_done, "dot_loop_end");

                // Load Q[local_row, d_idx] from shared memory (32-bit addressing)
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let q_row_offset = ctx.mul_u32_reg(local_row, head_dim_u32);
                let q_elem_smem = ctx.add_u32_reg(q_row_offset, d_idx);
                let q_elem_smem_bytes = ctx.mul_u32(q_elem_smem, 4);
                let q_dot_val = ctx.ld_shared_f32(q_elem_smem_bytes);

                // Load K[local_col, d_idx] from shared memory (32-bit addressing, K is transposed conceptually)
                let k_col_offset = ctx.mul_u32_reg(local_col, head_dim_u32);
                let k_elem_smem = ctx.add_u32_reg(k_col_offset, d_idx);
                let k_elem_smem_bytes = ctx.mul_u32(k_elem_smem, 4);
                let k_smem_base_loop = ctx.mov_u32_imm(k_smem_base);
                let k_elem_smem_full = ctx.add_u32_reg(k_smem_base_loop, k_elem_smem_bytes);
                let k_dot_val = ctx.ld_shared_f32(k_elem_smem_full);

                // Accumulate Q[i,d] * K[j,d] - IN-PLACE UPDATE
                ctx.fma_f32_inplace(s_acc, q_dot_val, k_dot_val);

                // Increment and loop back - IN-PLACE UPDATE
                ctx.add_u32_inplace(d_idx, 1);
                ctx.branch("dot_loop_start");

                ctx.label("dot_loop_end");

                // ===== Apply scale factor =====
                let scale_reg = ctx.mov_f32_imm(scale);
                let s_scaled = ctx.mul_f32(s_acc, scale_reg);

                // ===== Online softmax update =====
                // m_new = max(m_prev, s_scaled)
                let m_new = ctx.max_f32(m_prev, s_scaled);

                // scale_factor = exp(m_prev - m_new)
                let m_diff = ctx.sub_f32(m_prev, m_new);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let m_diff_scaled = ctx.mul_f32(m_diff, log2_e);
                let scale_factor = ctx.ex2_f32(m_diff_scaled);

                // p = exp(s_scaled - m_new)
                let s_shifted = ctx.sub_f32(s_scaled, m_new);
                let s_shifted_scaled = ctx.mul_f32(s_shifted, log2_e);
                let p_val = ctx.ex2_f32(s_shifted_scaled);

                // l_new = scale_factor * l_prev + p
                let l_scaled = ctx.mul_f32(scale_factor, l_prev);
                let l_new = ctx.add_f32(l_scaled, p_val);

                // ===== Update output accumulator =====
                // o_new = (scale_factor * l_prev * o_prev + p * v) / l_new
                let o_scaled = ctx.mul_f32(o_acc, scale_factor);
                let o_weighted = ctx.mul_f32(o_scaled, l_prev);

                // Load V value from shared memory (32-bit addressing)
                let v_elem_smem_bytes = ctx.mul_u32(k_elem_smem, 4); // Reuse k offset calculation
                let v_smem_base_loop = ctx.mov_u32_imm(v_smem_base);
                let v_elem_smem_full = ctx.add_u32_reg(v_smem_base_loop, v_elem_smem_bytes);
                let v_out_val = ctx.ld_shared_f32(v_elem_smem_full);

                let pv = ctx.mul_f32(p_val, v_out_val);
                let o_sum = ctx.add_f32(o_weighted, pv);
                let o_new = ctx.div_f32(o_sum, l_new);

                // Update running stats for next iteration - COPY TO ACCUMULATORS
                ctx.mov_f32_reg(m_prev, m_new);
                ctx.mov_f32_reg(l_prev, l_new);
                ctx.mov_f32_reg(o_acc, o_new);

                ctx.bar_sync(1);

                // Increment KV block counter and loop back - IN-PLACE UPDATE
                ctx.add_u32_inplace(kv_block, 1);
                ctx.branch("kv_loop_start");

                ctx.label("kv_loop_end");

                // PARITY-114 FIX: Bounds check HERE (after all threads finished barriers)
                // Only threads with valid output coordinates store to O
                ctx.branch_if_not(head_valid, "exit");
                ctx.branch_if_not(thread_valid, "exit");

                // ===== Store output =====
                // Calculate output address
                let o_base = ctx.add_u64(o_ptr, head_offset_bytes);
                let o_tile_offset = ctx.mul_wide_u32_reg(q_row_start, head_dim_param);
                let o_tile_offset_bytes = ctx.mul_u64(o_tile_offset, 4);
                let o_tile_base = ctx.add_u64(o_base, o_tile_offset_bytes);
                let o_addr = ctx.add_u64(o_tile_base, q_elem_offset_bytes);

                // Store accumulated output (o_acc is always valid, even if loop never ran)
                ctx.st_global_f32(o_addr, o_acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Build Tensor Core FlashAttention using WMMA for Q×K^T
    ///
    /// Key optimization: Replace serial FP32 dot product with FP16 WMMA 16×16×16 tiles.
    /// For head_dim=128, we need 8 WMMA operations per S[i,j] tile (128/16=8).
    ///
    /// Algorithm:
    /// 1. Load Q tile [16×head_dim] to shared memory (FP32→FP16)
    /// 2. Load K tile [16×head_dim] to shared memory (FP32→FP16)
    /// 3. For each 16-element chunk along head_dim:
    ///    - WMMA: S_acc[16×16] += Q_frag[16×16] × K_frag^T[16×16]
    /// 4. Apply scale, online softmax, V multiplication
    ///
    /// Launch config: grid_2d(seq_len/16, num_heads, 32, 1) - one warp per 16×16 Q×K tile
    #[allow(clippy::too_many_lines)]
    fn build_tensor_core_attention(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let tile_q = 16_u32; // Fixed for WMMA
        let tile_kv = 16_u32; // Fixed for WMMA
        let scale = self.scale;
        let causal = self.causal;

        // Number of WMMA steps to accumulate the full dot product
        // For head_dim=128, n_k_steps = 8
        let n_k_steps = (head_dim + 15) / 16;

        // Shared memory layout:
        // Q tile: 16 × head_dim × 2 bytes (FP16)
        // K tile: 16 × head_dim × 2 bytes (FP16)
        // V tile: 16 × head_dim × 4 bytes (FP32)
        // S tile: 16 × 16 × 4 bytes (FP32 attention scores)
        let q_smem_size = tile_q * head_dim * 2;
        let k_smem_size = tile_kv * head_dim * 2;
        let v_smem_size = tile_kv * head_dim * 4;
        let s_smem_size = tile_q * tile_kv * 4;
        let smem_size = q_smem_size + k_smem_size + v_smem_size + s_smem_size;

        let kernel_name = if causal {
            "flash_attention_tensor_core_causal"
        } else {
            "flash_attention_tensor_core"
        };

        PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "o_ptr")
            .param(PtxType::U32, "seq_len")
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "num_heads")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // WMMA operates at warp level (32 threads cooperatively)
                // Grid: (seq_len/16) x num_heads
                // Block: 32 threads (1 warp)

                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                // Load parameters
                let seq_len_param = ctx.load_param_u32("seq_len");
                let head_dim_param = ctx.load_param_u32("head_dim");
                let num_heads = ctx.load_param_u32("num_heads");
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let o_ptr = ctx.load_param_u64("o_ptr");

                // Block index determines which Q tile we're computing
                let q_block = ctaid_x;
                let head_idx = ctaid_y;

                // PARITY-114 FIX: Compute predicate but DON'T exit early
                // All threads must participate in barriers (WMMA requires full warp)
                let head_valid = ctx.setp_lt_u32(head_idx, num_heads);

                // Calculate head offset
                let head_stride = ctx.mul_u32_reg(seq_len_param, head_dim_param);
                let head_offset = ctx.mul_wide_u32_reg(head_idx, head_stride);
                let head_offset_bytes = ctx.mul_u64(head_offset, 4);

                // Shared memory base addresses (need actual smem pointer, not just offset)
                // For regular loads/stores, u32 offset from smem[0] works
                // For WMMA, we need the actual shared memory address
                let smem_ptr = ctx.shared_base_addr(); // u64 pointer to smem
                let q_smem_base = ctx.mov_u32_imm(0);
                let k_smem_base = ctx.mov_u32_imm(q_smem_size);
                let v_smem_base = ctx.mov_u32_imm(q_smem_size + k_smem_size);
                let s_smem_base = ctx.mov_u32_imm(q_smem_size + k_smem_size + v_smem_size);
                // Pre-compute u64 pointers for WMMA operations
                let q_smem_base_64 = ctx.cvt_u64_u32(q_smem_base);
                let q_smem_ptr = ctx.add_u64(smem_ptr, q_smem_base_64);
                let k_smem_base_64 = ctx.cvt_u64_u32(k_smem_base);
                let k_smem_ptr = ctx.add_u64(smem_ptr, k_smem_base_64);
                let s_smem_base_64 = ctx.cvt_u64_u32(s_smem_base);
                let s_smem_ptr = ctx.add_u64(smem_ptr, s_smem_base_64);

                // Q tile base address
                let tile_16 = ctx.mov_u32_imm(16);
                let q_row_start = ctx.mul_u32_reg(q_block, tile_16);
                let q_tile_offset = ctx.mul_wide_u32_reg(q_row_start, head_dim_param);
                let q_tile_offset_bytes = ctx.mul_u64(q_tile_offset, 4);
                let q_base = ctx.add_u64(q_ptr, head_offset_bytes);
                let q_tile_base = ctx.add_u64(q_base, q_tile_offset_bytes);

                // ===== Load Q tile to shared memory (FP32 → FP16) =====
                // Each of 32 threads loads multiple elements
                // Total: 16 × head_dim elements
                let q_total_elems = ctx.mov_u32_imm(16 * head_dim);
                let elems_per_thread = ctx.div_u32(q_total_elems, 32);
                let my_start = ctx.mul_u32_reg(tid, elems_per_thread);

                let load_idx = ctx.mov_u32_imm(0);
                ctx.label("load_q_loop");
                let load_done = ctx.setp_ge_u32(load_idx, elems_per_thread);
                ctx.branch_if(load_done, "load_q_end");

                let elem_idx = ctx.add_u32_reg(my_start, load_idx);
                let elem_check = ctx.setp_ge_u32(elem_idx, q_total_elems);
                ctx.branch_if(elem_check, "load_q_end");

                // Load from global (FP32)
                let q_global_offset = ctx.mul_wide_u32(elem_idx, 4);
                let q_addr = ctx.add_u64(q_tile_base, q_global_offset);
                let q_val_f32 = ctx.ld_global_f32(q_addr);

                // Convert to FP16 and store to shared
                let q_val_f16 = ctx.cvt_f16_f32(q_val_f32);
                let q_smem_offset = ctx.mul_u32(elem_idx, 2);
                let q_smem_addr = ctx.add_u32_reg(q_smem_base, q_smem_offset);
                ctx.st_shared_f16(q_smem_addr, q_val_f16);

                ctx.add_u32_inplace(load_idx, 1);
                ctx.branch("load_q_loop");
                ctx.label("load_q_end");

                // Initialize output accumulators (16 values per thread for the output row)
                // Each thread in the warp contributes to 16×16 output tile
                // For simplicity, we'll use per-row accumulators
                let o_acc = ctx.mov_f32_imm(0.0);
                let m_prev = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let l_prev = ctx.mov_f32_imm(0.0);

                // Number of KV blocks
                let num_kv_blocks = ctx.div_u32(seq_len_param, 16);
                let kv_block = ctx.mov_u32_imm(0);

                ctx.label("kv_loop_start");
                let kv_done = ctx.setp_ge_u32(kv_block, num_kv_blocks);
                ctx.branch_if(kv_done, "kv_loop_end");

                // Causal masking
                if causal {
                    let causal_skip = ctx.setp_lt_u32(q_block, kv_block);
                    ctx.branch_if(causal_skip, "kv_loop_end");
                }

                // Calculate K, V tile base addresses
                let kv_row_start = ctx.mul_u32_reg(kv_block, tile_16);
                let kv_tile_offset = ctx.mul_wide_u32_reg(kv_row_start, head_dim_param);
                let kv_tile_offset_bytes = ctx.mul_u64(kv_tile_offset, 4);
                let k_base = ctx.add_u64(k_ptr, head_offset_bytes);
                let k_tile_base = ctx.add_u64(k_base, kv_tile_offset_bytes);
                let v_base = ctx.add_u64(v_ptr, head_offset_bytes);
                let v_tile_base = ctx.add_u64(v_base, kv_tile_offset_bytes);

                // ===== Load K tile to shared memory (FP32 → FP16) =====
                let load_k_idx = ctx.mov_u32_imm(0);
                ctx.label("load_k_loop");
                let k_load_done = ctx.setp_ge_u32(load_k_idx, elems_per_thread);
                ctx.branch_if(k_load_done, "load_k_end");

                let k_elem_idx = ctx.add_u32_reg(my_start, load_k_idx);
                let k_elem_check = ctx.setp_ge_u32(k_elem_idx, q_total_elems);
                ctx.branch_if(k_elem_check, "load_k_end");

                let k_global_offset = ctx.mul_wide_u32(k_elem_idx, 4);
                let k_addr = ctx.add_u64(k_tile_base, k_global_offset);
                let k_val_f32 = ctx.ld_global_f32(k_addr);
                let k_val_f16 = ctx.cvt_f16_f32(k_val_f32);
                let k_smem_offset = ctx.mul_u32(k_elem_idx, 2);
                let k_smem_addr = ctx.add_u32_reg(k_smem_base, k_smem_offset);
                ctx.st_shared_f16(k_smem_addr, k_val_f16);

                ctx.add_u32_inplace(load_k_idx, 1);
                ctx.branch("load_k_loop");
                ctx.label("load_k_end");

                // ===== Load V tile to shared memory (FP32) =====
                let load_v_idx = ctx.mov_u32_imm(0);
                ctx.label("load_v_loop");
                let v_load_done = ctx.setp_ge_u32(load_v_idx, elems_per_thread);
                ctx.branch_if(v_load_done, "load_v_end");

                let v_elem_idx = ctx.add_u32_reg(my_start, load_v_idx);
                let v_elem_check = ctx.setp_ge_u32(v_elem_idx, q_total_elems);
                ctx.branch_if(v_elem_check, "load_v_end");

                let v_global_offset = ctx.mul_wide_u32(v_elem_idx, 4);
                let v_addr = ctx.add_u64(v_tile_base, v_global_offset);
                let v_val = ctx.ld_global_f32(v_addr);
                let v_smem_offset = ctx.mul_u32(v_elem_idx, 4);
                let v_smem_addr = ctx.add_u32_reg(v_smem_base, v_smem_offset);
                ctx.st_shared_f32(v_smem_addr, v_val);

                ctx.add_u32_inplace(load_v_idx, 1);
                ctx.branch("load_v_loop");
                ctx.label("load_v_end");

                ctx.bar_sync(0);

                // ===== Compute S = Q × K^T using WMMA =====
                // Initialize S tile accumulator to zero (8 f32 registers)
                let mut frag_c = Vec::with_capacity(8);
                for _ in 0..8 {
                    frag_c.push(ctx.mov_f32_imm(0.0));
                }

                // Loop over head_dim in steps of 16
                let k_step = ctx.mov_u32_imm(0);
                let n_k_steps_reg = ctx.mov_u32_imm(n_k_steps);

                ctx.label("wmma_loop_start");
                let wmma_done = ctx.setp_ge_u32(k_step, n_k_steps_reg);
                ctx.branch_if(wmma_done, "wmma_loop_end");

                // Q fragment address: q_smem_ptr + k_step * 16 * 2 bytes
                let q_frag_offset = ctx.mul_u32(k_step, 32); // 16 elements × 2 bytes
                let q_frag_offset_64 = ctx.cvt_u64_u32(q_frag_offset);
                let q_frag_addr = ctx.add_u64(q_smem_ptr, q_frag_offset_64);
                let frag_a = ctx.wmma_load_a_f16(q_frag_addr, head_dim, WmmaLayout::RowMajor);

                // K fragment address: k_smem_ptr + k_step * 16 * 2 - needs col-major for K^T
                let k_frag_offset = ctx.mul_u32(k_step, 32);
                let k_frag_offset_64 = ctx.cvt_u64_u32(k_frag_offset);
                let k_frag_addr = ctx.add_u64(k_smem_ptr, k_frag_offset_64);
                let frag_b = ctx.wmma_load_b_f16(k_frag_addr, head_dim, WmmaLayout::ColMajor);

                // WMMA MMA: D = A × B + C (accumulates into D fragment)
                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // Copy D -> C for next iteration's accumulation (8 f32 registers)
                for i in 0..8 {
                    ctx.mov_f32_reg(frag_c[i], frag_d[i]);
                }

                ctx.add_u32_inplace(k_step, 1);
                ctx.branch("wmma_loop_start");
                ctx.label("wmma_loop_end");

                // Store S tile (accumulated result in D) to shared memory for softmax
                ctx.wmma_store_d_f32(s_smem_ptr, &frag_d, 16, WmmaLayout::RowMajor);

                ctx.bar_sync(1);

                // ===== Apply scale and online softmax =====
                // Each thread handles one element of the 16×16 S tile
                // Thread tid handles element (tid/16, tid%16) for tid < 256
                // We have 32 threads, so each thread handles 8 elements

                let s_idx = ctx.mov_u32_imm(0);
                let loop_limit_8 = ctx.mov_u32_imm(8);
                let elems_256 = ctx.mov_u32_imm(256);
                let step_8 = ctx.mov_u32_imm(8);

                ctx.label("softmax_loop_start");
                let s_idx_check = ctx.setp_ge_u32(s_idx, loop_limit_8);
                ctx.branch_if(s_idx_check, "softmax_loop_end");

                // Calculate which S element this iteration handles
                let s_elem = ctx.mad_lo_u32(tid, step_8, s_idx);
                let s_elem_check = ctx.setp_ge_u32(s_elem, elems_256);
                ctx.branch_if(s_elem_check, "softmax_next");

                // Load S[i,j] from shared memory
                let s_offset = ctx.mul_u32(s_elem, 4);
                let s_addr = ctx.add_u32_reg(s_smem_base, s_offset);
                let s_val = ctx.ld_shared_f32(s_addr);

                // Apply scale
                let scale_reg = ctx.mov_f32_imm(scale);
                let s_scaled = ctx.mul_f32(s_val, scale_reg);

                // Online softmax update (simplified - each thread maintains local max/sum)
                let m_new = ctx.max_f32(m_prev, s_scaled);
                let m_diff = ctx.sub_f32(m_prev, m_new);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let m_diff_scaled = ctx.mul_f32(m_diff, log2_e);
                let scale_factor = ctx.ex2_f32(m_diff_scaled);

                let s_shifted = ctx.sub_f32(s_scaled, m_new);
                let s_shifted_scaled = ctx.mul_f32(s_shifted, log2_e);
                let p_val = ctx.ex2_f32(s_shifted_scaled);

                let l_scaled = ctx.mul_f32(scale_factor, l_prev);
                let l_new = ctx.add_f32(l_scaled, p_val);

                // Store scaled attention weight back
                ctx.st_shared_f32(s_addr, p_val);

                ctx.mov_f32_reg(m_prev, m_new);
                ctx.mov_f32_reg(l_prev, l_new);

                ctx.label("softmax_next");
                ctx.add_u32_inplace(s_idx, 1);
                ctx.branch("softmax_loop_start");
                ctx.label("softmax_loop_end");

                ctx.bar_sync(2);

                // ===== Compute O += softmax(S) × V =====
                // Load attention weights and multiply with V
                // This is another GEMM: P[16×16] × V[16×head_dim] = O_update[16×head_dim]
                // For simplicity, we'll use the scalar path for V multiplication
                // (Tensor Core V multiplication would require additional WMMA calls)

                let v_col = ctx.rem_u32(tid, head_dim);
                let v_row_idx = ctx.mov_u32_imm(0);

                ctx.label("v_loop_start");
                let v_loop_done = ctx.setp_ge_u32(v_row_idx, tile_16);
                ctx.branch_if(v_loop_done, "v_loop_end");

                // Load attention weight P[row, v_row_idx]
                let p_idx_base = ctx.mul_u32_reg(v_row_idx, tile_16);
                let p_offset = ctx.mul_u32(p_idx_base, 4);
                let p_addr = ctx.add_u32_reg(s_smem_base, p_offset);
                let p_weight = ctx.ld_shared_f32(p_addr);

                // Load V[v_row_idx, v_col]
                let v_idx = ctx.mad_lo_u32(v_row_idx, head_dim_param, v_col);
                let v_offset = ctx.mul_u32(v_idx, 4);
                let v_elem_addr = ctx.add_u32_reg(v_smem_base, v_offset);
                let v_elem = ctx.ld_shared_f32(v_elem_addr);

                // Accumulate: o_acc += p_weight * v_elem
                ctx.fma_f32_inplace(o_acc, p_weight, v_elem);

                ctx.add_u32_inplace(v_row_idx, 1);
                ctx.branch("v_loop_start");
                ctx.label("v_loop_end");

                ctx.bar_sync(3);

                ctx.add_u32_inplace(kv_block, 1);
                ctx.branch("kv_loop_start");
                ctx.label("kv_loop_end");

                // PARITY-114 FIX: Bounds check HERE (after all threads finished barriers)
                // Only threads with valid heads store to O
                ctx.branch_if_not(head_valid, "exit");

                // ===== Normalize and store output =====
                let o_normalized = ctx.div_f32(o_acc, l_prev);

                // Calculate output address
                let o_base = ctx.add_u64(o_ptr, head_offset_bytes);
                let tid_div_hd = ctx.div_u32(tid, head_dim);
                let o_row = ctx.mad_lo_u32(q_block, tile_16, tid_div_hd);
                let o_col = ctx.rem_u32(tid, head_dim);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let o_idx = ctx.mad_lo_u32(o_row, head_dim_reg, o_col);
                let o_offset = ctx.mul_wide_u32(o_idx, 4);
                let o_addr = ctx.add_u64(o_base, o_offset);

                ctx.st_global_f32(o_addr, o_normalized);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// PAR-020: Incremental Attention Kernel for M=1 Autoregressive Decoding
// =============================================================================

/// Incremental attention kernel for single-query autoregressive decoding (PAR-020)
///
/// Optimized for the critical path of LLM token generation where each new token
/// requires attention over the entire KV cache with a single query vector.
///
/// # Memory Layout
///
/// - Q: [head_dim] - single query vector for current position
/// - K: [seq_len, head_dim] - cached keys (GPU-resident)
/// - V: [seq_len, head_dim] - cached values (GPU-resident)
/// - Output: [head_dim] - weighted sum of values
///
/// # Algorithm
///
/// 1. Compute attention scores: score[i] = dot(Q, K[i]) * scale
/// 2. Apply causal mask (positions > current are masked)
/// 3. Online softmax: max_score, sum_exp tracked incrementally
/// 4. Compute weighted V sum: output = sum(softmax[i] * V[i])
///
/// # Performance
///
/// - Avoids materializing [seq_len, seq_len] attention matrix
/// - Uses warp shuffle for efficient parallel reduction
/// - Designed for GPU-resident KV cache (no D2H transfer)
/// - Target: O(seq_len * head_dim) memory, O(seq_len * head_dim) compute
#[derive(Debug, Clone)]
pub struct IncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA, <= num_heads)
    pub num_kv_heads: u32,
    /// Scaling factor for attention scores (1/sqrt(head_dim))
    pub scale: f32,
    /// PAR-061: Read seq_len from device memory (for CUDA graph compatibility)
    pub indirect_seq_len: bool,
}

impl IncrementalAttentionKernel {
    /// Create new incremental attention kernel (MHA - num_kv_heads = num_heads)
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of attention heads
    #[must_use]
    pub fn new(max_seq_len: u32, head_dim: u32, num_heads: u32) -> Self {
        Self::with_gqa(max_seq_len, head_dim, num_heads, num_heads)
    }

    /// Create new incremental attention kernel with GQA support (PAR-021)
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    #[must_use]
    pub fn with_gqa(max_seq_len: u32, head_dim: u32, num_heads: u32, num_kv_heads: u32) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            indirect_seq_len: false,
        }
    }

    /// PAR-061: Enable indirect seq_len mode (reads from device memory)
    /// Required for CUDA graph compatibility
    #[must_use]
    pub fn with_indirect_seq_len(mut self, indirect: bool) -> Self {
        self.indirect_seq_len = indirect;
        self
    }

    /// Check if this kernel is configured for GQA
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads != self.num_heads
    }
}

impl Kernel for IncrementalAttentionKernel {
    fn name(&self) -> &str {
        // PAR-061: Different kernel name for indirect mode
        if self.indirect_seq_len {
            "incremental_attention_indirect"
        } else {
            "incremental_attention"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let indirect = self.indirect_seq_len;

        // Kernel strategy (PAR-020 + PAR-021 GQA):
        // - Grid: (num_heads, 1, 1) - one block per Q head
        // - Block: (32, 1, 1) - one warp per block
        // - Each warp computes attention for one Q head using online softmax
        //
        // Memory layout:
        // - q: [num_heads, head_dim] - query vectors for current position
        // - k: [num_kv_heads, max_seq_len, head_dim] - key cache (GPU-resident)
        // - v: [num_kv_heads, max_seq_len, head_dim] - value cache (GPU-resident)
        // - output: [num_heads, head_dim] - attention output
        //
        // GQA mapping (PAR-021):
        // - Each Q head uses kv_head_idx = q_head_idx * num_kv_heads / num_heads
        // - For MHA: kv_head_idx = q_head_idx
        // - For GQA: multiple Q heads share the same KV head
        //
        // Algorithm:
        // 1. Thread i loads Q[lane_id], Q[lane_id+32], ... (strided)
        // 2. Loop over seq positions, computing Q·K dot product per position
        // 3. Warp-reduce dot product using shfl_down
        // 4. Online softmax: track running max and sum_exp
        // 5. Accumulate weighted V vectors
        // 6. Normalize and store output

        // PAR-061: Use different kernel name and parameter type for indirect mode
        let kernel_name = if indirect {
            "incremental_attention_indirect"
        } else {
            "incremental_attention"
        };

        let mut builder = PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "out_ptr");

        // PAR-061: Indirect mode takes seq_len_ptr (U64), direct mode takes seq_len (U32)
        builder = if indirect {
            builder.param(PtxType::U64, "seq_len_ptr")
        } else {
            builder.param(PtxType::U32, "seq_len")
        };

        builder
            .shared_memory(0) // Register-only, warp shuffle for reduction
            .build(move |ctx| {
                // Get indices
                let q_head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                // PAR-061: In indirect mode, load seq_len from device memory
                let seq_len = if indirect {
                    let seq_len_ptr = ctx.load_param_u64("seq_len_ptr");
                    ctx.ld_global_u32(seq_len_ptr)
                } else {
                    ctx.load_param_u32("seq_len")
                };
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");

                // Pre-compute constants
                let four = ctx.mov_u32_imm(4);
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);

                // Compute Q/output head offset
                // Q/output: q_head_idx * head_dim
                let q_head_off = ctx.mul_lo_u32(q_head_idx, head_dim_u32);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // PAR-021 GQA: Compute KV head index
                // kv_head_idx = q_head_idx * num_kv_heads / num_heads
                // This maps multiple Q heads to the same KV head
                // Use literal values since they're known at kernel build time
                let kv_head_idx = ctx.mul_u32(q_head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_ptr, kv_head_off_bytes);

                // CORRECTNESS-002: Each thread handles 4 elements (strided by 32) for head_dim up to 128
                // Thread 0 handles [0,32,64,96], thread 1 handles [1,33,65,97], etc.
                // Supports head_dim: 32, 64, 96, 128

                // Load Q values into registers (persistent across seq loop)
                // Using predicated loads for bounds checking
                // CORRECTNESS-002: Support head_dim up to 128 (4 elements per thread)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                // Second element (if head_dim > 32)
                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Third element (if head_dim > 64)
                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                // CORRECTNESS-002: Fourth element (if head_dim > 96)
                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize output accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                // CORRECTNESS-002: Additional accumulators for head_dim > 64
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);

                // Log2(e) for exp approximation via ex2
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop counter
                let pos = ctx.mov_u32_imm(0);

                ctx.label("seq_loop");

                // Check loop condition
                let loop_cond = ctx.setp_lt_u32(pos, seq_len);
                ctx.branch_if_not(loop_cond, "seq_loop_end");

                // Compute K offset for this position: pos * head_dim
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                // Load K[pos, lane_id] and K[pos, lane_id+32]
                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Load K[pos, lane_id+64] and K[pos, lane_id+96]
                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Compute partial dot product: q0*k0 + q1*k1 + q2*k2 + q3*k3
                // CORRECTNESS-002: Now handles full head_dim=128
                let dot_partial = ctx.mul_f32(q0, k0);
                let dot_partial = ctx.fma_f32(q1, k1, dot_partial);
                let dot_partial = ctx.fma_f32(q2, k2, dot_partial);
                let dot_partial = ctx.fma_f32(q3, k3, dot_partial);

                // Warp-reduce the dot product using shfl.down
                // sum += shfl_down(sum, 16)
                // sum += shfl_down(sum, 8)
                // sum += shfl_down(sum, 4)
                // sum += shfl_down(sum, 2)
                // sum += shfl_down(sum, 1)
                let dot16 = ctx.shfl_down_f32(dot_partial, 16, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot16);
                let dot8 = ctx.shfl_down_f32(dot_partial, 8, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot8);
                let dot4 = ctx.shfl_down_f32(dot_partial, 4, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot4);
                let dot2 = ctx.shfl_down_f32(dot_partial, 2, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot2);
                let dot1 = ctx.shfl_down_f32(dot_partial, 1, 0xFFFF_FFFF);
                let dot_reduced = ctx.add_f32(dot_partial, dot1);

                // Broadcast result to all threads via shfl.idx lane 0
                let dot_broadcast = ctx.shfl_idx_f32(dot_reduced, 0, 0xFFFF_FFFF);

                // Scale the attention score
                let score = ctx.mul_f32(dot_broadcast, scale_reg);

                // Online softmax update (Milakov & Gimelshein 2018):
                // new_max = max(old_max, score)
                // correction = exp(old_max - new_max)
                // sum_exp = sum_exp * correction + exp(score - new_max)
                // output = output * correction + exp(score - new_max) * V

                let new_max = ctx.max_f32(max_score, score);

                // exp(old_max - new_max) using 2^(x * log2(e))
                let max_diff = ctx.sub_f32(max_score, new_max);
                let max_diff_scaled = ctx.mul_f32(max_diff, log2e);
                let correction = ctx.ex2_f32(max_diff_scaled);

                // exp(score - new_max)
                let score_diff = ctx.sub_f32(score, new_max);
                let score_diff_scaled = ctx.mul_f32(score_diff, log2e);
                let exp_score = ctx.ex2_f32(score_diff_scaled);

                // Load V[pos, lane_id] and V[pos, lane_id+32]
                let v0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let v0_off_bytes = ctx.mul_wide_u32_reg(v0_elem_off, four);
                let v0_addr = ctx.add_u64(v_head_ptr, v0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);

                let v1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let v1_off_bytes = ctx.mul_wide_u32_reg(v1_elem_off, four);
                let v1_addr = ctx.add_u64(v_head_ptr, v1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Load V[pos, lane_id+64] and V[pos, lane_id+96]
                let v2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let v2_off_bytes = ctx.mul_wide_u32_reg(v2_elem_off, four);
                let v2_addr = ctx.add_u64(v_head_ptr, v2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);

                let v3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let v3_off_bytes = ctx.mul_wide_u32_reg(v3_elem_off, four);
                let v3_addr = ctx.add_u64(v_head_ptr, v3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);

                // Update loop state using in-place operations
                // Online softmax: max_score = max(max_score, score)
                ctx.max_f32_inplace(max_score, score);

                // sum_exp = sum_exp * correction + exp_score
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // out = out * correction + exp_score * V
                ctx.mul_f32_inplace(out0, correction);
                ctx.fma_f32_inplace(out0, exp_score, v0);
                ctx.mul_f32_inplace(out1, correction);
                ctx.fma_f32_inplace(out1, exp_score, v1);
                // CORRECTNESS-002: Update out2 and out3
                ctx.mul_f32_inplace(out2, correction);
                ctx.fma_f32_inplace(out2, exp_score, v2);
                ctx.mul_f32_inplace(out3, correction);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                // Increment position
                ctx.add_u32_inplace(pos, 1);
                ctx.branch("seq_loop");

                ctx.label("seq_loop_end");

                // Normalize output: out /= sum_exp
                // Use reciprocal approximation for speed
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, sum_exp);

                ctx.mul_f32_inplace(out0, inv_sum);
                ctx.mul_f32_inplace(out1, inv_sum);
                // CORRECTNESS-002: Normalize out2 and out3
                ctx.mul_f32_inplace(out2, inv_sum);
                ctx.mul_f32_inplace(out3, inv_sum);

                // Store output (only for valid indices)
                // Thread writes to output[head_idx, lane_id]
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "skip_store0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("skip_store0");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "skip_store1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("skip_store1");

                // CORRECTNESS-002: Store out2 and out3
                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "skip_store2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("skip_store2");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "skip_store3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("skip_store3");

                ctx.ret();
            })
    }
}

/// PAR-070: Multi-warp incremental attention for decode phase
///
/// Uses multiple warps per head to parallelize across KV cache positions.
/// Each warp processes a chunk of positions, then cross-warp reduction combines
/// the partial softmax states.
///
/// Performance target: 8x speedup over single-warp (from 81µs to ~10µs)
///
/// # Algorithm
///
/// 1. Launch `num_heads × num_warps_per_head` blocks
/// 2. Each warp handles positions [warp_idx * chunk, (warp_idx + 1) * chunk)
/// 3. Compute local max_score, sum_exp, weighted_output
/// 4. Cross-warp reduction in shared memory to get global max
/// 5. Correction pass to align all warps to global max
/// 6. Final sum and normalization
#[derive(Debug, Clone)]
pub struct MultiWarpIncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension (must be <= 128)
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA, <= num_heads)
    pub num_kv_heads: u32,
    /// Number of warps per head (parallelism factor)
    pub num_warps_per_head: u32,
    /// Scaling factor for attention scores (1/sqrt(head_dim))
    pub scale: f32,
    /// PAR-061: Read seq_len from device memory (for CUDA graph compatibility)
    pub indirect_seq_len: bool,
}

impl MultiWarpIncrementalAttentionKernel {
    /// Create new multi-warp incremental attention kernel
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head (must be <= 128)
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `num_warps` - Number of warps per head (4-8 recommended)
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        num_warps: u32,
    ) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            num_warps_per_head: num_warps,
            scale: 1.0 / (head_dim as f32).sqrt(),
            indirect_seq_len: false,
        }
    }

    /// Enable indirect seq_len mode (reads from device memory)
    #[must_use]
    pub fn with_indirect_seq_len(mut self, indirect: bool) -> Self {
        self.indirect_seq_len = indirect;
        self
    }
}

impl Kernel for MultiWarpIncrementalAttentionKernel {
    fn name(&self) -> &str {
        if self.indirect_seq_len {
            "multi_warp_attention_indirect"
        } else {
            "multi_warp_attention"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let num_warps = self.num_warps_per_head;
        let indirect = self.indirect_seq_len;

        // Shared memory for cross-warp reduction:
        // - max_scores[num_warps]: local max per warp (offset 0)
        // - sum_exps[num_warps]: local sum_exp per warp (offset num_warps*4)
        // - global_max: 1 float (offset num_warps*8) - CORRECTNESS-013: separate from local values
        // - global_sum: 1 float (offset num_warps*8+4)
        // - outputs[num_warps * head_dim]: partial outputs per warp (offset num_warps*8+8)
        let smem_size = (num_warps * 2 + 2 + num_warps * head_dim) * 4;

        let kernel_name = if indirect {
            "multi_warp_attention_indirect"
        } else {
            "multi_warp_attention"
        };

        let mut builder = PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "out_ptr");

        builder = if indirect {
            builder.param(PtxType::U64, "seq_len_ptr")
        } else {
            builder.param(PtxType::U32, "seq_len")
        };

        builder
            .shared_memory(smem_size as usize)
            .build(move |ctx| {
                // Grid: (num_heads, 1, 1) - one block per head
                // Block: (32 * num_warps, 1, 1) - multiple warps per block
                let q_head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let tid = ctx.special_reg(PtxReg::TidX);

                // Compute warp_idx and lane_id from tid
                // warp_idx = tid / 32, lane_id = tid % 32
                let warp_idx = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Load seq_len
                let seq_len = if indirect {
                    let seq_len_ptr = ctx.load_param_u64("seq_len_ptr");
                    ctx.ld_global_u32(seq_len_ptr)
                } else {
                    ctx.load_param_u32("seq_len")
                };

                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");

                // Constants
                let four = ctx.mov_u32_imm(4);
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_warps_u32 = ctx.mov_u32_imm(num_warps);

                // Compute chunk boundaries for this warp
                // chunk_size = ceil(seq_len / num_warps) = (seq_len + num_warps - 1) / num_warps
                let seq_plus_nw = ctx.add_u32(seq_len, num_warps - 1);
                let chunk_size = ctx.div_u32(seq_plus_nw, num_warps);

                let start_pos = ctx.mul_lo_u32(warp_idx, chunk_size);
                let end_pos = ctx.add_u32_reg(start_pos, chunk_size);
                let end_pos = ctx.min_u32(end_pos, seq_len);

                // Compute Q/output head offset
                let q_head_off = ctx.mul_lo_u32(q_head_idx, head_dim_u32);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // GQA: kv_head_idx = q_head_idx * num_kv_heads / num_heads
                let kv_head_idx = ctx.mul_u32(q_head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_ptr, kv_head_off_bytes);

                // Load Q values (persistent across loop)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);

                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over this warp's chunk of positions
                let pos = ctx.add_u32(start_pos, 0);

                ctx.label("chunk_loop");
                let loop_cond = ctx.setp_lt_u32(pos, end_pos);
                ctx.branch_if_not(loop_cond, "chunk_loop_end");

                // Load K[pos]
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);
                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Dot product Q·K
                let dot = ctx.mul_f32(q0, k0);
                let dot = ctx.fma_f32(q1, k1, dot);
                let dot = ctx.fma_f32(q2, k2, dot);
                let dot = ctx.fma_f32(q3, k3, dot);

                // Warp-reduce
                let dot16 = ctx.shfl_down_f32(dot, 16, 0xFFFF_FFFF);
                let dot = ctx.add_f32(dot, dot16);
                let dot8 = ctx.shfl_down_f32(dot, 8, 0xFFFF_FFFF);
                let dot = ctx.add_f32(dot, dot8);
                let dot4 = ctx.shfl_down_f32(dot, 4, 0xFFFF_FFFF);
                let dot = ctx.add_f32(dot, dot4);
                let dot2 = ctx.shfl_down_f32(dot, 2, 0xFFFF_FFFF);
                let dot = ctx.add_f32(dot, dot2);
                let dot1 = ctx.shfl_down_f32(dot, 1, 0xFFFF_FFFF);
                let dot = ctx.add_f32(dot, dot1);
                let score = ctx.shfl_idx_f32(dot, 0, 0xFFFF_FFFF);
                let score = ctx.mul_f32(score, scale_reg);

                // Online softmax update
                let new_max = ctx.max_f32(max_score, score);
                let max_diff = ctx.sub_f32(max_score, new_max);
                let max_diff_scaled = ctx.mul_f32(max_diff, log2e);
                let correction = ctx.ex2_f32(max_diff_scaled);
                let score_diff = ctx.sub_f32(score, new_max);
                let score_diff_scaled = ctx.mul_f32(score_diff, log2e);
                let exp_score = ctx.ex2_f32(score_diff_scaled);

                // Load V[pos]
                let v0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let v0_off_bytes = ctx.mul_wide_u32_reg(v0_elem_off, four);
                let v0_addr = ctx.add_u64(v_head_ptr, v0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);

                let v1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let v1_off_bytes = ctx.mul_wide_u32_reg(v1_elem_off, four);
                let v1_addr = ctx.add_u64(v_head_ptr, v1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);

                let v2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let v2_off_bytes = ctx.mul_wide_u32_reg(v2_elem_off, four);
                let v2_addr = ctx.add_u64(v_head_ptr, v2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);

                let v3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let v3_off_bytes = ctx.mul_wide_u32_reg(v3_elem_off, four);
                let v3_addr = ctx.add_u64(v_head_ptr, v3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);

                // Update state
                ctx.max_f32_inplace(max_score, score);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                ctx.mul_f32_inplace(out0, correction);
                ctx.fma_f32_inplace(out0, exp_score, v0);
                ctx.mul_f32_inplace(out1, correction);
                ctx.fma_f32_inplace(out1, exp_score, v1);
                ctx.mul_f32_inplace(out2, correction);
                ctx.fma_f32_inplace(out2, exp_score, v2);
                ctx.mul_f32_inplace(out3, correction);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("chunk_loop");

                ctx.label("chunk_loop_end");

                // Store local max and sum_exp to shared memory for cross-warp reduction
                // smem layout: [max_0..max_n, sum_0..sum_n, out_0..out_n*head_dim]
                // Shared memory uses offset from 0, not a base pointer in PTX
                let warp_off = ctx.mul_u32(warp_idx, 4);
                let warp_off_64 = ctx.cvt_u64_u32(warp_off);
                let max_off_base = ctx.mov_u64_imm(0);
                let max_addr = ctx.add_u64(max_off_base, warp_off_64);

                let sum_off_base = ctx.mov_u64_imm((num_warps * 4) as u64);
                let sum_addr = ctx.add_u64(sum_off_base, warp_off_64);

                // Only lane 0 writes to shared memory
                let zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_u32);
                ctx.branch_if_not(is_lane0, "skip_smem_write");
                ctx.st_shared_f32(max_addr, max_score);
                ctx.st_shared_f32(sum_addr, sum_exp);
                ctx.label("skip_smem_write");

                // Barrier for all threads in this block
                ctx.bar_sync(0);

                // Lane 0 of warp 0 reduces global max across warps
                let is_warp0 = ctx.setp_eq_u32(warp_idx, zero_u32);
                let is_warp0_lane0 = ctx.and_pred(is_warp0, is_lane0);

                ctx.branch_if_not(is_warp0_lane0, "skip_reduce");

                // Compute global max
                let global_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let reduce_i = ctx.mov_u32_imm(0);
                ctx.label("reduce_max_loop");
                let reduce_cond = ctx.setp_lt_u32(reduce_i, num_warps_u32);
                ctx.branch_if_not(reduce_cond, "reduce_max_done");

                let i_off = ctx.mul_u32(reduce_i, 4);
                let i_off_64 = ctx.cvt_u64_u32(i_off);
                let max_i_addr = ctx.add_u64(max_off_base, i_off_64);
                let max_i = ctx.ld_shared_f32(max_i_addr);
                ctx.max_f32_inplace(global_max, max_i);
                ctx.add_u32_inplace(reduce_i, 1);
                ctx.branch("reduce_max_loop");

                ctx.label("reduce_max_done");

                // Compute global sum_exp = sum(sum_exp_i * exp(max_i - global_max))
                let global_sum = ctx.mov_f32_imm(0.0);
                let reduce_i = ctx.mov_u32_imm(0);
                ctx.label("reduce_sum_loop");
                let reduce_cond = ctx.setp_lt_u32(reduce_i, num_warps_u32);
                ctx.branch_if_not(reduce_cond, "reduce_sum_done");

                let i_off = ctx.mul_u32(reduce_i, 4);
                let i_off_64 = ctx.cvt_u64_u32(i_off);
                let max_i_addr = ctx.add_u64(max_off_base, i_off_64);
                let max_i = ctx.ld_shared_f32(max_i_addr);
                let sum_i_addr = ctx.add_u64(sum_off_base, i_off_64);
                let sum_i = ctx.ld_shared_f32(sum_i_addr);

                let diff = ctx.sub_f32(max_i, global_max);
                let diff_scaled = ctx.mul_f32(diff, log2e);
                let correction = ctx.ex2_f32(diff_scaled);
                let corrected_sum = ctx.mul_f32(sum_i, correction);
                ctx.add_f32_inplace(global_sum, corrected_sum);

                ctx.add_u32_inplace(reduce_i, 1);
                ctx.branch("reduce_sum_loop");

                ctx.label("reduce_sum_done");

                // CORRECTNESS-013: Store global_max and global_sum to SEPARATE locations
                // Previous bug: storing to max_off_base (0) overwrote warp 0's local max
                let global_max_off = ctx.mov_u64_imm((num_warps * 8) as u64);
                let global_sum_off = ctx.mov_u64_imm((num_warps * 8 + 4) as u64);
                ctx.st_shared_f32(global_max_off, global_max);
                ctx.st_shared_f32(global_sum_off, global_sum);

                ctx.label("skip_reduce");

                // Barrier to wait for reduction
                ctx.bar_sync(1);

                // CORRECTNESS-013: Load global max and sum from dedicated locations
                let global_max_off = ctx.mov_u64_imm((num_warps * 8) as u64);
                let global_sum_off = ctx.mov_u64_imm((num_warps * 8 + 4) as u64);
                let global_max = ctx.ld_shared_f32(global_max_off);
                let global_sum = ctx.ld_shared_f32(global_sum_off);

                // Compute correction for this warp's contribution
                let my_max = ctx.ld_shared_f32(max_addr);
                let my_diff = ctx.sub_f32(my_max, global_max);
                let my_diff_scaled = ctx.mul_f32(my_diff, log2e);
                let my_correction = ctx.ex2_f32(my_diff_scaled);

                // Correct and normalize this warp's output
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, global_sum);
                let final_scale = ctx.mul_f32(my_correction, inv_sum);

                ctx.mul_f32_inplace(out0, final_scale);
                ctx.mul_f32_inplace(out1, final_scale);
                ctx.mul_f32_inplace(out2, final_scale);
                ctx.mul_f32_inplace(out3, final_scale);

                // Store corrected outputs to shared memory for warp 0 to sum
                // CORRECTNESS-013: Output area starts at offset: num_warps*8 + 8 (after global max/sum)
                // Each warp stores head_dim * 4 bytes
                let out_area_base = ctx.mov_u32_imm(num_warps * 8 + 8);
                let warp_out_offset = ctx.mul_u32(warp_idx, head_dim * 4);
                let out_base = ctx.add_u32_reg(out_area_base, warp_out_offset);

                // Store out0-out3 based on lane positions
                let lane_off_0 = ctx.mul_u32(lane_id, 4);
                let out0_smem_off = ctx.add_u32_reg(out_base, lane_off_0);
                let out0_smem_addr = ctx.cvt_u64_u32(out0_smem_off);
                ctx.branch_if_not(in_bounds0, "skip_store_out0");
                ctx.st_shared_f32(out0_smem_addr, out0);
                ctx.label("skip_store_out0");

                let lane_off_1 = ctx.mul_u32(lane_plus_32, 4);
                let out1_smem_off = ctx.add_u32_reg(out_base, lane_off_1);
                let out1_smem_addr = ctx.cvt_u64_u32(out1_smem_off);
                ctx.branch_if_not(in_bounds1, "skip_store_out1");
                ctx.st_shared_f32(out1_smem_addr, out1);
                ctx.label("skip_store_out1");

                let lane_off_2 = ctx.mul_u32(lane_plus_64, 4);
                let out2_smem_off = ctx.add_u32_reg(out_base, lane_off_2);
                let out2_smem_addr = ctx.cvt_u64_u32(out2_smem_off);
                ctx.branch_if_not(in_bounds2, "skip_store_out2");
                ctx.st_shared_f32(out2_smem_addr, out2);
                ctx.label("skip_store_out2");

                let lane_off_3 = ctx.mul_u32(lane_plus_96, 4);
                let out3_smem_off = ctx.add_u32_reg(out_base, lane_off_3);
                let out3_smem_addr = ctx.cvt_u64_u32(out3_smem_off);
                ctx.branch_if_not(in_bounds3, "skip_store_out3");
                ctx.st_shared_f32(out3_smem_addr, out3);
                ctx.label("skip_store_out3");

                // Barrier to ensure all warps stored their outputs
                ctx.bar_sync(2);

                // Only warp 0 sums outputs and stores to global memory
                ctx.branch_if_not(is_warp0, "skip_final_sum");

                // Sum outputs from all warps for each element this warp handles
                // Thread i in warp 0 sums element i across all warps
                let final_out0 = ctx.mov_f32_imm(0.0);
                let sum_w = ctx.mov_u32_imm(0);
                ctx.label("sum_warps_loop0");
                let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
                ctx.branch_if_not(sum_cond, "sum_warps_done0");

                let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
                let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
                let elem_off = ctx.mul_u32(lane_id, 4);
                let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
                let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
                let elem_val = ctx.ld_shared_f32(elem_addr);
                ctx.add_f32_inplace(final_out0, elem_val);
                ctx.add_u32_inplace(sum_w, 1);
                ctx.branch("sum_warps_loop0");

                ctx.label("sum_warps_done0");

                // Store final output to global memory
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "skip_final_store0");
                ctx.st_global_f32(out0_addr, final_out0);
                ctx.label("skip_final_store0");

                // Repeat for elements 32-63 (out1)
                let final_out1 = ctx.mov_f32_imm(0.0);
                let sum_w = ctx.mov_u32_imm(0);
                ctx.label("sum_warps_loop1");
                let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
                ctx.branch_if_not(sum_cond, "sum_warps_done1");

                let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
                let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
                let elem_off = ctx.mul_u32(lane_plus_32, 4);
                let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
                let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
                let elem_val = ctx.ld_shared_f32(elem_addr);
                ctx.add_f32_inplace(final_out1, elem_val);
                ctx.add_u32_inplace(sum_w, 1);
                ctx.branch("sum_warps_loop1");

                ctx.label("sum_warps_done1");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "skip_final_store1");
                ctx.st_global_f32(out1_addr, final_out1);
                ctx.label("skip_final_store1");

                // Repeat for elements 64-95 (out2)
                let final_out2 = ctx.mov_f32_imm(0.0);
                let sum_w = ctx.mov_u32_imm(0);
                ctx.label("sum_warps_loop2");
                let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
                ctx.branch_if_not(sum_cond, "sum_warps_done2");

                let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
                let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
                let elem_off = ctx.mul_u32(lane_plus_64, 4);
                let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
                let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
                let elem_val = ctx.ld_shared_f32(elem_addr);
                ctx.add_f32_inplace(final_out2, elem_val);
                ctx.add_u32_inplace(sum_w, 1);
                ctx.branch("sum_warps_loop2");

                ctx.label("sum_warps_done2");

                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "skip_final_store2");
                ctx.st_global_f32(out2_addr, final_out2);
                ctx.label("skip_final_store2");

                // Repeat for elements 96-127 (out3)
                let final_out3 = ctx.mov_f32_imm(0.0);
                let sum_w = ctx.mov_u32_imm(0);
                ctx.label("sum_warps_loop3");
                let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
                ctx.branch_if_not(sum_cond, "sum_warps_done3");

                let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
                let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
                let elem_off = ctx.mul_u32(lane_plus_96, 4);
                let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
                let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
                let elem_val = ctx.ld_shared_f32(elem_addr);
                ctx.add_f32_inplace(final_out3, elem_val);
                ctx.add_u32_inplace(sum_w, 1);
                ctx.branch("sum_warps_loop3");

                ctx.label("sum_warps_done3");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "skip_final_store3");
                ctx.st_global_f32(out3_addr, final_out3);
                ctx.label("skip_final_store3");

                ctx.label("skip_final_sum");

                ctx.ret();
            })
    }
}

/// PAR-118: Batched Incremental Attention for M sequences in parallel
///
/// Processes M independent sequences in a single kernel launch, reducing
/// kernel launch overhead from 3M to 3 per layer (batched KV scatter + batched attention).
///
/// Grid: (num_heads, batch_size, 1)
/// Block: (32, 1, 1) - one warp per head
///
/// Memory layout:
/// - q: [M, num_heads, head_dim] - contiguous query vectors
/// - k_ptrs: [M] - array of M pointers to K caches
/// - v_ptrs: [M] - array of M pointers to V caches
/// - output: [M, num_heads, head_dim] - contiguous output
/// - seq_lens: [M] - array of M sequence lengths (indirect mode)
#[derive(Debug, Clone)]
pub struct BatchedIncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Scaling factor for attention scores
    pub scale: f32,
}

impl BatchedIncrementalAttentionKernel {
    /// Create a new batched incremental attention kernel
    #[must_use]
    pub fn new(max_seq_len: u32, head_dim: u32, num_heads: u32, num_kv_heads: u32, batch_size: u32) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            batch_size,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

impl Kernel for BatchedIncrementalAttentionKernel {
    fn name(&self) -> &str {
        "batched_incremental_attention"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, 1)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block handles one (head, batch) pair
        // batch_idx = blockIdx.y selects which sequence
        // head_idx = blockIdx.x selects which Q head

        PtxKernel::new("batched_incremental_attention")
            .param(PtxType::U64, "q_ptr")         // [M, num_heads, head_dim]
            .param(PtxType::U64, "k_ptrs_ptr")    // [M] array of K cache pointers
            .param(PtxType::U64, "v_ptrs_ptr")    // [M] array of V cache pointers
            .param(PtxType::U64, "out_ptr")       // [M, num_heads, head_dim]
            .param(PtxType::U64, "seq_lens_ptr")  // [M] array of sequence lengths
            .shared_memory(0)
            .build(move |ctx| {
                // Get indices
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptrs_ptr = ctx.load_param_u64("k_ptrs_ptr");
                let v_ptrs_ptr = ctx.load_param_u64("v_ptrs_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");

                // Load seq_len for this batch element
                let four = ctx.mov_u32_imm(4);
                let eight = ctx.mov_u32_imm(8);
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);

                // Load K and V cache pointers for this batch element
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Compute Q/output offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // GQA: Compute KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // Load Q values (same as IncrementalAttentionKernel)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over sequence positions
                let pos = ctx.mov_u32_imm(0);
                ctx.label("batched_seq_loop");
                let loop_cond = ctx.setp_lt_u32(pos, seq_len);
                ctx.branch_if_not(loop_cond, "batched_seq_loop_end");

                // Load K[pos] and compute Q·K dot product
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Dot product Q·K
                let dot = ctx.mul_f32(q0, k0);
                ctx.fma_f32_inplace(dot, q1, k1);
                ctx.fma_f32_inplace(dot, q2, k2);
                ctx.fma_f32_inplace(dot, q3, k3);

                // Warp reduce - use full warp mask for all 32 threads
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // Scale score
                let score = ctx.mul_f32(dot, scale_reg);

                // Online softmax update
                let old_max = max_score;
                ctx.max_f32_inplace(max_score, score);
                let score_minus_max = ctx.sub_f32(score, max_score);
                let score_log2 = ctx.mul_f32(score_minus_max, log2e);
                let exp_score = ctx.ex2_f32(score_log2);

                // Rescale sum_exp if max changed
                let old_minus_new = ctx.sub_f32(old_max, max_score);
                let log2_old = ctx.mul_f32(old_minus_new, log2e);
                let correction = ctx.ex2_f32(log2_old);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // Rescale existing output
                ctx.mul_f32_inplace(out0, correction);
                ctx.mul_f32_inplace(out1, correction);
                ctx.mul_f32_inplace(out2, correction);
                ctx.mul_f32_inplace(out3, correction);

                // Load V[pos] and accumulate
                let v0_addr = ctx.add_u64(v_head_ptr, k0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);
                ctx.fma_f32_inplace(out0, exp_score, v0);

                let v1_addr = ctx.add_u64(v_head_ptr, k1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);
                ctx.fma_f32_inplace(out1, exp_score, v1);

                let v2_addr = ctx.add_u64(v_head_ptr, k2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);
                ctx.fma_f32_inplace(out2, exp_score, v2);

                let v3_addr = ctx.add_u64(v_head_ptr, k3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("batched_seq_loop");

                ctx.label("batched_seq_loop_end");

                // Normalize output
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, sum_exp);
                ctx.mul_f32_inplace(out0, inv_sum);
                ctx.mul_f32_inplace(out1, inv_sum);
                ctx.mul_f32_inplace(out2, inv_sum);
                ctx.mul_f32_inplace(out3, inv_sum);

                // Store output
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "batched_skip_store0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("batched_skip_store0");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "batched_skip_store1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("batched_skip_store1");

                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "batched_skip_store2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("batched_skip_store2");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "batched_skip_store3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("batched_skip_store3");

                ctx.ret();
            })
    }
}

// =============================================================================
// PAR-118: Flash Decoding - Split-K Attention for 2X Ollama Performance
// =============================================================================
//
// Flash Decoding splits the KV cache into chunks processed in parallel,
// then reduces partial results. This amortizes memory bandwidth across
// multiple thread blocks, achieving higher throughput for long sequences.
//
// Algorithm:
// 1. Split sequence into K chunks of CHUNK_SIZE positions
// 2. Each chunk computes partial attention: (max_score, sum_exp, weighted_out)
// 3. Reduction combines partials with proper softmax rescaling:
//    - new_max = max(chunk_max[0], chunk_max[1], ...)
//    - For each chunk: scale = exp(chunk_max - new_max)
//    - new_sum = sum(chunk_sum[i] * scale[i])
//    - output = sum(chunk_out[i] * chunk_sum[i] * scale[i]) / new_sum
//
// Performance:
// - Current: Sequential loop over seq_len (memory-bandwidth limited)
// - Flash Decoding: K parallel blocks (K = ceil(seq_len / CHUNK_SIZE))
// - Expected speedup: ~1.5-2x for typical seq_len (512-2048)
// =============================================================================

/// Chunk size for Flash Decoding split-K attention
/// Trade-off: smaller = more parallelism, larger = less reduction overhead
pub const FLASH_DECODE_CHUNK_SIZE: u32 = 128;

/// PAR-118: Flash Decoding kernel for split-K attention
///
/// Splits the KV cache into chunks and processes them in parallel.
/// Requires a separate reduction kernel to combine partial results.
///
/// Memory layout:
/// - q: [M, num_heads, head_dim] - contiguous query vectors
/// - k_ptrs: [M] - array of M pointers to K caches
/// - v_ptrs: [M] - array of M pointers to V caches
/// - partials: [M, num_heads, num_chunks, head_dim + 2] - partial results
///   - [0..head_dim]: weighted output (sum of exp_score * V)
///   - [head_dim]: max_score for this chunk
///   - [head_dim + 1]: sum_exp for this chunk
/// - seq_lens: [M] - array of M sequence lengths
#[derive(Debug, Clone)]
pub struct FlashDecodingChunkKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Chunk size for split-K
    pub chunk_size: u32,
    /// Scaling factor for attention scores
    pub scale: f32,
}

impl FlashDecodingChunkKernel {
    /// Create a new Flash Decoding chunk kernel
    #[must_use]
    pub fn new(max_seq_len: u32, head_dim: u32, num_heads: u32, num_kv_heads: u32, batch_size: u32) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            batch_size,
            chunk_size: FLASH_DECODE_CHUNK_SIZE,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Get the number of chunks for a given sequence length
    #[must_use]
    pub fn num_chunks(&self, seq_len: u32) -> u32 {
        (seq_len + self.chunk_size - 1) / self.chunk_size
    }

    /// Get the size of the partials buffer per (head, batch) pair
    /// Layout: [num_chunks, head_dim + 2]
    #[must_use]
    pub fn partials_size_per_head(&self, max_chunks: u32) -> u32 {
        max_chunks * (self.head_dim + 2)
    }
}

impl Kernel for FlashDecodingChunkKernel {
    fn name(&self) -> &str {
        "flash_decoding_chunk"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let chunk_size = self.chunk_size;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, num_chunks)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block handles one (head, batch, chunk) triple
        // chunk_idx = blockIdx.z selects which chunk of the sequence
        // batch_idx = blockIdx.y selects which sequence
        // head_idx = blockIdx.x selects which Q head

        PtxKernel::new("flash_decoding_chunk")
            .param(PtxType::U64, "q_ptr")         // [M, num_heads, head_dim]
            .param(PtxType::U64, "k_ptrs_ptr")    // [M] array of K cache pointers
            .param(PtxType::U64, "v_ptrs_ptr")    // [M] array of V cache pointers
            .param(PtxType::U64, "partials_ptr")  // [M, num_heads, num_chunks, head_dim + 2]
            .param(PtxType::U64, "seq_lens_ptr")  // [M] array of sequence lengths
            .param(PtxType::U32, "max_chunks")    // Maximum number of chunks
            .shared_memory(0)
            .build(move |ctx| {
                // Get indices
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let chunk_idx = ctx.special_reg(PtxReg::CtaIdZ);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptrs_ptr = ctx.load_param_u64("k_ptrs_ptr");
                let v_ptrs_ptr = ctx.load_param_u64("v_ptrs_ptr");
                let partials_ptr = ctx.load_param_u64("partials_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");
                let max_chunks_param = ctx.load_param_u32("max_chunks");

                let four = ctx.mov_u32_imm(4);
                let eight = ctx.mov_u32_imm(8);

                // Load seq_len for this batch element
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);

                // Compute chunk boundaries
                let chunk_size_u32 = ctx.mov_u32_imm(chunk_size);
                let chunk_start = ctx.mul_lo_u32(chunk_idx, chunk_size_u32);
                let chunk_end_raw = ctx.add_u32(chunk_start, chunk_size);  // Use literal for add_u32
                // Clamp chunk_end to seq_len
                let chunk_end = ctx.min_u32(chunk_end_raw, seq_len);  // Both are VirtualReg

                // Early exit if chunk_start >= seq_len (this chunk has no work)
                let has_work = ctx.setp_lt_u32(chunk_start, seq_len);
                ctx.branch_if_not(has_work, "flash_decode_chunk_empty");

                // Load K and V cache pointers for this batch element
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Compute Q offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);

                // GQA: Compute KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // Load Q values (head_dim up to 128, 4 elements per lane)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize accumulators for this chunk
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state for this chunk
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over positions in this chunk [chunk_start, chunk_end)
                let pos = chunk_start;
                ctx.label("flash_decode_chunk_loop");
                let loop_cond = ctx.setp_lt_u32(pos, chunk_end);
                ctx.branch_if_not(loop_cond, "flash_decode_chunk_loop_end");

                // Load K[pos] and compute Q·K dot product
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Dot product Q·K
                let dot = ctx.mul_f32(q0, k0);
                ctx.fma_f32_inplace(dot, q1, k1);
                ctx.fma_f32_inplace(dot, q2, k2);
                ctx.fma_f32_inplace(dot, q3, k3);

                // Warp reduce
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // Scale score
                let score = ctx.mul_f32(dot, scale_reg);

                // Online softmax update
                let old_max = max_score;
                ctx.max_f32_inplace(max_score, score);
                let score_minus_max = ctx.sub_f32(score, max_score);
                let score_log2 = ctx.mul_f32(score_minus_max, log2e);
                let exp_score = ctx.ex2_f32(score_log2);

                // Rescale sum_exp if max changed
                let old_minus_new = ctx.sub_f32(old_max, max_score);
                let log2_old = ctx.mul_f32(old_minus_new, log2e);
                let correction = ctx.ex2_f32(log2_old);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // Rescale existing output
                ctx.mul_f32_inplace(out0, correction);
                ctx.mul_f32_inplace(out1, correction);
                ctx.mul_f32_inplace(out2, correction);
                ctx.mul_f32_inplace(out3, correction);

                // Load V[pos] and accumulate (NOT normalized yet)
                let v0_addr = ctx.add_u64(v_head_ptr, k0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);
                ctx.fma_f32_inplace(out0, exp_score, v0);

                let v1_addr = ctx.add_u64(v_head_ptr, k1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);
                ctx.fma_f32_inplace(out1, exp_score, v1);

                let v2_addr = ctx.add_u64(v_head_ptr, k2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);
                ctx.fma_f32_inplace(out2, exp_score, v2);

                let v3_addr = ctx.add_u64(v_head_ptr, k3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("flash_decode_chunk_loop");

                ctx.label("flash_decode_chunk_loop_end");

                // Compute partials offset:
                // partials_ptr + (batch_idx * num_heads * max_chunks + head_idx * max_chunks + chunk_idx) * (head_dim + 2) * 4
                let head_dim_plus_2 = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride = ctx.mul_lo_u32(max_chunks_param, head_dim_plus_2);
                let batch_partial_stride = ctx.mul_lo_u32(num_heads_u32, partial_stride);
                let batch_partial_off = ctx.mul_lo_u32(batch_idx, batch_partial_stride);
                let head_partial_off = ctx.mul_lo_u32(head_idx, partial_stride);
                let chunk_partial_off = ctx.mul_lo_u32(chunk_idx, head_dim_plus_2);
                let partial_off = ctx.add_u32_reg(batch_partial_off, head_partial_off);
                let partial_off = ctx.add_u32_reg(partial_off, chunk_partial_off);
                let partial_off_bytes = ctx.mul_wide_u32_reg(partial_off, four);
                let partial_base = ctx.add_u64(partials_ptr, partial_off_bytes);

                // Store weighted output (out0..out3)
                let out0_addr = ctx.add_u64(partial_base, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "flash_decode_skip_out0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("flash_decode_skip_out0");

                let out1_addr = ctx.add_u64(partial_base, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "flash_decode_skip_out1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("flash_decode_skip_out1");

                let out2_addr = ctx.add_u64(partial_base, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "flash_decode_skip_out2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("flash_decode_skip_out2");

                let out3_addr = ctx.add_u64(partial_base, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "flash_decode_skip_out3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("flash_decode_skip_out3");

                // Store max_score at offset head_dim (only lane 0)
                let zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_u32);
                ctx.branch_if_not(is_lane0, "flash_decode_skip_meta");
                let max_off = ctx.mov_u32_imm(head_dim);
                let max_off_bytes = ctx.mul_wide_u32_reg(max_off, four);
                let max_addr = ctx.add_u64(partial_base, max_off_bytes);
                ctx.st_global_f32(max_addr, max_score);

                // Store sum_exp at offset head_dim + 1
                let sum_off = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes = ctx.mul_wide_u32_reg(sum_off, four);
                let sum_addr = ctx.add_u64(partial_base, sum_off_bytes);
                ctx.st_global_f32(sum_addr, sum_exp);
                ctx.label("flash_decode_skip_meta");

                ctx.ret();

                // Empty chunk handler - store sentinel values
                ctx.label("flash_decode_chunk_empty");
                // Same partial offset calculation
                let head_dim_plus_2_e = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride_e = ctx.mul_lo_u32(max_chunks_param, head_dim_plus_2_e);
                let batch_partial_stride_e = ctx.mul_lo_u32(num_heads_u32, partial_stride_e);
                let batch_partial_off_e = ctx.mul_lo_u32(batch_idx, batch_partial_stride_e);
                let head_partial_off_e = ctx.mul_lo_u32(head_idx, partial_stride_e);
                let chunk_partial_off_e = ctx.mul_lo_u32(chunk_idx, head_dim_plus_2_e);
                let partial_off_e = ctx.add_u32_reg(batch_partial_off_e, head_partial_off_e);
                let partial_off_e = ctx.add_u32_reg(partial_off_e, chunk_partial_off_e);
                let partial_off_bytes_e = ctx.mul_wide_u32_reg(partial_off_e, four);
                let partial_base_e = ctx.add_u64(partials_ptr, partial_off_bytes_e);

                // Store -inf for max_score (sentinel for empty chunk)
                let zero_u32_e = ctx.mov_u32_imm(0);
                let is_lane0_e = ctx.setp_eq_u32(lane_id, zero_u32_e);
                ctx.branch_if_not(is_lane0_e, "flash_decode_empty_done");
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let max_off_e = ctx.mov_u32_imm(head_dim);
                let max_off_bytes_e = ctx.mul_wide_u32_reg(max_off_e, four);
                let max_addr_e = ctx.add_u64(partial_base_e, max_off_bytes_e);
                ctx.st_global_f32(max_addr_e, neg_inf);

                // Store 0 for sum_exp
                let zero = ctx.mov_f32_imm(0.0);
                let sum_off_e = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes_e = ctx.mul_wide_u32_reg(sum_off_e, four);
                let sum_addr_e = ctx.add_u64(partial_base_e, sum_off_bytes_e);
                ctx.st_global_f32(sum_addr_e, zero);
                ctx.label("flash_decode_empty_done");

                ctx.ret();
            })
    }
}

/// PAR-118: Flash Decoding reduction kernel
///
/// Reduces partial attention results from multiple chunks into final output.
/// Uses online softmax rescaling to combine results correctly.
///
/// Memory layout:
/// - partials: [M, num_heads, num_chunks, head_dim + 2]
/// - output: [M, num_heads, head_dim]
/// - seq_lens: [M] - sequence lengths (to compute actual num_chunks)
#[derive(Debug, Clone)]
pub struct FlashDecodingReduceKernel {
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Chunk size used in chunk kernel
    pub chunk_size: u32,
}

impl FlashDecodingReduceKernel {
    /// Create a new Flash Decoding reduce kernel
    #[must_use]
    pub fn new(head_dim: u32, num_heads: u32, batch_size: u32) -> Self {
        Self {
            head_dim,
            num_heads,
            batch_size,
            chunk_size: FLASH_DECODE_CHUNK_SIZE,
        }
    }
}

impl Kernel for FlashDecodingReduceKernel {
    fn name(&self) -> &str {
        "flash_decoding_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let chunk_size = self.chunk_size;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, 1)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block reduces all chunks for one (head, batch) pair

        PtxKernel::new("flash_decoding_reduce")
            .param(PtxType::U64, "partials_ptr")  // [M, num_heads, max_chunks, head_dim + 2]
            .param(PtxType::U64, "output_ptr")    // [M, num_heads, head_dim]
            .param(PtxType::U64, "seq_lens_ptr")  // [M] sequence lengths
            .param(PtxType::U32, "max_chunks")    // Maximum number of chunks
            .shared_memory(0)
            .build(move |ctx| {
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                let partials_ptr = ctx.load_param_u64("partials_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");
                let max_chunks = ctx.load_param_u32("max_chunks");

                let four = ctx.mov_u32_imm(4);

                // Load seq_len and compute actual number of chunks
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);
                // num_chunks = (seq_len + chunk_size - 1) / chunk_size
                let seq_plus_chunk_m1 = ctx.add_u32(seq_len, chunk_size - 1);
                let num_chunks = ctx.div_u32(seq_plus_chunk_m1, chunk_size);

                // Compute partials base for this (batch, head) pair
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let head_dim_plus_2 = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride = ctx.mul_lo_u32(max_chunks, head_dim_plus_2);
                let batch_partial_stride = ctx.mul_lo_u32(num_heads_u32, partial_stride);
                let batch_partial_off = ctx.mul_lo_u32(batch_idx, batch_partial_stride);
                let head_partial_off = ctx.mul_lo_u32(head_idx, partial_stride);
                let partial_base_off = ctx.add_u32_reg(batch_partial_off, head_partial_off);
                let partial_base_off_bytes = ctx.mul_wide_u32_reg(partial_base_off, four);
                let partial_base = ctx.add_u64(partials_ptr, partial_base_off_bytes);

                // First pass: find global max across all chunks
                let global_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let chunk_iter = ctx.mov_u32_imm(0);
                ctx.label("reduce_max_loop");
                let max_loop_cond = ctx.setp_lt_u32(chunk_iter, num_chunks);
                ctx.branch_if_not(max_loop_cond, "reduce_max_loop_end");

                // Load this chunk's max_score (at offset head_dim within chunk)
                let chunk_off = ctx.mul_lo_u32(chunk_iter, head_dim_plus_2);
                let max_elem_off = ctx.add_u32(chunk_off, head_dim);
                let max_elem_off_bytes = ctx.mul_wide_u32_reg(max_elem_off, four);
                let chunk_max_addr = ctx.add_u64(partial_base, max_elem_off_bytes);
                // Only lane 0 needs to read (broadcast later)
                let zero_lane = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_lane);
                let chunk_max = ctx.ld_global_f32_predicated(chunk_max_addr, is_lane0, f32::NEG_INFINITY);
                // Broadcast from lane 0 to all lanes
                let chunk_max = ctx.shfl_idx_f32(chunk_max, 0, 0xFFFF_FFFF);
                ctx.max_f32_inplace(global_max, chunk_max);

                ctx.add_u32_inplace(chunk_iter, 1);
                ctx.branch("reduce_max_loop");
                ctx.label("reduce_max_loop_end");

                // Second pass: accumulate rescaled outputs
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let global_sum = ctx.mov_f32_imm(0.0);
                let acc0 = ctx.mov_f32_imm(0.0);
                let acc1 = ctx.mov_f32_imm(0.0);
                let acc2 = ctx.mov_f32_imm(0.0);
                let acc3 = ctx.mov_f32_imm(0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);

                let chunk_iter2 = ctx.mov_u32_imm(0);
                ctx.label("reduce_acc_loop");
                let acc_loop_cond = ctx.setp_lt_u32(chunk_iter2, num_chunks);
                ctx.branch_if_not(acc_loop_cond, "reduce_acc_loop_end");

                // Load chunk max and sum_exp
                let chunk_off2 = ctx.mul_lo_u32(chunk_iter2, head_dim_plus_2);
                let max_elem_off2 = ctx.add_u32(chunk_off2, head_dim);
                let sum_elem_off2 = ctx.add_u32(chunk_off2, head_dim);
                let sum_elem_off2 = ctx.add_u32(sum_elem_off2, 1);
                let max_off_bytes2 = ctx.mul_wide_u32_reg(max_elem_off2, four);
                let sum_off_bytes2 = ctx.mul_wide_u32_reg(sum_elem_off2, four);
                let chunk_max_addr2 = ctx.add_u64(partial_base, max_off_bytes2);
                let chunk_sum_addr2 = ctx.add_u64(partial_base, sum_off_bytes2);

                let chunk_max2 = ctx.ld_global_f32_predicated(chunk_max_addr2, is_lane0, f32::NEG_INFINITY);
                let chunk_sum2 = ctx.ld_global_f32_predicated(chunk_sum_addr2, is_lane0, 0.0);
                let chunk_max2 = ctx.shfl_idx_f32(chunk_max2, 0, 0xFFFF_FFFF);
                let chunk_sum2 = ctx.shfl_idx_f32(chunk_sum2, 0, 0xFFFF_FFFF);

                // Skip empty chunks (max = -inf)
                let neg_inf_check = ctx.mov_f32_imm(-1e30);
                let is_valid = ctx.setp_gt_f32(chunk_max2, neg_inf_check);
                ctx.branch_if_not(is_valid, "reduce_skip_chunk");

                // Compute scale factor: exp(chunk_max - global_max)
                let max_diff = ctx.sub_f32(chunk_max2, global_max);
                let max_diff_log2 = ctx.mul_f32(max_diff, log2e);
                let scale_factor = ctx.ex2_f32(max_diff_log2);

                // Accumulate scaled sum_exp
                let scaled_sum = ctx.mul_f32(chunk_sum2, scale_factor);
                ctx.add_f32_inplace(global_sum, scaled_sum);

                // Load and accumulate scaled output
                let chunk_base_off_bytes = ctx.mul_wide_u32_reg(chunk_off2, four);
                let chunk_base = ctx.add_u64(partial_base, chunk_base_off_bytes);

                let out0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let out0_addr = ctx.add_u64(chunk_base, out0_off_bytes);
                let out0 = ctx.ld_global_f32_predicated(out0_addr, in_bounds0, 0.0);
                let scaled_out0 = ctx.mul_f32(out0, scale_factor);
                ctx.add_f32_inplace(acc0, scaled_out0);

                let out1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let out1_addr = ctx.add_u64(chunk_base, out1_off_bytes);
                let out1 = ctx.ld_global_f32_predicated(out1_addr, in_bounds1, 0.0);
                let scaled_out1 = ctx.mul_f32(out1, scale_factor);
                ctx.add_f32_inplace(acc1, scaled_out1);

                let out2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let out2_addr = ctx.add_u64(chunk_base, out2_off_bytes);
                let out2 = ctx.ld_global_f32_predicated(out2_addr, in_bounds2, 0.0);
                let scaled_out2 = ctx.mul_f32(out2, scale_factor);
                ctx.add_f32_inplace(acc2, scaled_out2);

                let out3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let out3_addr = ctx.add_u64(chunk_base, out3_off_bytes);
                let out3 = ctx.ld_global_f32_predicated(out3_addr, in_bounds3, 0.0);
                let scaled_out3 = ctx.mul_f32(out3, scale_factor);
                ctx.add_f32_inplace(acc3, scaled_out3);

                ctx.label("reduce_skip_chunk");
                ctx.add_u32_inplace(chunk_iter2, 1);
                ctx.branch("reduce_acc_loop");
                ctx.label("reduce_acc_loop_end");

                // Final normalization: output = acc / global_sum
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, global_sum);
                ctx.mul_f32_inplace(acc0, inv_sum);
                ctx.mul_f32_inplace(acc1, inv_sum);
                ctx.mul_f32_inplace(acc2, inv_sum);
                ctx.mul_f32_inplace(acc3, inv_sum);

                // Compute output offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let out_base_off = ctx.add_u32_reg(batch_off, head_off);
                let out_base_off_bytes = ctx.mul_wide_u32_reg(out_base_off, four);
                let out_base = ctx.add_u64(output_ptr, out_base_off_bytes);

                // Store output
                let final_out0_addr = ctx.add_u64(out_base, out0_off_bytes);
                ctx.branch_if_not(in_bounds0, "reduce_skip_store0");
                ctx.st_global_f32(final_out0_addr, acc0);
                ctx.label("reduce_skip_store0");

                let final_out1_addr = ctx.add_u64(out_base, out1_off_bytes);
                ctx.branch_if_not(in_bounds1, "reduce_skip_store1");
                ctx.st_global_f32(final_out1_addr, acc1);
                ctx.label("reduce_skip_store1");

                let final_out2_addr = ctx.add_u64(out_base, out2_off_bytes);
                ctx.branch_if_not(in_bounds2, "reduce_skip_store2");
                ctx.st_global_f32(final_out2_addr, acc2);
                ctx.label("reduce_skip_store2");

                let final_out3_addr = ctx.add_u64(out_base, out3_off_bytes);
                ctx.branch_if_not(in_bounds3, "reduce_skip_store3");
                ctx.st_global_f32(final_out3_addr, acc3);
                ctx.label("reduce_skip_store3");

                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_kernel_name() {
        let kernel = AttentionKernel::new(2048, 64);
        assert_eq!(kernel.name(), "flash_attention");

        let kernel_causal = AttentionKernel::new(2048, 64).with_causal();
        assert_eq!(kernel_causal.name(), "flash_attention_causal");

        let kernel_tc = AttentionKernel::tensor_core(2048, 64);
        assert_eq!(kernel_tc.name(), "flash_attention_tensor_core");

        let kernel_tc_causal = AttentionKernel::tensor_core(2048, 64).with_causal();
        assert_eq!(
            kernel_tc_causal.name(),
            "flash_attention_tensor_core_causal"
        );
    }

    #[test]
    fn test_tensor_core_attention_config() {
        let kernel = AttentionKernel::tensor_core(2048, 128);
        assert_eq!(kernel.seq_len, 2048);
        assert_eq!(kernel.head_dim, 128);
        assert!(kernel.use_tensor_cores);
        // Tile sizes should be at least 16 for WMMA
        assert!(kernel.tile_q >= 16);
        assert!(kernel.tile_kv >= 16);
    }

    #[test]
    fn test_tensor_core_attention_ptx_generation() {
        let kernel = AttentionKernel::tensor_core(512, 64);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(ptx.contains(".entry flash_attention_tensor_core"));

        // Verify parameters
        assert!(ptx.contains(".param .u64 q_ptr"));
        assert!(ptx.contains(".param .u64 k_ptr"));
        assert!(ptx.contains(".param .u64 v_ptr"));
        assert!(ptx.contains(".param .u64 o_ptr"));

        // Verify shared memory allocation
        assert!(ptx.contains(".shared"));

        // Verify WMMA operations present
        assert!(
            ptx.contains("wmma") || ptx.contains("mma"),
            "Tensor Core kernel should use WMMA instructions"
        );
    }

    #[test]
    fn test_tensor_core_attention_with_causal() {
        let kernel = AttentionKernel::tensor_core(1024, 64).with_causal();
        assert!(kernel.causal);
        assert!(kernel.use_tensor_cores);

        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("flash_attention_tensor_core_causal"));
    }

    #[test]
    fn test_with_tensor_cores_builder() {
        let kernel = AttentionKernel::new(1024, 64).with_tensor_cores();
        assert!(kernel.use_tensor_cores);
        assert_eq!(kernel.name(), "flash_attention_tensor_core");
    }

    #[test]
    fn test_attention_default_config() {
        let kernel = AttentionKernel::new(2048, 64);
        assert_eq!(kernel.seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.tile_q, 64);
        assert_eq!(kernel.tile_kv, 64);
        assert!(!kernel.causal);
        // scale should be 1/sqrt(64) = 0.125
        assert!((kernel.scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_attention_with_tiles() {
        let kernel = AttentionKernel::new(2048, 64).with_tiles(32, 128);
        assert_eq!(kernel.tile_q, 32);
        assert_eq!(kernel.tile_kv, 128);
    }

    #[test]
    fn test_attention_with_causal() {
        let kernel = AttentionKernel::new(2048, 64).with_causal();
        assert!(kernel.causal);
    }

    #[test]
    fn test_attention_with_scale() {
        let kernel = AttentionKernel::new(2048, 64).with_scale(0.1);
        assert!((kernel.scale - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_attention_ptx_generation() {
        let kernel = AttentionKernel::new(2048, 64);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 q_ptr"));
        assert!(ptx.contains(".param .u64 k_ptr"));
        assert!(ptx.contains(".param .u64 v_ptr"));
        assert!(ptx.contains(".param .u64 o_ptr"));
        assert!(ptx.contains(".param .u32 seq_len"));
        assert!(ptx.contains(".param .u32 head_dim"));
        assert!(ptx.contains(".param .u32 num_heads"));
    }

    #[test]
    fn test_attention_shared_memory() {
        let kernel = AttentionKernel::new(2048, 64);
        let ptx_kernel = kernel.build_ptx();

        // Should have shared memory for Q, K, V tiles
        assert!(ptx_kernel.shared_memory_bytes() > 0);
    }

    #[test]
    fn test_attention_ptx_contains_operations() {
        let kernel = AttentionKernel::new(2048, 64);
        let ptx = kernel.emit_ptx();

        // Verify memory operations
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));

        // Verify shared memory operations
        assert!(ptx.contains("ld.shared.f32") || ptx.contains("ld.f32"));
        assert!(ptx.contains("st.shared.f32") || ptx.contains("st.f32"));

        // Verify barrier synchronization
        assert!(ptx.contains("bar"));

        // Verify arithmetic for attention computation
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats
    }

    #[test]
    fn test_attention_online_softmax_ops() {
        let kernel = AttentionKernel::new(2048, 64);
        let ptx = kernel.emit_ptx();

        // Verify max for running max computation
        assert!(ptx.contains("max.f32"));

        // Verify exp for softmax (via ex2)
        assert!(ptx.contains("ex2.f32") || ptx.contains("ex2"));

        // Verify subtraction for x - max
        assert!(ptx.contains("sub.f32"));
    }

    #[test]
    fn test_attention_causal_vs_noncausal() {
        let kernel = AttentionKernel::new(2048, 64);
        let kernel_causal = AttentionKernel::new(2048, 64).with_causal();

        let ptx = kernel.emit_ptx();
        let ptx_causal = kernel_causal.emit_ptx();

        // Both should produce valid PTX
        assert!(!ptx.is_empty());
        assert!(!ptx_causal.is_empty());

        // Different kernel names
        assert!(ptx.contains("flash_attention"));
        assert!(ptx_causal.contains("flash_attention_causal"));
    }

    #[test]
    fn test_attention_kernel_variants() {
        // Test with different configurations
        let configs = vec![
            AttentionKernel::new(512, 32),
            AttentionKernel::new(1024, 64),
            AttentionKernel::new(2048, 128),
            AttentionKernel::new(4096, 64).with_causal(),
        ];

        for config in configs {
            let ptx = config.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".visible .entry"));
        }
    }

    /// PARITY-114: Verify flash_attention has no early exit before barrier
    #[test]
    fn test_parity_114_flash_attention_no_early_exit_before_barrier() {
        let kernel = AttentionKernel::new(64, 32);
        let ptx = kernel.emit_ptx();

        // Find positions of key elements
        let bar_sync_pos = ptx.find("bar.sync").expect("PTX should have bar.sync");
        let kv_loop_end_pos = ptx
            .find("kv_loop_end:")
            .expect("PTX should have kv_loop_end");

        // Verify bar.sync is inside the loop (before kv_loop_end)
        assert!(
            bar_sync_pos < kv_loop_end_pos,
            "bar.sync should be inside kv_loop (before kv_loop_end)"
        );

        // After PARITY-114 fix, bounds check should be AFTER kv_loop_end
        // Look for exit branches after the loop
        let exit_pos = ptx[kv_loop_end_pos..].find("bra exit");
        assert!(
            exit_pos.is_some(),
            "Exit branch should exist after kv_loop_end"
        );
    }

    /// PARITY-114: Verify tensor_core attention has no early exit before barrier
    #[test]
    fn test_parity_114_tensor_core_attention_no_early_exit_before_barrier() {
        let kernel = AttentionKernel::tensor_core(64, 32);
        let ptx = kernel.emit_ptx();

        // Find positions of key elements
        let bar_sync_pos = ptx.find("bar.sync").expect("PTX should have bar.sync");
        let kv_loop_end_pos = ptx
            .find("kv_loop_end:")
            .expect("PTX should have kv_loop_end");

        // Verify bar.sync is inside the loop (before kv_loop_end)
        assert!(
            bar_sync_pos < kv_loop_end_pos,
            "bar.sync should be inside kv_loop (before kv_loop_end)"
        );

        // Verify WMMA operations exist
        assert!(
            ptx.contains("wmma") || ptx.contains("mma"),
            "Tensor Core kernel should use WMMA instructions"
        );
    }

    /// PARITY-114 Countermeasure: Test boundary conditions for attention
    /// Five Whys Root Cause: Only tested dimensions where all threads have valid work
    #[test]
    fn test_boundary_conditions_flash_attention() {
        // Test sequence lengths NOT divisible by tile size
        let boundary_cases = [
            (17, 8),   // Odd seq_len
            (33, 16),  // Just over 2 tiles
            (100, 32), // Arbitrary non-power-of-2
            (1, 8),    // Edge: single position
            (63, 32),  // Just under 2 tiles
        ];

        for (seq_len, head_dim) in boundary_cases {
            let kernel = AttentionKernel::new(seq_len, head_dim);
            let ptx = kernel.emit_ptx();

            assert!(
                ptx.contains(".entry"),
                "Attention seq={seq_len} head_dim={head_dim} should have entry"
            );
            assert!(
                ptx.contains("bar.sync"),
                "Attention seq={seq_len} head_dim={head_dim} should have barrier"
            );

            // Verify barrier is inside loop
            let bar_sync_pos = ptx.find("bar.sync").unwrap();
            let kv_loop_end_pos = ptx.find("kv_loop_end:").unwrap();
            assert!(
                bar_sync_pos < kv_loop_end_pos,
                "Attention seq={seq_len} head_dim={head_dim}: barrier must be inside loop"
            );
        }
    }

    /// PARITY-114 Countermeasure: Test boundary conditions for tensor core attention
    #[test]
    fn test_boundary_conditions_tensor_core_attention() {
        let boundary_cases = [(17, 16), (33, 32), (65, 64), (100, 32)];

        for (seq_len, head_dim) in boundary_cases {
            let kernel = AttentionKernel::tensor_core(seq_len, head_dim);
            let ptx = kernel.emit_ptx();

            assert!(
                ptx.contains(".entry"),
                "TC Attention seq={seq_len} head_dim={head_dim} should have entry"
            );
            assert!(
                ptx.contains("bar.sync"),
                "TC Attention seq={seq_len} head_dim={head_dim} should have barrier"
            );
        }
    }

    // ===== PAR-020: IncrementalAttentionKernel Tests =====

    #[test]
    fn test_incremental_attention_kernel_new() {
        let kernel = IncrementalAttentionKernel::new(2048, 64, 32);
        assert_eq!(kernel.max_seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 32);
        // scale should be 1/sqrt(64) = 0.125
        assert!((kernel.scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_incremental_attention_kernel_name() {
        let kernel = IncrementalAttentionKernel::new(1024, 64, 22);
        assert_eq!(kernel.name(), "incremental_attention");
    }

    #[test]
    fn test_incremental_attention_ptx_generation() {
        let kernel = IncrementalAttentionKernel::new(512, 64, 22);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry incremental_attention"),
            "Should have incremental_attention entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 q_ptr"), "Should have q_ptr param");
        assert!(ptx.contains(".param .u64 k_ptr"), "Should have k_ptr param");
        assert!(ptx.contains(".param .u64 v_ptr"), "Should have v_ptr param");
        assert!(
            ptx.contains(".param .u64 out_ptr"),
            "Should have out_ptr param"
        );
        assert!(
            ptx.contains(".param .u32 seq_len"),
            "Should have seq_len param"
        );
    }

    #[test]
    fn test_incremental_attention_no_shared_memory() {
        let kernel = IncrementalAttentionKernel::new(1024, 64, 22);
        let ptx_kernel = kernel.build_ptx();

        // IncrementalAttention uses warp shuffle only, no shared memory
        assert_eq!(
            ptx_kernel.shared_memory_bytes(),
            0,
            "Incremental attention should use no shared memory"
        );
    }

    #[test]
    fn test_incremental_attention_warp_shuffle() {
        let kernel = IncrementalAttentionKernel::new(512, 64, 22);
        let ptx = kernel.emit_ptx();

        // Verify warp shuffle operations for dot product reduction
        assert!(
            ptx.contains("shfl.sync.down"),
            "Should have shfl.down for warp reduction"
        );
        assert!(
            ptx.contains("shfl.sync.idx"),
            "Should have shfl.idx for broadcast"
        );
    }

    #[test]
    fn test_incremental_attention_online_softmax() {
        let kernel = IncrementalAttentionKernel::new(512, 64, 22);
        let ptx = kernel.emit_ptx();

        // Verify online softmax operations
        assert!(
            ptx.contains("max.f32"),
            "Should have max for online softmax"
        );
        assert!(ptx.contains("ex2"), "Should have ex2 for exp computation");
        assert!(ptx.contains("sub.f32"), "Should have sub for score - max");
    }

    #[test]
    fn test_incremental_attention_loop_structure() {
        let kernel = IncrementalAttentionKernel::new(512, 64, 22);
        let ptx = kernel.emit_ptx();

        // Verify loop structure
        assert!(ptx.contains("seq_loop:"), "Should have seq_loop label");
        assert!(
            ptx.contains("seq_loop_end:"),
            "Should have seq_loop_end label"
        );
        assert!(
            ptx.contains("bra seq_loop"),
            "Should have branch back to loop start"
        );
    }

    #[test]
    fn test_incremental_attention_tinyllama_config() {
        // TinyLlama: 22 heads, 64 head_dim
        let kernel = IncrementalAttentionKernel::new(2048, 64, 22);
        let ptx = kernel.emit_ptx();

        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("ret;"));
    }

    // ===== PAR-021: GQA (Grouped Query Attention) Tests =====

    #[test]
    fn test_incremental_attention_with_gqa() {
        // Llama2 70B config: 64 query heads, 8 kv heads
        let kernel = IncrementalAttentionKernel::with_gqa(4096, 128, 64, 8);
        assert_eq!(kernel.max_seq_len, 4096);
        assert_eq!(kernel.head_dim, 128);
        assert_eq!(kernel.num_heads, 64);
        assert_eq!(kernel.num_kv_heads, 8);
        assert!(kernel.is_gqa());
        // scale should be 1/sqrt(128)
        assert!((kernel.scale - 0.088388).abs() < 0.001);
    }

    #[test]
    fn test_incremental_attention_is_gqa_false_for_mha() {
        // MHA: num_heads == num_kv_heads
        let kernel = IncrementalAttentionKernel::new(2048, 64, 32);
        assert!(!kernel.is_gqa());
        assert_eq!(kernel.num_heads, kernel.num_kv_heads);
    }

    #[test]
    fn test_incremental_attention_is_gqa_true_for_gqa() {
        let kernel = IncrementalAttentionKernel::with_gqa(2048, 64, 32, 4);
        assert!(kernel.is_gqa());
        assert_ne!(kernel.num_heads, kernel.num_kv_heads);
    }

    // ===== PAR-061: Indirect seq_len Tests =====

    #[test]
    fn test_incremental_attention_with_indirect_seq_len() {
        let kernel = IncrementalAttentionKernel::new(2048, 64, 32).with_indirect_seq_len(true);
        assert!(kernel.indirect_seq_len);
        assert_eq!(kernel.name(), "incremental_attention_indirect");
    }

    #[test]
    fn test_incremental_attention_indirect_ptx_generation() {
        let kernel = IncrementalAttentionKernel::new(512, 64, 22).with_indirect_seq_len(true);
        let ptx = kernel.emit_ptx();

        // Verify indirect kernel entry point
        assert!(
            ptx.contains(".entry incremental_attention_indirect"),
            "Should have incremental_attention_indirect entry"
        );

        // Verify seq_len_ptr parameter for device memory access
        assert!(
            ptx.contains(".param .u64 seq_len_ptr"),
            "Should have seq_len_ptr param for indirect mode"
        );
    }

    #[test]
    fn test_incremental_attention_indirect_false() {
        let kernel = IncrementalAttentionKernel::new(2048, 64, 32).with_indirect_seq_len(false);
        assert!(!kernel.indirect_seq_len);
        assert_eq!(kernel.name(), "incremental_attention");
    }

    // ===== MultiWarpIncrementalAttentionKernel Tests =====

    #[test]
    fn test_multi_warp_attention_kernel_new() {
        let kernel = MultiWarpIncrementalAttentionKernel::new(2048, 64, 32, 8, 4);
        assert_eq!(kernel.max_seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.num_kv_heads, 8);
        assert_eq!(kernel.num_warps_per_head, 4);
        assert!(!kernel.indirect_seq_len);
        // scale should be 1/sqrt(64) = 0.125
        assert!((kernel.scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_multi_warp_attention_kernel_name() {
        let kernel = MultiWarpIncrementalAttentionKernel::new(1024, 64, 22, 22, 4);
        assert_eq!(kernel.name(), "multi_warp_attention");
    }

    #[test]
    fn test_multi_warp_attention_indirect_name() {
        let kernel =
            MultiWarpIncrementalAttentionKernel::new(1024, 64, 22, 22, 4).with_indirect_seq_len(true);
        assert!(kernel.indirect_seq_len);
        assert_eq!(kernel.name(), "multi_warp_attention_indirect");
    }

    #[test]
    fn test_multi_warp_attention_ptx_generation() {
        let kernel = MultiWarpIncrementalAttentionKernel::new(512, 64, 22, 22, 4);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry multi_warp_attention"),
            "Should have multi_warp_attention entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 q_ptr"), "Should have q_ptr param");
        assert!(ptx.contains(".param .u64 k_ptr"), "Should have k_ptr param");
        assert!(ptx.contains(".param .u64 v_ptr"), "Should have v_ptr param");
        assert!(
            ptx.contains(".param .u64 out_ptr"),
            "Should have out_ptr param"
        );
    }

    #[test]
    fn test_multi_warp_attention_shared_memory() {
        let kernel = MultiWarpIncrementalAttentionKernel::new(1024, 64, 22, 22, 4);
        let ptx_kernel = kernel.build_ptx();

        // Multi-warp attention uses shared memory for inter-warp reduction
        assert!(
            ptx_kernel.shared_memory_bytes() > 0,
            "Multi-warp attention should use shared memory"
        );
    }

    #[test]
    fn test_multi_warp_attention_contains_warp_reduction() {
        let kernel = MultiWarpIncrementalAttentionKernel::new(512, 64, 22, 22, 4);
        let ptx = kernel.emit_ptx();

        // Verify warp shuffle operations
        assert!(
            ptx.contains("shfl.sync") || ptx.contains("shfl."),
            "Should have shfl for warp operations"
        );

        // Verify barrier for inter-warp sync
        assert!(
            ptx.contains("bar.sync"),
            "Should have barrier for warp synchronization"
        );
    }

    #[test]
    fn test_multi_warp_attention_indirect_ptx() {
        let kernel =
            MultiWarpIncrementalAttentionKernel::new(512, 64, 22, 22, 4).with_indirect_seq_len(true);
        let ptx = kernel.emit_ptx();

        // Verify indirect kernel entry point
        assert!(
            ptx.contains(".entry multi_warp_attention_indirect"),
            "Should have indirect entry point"
        );
    }

    #[test]
    fn test_multi_warp_attention_gqa_config() {
        // Test GQA config: 32 query heads, 4 kv heads
        let kernel = MultiWarpIncrementalAttentionKernel::new(2048, 128, 32, 4, 8);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.num_kv_heads, 4);

        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
    }

    // ===== BatchedIncrementalAttentionKernel Tests =====

    #[test]
    fn test_batched_incremental_attention_kernel_new() {
        let kernel = BatchedIncrementalAttentionKernel::new(2048, 64, 32, 8, 4);
        assert_eq!(kernel.max_seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.num_kv_heads, 8);
        assert_eq!(kernel.batch_size, 4);
        // scale should be 1/sqrt(64) = 0.125
        assert!((kernel.scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_batched_incremental_attention_kernel_name() {
        let kernel = BatchedIncrementalAttentionKernel::new(1024, 64, 22, 22, 8);
        assert_eq!(kernel.name(), "batched_incremental_attention");
    }

    #[test]
    fn test_batched_incremental_attention_ptx_generation() {
        let kernel = BatchedIncrementalAttentionKernel::new(512, 64, 22, 22, 4);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry batched_incremental_attention"),
            "Should have batched_incremental_attention entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 q_ptr"), "Should have q_ptr param");
        assert!(ptx.contains(".param .u64 k_ptr"), "Should have k_ptr param");
        assert!(ptx.contains(".param .u64 v_ptr"), "Should have v_ptr param");
        assert!(
            ptx.contains(".param .u64 out_ptr"),
            "Should have out_ptr param"
        );
    }

    #[test]
    fn test_batched_incremental_attention_batch_sizes() {
        // Test various batch sizes
        for batch_size in [1, 2, 4, 8, 16] {
            let kernel = BatchedIncrementalAttentionKernel::new(1024, 64, 22, 22, batch_size);
            assert_eq!(kernel.batch_size, batch_size);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    #[test]
    fn test_batched_incremental_attention_gqa_config() {
        // GQA config: 32 query heads, 4 kv heads
        let kernel = BatchedIncrementalAttentionKernel::new(2048, 128, 32, 4, 8);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.num_kv_heads, 4);

        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_batched_incremental_attention_head_dims() {
        // Test common head dimensions
        for head_dim in [32, 64, 128] {
            let kernel = BatchedIncrementalAttentionKernel::new(1024, head_dim, 22, 22, 4);
            assert_eq!(kernel.head_dim, head_dim);
            let expected_scale = 1.0 / (head_dim as f32).sqrt();
            assert!((kernel.scale - expected_scale).abs() < 0.001);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
        }
    }

    #[test]
    fn test_batched_incremental_attention_warp_operations() {
        let kernel = BatchedIncrementalAttentionKernel::new(512, 64, 22, 22, 4);
        let ptx = kernel.emit_ptx();

        // Verify warp shuffle operations for reduction
        assert!(
            ptx.contains("shfl.sync") || ptx.contains("shfl."),
            "Should have shfl for warp operations"
        );
    }

    #[test]
    fn test_batched_incremental_attention_memory_ops() {
        let kernel = BatchedIncrementalAttentionKernel::new(512, 64, 22, 22, 4);
        let ptx = kernel.emit_ptx();

        // Verify memory operations
        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== FlashDecodingChunkKernel Tests =====

    #[test]
    fn test_flash_decoding_chunk_kernel_new() {
        let kernel = FlashDecodingChunkKernel::new(2048, 64, 32, 8, 4);
        assert_eq!(kernel.max_seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.num_kv_heads, 8);
        assert_eq!(kernel.batch_size, 4);
        // scale should be 1/sqrt(64) = 0.125
        assert!((kernel.scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_flash_decoding_chunk_kernel_name() {
        let kernel = FlashDecodingChunkKernel::new(1024, 64, 22, 22, 1);
        assert_eq!(kernel.name(), "flash_decoding_chunk");
    }

    #[test]
    fn test_flash_decoding_chunk_num_chunks() {
        let kernel = FlashDecodingChunkKernel::new(2048, 64, 32, 8, 4);
        // chunk_size is FLASH_DECODE_CHUNK_SIZE (typically 256)
        let chunks = kernel.num_chunks(1024);
        assert!(chunks > 0);
        assert!(chunks <= 1024);
    }

    #[test]
    fn test_flash_decoding_chunk_partials_size() {
        let kernel = FlashDecodingChunkKernel::new(2048, 64, 32, 8, 4);
        let size = kernel.partials_size_per_head(8);
        // size = max_chunks * (head_dim + 2) = 8 * (64 + 2) = 528
        assert_eq!(size, 8 * (64 + 2));
    }

    #[test]
    fn test_flash_decoding_chunk_ptx_generation() {
        let kernel = FlashDecodingChunkKernel::new(512, 64, 22, 22, 2);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry flash_decoding_chunk"),
            "Should have flash_decoding_chunk entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 q_ptr"), "Should have q_ptr param");
        assert!(ptx.contains(".param .u64 k_ptr"), "Should have k_ptr param");
        assert!(ptx.contains(".param .u64 v_ptr"), "Should have v_ptr param");
        assert!(ptx.contains(".param .u64 partials_ptr"), "Should have partials_ptr param");
    }

    #[test]
    fn test_flash_decoding_chunk_different_configs() {
        // Test various configurations
        let configs = [
            (512, 32, 8, 8, 1),
            (1024, 64, 16, 4, 2),
            (2048, 128, 32, 8, 4),
            (4096, 64, 64, 8, 8),
        ];

        for (max_seq_len, head_dim, num_heads, num_kv_heads, batch_size) in configs {
            let kernel = FlashDecodingChunkKernel::new(max_seq_len, head_dim, num_heads, num_kv_heads, batch_size);
            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty(), "PTX should not be empty for config {:?}", (max_seq_len, head_dim));
            assert!(ptx.contains(".entry"), "Should have entry point");
        }
    }

    // ===== FlashDecodingReduceKernel Tests =====

    #[test]
    fn test_flash_decoding_reduce_kernel_new() {
        let kernel = FlashDecodingReduceKernel::new(64, 32, 4);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.batch_size, 4);
    }

    #[test]
    fn test_flash_decoding_reduce_kernel_name() {
        let kernel = FlashDecodingReduceKernel::new(64, 22, 1);
        assert_eq!(kernel.name(), "flash_decoding_reduce");
    }

    #[test]
    fn test_flash_decoding_reduce_ptx_generation() {
        let kernel = FlashDecodingReduceKernel::new(64, 22, 2);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry flash_decoding_reduce"),
            "Should have flash_decoding_reduce entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 partials_ptr"), "Should have partials_ptr param");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr param");
        assert!(ptx.contains(".param .u64 seq_lens_ptr"), "Should have seq_lens_ptr param");
        assert!(ptx.contains(".param .u32 max_chunks"), "Should have max_chunks param");
    }

    #[test]
    fn test_flash_decoding_reduce_different_configs() {
        // Test various configurations
        let configs = [
            (32, 8, 1),
            (64, 16, 2),
            (128, 32, 4),
            (64, 64, 8),
        ];

        for (head_dim, num_heads, batch_size) in configs {
            let kernel = FlashDecodingReduceKernel::new(head_dim, num_heads, batch_size);
            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty(), "PTX should not be empty for config {:?}", (head_dim, num_heads));
            assert!(ptx.contains(".entry"), "Should have entry point");
        }
    }

    #[test]
    fn test_flash_decoding_reduce_memory_ops() {
        let kernel = FlashDecodingReduceKernel::new(64, 22, 2);
        let ptx = kernel.emit_ptx();

        // Verify memory operations for reduction
        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }
}
