//! Layout Transform and Batched Kernels
//!
//! GPU kernels for tensor layout transformations in multi-head attention.
//!
//! ## Transform Kernels
//!
//! - `TransposeKernel`: Matrix transpose
//! - `InterleavedToBatchedKernel`: Convert interleaved to batched layout
//! - `BatchedToInterleavedKernel`: Convert batched to interleaved layout
//! - `ExtractSingleHeadKernel`: Extract one head from interleaved tensor
//! - `CopySingleHeadKernel`: Copy to head position in interleaved tensor
//!
//! ## Batched Kernels
//!
//! - `BatchedTransposeKernel`: Transpose multiple matrices
//! - `BatchedScaleKernel`: Scale all elements by scalar
//! - `BatchedSoftmaxKernel`: Row-wise softmax for attention

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Transpose Kernel: output[j, i] = input[i, j]
///
/// Simple matrix transpose for attention K^T computation.
#[derive(Debug, Clone)]
pub struct TransposeKernel {
    /// Number of rows in input
    pub rows: u32,
    /// Number of columns in input
    pub cols: u32,
}

impl TransposeKernel {
    /// Create a new transpose kernel
    #[must_use]
    pub const fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }
}

impl Kernel for TransposeKernel {
    fn name(&self) -> &str {
        "transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        let rows = self.rows;
        let cols = self.cols;
        let total_elems = rows * cols;

        PtxKernel::new("transpose")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "cols")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let total = ctx.mov_u32_imm(total_elems);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let row_idx = ctx.div_u32(gid, cols);
                let col_idx = ctx.rem_u32(gid, cols);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);

                let rows_reg = ctx.mov_u32_imm(rows);
                let out_linear = ctx.mad_lo_u32(col_idx, rows_reg, row_idx);
                let output_offset = ctx.mul_wide_u32_reg(out_linear, four);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Interleaved to Batched Kernel: [seq_len, n_heads * head_dim] -> [n_heads, seq_len, head_dim]
#[derive(Debug, Clone)]
pub struct InterleavedToBatchedKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl InterleavedToBatchedKernel {
    /// Create a new interleaved-to-batched kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self {
            seq_len,
            n_heads,
            head_dim,
        }
    }
}

impl Kernel for InterleavedToBatchedKernel {
    fn name(&self) -> &str {
        "interleaved_to_batched"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let d_model = n_heads * head_dim;
        let total_elems = seq_len * d_model;

        PtxKernel::new("interleaved_to_batched")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.mov_u32_imm(total_elems);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let s = ctx.div_u32(gid, d_model);
                let remainder = ctx.rem_u32(gid, d_model);
                let h = ctx.div_u32(remainder, head_dim);
                let d = ctx.rem_u32(remainder, head_dim);

                let seq_head = ctx.mov_u32_imm(seq_len * head_dim);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let out_base = ctx.mul_lo_u32(h, seq_head);
                let out_row = ctx.mad_lo_u32(s, head_dim_reg, d);
                let out_idx = ctx.add_u32_reg(out_base, out_row);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let output_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Extract Single Head Kernel: extract head h from interleaved [seq_len, n_heads * head_dim]
#[derive(Debug, Clone)]
pub struct ExtractSingleHeadKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl ExtractSingleHeadKernel {
    /// Create kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self {
            seq_len,
            n_heads,
            head_dim,
        }
    }
}

impl Kernel for ExtractSingleHeadKernel {
    fn name(&self) -> &str {
        "extract_single_head"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let d_model = self.n_heads * head_dim;
        let output_size = seq_len * head_dim;

        PtxKernel::new("extract_single_head")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "head_idx")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.mov_u32_imm(output_size);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let head_idx = ctx.load_param_u32("head_idx");

                let s = ctx.div_u32(gid, head_dim);
                let d = ctx.rem_u32(gid, head_dim);

                let d_model_reg = ctx.mov_u32_imm(d_model);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, head_dim_reg);
                let row_offset = ctx.mul_lo_u32(s, d_model_reg);
                let in_idx = ctx.add_u32_reg(row_offset, head_offset);
                let in_idx = ctx.add_u32_reg(in_idx, d);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let output_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Copy Single Head Kernel: copy [seq_len, head_dim] to head position in interleaved output
#[derive(Debug, Clone)]
pub struct CopySingleHeadKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl CopySingleHeadKernel {
    /// Create kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self {
            seq_len,
            n_heads,
            head_dim,
        }
    }
}

impl Kernel for CopySingleHeadKernel {
    fn name(&self) -> &str {
        "copy_single_head"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let d_model = self.n_heads * head_dim;
        let input_size = seq_len * head_dim;

        PtxKernel::new("copy_single_head")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "head_idx")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.mov_u32_imm(input_size);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let head_idx = ctx.load_param_u32("head_idx");

                let s = ctx.div_u32(gid, head_dim);
                let d = ctx.rem_u32(gid, head_dim);

                let d_model_reg = ctx.mov_u32_imm(d_model);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, head_dim_reg);
                let row_offset = ctx.mul_lo_u32(s, d_model_reg);
                let out_idx = ctx.add_u32_reg(row_offset, head_offset);
                let out_idx = ctx.add_u32_reg(out_idx, d);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let output_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched to Interleaved Kernel: [n_heads, seq_len, head_dim] -> [seq_len, n_heads * head_dim]
#[derive(Debug, Clone)]
pub struct BatchedToInterleavedKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl BatchedToInterleavedKernel {
    /// Create a new batched-to-interleaved kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self {
            seq_len,
            n_heads,
            head_dim,
        }
    }
}

impl Kernel for BatchedToInterleavedKernel {
    fn name(&self) -> &str {
        "batched_to_interleaved"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let d_model = n_heads * head_dim;
        let total_elems = seq_len * d_model;

        PtxKernel::new("batched_to_interleaved")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.mov_u32_imm(total_elems);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let s = ctx.div_u32(gid, d_model);
                let remainder = ctx.rem_u32(gid, d_model);
                let h = ctx.div_u32(remainder, head_dim);
                let d = ctx.rem_u32(remainder, head_dim);

                let seq_head = ctx.mov_u32_imm(seq_len * head_dim);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let in_base = ctx.mul_lo_u32(h, seq_head);
                let in_row = ctx.mad_lo_u32(s, head_dim_reg, d);
                let in_idx = ctx.add_u32_reg(in_base, in_row);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let output_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// Batched Kernels for Multi-Head Attention (WAPR-PERF-008)
// =============================================================================

/// Batched transpose kernel: transpose multiple matrices in one launch
#[derive(Debug, Clone)]
pub struct BatchedTransposeKernel {
    /// Number of batches (e.g., n_heads)
    pub batch: u32,
    /// Input rows (becomes output cols)
    pub rows: u32,
    /// Input cols (becomes output rows)
    pub cols: u32,
}

impl BatchedTransposeKernel {
    /// Create a new batched transpose kernel
    #[must_use]
    pub const fn new(batch: u32, rows: u32, cols: u32) -> Self {
        Self { batch, rows, cols }
    }
}

impl Kernel for BatchedTransposeKernel {
    fn name(&self) -> &str {
        "batched_transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        let rows = self.rows;
        let cols = self.cols;
        let total_per_batch = rows * cols;

        PtxKernel::new("batched_transpose")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "cols")
            .build(move |ctx| {
                let batch_idx = ctx.special_reg(PtxReg::CtaIdZ);
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total = ctx.mov_u32_imm(total_per_batch);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                let batch_param = ctx.load_param_u32("batch");
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let valid = ctx.and_pred(in_bounds, batch_valid);
                ctx.branch_if_not(valid, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                let row = ctx.div_u32(gid, cols);
                let col = ctx.rem_u32(gid, cols);

                let batch_offset = ctx.mul_wide_u32(batch_idx, total_per_batch * 4);
                let in_batch_ptr = ctx.add_u64(input_ptr, batch_offset);
                let out_batch_ptr = ctx.add_u64(output_ptr, batch_offset);

                let cols_reg = ctx.mov_u32_imm(cols);
                let in_idx = ctx.mad_lo_u32(row, cols_reg, col);
                let rows_reg = ctx.mov_u32_imm(rows);
                let out_idx = ctx.mad_lo_u32(col, rows_reg, row);

                let four = ctx.mov_u32_imm(4);
                let in_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let out_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let in_addr = ctx.add_u64(in_batch_ptr, in_offset);
                let out_addr = ctx.add_u64(out_batch_ptr, out_offset);

                let val = ctx.ld_global_f32(in_addr);
                ctx.st_global_f32(out_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched scale kernel: multiply all elements by a scalar
#[derive(Debug, Clone)]
pub struct BatchedScaleKernel {
    /// Total number of elements (batch * rows * cols)
    pub n: u32,
}

impl BatchedScaleKernel {
    /// Create a new batched scale kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for BatchedScaleKernel {
    fn name(&self) -> &str {
        "batched_scale"
    }

    fn build_ptx(&self) -> PtxKernel {
        let total = self.n;

        PtxKernel::new("batched_scale")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::F32, "scale")
            .param(PtxType::U32, "n")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total_reg = ctx.mov_u32_imm(total);
                let in_bounds = ctx.setp_lt_u32(gid, total_reg);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let scale = ctx.load_param_f32("scale");

                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                let val = ctx.ld_global_f32(in_addr);
                let scaled = ctx.mul_f32(val, scale);
                ctx.st_global_f32(out_addr, scaled);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched softmax kernel: softmax for multiple independent rows
///
/// Uses warp shuffle for reduction. One warp per row.
#[derive(Debug, Clone)]
pub struct BatchedSoftmaxKernel {
    /// Total number of rows to process (batch * n_rows)
    pub total_rows: u32,
    /// Size of each row
    pub row_size: u32,
}

impl BatchedSoftmaxKernel {
    /// Create a new batched softmax kernel
    #[must_use]
    pub const fn new(total_rows: u32, row_size: u32) -> Self {
        Self {
            total_rows,
            row_size,
        }
    }
}

impl Kernel for BatchedSoftmaxKernel {
    fn name(&self) -> &str {
        "batched_softmax"
    }

    fn build_ptx(&self) -> PtxKernel {
        let total_rows = self.total_rows;
        let row_size = self.row_size;

        PtxKernel::new("batched_softmax")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "total_rows")
            .param(PtxType::U32, "row_size")
            .shared_memory(72)
            .build(move |ctx| {
                let row_idx = ctx.special_reg(PtxReg::CtaIdX);
                let tid = ctx.special_reg(PtxReg::TidX);

                let total_rows_reg = ctx.mov_u32_imm(total_rows);
                let valid = ctx.setp_lt_u32(row_idx, total_rows_reg);
                ctx.branch_if_not(valid, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let row_size_reg = ctx.mov_u32_imm(row_size);

                let row_offset = ctx.mul_wide_u32(row_idx, row_size * 4);
                let row_input_ptr = ctx.add_u64(input_ptr, row_offset);
                let row_output_ptr = ctx.add_u64(output_ptr, row_offset);

                let four = ctx.mov_u32_imm(4);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // Pass 1: Find max
                let local_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let i_max = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_max, tid);
                ctx.label("max_loop");
                let max_done = ctx.setp_ge_u32(i_max, row_size_reg);
                ctx.branch_if(max_done, "max_done");

                let offset = ctx.mul_wide_u32_reg(i_max, four);
                let addr = ctx.add_u64(row_input_ptr, offset);
                let val = ctx.ld_global_f32(addr);
                ctx.max_f32_inplace(local_max, val);
                ctx.add_u32_inplace(i_max, 32);
                ctx.branch("max_loop");

                ctx.label("max_done");

                // Warp reduce for max
                let tmp16 = ctx.shfl_down_f32(local_max, 16, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp16);
                let tmp8 = ctx.shfl_down_f32(local_max, 8, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp8);
                let tmp4 = ctx.shfl_down_f32(local_max, 4, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp4);
                let tmp2 = ctx.shfl_down_f32(local_max, 2, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp2);
                let tmp1 = ctx.shfl_down_f32(local_max, 1, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp1);

                let row_max = ctx.shfl_idx_f32(local_max, 0, 0xFFFF_FFFF);

                // Pass 2: Sum of exp(x - max)
                let local_sum = ctx.mov_f32_imm(0.0);
                let i_sum = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_sum, tid);
                ctx.label("sum_loop");
                let sum_done = ctx.setp_ge_u32(i_sum, row_size_reg);
                ctx.branch_if(sum_done, "sum_done");

                let offset = ctx.mul_wide_u32_reg(i_sum, four);
                let addr = ctx.add_u64(row_input_ptr, offset);
                let val = ctx.ld_global_f32(addr);
                let diff = ctx.sub_f32(val, row_max);
                let exp_arg = ctx.mul_f32(diff, log2e);
                let exp_val = ctx.ex2_f32(exp_arg);
                ctx.add_f32_inplace(local_sum, exp_val);
                ctx.add_u32_inplace(i_sum, 32);
                ctx.branch("sum_loop");

                ctx.label("sum_done");

                // Warp reduce for sum
                let stmp16 = ctx.shfl_down_f32(local_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp16);
                let stmp8 = ctx.shfl_down_f32(local_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp8);
                let stmp4 = ctx.shfl_down_f32(local_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp4);
                let stmp2 = ctx.shfl_down_f32(local_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp2);
                let stmp1 = ctx.shfl_down_f32(local_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp1);

                let row_sum = ctx.shfl_idx_f32(local_sum, 0, 0xFFFF_FFFF);

                // Pass 3: Write normalized values
                let i_write = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(i_write, tid);
                ctx.label("write_loop");
                let write_done = ctx.setp_ge_u32(i_write, row_size_reg);
                ctx.branch_if(write_done, "exit");

                let offset = ctx.mul_wide_u32_reg(i_write, four);
                let in_addr = ctx.add_u64(row_input_ptr, offset);
                let out_addr = ctx.add_u64(row_output_ptr, offset);

                let val = ctx.ld_global_f32(in_addr);
                let diff = ctx.sub_f32(val, row_max);
                let exp_arg = ctx.mul_f32(diff, log2e);
                let exp_val = ctx.ex2_f32(exp_arg);
                let normalized = ctx.div_f32(exp_val, row_sum);
                ctx.st_global_f32(out_addr, normalized);

                ctx.add_u32_inplace(i_write, 32);
                ctx.branch("write_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

