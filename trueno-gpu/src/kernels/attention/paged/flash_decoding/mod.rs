//! Flash Decoding - Split-K Attention for 2X Ollama Performance (PAR-118)
//!
//! Flash Decoding splits the KV cache into chunks processed in parallel,
//! then reduces partial results. This amortizes memory bandwidth across
//! multiple thread blocks, achieving higher throughput for long sequences.
//!
//! Algorithm:
//! 1. Split sequence into K chunks of CHUNK_SIZE positions
//! 2. Each chunk computes partial attention: (max_score, sum_exp, weighted_out)
//! 3. Reduction combines partials with proper softmax rescaling:
//!    - new_max = max(chunk_max[0], chunk_max[1], ...)
//!    - For each chunk: scale = exp(chunk_max - new_max)
//!    - new_sum = sum(chunk_sum[i] * scale[i])
//!    - output = sum(chunk_out[i] * chunk_sum[i] * scale[i]) / new_sum
//!
//! Performance:
//! - Current: Sequential loop over seq_len (memory-bandwidth limited)
//! - Flash Decoding: K parallel blocks (K = ceil(seq_len / CHUNK_SIZE))
//! - Expected speedup: ~1.5-2x for typical seq_len (512-2048)

mod chunk_kernel;
mod reduce_kernel;

pub use chunk_kernel::FlashDecodingChunkKernel;
pub use reduce_kernel::FlashDecodingReduceKernel;

/// Chunk size for Flash Decoding split-K attention
/// Trade-off: smaller = more parallelism, larger = less reduction overhead
pub const FLASH_DECODE_CHUNK_SIZE: u32 = 128;
