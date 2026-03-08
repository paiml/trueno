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

/// Chunk size for Flash Decoding split-K attention.
///
/// PMAT-040: Reduced from 128 to 32 to enable actual parallelism at typical
/// decode sequence lengths (32-256 tokens). With chunk_size=128, sequences <128
/// got only 1 chunk = zero parallelism (Flash Decoding degenerated to sequential).
///
/// Trade-offs at chunk_size=32:
/// - seq_len=64: 2 chunks (2x parallelism vs 1x with 128)
/// - seq_len=128: 4 chunks × 28 heads = 112 blocks → 88% SM util on 4090
/// - max_seq_len=4096: 128 chunks → partials buffer ~945KB (negligible)
/// - Reduction overhead: max 128 iterations in reduce kernel (single block)
pub const FLASH_DECODE_CHUNK_SIZE: u32 = 32;
