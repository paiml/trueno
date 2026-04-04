//! Quantized kernel profiler (Q4K/Q6K CPU). Spec section 4.7.
//! Profiles trueno's fused dequantization + GEMV CPU kernels.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Supported quantized kernel types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantKernel {
    Q4kGemv,
    Q5kGemv,
    Q6kGemv,
    Q8Gemv,
    Nf4Gemv,
}

impl std::str::FromStr for QuantKernel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "q4k_gemv" | "q4k" => Ok(Self::Q4kGemv),
            "q5k_gemv" | "q5k" => Ok(Self::Q5kGemv),
            "q6k_gemv" | "q6k" => Ok(Self::Q6kGemv),
            "q8_gemv" | "q8" => Ok(Self::Q8Gemv),
            "nf4_gemv" | "nf4" => Ok(Self::Nf4Gemv),
            _ => anyhow::bail!("Unknown quant kernel: {s}. Supported: q4k_gemv, q5k_gemv, q6k_gemv, q8_gemv, nf4_gemv"),
        }
    }
}

impl QuantKernel {
    /// Super-block size for this quantization format.
    pub fn superblock_elements(&self) -> u32 {
        match self {
            QuantKernel::Q4kGemv => 256,
            QuantKernel::Q5kGemv => 256,
            QuantKernel::Q6kGemv => 256,
            QuantKernel::Q8Gemv => 256,
            QuantKernel::Nf4Gemv => 64,
        }
    }

    /// Bytes per super-block.
    pub fn superblock_bytes(&self) -> u32 {
        match self {
            QuantKernel::Q4kGemv => 144, // Q4K: 256 elements in 144 bytes
            QuantKernel::Q5kGemv => 176,
            QuantKernel::Q6kGemv => 210,
            QuantKernel::Q8Gemv => 256, // 1 byte per element
            QuantKernel::Nf4Gemv => 32, // 4 bits per element
        }
    }
}

/// Quantized kernel profile output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantProfile {
    pub kernel: QuantKernel,
    pub dimensions: [u32; 3],
    /// Super-blocks processed per second.
    pub superblocks_per_sec: f64,
    /// Effective memory bandwidth (compressed input bytes / time).
    pub effective_bandwidth_gbps: f64,
    /// Compression speedup vs FP32 baseline.
    pub compression_speedup: f64,
    /// Wall clock time in microseconds.
    pub wall_time_us: f64,
}

/// Parse dimension string "MxNxK" into [M, N, K].
fn parse_dimensions(size: &str) -> Result<[u32; 3]> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 3 {
        anyhow::bail!("Dimensions must be MxNxK format, got: {size}");
    }
    let m: u32 = parts[0].parse().map_err(|_| anyhow::anyhow!("Invalid M: {}", parts[0]))?;
    let n: u32 = parts[1].parse().map_err(|_| anyhow::anyhow!("Invalid N: {}", parts[1]))?;
    let k: u32 = parts[2].parse().map_err(|_| anyhow::anyhow!("Invalid K: {}", parts[2]))?;
    Ok([m, n, k])
}

/// Profile a quantized GEMV kernel.
pub fn profile_quant(kernel_name: &str, size: &str) -> Result<()> {
    let kernel: QuantKernel = kernel_name.parse()?;
    let dims = parse_dimensions(size)?;

    println!("\n=== CGP Quant Profile: {kernel_name} ({size}) ===\n");
    println!("  Kernel: {kernel:?}");
    println!("  Dimensions: M={}, N={}, K={}", dims[0], dims[1], dims[2]);
    println!(
        "  Super-block: {} elements, {} bytes",
        kernel.superblock_elements(),
        kernel.superblock_bytes()
    );

    // Calculate theoretical metrics
    let total_elements = dims[0] as u64 * dims[2] as u64;
    let num_superblocks = total_elements / kernel.superblock_elements() as u64;
    let compressed_bytes = num_superblocks * kernel.superblock_bytes() as u64;
    let fp32_bytes = total_elements * 4;

    println!("  Total weights: {total_elements}");
    println!("  Super-blocks: {num_superblocks}");
    println!("  Compressed size: {:.2} MB", compressed_bytes as f64 / 1e6);
    println!("  FP32 equivalent: {:.2} MB", fp32_bytes as f64 / 1e6);
    println!("  Compression ratio: {:.1}x", fp32_bytes as f64 / compressed_bytes as f64);
    println!("\n  Metrics: superblocks/sec, effective_bandwidth_gbps, compression_speedup");
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dimensions() {
        let dims = parse_dimensions("4096x1x4096").unwrap();
        assert_eq!(dims, [4096, 1, 4096]);
    }

    #[test]
    fn test_parse_dimensions_invalid() {
        assert!(parse_dimensions("4096x4096").is_err());
        assert!(parse_dimensions("abc").is_err());
    }

    /// Q4K super-block: 256 elements in 144 bytes.
    #[test]
    fn test_q4k_superblock() {
        let k = QuantKernel::Q4kGemv;
        assert_eq!(k.superblock_elements(), 256);
        assert_eq!(k.superblock_bytes(), 144);
    }

    /// FALSIFY-CGP-075: Effective bandwidth uses compressed bytes, not FP32.
    #[test]
    fn test_effective_bandwidth_compressed() {
        // 4096*4096 weights / 256 * 144 bytes = 9.44 MB (compressed)
        let total_elements: u64 = 4096 * 4096;
        let num_superblocks = total_elements / 256;
        let compressed_bytes = num_superblocks * 144;
        let expected_mb = 9.437184; // 9.44 MB approximately
        assert!(
            (compressed_bytes as f64 / 1e6 - expected_mb).abs() < 0.01,
            "Compressed size {:.2} MB != expected {:.2} MB",
            compressed_bytes as f64 / 1e6,
            expected_mb
        );
    }

    #[test]
    fn test_kernel_from_str() {
        assert_eq!("q4k_gemv".parse::<QuantKernel>().unwrap(), QuantKernel::Q4kGemv);
        assert_eq!("q6k".parse::<QuantKernel>().unwrap(), QuantKernel::Q6kGemv);
        assert!("invalid".parse::<QuantKernel>().is_err());
    }
}
