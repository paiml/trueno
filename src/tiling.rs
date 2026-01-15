//! Tiling Compute Blocks (TCB) - Work Partitioning for High-Performance Kernels
//!
//! TCBs represent the fundamental unit of work partitioning within `ComputeBrick` kernels.
//! While a `ComputeBrick` defines a logical operation (e.g., Q4_K MatMul), a TCB defines
//! the physical execution strategy—how data is partitioned across the memory hierarchy.
//!
//! # Architecture
//!
//! Tiling occurs at three levels:
//! 1. **Macro-Tile (L3/Global Memory)**: Partitioning across CPU sockets or GPU SMs
//! 2. **Midi-Tile (L2/Shared Memory)**: Partitioning within a thread block or Rayon task
//! 3. **Micro-Tile (Registers)**: Smallest unit processed by SIMD or CUDA warps
//!
//! # Scientific Basis
//!
//! Per Lam et al. (1991), optimal tile sizes balance:
//! - Cache capacity (tile must fit in target cache level)
//! - Reuse factor (maximize arithmetic intensity)
//! - Alignment (match hardware vector width)
//!
//! # Example
//!
//! ```rust
//! use trueno::tiling::{TcbGeometry, TcbLevel, TilingConfig};
//!
//! // GPU Q4_K MatVec tiling
//! let config = TilingConfig::gpu_q4k_matvec();
//! assert_eq!(config.macro_tile.m, 1);
//! assert_eq!(config.macro_tile.k, 256);  // Q4_K superblock alignment
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// TILE-001: TcbGeometry Struct
// ============================================================================

/// Dimensions for a Tiling Compute Block
///
/// Represents the (M, N, K) dimensions of a tile in matrix operations:
/// - M: Output rows
/// - N: Output columns
/// - K: Reduction dimension (inner product)
///
/// # Alignment Constraints
///
/// Per the TCB-03 pattern (Tile Quantization Alignment), K must align with
/// the quantization superblock size:
/// - Q4_0: K % 32 == 0
/// - Q4_K: K % 256 == 0
/// - Q8_0: K % 32 == 0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcbGeometry {
    /// Items processed in M dimension (rows)
    pub m: u32,
    /// Items processed in N dimension (columns)
    pub n: u32,
    /// Reduction dimension (inner product)
    pub k: u32,
    /// Alignment requirement in bytes (typically 16 for SIMD, 32 for AVX2, 64 for AVX-512)
    pub alignment: u32,
}

impl TcbGeometry {
    /// Create a new TCB geometry
    ///
    /// # Panics
    /// Panics if any dimension is zero.
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        assert!(m > 0 && n > 0 && k > 0, "TCB dimensions must be non-zero");
        Self {
            m,
            n,
            k,
            alignment: 16, // Default to SSE/NEON alignment
        }
    }

    /// Create geometry with explicit alignment
    #[must_use]
    pub fn with_alignment(m: u32, n: u32, k: u32, alignment: u32) -> Self {
        assert!(m > 0 && n > 0 && k > 0, "TCB dimensions must be non-zero");
        assert!(
            alignment.is_power_of_two(),
            "Alignment must be power of 2"
        );
        Self { m, n, k, alignment }
    }

    /// Calculate arithmetic intensity (FLOPS per byte loaded)
    ///
    /// For GEMM: AI = (2 * M * N * K) / (M*K + K*N) * sizeof(f32)
    ///
    /// Higher AI means compute-bound; lower means memory-bound.
    #[must_use]
    pub fn arithmetic_intensity(&self) -> f32 {
        let flops = 2.0 * self.m as f64 * self.n as f64 * self.k as f64;
        let bytes = (self.m as f64 * self.k as f64 + self.k as f64 * self.n as f64) * 4.0;
        (flops / bytes) as f32
    }

    /// Calculate total elements in the tile
    #[must_use]
    pub fn total_elements(&self) -> u64 {
        self.m as u64 * self.n as u64
    }

    /// Calculate total FLOPs for this tile
    #[must_use]
    pub fn total_flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }

    /// Check if K dimension aligns with Q4_K superblock (256)
    #[must_use]
    pub fn is_q4k_aligned(&self) -> bool {
        self.k % 256 == 0
    }

    /// Check if K dimension aligns with Q4_0/Q8_0 block (32)
    #[must_use]
    pub fn is_q4_0_aligned(&self) -> bool {
        self.k % 32 == 0
    }

    /// Calculate bytes needed for A tile (M × K × sizeof(f32))
    #[must_use]
    pub fn a_tile_bytes(&self) -> usize {
        self.m as usize * self.k as usize * 4
    }

    /// Calculate bytes needed for B tile (K × N × sizeof(f32))
    #[must_use]
    pub fn b_tile_bytes(&self) -> usize {
        self.k as usize * self.n as usize * 4
    }

    /// Calculate bytes needed for C tile (M × N × sizeof(f32))
    #[must_use]
    pub fn c_tile_bytes(&self) -> usize {
        self.m as usize * self.n as usize * 4
    }

    /// Check if tile fits in given cache size (bytes)
    #[must_use]
    pub fn fits_in_cache(&self, cache_bytes: usize) -> bool {
        self.a_tile_bytes() + self.b_tile_bytes() <= cache_bytes
    }
}

impl Default for TcbGeometry {
    fn default() -> Self {
        // Sensible default: 4×4 micro-tile for SIMD
        Self {
            m: 4,
            n: 4,
            k: 4,
            alignment: 16,
        }
    }
}

impl fmt::Display for TcbGeometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TCB({}×{}×{}, align={}, AI={:.2})",
            self.m,
            self.n,
            self.k,
            self.alignment,
            self.arithmetic_intensity()
        )
    }
}

// ============================================================================
// TILE-001: Tiling Levels
// ============================================================================

/// Tiling hierarchy level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TcbLevel {
    /// Macro-tile: L3 cache / GPU global memory partitioning
    Macro,
    /// Midi-tile: L2 cache / GPU shared memory
    Midi,
    /// Micro-tile: Registers / SIMD lanes
    Micro,
}

impl TcbLevel {
    /// Get typical cache size for this level (x86_64)
    #[must_use]
    pub fn typical_cache_bytes(&self) -> usize {
        match self {
            TcbLevel::Macro => 32 * 1024 * 1024, // 32 MB L3
            TcbLevel::Midi => 256 * 1024,         // 256 KB L2
            TcbLevel::Micro => 32 * 1024,         // 32 KB L1
        }
    }
}

// ============================================================================
// TILE-001: Complete Tiling Configuration
// ============================================================================

/// Complete tiling configuration for a kernel
///
/// Contains geometry for all three tiling levels, enabling hierarchical
/// cache-aware execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilingConfig {
    /// Kernel name for identification
    pub name: String,
    /// Macro-tile geometry (L3/Global)
    pub macro_tile: TcbGeometry,
    /// Midi-tile geometry (L2/Shared)
    pub midi_tile: TcbGeometry,
    /// Micro-tile geometry (Registers)
    pub micro_tile: TcbGeometry,
    /// Target backend
    pub backend: TilingBackend,
}

/// Backend target for tiling configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TilingBackend {
    /// CPU with AVX2 (256-bit SIMD)
    CpuAvx2,
    /// CPU with AVX-512 (512-bit SIMD)
    CpuAvx512,
    /// CPU with NEON (128-bit SIMD)
    CpuNeon,
    /// GPU (CUDA/wgpu)
    Gpu,
    /// Scalar fallback
    Scalar,
}

impl TilingConfig {
    /// Create configuration for GPU Q4_K MatVec
    ///
    /// Optimized for single-token generation where M=1.
    #[must_use]
    pub fn gpu_q4k_matvec() -> Self {
        Self {
            name: "Q4K_MatVec_GPU".into(),
            macro_tile: TcbGeometry::with_alignment(1, 4096, 256, 64),
            midi_tile: TcbGeometry::with_alignment(1, 256, 256, 64),
            micro_tile: TcbGeometry::with_alignment(1, 32, 256, 64),
            backend: TilingBackend::Gpu,
        }
    }

    /// Create configuration for GPU Q4_K MatMul (batched)
    ///
    /// Optimized for prefill where M > 1.
    #[must_use]
    pub fn gpu_q4k_matmul() -> Self {
        Self {
            name: "Q4K_MatMul_GPU".into(),
            macro_tile: TcbGeometry::with_alignment(128, 128, 256, 64),
            midi_tile: TcbGeometry::with_alignment(32, 32, 256, 64),
            micro_tile: TcbGeometry::with_alignment(8, 8, 256, 64),
            backend: TilingBackend::Gpu,
        }
    }

    /// Create configuration for GPU Softmax
    #[must_use]
    pub fn gpu_softmax() -> Self {
        Self {
            name: "Softmax_GPU".into(),
            macro_tile: TcbGeometry::with_alignment(1, 32000, 1, 64),
            midi_tile: TcbGeometry::with_alignment(1, 1024, 1, 64),
            micro_tile: TcbGeometry::with_alignment(1, 32, 1, 64),
            backend: TilingBackend::Gpu,
        }
    }

    /// Create configuration for CPU AVX-512 MatMul
    ///
    /// Optimized for 512-bit wide SIMD:
    /// - 16 floats per ZMM register
    /// - 32 ZMM registers available
    /// - 4×16 micro-kernel uses 8 registers (4 accumulators + 4 scratch)
    #[must_use]
    pub fn cpu_avx512_matmul() -> Self {
        Self {
            name: "MatMul_AVX512".into(),
            macro_tile: TcbGeometry::with_alignment(512, 512, 512, 64),
            midi_tile: TcbGeometry::with_alignment(128, 128, 128, 64),
            // 16 floats wide × 4 rows = 64 elements in registers
            micro_tile: TcbGeometry::with_alignment(4, 16, 128, 64),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Create configuration for CPU AVX-512 Q4K MatVec
    ///
    /// Optimized for Q4_K quantized inference with 512-bit SIMD.
    /// Key differences from AVX2:
    /// - 64-byte aligned for cache line optimization
    /// - 4×1 micro-kernel processes 4 rows simultaneously
    /// - K=256 aligned to Q4_K superblock
    #[must_use]
    pub fn cpu_avx512_q4k_matvec() -> Self {
        Self {
            name: "Q4K_MatVec_AVX512".into(),
            // Large macro-tile to amortize L3 access
            macro_tile: TcbGeometry::with_alignment(4096, 1, 4096, 64),
            // Midi-tile fits in L2 (256KB)
            // 64 rows × 256 K × 0.5625 bytes/element ≈ 9KB weights
            midi_tile: TcbGeometry::with_alignment(64, 1, 256, 64),
            // 4 rows × 1 output, K=256 (Q4_K superblock)
            micro_tile: TcbGeometry::with_alignment(4, 1, 256, 64),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Create configuration for AVX-512 VNNI Q4K×Q8K integer dot product
    ///
    /// AVX-512 VNNI (Vector Neural Network Instructions) provides:
    /// - VPDPBUSD: 8-bit unsigned × 8-bit signed multiply-add to i32
    /// - VPDPWSSD: 16-bit signed × 16-bit signed multiply-add to i32
    ///
    /// This enables pure integer Q4K×Q8K without intermediate f32 conversion.
    #[must_use]
    pub fn cpu_avx512_vnni_q4k_q8k() -> Self {
        Self {
            name: "Q4K_Q8K_VNNI".into(),
            macro_tile: TcbGeometry::with_alignment(4096, 1, 4096, 64),
            midi_tile: TcbGeometry::with_alignment(64, 1, 256, 64),
            // VNNI processes 64 i8 values per ZMM register
            micro_tile: TcbGeometry::with_alignment(4, 1, 256, 64),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Create configuration for CPU AVX2 MatMul
    #[must_use]
    pub fn cpu_avx2_matmul() -> Self {
        Self {
            name: "MatMul_AVX2".into(),
            macro_tile: TcbGeometry::with_alignment(256, 256, 256, 32),
            midi_tile: TcbGeometry::with_alignment(64, 64, 64, 32),
            // 8 floats wide × 4 rows = 32 elements in registers
            micro_tile: TcbGeometry::with_alignment(4, 8, 64, 32),
            backend: TilingBackend::CpuAvx2,
        }
    }

    /// Create configuration for CPU Q4_K MatVec (AVX2)
    #[must_use]
    pub fn cpu_avx2_q4k_matvec() -> Self {
        Self {
            name: "Q4K_MatVec_AVX2".into(),
            // Process 4 rows at a time (4×1 micro-kernel)
            macro_tile: TcbGeometry::with_alignment(4096, 1, 4096, 32),
            midi_tile: TcbGeometry::with_alignment(64, 1, 256, 32),
            // 4 rows × 1 output, K=256 (Q4_K superblock)
            micro_tile: TcbGeometry::with_alignment(4, 1, 256, 32),
            backend: TilingBackend::CpuAvx2,
        }
    }

    /// Create configuration for RMSNorm (CPU)
    #[must_use]
    pub fn cpu_rmsnorm() -> Self {
        Self {
            name: "RMSNorm_CPU".into(),
            macro_tile: TcbGeometry::with_alignment(1, 4096, 1, 32),
            midi_tile: TcbGeometry::with_alignment(1, 256, 1, 32),
            micro_tile: TcbGeometry::with_alignment(1, 16, 1, 32),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Validate that tiling configuration is internally consistent
    pub fn validate(&self) -> Result<(), TilingError> {
        // Macro must be >= Midi >= Micro
        if self.midi_tile.m > self.macro_tile.m
            || self.midi_tile.n > self.macro_tile.n
            || self.midi_tile.k > self.macro_tile.k
        {
            return Err(TilingError::InvalidHierarchy {
                reason: "Midi-tile larger than macro-tile".into(),
            });
        }

        if self.micro_tile.m > self.midi_tile.m
            || self.micro_tile.n > self.midi_tile.n
            || self.micro_tile.k > self.midi_tile.k
        {
            return Err(TilingError::InvalidHierarchy {
                reason: "Micro-tile larger than midi-tile".into(),
            });
        }

        // Check divisibility
        if self.macro_tile.m % self.midi_tile.m != 0 {
            return Err(TilingError::DivisibilityError {
                level: "macro/midi",
                dimension: "M",
                larger: self.macro_tile.m,
                smaller: self.midi_tile.m,
            });
        }

        if self.midi_tile.m % self.micro_tile.m != 0 {
            return Err(TilingError::DivisibilityError {
                level: "midi/micro",
                dimension: "M",
                larger: self.midi_tile.m,
                smaller: self.micro_tile.m,
            });
        }

        Ok(())
    }

    /// Calculate total number of macro-tiles for given problem size
    #[must_use]
    pub fn num_macro_tiles(&self, m: u32, n: u32) -> u32 {
        let m_tiles = (m + self.macro_tile.m - 1) / self.macro_tile.m;
        let n_tiles = (n + self.macro_tile.n - 1) / self.macro_tile.n;
        m_tiles * n_tiles
    }

    /// Calculate total number of midi-tiles within a macro-tile
    #[must_use]
    pub fn midi_tiles_per_macro(&self) -> u32 {
        let m_tiles = self.macro_tile.m / self.midi_tile.m;
        let n_tiles = self.macro_tile.n / self.midi_tile.n;
        m_tiles * n_tiles
    }

    /// Calculate total number of micro-tiles within a midi-tile
    #[must_use]
    pub fn micro_tiles_per_midi(&self) -> u32 {
        let m_tiles = self.midi_tile.m / self.micro_tile.m;
        let n_tiles = self.midi_tile.n / self.micro_tile.n;
        m_tiles * n_tiles
    }
}

// ============================================================================
// TILE-002: Hierarchical Index Calculator
// ============================================================================

/// Index calculator for hierarchical tiling
///
/// Converts between linear indices and (row, col) coordinates at each tiling level.
#[derive(Debug, Clone)]
pub struct TcbIndexCalculator {
    /// Tiling configuration
    config: TilingConfig,
    /// Problem dimensions
    problem_m: u32,
    problem_n: u32,
    problem_k: u32,
}

impl TcbIndexCalculator {
    /// Create a new index calculator for the given problem size
    #[must_use]
    pub fn new(config: TilingConfig, m: u32, n: u32, k: u32) -> Self {
        Self {
            config,
            problem_m: m,
            problem_n: n,
            problem_k: k,
        }
    }

    /// Get macro-tile offset for a given block index
    ///
    /// Returns (row_offset, col_offset) in the output matrix.
    #[must_use]
    pub fn macro_tile_offset(&self, block_idx: u32) -> (u32, u32) {
        let tiles_per_row = (self.problem_n + self.config.macro_tile.n - 1) / self.config.macro_tile.n;
        let row = (block_idx / tiles_per_row) * self.config.macro_tile.m;
        let col = (block_idx % tiles_per_row) * self.config.macro_tile.n;
        (row, col)
    }

    /// Get midi-tile offset within a macro-tile
    #[must_use]
    pub fn midi_tile_offset(&self, midi_idx: u32) -> (u32, u32) {
        let tiles_per_row = self.config.macro_tile.n / self.config.midi_tile.n;
        let row = (midi_idx / tiles_per_row) * self.config.midi_tile.m;
        let col = (midi_idx % tiles_per_row) * self.config.midi_tile.n;
        (row, col)
    }

    /// Get micro-tile offset within a midi-tile
    #[must_use]
    pub fn micro_tile_offset(&self, micro_idx: u32) -> (u32, u32) {
        let tiles_per_row = self.config.midi_tile.n / self.config.micro_tile.n;
        let row = (micro_idx / tiles_per_row) * self.config.micro_tile.m;
        let col = (micro_idx % tiles_per_row) * self.config.micro_tile.n;
        (row, col)
    }

    /// Convert block index to linear memory offset
    ///
    /// For row-major C matrix with given stride.
    #[must_use]
    #[inline]
    pub fn block_to_linear_offset(&self, block_idx: u32, stride: u32) -> usize {
        let (row, col) = self.macro_tile_offset(block_idx);
        (row * stride + col) as usize
    }

    /// Calculate A matrix offset for K-dimension blocking
    #[must_use]
    #[inline]
    pub fn a_offset(&self, macro_row: u32, k_block: u32) -> usize {
        let row = macro_row * self.config.macro_tile.m;
        let col = k_block * self.config.macro_tile.k;
        (row * self.problem_k + col) as usize
    }

    /// Calculate B matrix offset for K-dimension blocking
    #[must_use]
    #[inline]
    pub fn b_offset(&self, k_block: u32, macro_col: u32) -> usize {
        let row = k_block * self.config.macro_tile.k;
        let col = macro_col * self.config.macro_tile.n;
        (row * self.problem_n + col) as usize
    }

    /// Get number of K blocks needed
    #[must_use]
    pub fn num_k_blocks(&self) -> u32 {
        (self.problem_k + self.config.macro_tile.k - 1) / self.config.macro_tile.k
    }

    /// Check if this is a boundary tile (may need masking)
    #[must_use]
    pub fn is_boundary_tile(&self, block_idx: u32) -> bool {
        let (row, col) = self.macro_tile_offset(block_idx);
        row + self.config.macro_tile.m > self.problem_m
            || col + self.config.macro_tile.n > self.problem_n
    }

    /// Get actual tile dimensions (may be smaller at boundaries)
    #[must_use]
    pub fn actual_tile_dims(&self, block_idx: u32) -> (u32, u32) {
        let (row, col) = self.macro_tile_offset(block_idx);
        let actual_m = (self.problem_m - row).min(self.config.macro_tile.m);
        let actual_n = (self.problem_n - col).min(self.config.macro_tile.n);
        (actual_m, actual_n)
    }
}

// ============================================================================
// TILE-002: Memory Layout Helpers
// ============================================================================

/// Memory layout for packed matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingLayout {
    /// Row-major (C-style)
    RowMajor,
    /// Column-major (Fortran-style)
    ColumnMajor,
    /// Panel-major for A (Goto algorithm)
    PanelMajorA,
    /// Panel-major for B (Goto algorithm)
    PanelMajorB,
}

/// Calculate packed index for panel-major A layout
///
/// Panel-major stores micro-panels contiguously for sequential access.
#[must_use]
#[inline]
pub fn pack_a_index(row: usize, col: usize, mr: usize, kc: usize, _mc: usize) -> usize {
    let panel = row / mr;
    let row_in_panel = row % mr;
    panel * mr * kc + col * mr + row_in_panel
}

/// Calculate packed index for panel-major B layout
#[must_use]
#[inline]
pub fn pack_b_index(row: usize, col: usize, nr: usize, kc: usize, _nc: usize) -> usize {
    let panel = col / nr;
    let col_in_panel = col % nr;
    panel * kc * nr + row * nr + col_in_panel
}

/// Apply XOR swizzling for shared memory bank conflict avoidance
///
/// Pattern: idx_swizzled = idx ^ (idx >> 5) for 32-bank architectures.
#[must_use]
#[inline]
pub fn swizzle_index(idx: usize) -> usize {
    idx ^ (idx >> 5)
}

// ============================================================================
// TILE-003: Prefetch Helpers
// ============================================================================

/// Prefetch locality hint
#[derive(Debug, Clone, Copy)]
pub enum PrefetchLocality {
    /// Non-temporal (streaming, evict soon)
    NonTemporal,
    /// L3 cache
    T2,
    /// L2 cache
    T1,
    /// L1 cache (highest priority)
    T0,
}

/// Calculate optimal prefetch distance based on tile geometry and cache level
///
/// Per Ding & Kennedy (2004): distance = memory_latency / compute_time_per_iter
#[must_use]
pub fn optimal_prefetch_distance(geometry: &TcbGeometry, level: TcbLevel) -> usize {
    // Approximate cycles per micro-tile
    let compute_cycles = geometry.m as usize * geometry.n as usize * geometry.k as usize / 8;

    // Memory latency in cycles (approximate for modern x86)
    let mem_latency = match level {
        TcbLevel::Micro => 4,    // L1: ~4 cycles
        TcbLevel::Midi => 12,   // L2: ~12 cycles
        TcbLevel::Macro => 40,  // L3: ~40 cycles
    };

    // Distance = latency / compute_time, minimum 1
    (mem_latency / compute_cycles.max(1)).max(1)
}

// ============================================================================
// TILE-003: Tiled Q4_K MatVec (TCB-01 Pattern)
// ============================================================================

/// Q4_K superblock constants (per GGML specification)
pub const Q4K_SUPERBLOCK_SIZE: usize = 256;
pub const Q4K_SUPERBLOCK_BYTES: usize = 144;

/// Tiled Q4_K MatVec executor
///
/// Implements TCB-01 pattern: Cache-blocked matvec with 4×1 micro-kernel.
///
/// # Memory Layout
///
/// Weights are stored in Q4_K superblock format (144 bytes per 256 elements):
/// - d: f16 (2 bytes) - block scale
/// - dmin: f16 (2 bytes) - block minimum
/// - scales: 12 bytes - 8 sub-block scales (6-bit packed)
/// - qs: 128 bytes - 256 quantized values (4-bit packed)
///
/// # Performance Characteristics
///
/// - L2-resident: Process midi_tile.m rows at a time
/// - Vectorized: 4×1 micro-kernel processes 4 output rows simultaneously
/// - Aligned: K dimension aligned to Q4_K superblock (256)
#[derive(Debug, Clone)]
pub struct TiledQ4KMatvec {
    /// Tiling configuration
    pub config: TilingConfig,
    /// Number of rows (M dimension)
    pub m: usize,
    /// Number of columns (K dimension)
    pub k: usize,
}

impl TiledQ4KMatvec {
    /// Create a new tiled Q4K matvec executor
    ///
    /// # Panics
    /// Panics if K is not aligned to Q4_K superblock size (256).
    #[must_use]
    pub fn new(m: usize, k: usize) -> Self {
        assert!(
            k % Q4K_SUPERBLOCK_SIZE == 0,
            "K dimension ({}) must be aligned to Q4_K superblock size ({})",
            k,
            Q4K_SUPERBLOCK_SIZE
        );

        Self {
            config: TilingConfig::cpu_avx2_q4k_matvec(),
            m,
            k,
        }
    }

    /// Get number of superblocks per row
    #[must_use]
    pub fn superblocks_per_row(&self) -> usize {
        self.k / Q4K_SUPERBLOCK_SIZE
    }

    /// Get total number of superblocks
    #[must_use]
    pub fn total_superblocks(&self) -> usize {
        self.m * self.superblocks_per_row()
    }

    /// Get weight bytes offset for a given row
    #[must_use]
    #[inline]
    pub fn weight_row_offset(&self, row: usize) -> usize {
        row * self.superblocks_per_row() * Q4K_SUPERBLOCK_BYTES
    }

    /// Calculate optimal number of parallel rows based on L2 cache
    ///
    /// Goal: Keep working set in L2 (256KB typical)
    /// Working set = midi_tile.m rows × K × sizeof(Q4K) + K × sizeof(f32)
    #[must_use]
    pub fn optimal_parallel_rows(&self, l2_bytes: usize) -> usize {
        // Q4K: 144 bytes per 256 elements = 0.5625 bytes/element
        let row_bytes = (self.k as f32 * 0.5625) as usize;
        // Input vector: K × 4 bytes
        let input_bytes = self.k * 4;
        // Available for rows
        let available = l2_bytes.saturating_sub(input_bytes);
        // Rows that fit (minimum 4 for micro-kernel)
        (available / row_bytes).max(4)
    }

    /// Execute tiled matvec (reference scalar implementation)
    ///
    /// This is the reference implementation for correctness testing.
    /// Actual SIMD implementation would be in the backends.
    pub fn execute_scalar(&self, weights: &[u8], input: &[f32], output: &mut [f32]) {
        assert_eq!(weights.len(), self.total_superblocks() * Q4K_SUPERBLOCK_BYTES);
        assert_eq!(input.len(), self.k);
        assert_eq!(output.len(), self.m);

        let superblocks_per_row = self.superblocks_per_row();

        for row in 0..self.m {
            let mut sum = 0.0f32;
            let row_offset = row * superblocks_per_row * Q4K_SUPERBLOCK_BYTES;

            for sb in 0..superblocks_per_row {
                let sb_offset = row_offset + sb * Q4K_SUPERBLOCK_BYTES;
                let sb_data = &weights[sb_offset..sb_offset + Q4K_SUPERBLOCK_BYTES];

                // Dequantize and dot product for this superblock
                let input_offset = sb * Q4K_SUPERBLOCK_SIZE;
                sum += self.scalar_superblock_dot(sb_data, &input[input_offset..input_offset + Q4K_SUPERBLOCK_SIZE]);
            }

            output[row] = sum;
        }
    }

    /// Scalar dot product for a single Q4_K superblock
    #[inline]
    fn scalar_superblock_dot(&self, sb_data: &[u8], input: &[f32]) -> f32 {
        // Read header
        let d = f16_to_f32(&sb_data[0..2]);
        let dmin = f16_to_f32(&sb_data[2..4]);
        let scales = &sb_data[4..16];
        let qs = &sb_data[16..144];

        let mut sum = 0.0f32;

        // Process 256 values in 8 chunks of 32
        for chunk in 0..8 {
            let (sc, m) = extract_scale_min_6bit(scales, chunk);
            let d_scale = d * sc;
            let dm = dmin * m;

            let q_offset = chunk * 16; // 32 nibbles = 16 bytes
            let input_offset = chunk * 32;

            // First 32 values: low nibbles then high nibbles
            for i in 0..16 {
                let byte = qs[q_offset + i];
                let q_lo = (byte & 0x0F) as f32;
                let q_hi = ((byte >> 4) & 0x0F) as f32;

                // Low nibble
                let val_lo = d_scale * q_lo - dm;
                sum += val_lo * input[input_offset + i];

                // High nibble
                let val_hi = d_scale * q_hi - dm;
                sum += val_hi * input[input_offset + 16 + i];
            }
        }

        sum
    }

    /// Get tiling statistics for profiling
    #[must_use]
    pub fn stats(&self) -> TilingStats {
        let bytes_per_row = self.superblocks_per_row() * Q4K_SUPERBLOCK_BYTES;
        let total_weight_bytes = self.m * bytes_per_row;
        let input_bytes = self.k * 4;
        let output_bytes = self.m * 4;

        TilingStats {
            total_weight_bytes,
            input_bytes,
            output_bytes,
            superblocks: self.total_superblocks(),
            arithmetic_ops: self.m * self.k * 2, // 2 ops per element (mul + add)
            arithmetic_intensity: (self.m * self.k * 2) as f32 / (total_weight_bytes + input_bytes) as f32,
        }
    }
}

/// Statistics for a tiled operation
#[derive(Debug, Clone)]
pub struct TilingStats {
    /// Total weight bytes
    pub total_weight_bytes: usize,
    /// Input vector bytes
    pub input_bytes: usize,
    /// Output vector bytes
    pub output_bytes: usize,
    /// Number of superblocks
    pub superblocks: usize,
    /// Total arithmetic operations
    pub arithmetic_ops: usize,
    /// Arithmetic intensity (FLOPS/byte)
    pub arithmetic_intensity: f32,
}

/// Convert 2 bytes (f16 IEEE 754) to f32
///
/// Manual implementation to avoid half crate dependency.
/// Format: 1 sign bit, 5 exponent bits, 10 mantissa bits.
#[inline]
fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);

    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal f16 -> normalized f32
        let m = mantissa as f32 / 1024.0;
        let result = m * 2.0f32.powi(-14);
        return if sign == 1 { -result } else { result };
    }

    if exponent == 31 {
        // Inf or NaN
        if mantissa == 0 {
            return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
        }
        return f32::NAN;
    }

    // Normal number
    // f16 bias = 15, f32 bias = 127
    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    let f32_mant = (mantissa as u32) << 13; // 10 bits -> 23 bits
    let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mant;

    f32::from_bits(f32_bits)
}

/// Extract 6-bit scale and min values from packed scales array
///
/// Q4_K uses 6-bit packed scales: 12 bytes encode 8 (scale, min) pairs.
#[inline]
fn extract_scale_min_6bit(scales: &[u8], idx: usize) -> (f32, f32) {
    // Simplified extraction - actual implementation matches GGML layout
    // Scale and min are packed in 6-bit format across the 12-byte array
    let base = idx * 3 / 2;
    let scale = if idx % 2 == 0 {
        (scales[base] & 0x3F) as f32
    } else {
        ((scales[base] >> 6) | ((scales[base + 1] & 0x0F) << 2)) as f32
    };
    let min = if idx % 2 == 0 {
        ((scales[base] >> 6) | ((scales[base + 1] & 0x0F) << 2)) as f32
    } else {
        ((scales[base + 1] >> 4) | ((scales.get(base + 2).unwrap_or(&0) & 0x03) << 4)) as f32
    };
    (scale, min)
}

// ============================================================================
// Error Types
// ============================================================================

/// Tiling configuration errors
#[derive(Debug, Clone)]
pub enum TilingError {
    /// Tile hierarchy is invalid (e.g., micro > midi)
    InvalidHierarchy { reason: String },
    /// Tile dimensions not divisible
    DivisibilityError {
        level: &'static str,
        dimension: &'static str,
        larger: u32,
        smaller: u32,
    },
    /// Tile doesn't fit in cache
    CacheOverflow {
        level: TcbLevel,
        required_bytes: usize,
        available_bytes: usize,
    },
    /// Alignment violation
    AlignmentError {
        required: u32,
        actual: u32,
    },
    /// Quantization alignment violated
    QuantAlignmentError {
        format: &'static str,
        required_k: u32,
        actual_k: u32,
    },
}

impl fmt::Display for TilingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TilingError::InvalidHierarchy { reason } => {
                write!(f, "Invalid tiling hierarchy: {}", reason)
            }
            TilingError::DivisibilityError {
                level,
                dimension,
                larger,
                smaller,
            } => {
                write!(
                    f,
                    "Tiling divisibility error at {}: {} ({}) not divisible by {}",
                    level, dimension, larger, smaller
                )
            }
            TilingError::CacheOverflow {
                level,
                required_bytes,
                available_bytes,
            } => {
                write!(
                    f,
                    "Tile exceeds {:?} cache: {} bytes required, {} available",
                    level, required_bytes, available_bytes
                )
            }
            TilingError::AlignmentError { required, actual } => {
                write!(
                    f,
                    "Alignment error: required {} bytes, actual {} bytes",
                    required, actual
                )
            }
            TilingError::QuantAlignmentError {
                format,
                required_k,
                actual_k,
            } => {
                write!(
                    f,
                    "Quantization alignment error for {}: K must be multiple of {}, got {}",
                    format, required_k, actual_k
                )
            }
        }
    }
}

impl std::error::Error for TilingError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // F301: TCB Output Equivalence - tested via property tests
    // F302: Tile Size Power of 2 - static analysis

    #[test]
    fn test_tcb_geometry_creation() {
        let geom = TcbGeometry::new(4, 8, 256);
        assert_eq!(geom.m, 4);
        assert_eq!(geom.n, 8);
        assert_eq!(geom.k, 256);
        assert_eq!(geom.alignment, 16);
    }

    #[test]
    fn test_tcb_geometry_alignment() {
        let geom = TcbGeometry::with_alignment(4, 16, 128, 64);
        assert_eq!(geom.alignment, 64);
    }

    #[test]
    #[should_panic(expected = "TCB dimensions must be non-zero")]
    fn test_tcb_geometry_zero_dimension() {
        let _ = TcbGeometry::new(0, 8, 256);
    }

    #[test]
    #[should_panic(expected = "Alignment must be power of 2")]
    fn test_tcb_geometry_invalid_alignment() {
        let _ = TcbGeometry::with_alignment(4, 8, 256, 17);
    }

    #[test]
    fn test_arithmetic_intensity() {
        // 4×8×256 tile
        let geom = TcbGeometry::new(4, 8, 256);
        let ai = geom.arithmetic_intensity();
        // AI = 2*4*8*256 / ((4*256 + 256*8) * 4) = 16384 / 12288 ≈ 1.33
        assert!((ai - 1.33).abs() < 0.1);
    }

    #[test]
    fn test_q4k_alignment() {
        let aligned = TcbGeometry::new(4, 8, 256);
        assert!(aligned.is_q4k_aligned());

        let unaligned = TcbGeometry::new(4, 8, 128);
        assert!(!unaligned.is_q4k_aligned());
    }

    #[test]
    fn test_cache_fitting() {
        let geom = TcbGeometry::new(64, 64, 64);
        // A: 64*64*4 = 16KB, B: 64*64*4 = 16KB, total = 32KB
        assert!(geom.fits_in_cache(64 * 1024)); // 64KB cache
        assert!(!geom.fits_in_cache(16 * 1024)); // 16KB cache
    }

    #[test]
    fn test_tiling_config_gpu_q4k_matvec() {
        let config = TilingConfig::gpu_q4k_matvec();
        assert_eq!(config.macro_tile.m, 1);
        assert_eq!(config.macro_tile.k, 256);
        assert!(config.macro_tile.is_q4k_aligned());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tiling_config_cpu_avx2() {
        let config = TilingConfig::cpu_avx2_matmul();
        assert_eq!(config.micro_tile.n, 8); // AVX2 = 8 floats
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tiling_config_validation_failure() {
        let mut config = TilingConfig::cpu_avx2_matmul();
        // Make midi larger than macro (invalid)
        config.midi_tile.m = config.macro_tile.m + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_index_calculator_macro_offset() {
        let config = TilingConfig::cpu_avx2_matmul();
        let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

        let (row, col) = calc.macro_tile_offset(0);
        assert_eq!((row, col), (0, 0));

        let (_row, col) = calc.macro_tile_offset(1);
        assert_eq!(col, config.macro_tile.n);
    }

    #[test]
    fn test_index_calculator_boundary() {
        let config = TilingConfig::cpu_avx2_matmul();

        // With 512×512 problem and 256×256 tiles, first tile is NOT a boundary
        let calc_large = TcbIndexCalculator::new(config.clone(), 512, 512, 256);
        assert!(!calc_large.is_boundary_tile(0));

        // With 100×100 problem and 256×256 tiles, first (only) tile IS a boundary
        let calc_small = TcbIndexCalculator::new(config, 100, 100, 256);
        assert!(calc_small.is_boundary_tile(0));

        // Actual dimensions should be clamped to problem size
        let (actual_m, actual_n) = calc_small.actual_tile_dims(0);
        assert_eq!(actual_m, 100);
        assert_eq!(actual_n, 100);
    }

    #[test]
    fn test_pack_a_index() {
        // mr=4, kc=256, panel 0
        let idx = pack_a_index(0, 0, 4, 256, 64);
        assert_eq!(idx, 0);

        // Second element in first panel
        let idx = pack_a_index(1, 0, 4, 256, 64);
        assert_eq!(idx, 1);

        // First element, second k
        let idx = pack_a_index(0, 1, 4, 256, 64);
        assert_eq!(idx, 4);
    }

    #[test]
    fn test_swizzle_index() {
        // XOR swizzling should avoid bank conflicts
        let idx0 = swizzle_index(0);
        let idx32 = swizzle_index(32);
        // These would conflict without swizzling (both bank 0)
        // With swizzling: 0 ^ 0 = 0, 32 ^ 1 = 33
        assert_ne!(idx0 % 32, idx32 % 32);
    }

    #[test]
    fn test_optimal_prefetch_distance() {
        let geom = TcbGeometry::new(4, 8, 64);
        let dist = optimal_prefetch_distance(&geom, TcbLevel::Midi);
        assert!(dist >= 1);
    }

    // F321: Odd-Sized Matrix Handling
    #[test]
    fn test_odd_sized_matrices() {
        let config = TilingConfig::cpu_avx2_matmul();

        // Test various odd sizes
        for (m, n, k) in [(127, 255, 513), (1, 1, 1), (7, 13, 31)] {
            let calc = TcbIndexCalculator::new(config.clone(), m, n, k);
            let num_tiles = calc.num_k_blocks();
            assert!(num_tiles >= 1);
        }
    }

    // F322: Zero-Padding Efficiency
    #[test]
    fn test_tile_count_calculation() {
        let config = TilingConfig::cpu_avx2_matmul();
        let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

        let num_macro = calc.config.num_macro_tiles(1024, 1024);
        let num_midi = calc.config.midi_tiles_per_macro();
        let num_micro = calc.config.micro_tiles_per_midi();

        assert!(num_macro > 0);
        assert!(num_midi > 0);
        assert!(num_micro > 0);
    }

    // TILE-003: Q4K MatVec Tests
    #[test]
    fn test_tiled_q4k_matvec_creation() {
        let matvec = TiledQ4KMatvec::new(4096, 4096);
        assert_eq!(matvec.m, 4096);
        assert_eq!(matvec.k, 4096);
        assert_eq!(matvec.superblocks_per_row(), 16); // 4096 / 256
        assert_eq!(matvec.total_superblocks(), 4096 * 16);
    }

    #[test]
    #[should_panic(expected = "K dimension")]
    fn test_tiled_q4k_matvec_unaligned_k() {
        let _ = TiledQ4KMatvec::new(4096, 100); // Not aligned to 256
    }

    #[test]
    fn test_tiled_q4k_matvec_weight_offset() {
        let matvec = TiledQ4KMatvec::new(100, 512);
        // Row 0: offset 0
        assert_eq!(matvec.weight_row_offset(0), 0);
        // Row 1: offset = 2 superblocks * 144 bytes = 288
        assert_eq!(matvec.weight_row_offset(1), 2 * Q4K_SUPERBLOCK_BYTES);
    }

    #[test]
    fn test_tiled_q4k_matvec_optimal_rows() {
        let matvec = TiledQ4KMatvec::new(4096, 4096);
        // With 256KB L2, should fit many rows
        let rows = matvec.optimal_parallel_rows(256 * 1024);
        assert!(rows >= 4); // At least micro-kernel size
        assert!(rows <= 4096); // At most all rows
    }

    #[test]
    fn test_tiled_q4k_matvec_stats() {
        let matvec = TiledQ4KMatvec::new(4096, 4096);
        let stats = matvec.stats();

        // Weight bytes: 4096 * 16 * 144 = 9,437,184 bytes
        assert_eq!(stats.superblocks, 4096 * 16);
        // Arithmetic ops: 4096 * 4096 * 2 = 33,554,432
        assert_eq!(stats.arithmetic_ops, 4096 * 4096 * 2);
        // AI should be reasonable for Q4K
        assert!(stats.arithmetic_intensity > 1.0);
    }

    #[test]
    fn test_q4k_constants() {
        assert_eq!(Q4K_SUPERBLOCK_SIZE, 256);
        assert_eq!(Q4K_SUPERBLOCK_BYTES, 144);
    }

    // TILE-004: AVX-512 Register Tiling Tests
    #[test]
    fn test_tiling_config_avx512_matmul() {
        let config = TilingConfig::cpu_avx512_matmul();
        assert_eq!(config.micro_tile.n, 16); // AVX-512 = 16 floats
        assert_eq!(config.micro_tile.alignment, 64); // 64-byte alignment
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tiling_config_avx512_q4k_matvec() {
        let config = TilingConfig::cpu_avx512_q4k_matvec();
        assert!(config.micro_tile.is_q4k_aligned());
        assert_eq!(config.micro_tile.m, 4); // 4×1 micro-kernel
        assert_eq!(config.micro_tile.n, 1); // Single output column (matvec)
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tiling_config_avx512_vnni() {
        let config = TilingConfig::cpu_avx512_vnni_q4k_q8k();
        assert!(config.micro_tile.is_q4k_aligned());
        assert_eq!(config.backend, TilingBackend::CpuAvx512);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_avx512_vs_avx2_tile_sizes() {
        let avx2 = TilingConfig::cpu_avx2_matmul();
        let avx512 = TilingConfig::cpu_avx512_matmul();

        // AVX-512 should have 2x wider micro-tiles
        assert_eq!(avx512.micro_tile.n, avx2.micro_tile.n * 2);

        // AVX-512 should have stricter alignment
        assert!(avx512.micro_tile.alignment >= avx2.micro_tile.alignment);
    }

    // TILE-005: F321-F340 Boundary Handling Tests
    // F321: Odd-Sized Matrix Handling (already exists above)

    // F323: Single-element matrices
    #[test]
    fn test_single_element_matrix() {
        let config = TilingConfig::cpu_avx2_matmul();
        let calc = TcbIndexCalculator::new(config, 1, 1, 256);

        assert!(calc.is_boundary_tile(0));
        let (actual_m, actual_n) = calc.actual_tile_dims(0);
        assert_eq!(actual_m, 1);
        assert_eq!(actual_n, 1);
    }

    // F324: Prime-sized matrices (no clean tiling)
    #[test]
    fn test_prime_sized_matrices() {
        let config = TilingConfig::cpu_avx2_matmul();

        // Prime sizes: 127, 251, 509 (all < macro_tile.m which is 256)
        for size in [127, 251] {
            let calc = TcbIndexCalculator::new(config.clone(), size, size, 256);
            let num_tiles = config.num_macro_tiles(size, size);
            assert!(num_tiles >= 1);

            // Tiles smaller than macro size are boundary tiles
            assert!(calc.is_boundary_tile(0));
        }

        // 509 > 256, so first tile is NOT a boundary, but second tile IS
        let calc = TcbIndexCalculator::new(config.clone(), 509, 509, 256);
        // First tile (0,0 to 255,255) is not boundary for 509×509
        assert!(!calc.is_boundary_tile(0));
        // Second tile (0,256 to 255,508) IS boundary (509-256=253 < 256)
        assert!(calc.is_boundary_tile(1));
    }

    // F325: K dimension exactly equals superblock
    #[test]
    fn test_k_equals_superblock() {
        let matvec = TiledQ4KMatvec::new(100, 256);
        assert_eq!(matvec.superblocks_per_row(), 1);
        assert_eq!(matvec.total_superblocks(), 100);
    }

    // F326: Very large M dimension
    #[test]
    fn test_large_m_dimension() {
        let matvec = TiledQ4KMatvec::new(100_000, 256);
        assert_eq!(matvec.superblocks_per_row(), 1);
        assert_eq!(matvec.total_superblocks(), 100_000);
        // Should still compute optimal rows
        let rows = matvec.optimal_parallel_rows(256 * 1024);
        assert!(rows >= 4);
    }

    // F327: Very large K dimension
    #[test]
    fn test_large_k_dimension() {
        let matvec = TiledQ4KMatvec::new(10, 32768); // 32K hidden dim
        assert_eq!(matvec.superblocks_per_row(), 128);
        let stats = matvec.stats();
        assert!(stats.arithmetic_intensity > 0.0);
    }

    // F328: Tile offset at boundaries
    #[test]
    fn test_tile_offset_boundaries() {
        let config = TilingConfig::cpu_avx2_matmul();
        let calc = TcbIndexCalculator::new(config.clone(), 1000, 1000, 256);

        // Last tile index
        let num_tiles = config.num_macro_tiles(1000, 1000);
        let last_idx = num_tiles - 1;

        let (row, col) = calc.macro_tile_offset(last_idx);
        // Should be within bounds
        assert!(row < 1000 + config.macro_tile.m);
        assert!(col < 1000 + config.macro_tile.n);
    }

    // F329: Index calculator consistency
    #[test]
    fn test_index_calculator_consistency() {
        let config = TilingConfig::cpu_avx2_matmul();
        let calc = TcbIndexCalculator::new(config.clone(), 512, 512, 256);

        // Macro offset for tile 0 should be (0, 0)
        let (r0, c0) = calc.macro_tile_offset(0);
        assert_eq!((r0, c0), (0, 0));

        // Linear offset should match
        let linear = calc.block_to_linear_offset(0, 512);
        assert_eq!(linear, 0);

        // A and B offsets at k_block=0 should also be 0
        let a_off = calc.a_offset(0, 0);
        let b_off = calc.b_offset(0, 0);
        assert_eq!(a_off, 0);
        assert_eq!(b_off, 0);
    }

    // F330: Midi/micro tile divisibility
    #[test]
    fn test_tile_divisibility() {
        let config = TilingConfig::cpu_avx512_matmul();

        // Macro should be divisible by midi
        assert_eq!(config.macro_tile.m % config.midi_tile.m, 0);
        assert_eq!(config.macro_tile.n % config.midi_tile.n, 0);

        // Midi should be divisible by micro
        assert_eq!(config.midi_tile.m % config.micro_tile.m, 0);
        assert_eq!(config.midi_tile.n % config.micro_tile.n, 0);
    }

    // F331: f16 to f32 conversion
    #[test]
    fn test_f16_conversion() {
        // Zero
        assert_eq!(f16_to_f32(&[0x00, 0x00]), 0.0);

        // One (0x3C00 in f16)
        let one = f16_to_f32(&[0x00, 0x3C]);
        assert!((one - 1.0).abs() < 0.001);

        // Negative one (0xBC00)
        let neg_one = f16_to_f32(&[0x00, 0xBC]);
        assert!((neg_one - (-1.0)).abs() < 0.001);

        // Infinity (0x7C00)
        assert!(f16_to_f32(&[0x00, 0x7C]).is_infinite());

        // NaN (0x7C01)
        assert!(f16_to_f32(&[0x01, 0x7C]).is_nan());
    }
}
