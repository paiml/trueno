# GPU Performance

This chapter presents empirical GPU performance findings from benchmarking on NVIDIA RTX 4090, documenting when GPU acceleration provides value versus SIMD.

## Executive Summary

**Date**: 2025-11-23
**Hardware**: NVIDIA GeForce RTX 4090 (24GB VRAM)
**Driver**: 570.195.03
**Platform**: Linux 6.8.0-87-generic
**Software**: Trueno v0.7.0, wgpu v27.0.1

### Key Findings

- ✅ **GPU wins for matrix operations**: 81x speedup on 1000×1000 matrix multiplication
- ❌ **GPU fails for vector operations**: 2000x+ slower than SIMD due to 3.5ms fixed overhead
- 🚀 **SIMD vastly superior** for vector ops: Zero transfer overhead, 200-400% speedup
- 💡 **Hybrid approach recommended**: Use SIMD by default, GPU only for matmul >500×500

## GPU Transfer Overhead

### Fixed Overhead Breakdown

Empirically measured per-operation costs:

| Component | Time | Description |
|-----------|------|-------------|
| Buffer creation | ~0.5 ms | Allocate GPU-side memory |
| CPU→GPU transfer | ~1.5 ms | PCIe bandwidth limitation |
| Kernel dispatch | ~0.3 ms | GPU scheduling overhead |
| GPU→CPU readback | ~1.2 ms | PCIe bandwidth limitation |
| **Total** | **~3.5 ms** | **Minimum per operation** |

### Implications for Different Workload Sizes

| Size | Data Volume | Overhead Impact | GPU Viable? |
|------|-------------|-----------------|-------------|
| 1K | 4 KB | 875 µs/KB | ❌ Never competitive |
| 10K | 40 KB | 87.5 µs/KB | ❌ Still dominated by overhead |
| 100K | 400 KB | 8.75 µs/KB | ⚠️ Marginal for complex ops |
| 1M | 4 MB | 0.875 µs/KB | ✅ Good amortization |

**Rule of thumb**: GPU only becomes competitive when **compute time >> 3.5ms**.

## Matrix Multiplication (GPU Excels)

Matrix multiplication has O(n³) complexity, which overwhelms the fixed 3.5ms overhead at large scales.

### Benchmark Results

| Size | GPU Time | Scalar Time | Speedup | GPU Throughput | Scalar Throughput |
|------|----------|-------------|---------|----------------|-------------------|
| 100×100 | 4.14 ms | 530.8 µs | **0.13x** ❌ | 241.7 Gelem/s | 1.88 Gelem/s |
| 500×500 | 4.59 ms | 77.4 ms | **16.9x** ✅ | 27.2 Gelem/s | 1.61 Gelem/s |
| 1000×1000 | 7.84 ms | 638.7 ms | **81.5x** ✅ | 127.6 Gelem/s | 1.57 Gelem/s |

### Why GPU Wins for Matrix Multiplication

**Compute complexity dominates transfer cost:**

- 100×100: 1M operations → 531µs scalar → GPU overhead too high
- 500×500: 125M operations → 77ms scalar → GPU wins at 4.6ms
- 1000×1000: 1B operations → 639ms scalar → GPU wins at 7.8ms

**Threshold**: GPU becomes competitive at **>500×500 (250,000 elements)**.

## Vector Operations (GPU Fails)

Simple vector operations are dominated by the 3.5ms fixed transfer overhead.

### Vector Addition Results

| Size | GPU Time | Scalar Time | Speedup | GPU Throughput | Scalar Throughput |
|------|----------|-------------|---------|----------------|-------------------|
| 1K | 3.26 ms | 71.0 ns | **0.00002x** ❌ | 306.4 Kelem/s | 14.09 Gelem/s |
| 10K | 3.44 ms | 819.0 ns | **0.0002x** ❌ | 2.91 Melem/s | 12.21 Gelem/s |
| 100K | 3.51 ms | 10.06 µs | **0.003x** ❌ | 28.45 Melem/s | 9.94 Gelem/s |
| 1M | 5.98 ms | 96.5 µs | **0.016x** ❌ | 167.3 Melem/s | 10.37 Gelem/s |

### Dot Product Results

| Size | GPU Time | Scalar Time | Speedup |
|------|----------|-------------|---------|
| 1K | 3.45 ms | 567.4 ns | **0.0002x** ❌ |
| 10K | 3.32 ms | 6.30 µs | **0.002x** ❌ |
| 100K | 4.81 ms | 63.2 µs | **0.013x** ❌ |
| 1M | 6.25 ms | 614.1 µs | **0.098x** ❌ |

**Key finding**: Even at 1M elements, GPU is still 62x slower than scalar due to transfer overhead. Reduction overhead compounds the problem.

## Activation Functions

Activation functions are more compute-intensive than simple vector operations, but still suffer from transfer overhead.

### ReLU (Simple Operation)

| Size | GPU Time | Scalar Time | Speedup |
|------|----------|-------------|---------|
| 10K | 3.49 ms | 559.9 ns | **0.0002x** ❌ |
| 100K | 3.75 ms | 6.37 µs | **0.002x** ❌ |
| 1M | 6.03 ms | 67.1 µs | **0.011x** ❌ |

### Sigmoid (Transcendental)

| Size | GPU Time | Scalar Time | Speedup |
|------|----------|-------------|---------|
| 10K | 3.64 ms | 20.99 µs | **0.006x** ❌ |
| 100K | 3.75 ms | 207.4 µs | **0.055x** ❌ |
| 1M | 5.81 ms | 3.18 ms | **0.55x** ❌ |

### GELU (Very Compute-Heavy)

| Size | GPU Time | Scalar Time | Speedup |
|------|----------|-------------|---------|
| 10K | 3.60 ms | 101.2 µs | **0.028x** ❌ |
| 100K | 3.72 ms | 327.0 µs | **0.088x** ❌ |
| 1M | 5.81 ms | 3.19 ms | **0.55x** ❌ |

**Key finding**: Even compute-heavy operations like GELU and sigmoid are slower on GPU due to transfer overhead. At 1M elements, GPU barely reaches parity with scalar.

### Softmax (Multi-Pass Algorithm)

| Size | GPU Time | Scalar Time | Speedup |
|------|----------|-------------|---------|
| 10K | 16.75 ms | 29.2 µs | **0.002x** ❌ |
| 100K | 16.26 ms | 292.3 µs | **0.018x** ❌ |
| 1M | 22.79 ms | 3.01 ms | **0.13x** ❌ |

**Why softmax is even worse**: Multi-pass algorithms require 3 GPU dispatches (max, exp, sum), compounding transfer overhead to ~10ms base cost.

## SIMD vs GPU Comparison

Golden traces from Renacer v0.6.2 show SIMD baseline performance:

### SIMD Performance (SSE2)

From `golden_traces/performance_demo_summary.txt`:

| Operation | Size | Scalar | SSE2 | Speedup | Runtime | Syscalls |
|-----------|------|--------|------|---------|---------|----------|
| Dot Product | 10K | 6.26µs | 1.55µs | **303%** | 1.507ms | 138 |
| Sum Reduction | 10K | 7.12µs | 1.69µs | **320%** | 1.507ms | 138 |
| Max Finding | 10K | 4.19µs | 1.06µs | **297%** | 1.507ms | 138 |
| Element-wise Add | 10K | 1.44µs | 1.10µs | 30% | 1.507ms | 138 |
| Element-wise Mul | 10K | 1.10µs | 1.10µs | 0% | 1.507ms | 138 |

### Head-to-Head Comparison

| Operation | Size | SIMD (SSE2) | GPU (RTX 4090) | Winner |
|-----------|------|-------------|----------------|--------|
| Dot Product | 10K | 1.55µs | 3,324µs | **SIMD 2144x faster** |
| Vector Add | 10K | 1.10µs | 3,439µs | **SIMD 3127x faster** |
| Vector Add | 1M | 96.5µs | 5,978µs | **SIMD 62x faster** |
| Matrix Mul | 1000×1000 | 638.7ms | 7.84ms | **GPU 81x faster** |

### Key Insights

- ✅ **SIMD dominates** for vector operations at ALL sizes due to zero overhead
- ✅ **GPU wins** for matrix operations (O(n³) complexity) at large scales
- 💡 **Hybrid approach**: Use SIMD by default, GPU only for matmul >500×500

## Current GPU Thresholds in Trueno

Based on empirical findings, Trueno uses these thresholds:

```rust
// src/vector.rs:1316
const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower

// src/matrix.rs:268
const GPU_THRESHOLD: usize = 500; // Empirical: 2x at 500×500, 9.6x at 1000×1000
```

**Rationale**:
- Vector operations: Transfer overhead will always dominate → GPU disabled
- Matrix operations: O(n³) complexity amortizes overhead → GPU at 500×500

## When to Use GPU

Use GPU when **all** of these conditions are met:

1. **Operation complexity**: O(n²) or higher (matrix multiplication, convolution)
2. **Data size**: >500×500 elements for matrix ops
3. **Compute time**: Operation takes >10ms on CPU
4. **Batch processing**: Multiple operations can be batched (future v2.0 API)

### GPU is NOT recommended for:

- ❌ Vector operations (add, mul, dot, reduce) - use SIMD
- ❌ Activation functions (relu, sigmoid, tanh) - use SIMD
- ❌ Small matrices (<500×500) - overhead dominates
- ❌ Single operations - transfer overhead too high

## Async Batch API ✅ (v0.3.0 - AVAILABLE NOW)

**Status**: Fully implemented and tested (previously documented as "Future v2.0")

The async batch API solves the transfer overhead problem by queuing multiple operations and executing them in a single batch, amortizing the 3.5ms overhead across all operations.

### Transfer Overhead Reduction

**Traditional Synchronous API** (current default):
```rust
// ❌ 3 operations = 3 × 3.5ms = 10.5ms overhead
let a = gpu.vec_add(&input1, &input2)?;  // Upload → Compute → Download
let b = gpu.scale(&a, 2.0)?;             // Upload → Compute → Download
let c = gpu.relu(&b)?;                   // Upload → Compute → Download
// Total: 6 GPU transfers (3 uploads + 3 downloads)
```

**Async Batch API** (recommended for chained operations):
```rust
use trueno::backends::gpu::{GpuDevice, GpuCommandBatch};

// ✅ 3 operations = 1 × 3.5ms = 3.5ms overhead
let device = GpuDevice::new()?;
let mut batch = GpuCommandBatch::new(device);

// Queue operations (no GPU execution yet!)
let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
let a = batch.add(input, other);
let b = batch.scale(a, 2.0);
let c = batch.relu(b);

// Execute entire batch in one GPU round-trip
batch.execute().await?;

// Read final result
let result = batch.read(c).await?;
// Total: 2 GPU transfers (1 upload + 1 download)
```

### Performance Benefits

| Metric | Traditional API | Batch API | Improvement |
|--------|----------------|-----------|-------------|
| **GPU Transfers** | 6 (3↑ + 3↓) | 2 (1↑ + 1↓) | **3x fewer** |
| **Overhead** | 3 × 3.5ms = 10.5ms | 1 × 3.5ms = 3.5ms | **3x reduction** |
| **Expected Speedup** | Baseline | 1.5-2x faster | For GPU-bound workloads |

### When to Use Batch API

**✅ Use batch API when:**
- Chaining multiple GPU operations (>2 ops)
- Processing large workloads where GPU is beneficial (matmul >500×500)
- Amortizing transfer overhead is critical

**❌ Stick with traditional API when:**
- Single operation only
- Interactive/real-time workloads requiring immediate results
- Workloads small enough that SIMD is faster anyway

### Complete Example

See `examples/gpu_batch_demo.rs` for three comprehensive demonstrations:

1. **Single Operation** - Baseline batch API usage
2. **Batched Operations** - ReLU → Scale → Add pipeline
3. **ML Pipeline** - `y = ReLU(x * W + b)` simulation

```bash
# Run the demonstration
cargo run --example gpu_batch_demo --features gpu --release
```

### Implementation Details

- **Location**: `src/backends/gpu/batch.rs` (1,008 lines)
- **Tests**: 8 comprehensive tests (all passing)
- **Operations**: relu, scale, add, mul, dot
- **API**: Fully async with tokio integration
- **Safety**: Type-safe buffer IDs prevent invalid operations

### Future Enhancements (v0.4.0+)

While the batch API is complete, future improvements may include:

- **Automatic optimization**: Detect operation chains and auto-batch
- **More operations**: Expand beyond current 5 operations (relu, scale, add, mul, dot)
- **Graph optimization**: Reorder operations for maximum efficiency
- **Multi-GPU**: Distribute batches across multiple GPUs
- **Persistent buffers**: Reuse buffers across multiple batch executions

## Hardware Details

```
GPU: NVIDIA GeForce RTX 4090
├─ Architecture: Ada Lovelace
├─ CUDA Cores: 16,384
├─ Memory: 24GB GDDR6X
├─ Memory Bandwidth: 1,008 GB/s
├─ Boost Clock: 2.52 GHz
└─ TDP: 450W

Driver: 570.195.03
Platform: Linux 6.8.0-87-generic (x86_64)
```

## Validation and Testing

### Quality Gates

- ✅ All 13 GPU operations benchmarked
- ✅ 4 size ranges tested per operation
- ✅ Statistical significance (10 samples, CV <5%)
- ✅ Comparison against scalar baseline
- ✅ Clippy: Zero warnings
- ✅ Coverage: 90.40% (≥90% threshold)
- ✅ GPU initialization verified
- ✅ Correctness tests pass

### Golden Trace Integration

Performance budgets established via `renacer.toml`:

```toml
[performance.budgets]
# SIMD operations should complete in <2ms with <200 syscalls
backend_detection = { max_time_ms = 2.0, max_syscalls = 200 }
matrix_operations = { max_time_ms = 2.0, max_syscalls = 200 }
activation_functions = { max_time_ms = 2.0, max_syscalls = 200 }
```

Validation tests in `tests/golden_trace_validation.rs` ensure SIMD performance doesn't regress.

## Recommendations

### Immediate Actions

1. **Use SIMD by default** for all vector operations
2. **Reserve GPU for matrix operations** >500×500
3. **Document transfer overhead** prominently in API docs
4. **Educate users** that GPU is not always faster

### Future Enhancements (v2.0)

1. **Async batch API** to amortize transfer overhead
2. **Persistent GPU buffers** for frequently-used data
3. **Hybrid CPU/GPU scheduling** with overlap
4. **Profile-guided optimization** for dynamic thresholds

## References

- Full benchmark report: `docs/gpu-benchmark-report-2025-11-23.md`
- Golden traces: `golden_traces/` directory
- Golden trace analysis: `golden_traces/ANALYSIS.md`
- SIMD performance: `golden_traces/performance_demo_summary.txt`
- Renacer configuration: `renacer.toml`
- GPU bug fix: Commit b5ca0af (missing device.poll() in wgpu v27)

## WebGPU for WASM (v0.7.3)

Trueno v0.7.3 introduces the `gpu-wasm` feature enabling GPU compute in browsers via WebGPU.

### Feature Flag

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
trueno = { version = "0.7.3", features = ["gpu-wasm"] }
```

### Platform Differences

| Platform | Sync API | Async API | Runtime |
|----------|----------|-----------|---------|
| Native | ✅ `GpuDevice::new()` | ✅ `new_async()` | pollster |
| WASM | ❌ (can't block) | ✅ `new_async()` | wasm-bindgen-futures |

### Async-First Design

All GPU operations now have async variants (`*_async`) that work on both native and WASM:

```rust
// Works on all platforms
let device = GpuDevice::new_async().await?;
device.matmul_async(&a, &b, &mut result, m, k, n).await?;
device.relu_async(&input, &mut output).await?;
```

### Runtime Detection

```rust
use trueno::backends::gpu::runtime;

if runtime::sync_available() {
    // Native: can use sync APIs
    let device = GpuDevice::new()?;
} else {
    // WASM: must use async
    let device = GpuDevice::new_async().await?;
}
```

### Real-World Example: trueno-viz

[trueno-viz](https://github.com/paiml/trueno-viz) demonstrates browser-based GPU compute with Trueno:

- WebGPU-accelerated matrix operations
- WASM-compiled Rust for client-side processing
- Interactive visualizations with GPU compute

See [GPU Backend Architecture](../architecture/gpu-backend.md) for complete WebGPU documentation.

## Next Steps

- **[Backend Comparison](./backend-comparison.md)** - Detailed SIMD vs GPU trade-offs
- **[Benchmarks Overview](./benchmarks.md)** - Complete benchmark methodology
- **[Optimization Guide](./optimization-guide.md)** - How to choose the right backend
- **[Profiling](./profiling.md)** - Using Renacer for performance analysis
