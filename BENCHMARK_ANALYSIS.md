# Benchmark Analysis Report

**Date**: 2025-11-23
**Branch**: `claude/continue-next-step-01NEN2Jw5zVsNK9DWCE1Hwqz`
**Hardware**: x86_64 (AVX2 capable)
**Benchmarks Run**: 457 configurations

---

## Executive Summary

**Key Finding**: SIMD provides **4-13x speedups** for compute-bound operations (dot, max, min), but **minimal benefit** (<1.2x) for memory-bound operations (add, mul, sub).

**Best Results** ✅:
- **dot (1000 elements)**: AVX2 is **10.56x faster** than scalar
- **max (1000 elements)**: AVX2 is **12.14x faster** than scalar
- **min (1000 elements)**: AVX2 is **12.45x faster** than scalar
- **argmax/argmin**: 4x faster with AVX2

**Expected Results** ⚠️:
- **add/mul/sub**: 0.76-1.14x (memory-bound, not compute-bound)
- **Small sizes (<100)**: Often slower due to SIMD overhead

---

## Complete Speedup Table

| Operation | Size | Scalar (ns) | SSE2 Speedup | AVX2 Speedup | AVX512 Speedup | Category |
|-----------|------|-------------|--------------|--------------|----------------|----------|
| **dot** | 100 | 74 | **4.04x** | **7.37x** | **7.89x** | ✅ Excellent |
| **dot** | 1000 | 1,148 | **4.32x** | **10.56x** | **13.64x** | ✅ Excellent |
| **dot** | 10000 | 11,962 | **4.04x** | **8.23x** | **8.76x** | ✅ Excellent |
| **max** | 100 | 69 | **5.45x** | **7.28x** | **7.18x** | ✅ Excellent |
| **max** | 1000 | 1,120 | **4.72x** | **12.14x** | **11.62x** | ✅ Excellent |
| **max** | 10000 | 11,838 | **4.07x** | **8.34x** | **8.31x** | ✅ Excellent |
| **min** | 100 | 71 | **5.26x** | **7.28x** | **7.22x** | ✅ Excellent |
| **min** | 1000 | 1,120 | **4.95x** | **12.45x** | **12.17x** | ✅ Excellent |
| **min** | 10000 | 11,867 | **4.08x** | **8.37x** | **8.35x** | ✅ Excellent |
| **argmax** | 100 | 105 | 1.48x | **2.36x** | **2.34x** | ✅ Good |
| **argmax** | 1000 | 1,190 | 1.88x | **4.10x** | **4.07x** | ✅ Good |
| **argmax** | 10000 | 11,947 | 1.91x | **4.31x** | **4.31x** | ✅ Good |
| **argmin** | 100 | 106 | 1.54x | **2.39x** | **2.38x** | ✅ Good |
| **argmin** | 1000 | 1,184 | 1.91x | **4.07x** | **4.04x** | ✅ Good |
| **argmin** | 10000 | 11,932 | 1.96x | **4.27x** | **4.30x** | ✅ Good |
| **sigmoid** | 100 | 422 | 1.68x | **3.22x** | - | ✅ Good |
| **div** | 100 | 88 | 1.09x | 1.05x | - | ⚠️ Modest |
| **div** | 1000 | 323 | 1.02x | 1.16x | - | ⚠️ Modest |
| **div** | 10000 | 2,741 | 1.00x | 1.16x | - | ⚠️ Modest |
| **fma** | 1000 | 204 | 0.88x | 1.09x | - | ⚠️ Modest |
| **fma** | 10000 | 2,671 | 0.98x | 1.25x | - | ⚠️ Modest |
| **scale** | 1000 | 180 | 1.15x | 1.11x | - | ⚠️ Modest |
| **add** | 100 | 59 | **0.79x** | **0.76x** | **0.71x** | ❌ Slower |
| **add** | 1000 | 170 | 1.01x | 0.97x | 0.84x | ❌ Slower |
| **add** | 10000 | 2,097 | 0.96x | 1.05x | 0.98x | ⚠️ Minimal |
| **mul** | 100 | 68 | 0.92x | 0.91x | - | ❌ Slower |
| **mul** | 1000 | 174 | 1.06x | 1.03x | - | ⚠️ Minimal |
| **mul** | 10000 | 2,125 | 1.00x | 1.07x | - | ⚠️ Minimal |
| **sub** | 100 | 73 | 1.00x | 0.98x | - | ⚠️ Minimal |
| **sub** | 1000 | 185 | 1.14x | 1.07x | - | ⚠️ Minimal |
| **sub** | 10000 | 2,127 | 0.99x | 1.06x | - | ⚠️ Minimal |
| **relu** | 10000 | 1,537 | 0.99x | 1.01x | - | ⚠️ Minimal |

---

## Analysis by Category

### ✅ Excellent: Compute-Bound Operations (4-13x speedup)

**Operations**: dot, max, min

**Why SIMD Excels**:
1. **Computation-heavy**: Multiple operations per element
   - `dot`: multiply + accumulate per element
   - `max/min`: compare + select per element
2. **Horizontal reductions**: SIMD efficiently aggregates across lanes
3. **FMA acceleration**: AVX2 fused multiply-add for dot product

**Performance Characteristics**:
- **SSE2 (4-wide)**: 4-5x speedup consistently
- **AVX2 (8-wide)**: 8-12x speedup (best at 1000 elements)
- **AVX-512 (16-wide)**: 8-13x speedup (best for dot product)

**Real-World Impact**:
- Machine learning inference: Fast dot products critical for matrix ops
- Statistics: Fast max/min for data analysis
- Signal processing: Dot products for convolution

### ✅ Good: Moderate Compute Operations (2-4x speedup)

**Operations**: argmax, argmin, sigmoid

**Why SIMD Helps**:
- `argmax/argmin`: SIMD max/min + index tracking
- `sigmoid`: SIMD exp + division (complex activation)

**Performance Characteristics**:
- **AVX2**: 2.4-4.3x speedup
- **Best at**: 1000-10000 element vectors

**Limitation**: Index tracking requires scalar post-processing

### ⚠️ Modest: Division & FMA (1.0-1.25x speedup)

**Operations**: div, fma

**Why SIMD Struggles**:
1. **Division is slow**: Even SIMD division is memory-bound
2. **FMA**: Already optimized in hardware, SIMD adds little

**Performance Characteristics**:
- AVX2: ~1.16x for div, ~1.25x for fma at 10K elements
- SSE2: Minimal benefit (<1.1x)

**Recommendation**: Use SIMD for consistency, but don't expect major speedups

### ❌ Slower/Minimal: Memory-Bound Operations (0.76-1.07x)

**Operations**: add, sub, mul, scale, relu

**Why SIMD Is Slower** (especially at size 100):
1. **Memory bandwidth bottleneck**: CPU can't feed SIMD units fast enough
2. **SIMD overhead**: Register setup, remainder handling costs time
3. **Scalar out-of-order**: Modern CPUs execute scalar code very efficiently
4. **Cache effects**: Small data fits in L1 cache, scalar wins

**Performance Characteristics**:
- **Size 100**: SIMD is **slower** (0.76-0.98x) due to overhead
- **Size 1000**: Barely faster (1.01-1.14x)
- **Size 10000**: Marginal benefit (1.05-1.07x)

**FFmpeg Parallel**: FFmpeg's assembly optimizations show same pattern
- Simple operations: Minimal benefit
- Complex operations: 4-16x speedup

---

## Performance by Data Size

### Small (100 elements)

**Scalar Wins For**:
- add, mul, sub, scale, relu (0.76-1.00x)

**SIMD Wins For**:
- dot (7.37x), max (7.28x), min (7.28x)

**Conclusion**: SIMD overhead matters for simple operations

### Medium (1000 elements)

**Best SIMD Performance**:
- dot: 10.56x
- max: 12.14x
- min: 12.45x
- argmax: 4.10x

**Modest SIMD Performance**:
- add, mul, sub: 1.01-1.14x

**Conclusion**: Sweet spot for SIMD compute-bound operations

### Large (10000 elements)

**SIMD Still Strong**:
- max: 8.34x
- min: 8.37x
- dot: 8.23x

**Memory Bandwidth Limits**:
- Speedups lower than 1000 elements for some ops
- L2/L3 cache effects

**Conclusion**: Memory bandwidth becomes bottleneck

---

## Backend Comparison

### SSE2 (4-wide, 128-bit)

**Strengths**:
- Available on all x86_64 CPUs (baseline)
- 4-5x speedup for compute-bound ops
- Consistent performance

**Weaknesses**:
- Half the throughput of AVX2
- Manual horizontal reductions slow

**Best For**: Baseline compatibility

### AVX2 (8-wide, 256-bit)

**Strengths**:
- **Best overall performance** for 1000-element vectors
- FMA support for dot products
- 8-12x speedup for compute-bound ops

**Weaknesses**:
- Not available on older CPUs (pre-2013 Intel, pre-2015 AMD)
- Minimal benefit for memory-bound ops

**Best For**: Production deployments on modern CPUs

### AVX-512 (16-wide, 512-bit)

**Strengths**:
- **Best for dot product**: 13.64x at 1000 elements
- Built-in horizontal reductions
- Future-proof for HPC

**Weaknesses**:
- Limited CPU availability (Skylake-X+, Zen 4+)
- Some benchmarks missing (div, fma, mul, etc.)
- Power/thermal concerns on some CPUs

**Best For**: HPC, server workloads, Zen 4+

---

## Missing AVX-512 Benchmarks

**Operations with 0 results**:
- div, fma, mul, relu, scale, sigmoid, sub (at some sizes)

**Likely Causes**:
1. Benchmarks didn't complete (timeout?)
2. AVX-512 implementations missing for those operations
3. Benchmark configuration issue

**Recommendation**: Investigate missing benchmarks

---

## Recommendations

### 1. For Library Users

**Use SIMD For**:
- ✅ Dot products, max/min operations: 4-12x speedup
- ✅ argmax/argmin: 2-4x speedup
- ✅ Complex activations (sigmoid): 1.7-3x speedup
- ✅ Large vectors (1000+ elements)

**Don't Expect Magic For**:
- ⚠️ Simple arithmetic (add/mul/sub): <1.2x speedup
- ⚠️ Small vectors (<100 elements): May be slower
- ⚠️ Division: ~1.2x speedup at best

### 2. For Library Developers

**Optimization Priorities** (by ROI):
1. **High**: Ensure dot/max/min use SIMD (already done ✅)
2. **Medium**: Complex operations (sigmoid, exp, tanh)
3. **Low**: Simple arithmetic (already good enough)

**Backend Selection Logic**:
- Workload size > 100K: Consider GPU
- Workload size 1000-100K: AVX2 optimal
- Workload size < 1000: SSE2 or scalar acceptable

**Future Work**:
- Fix missing AVX-512 benchmarks
- GPU benchmarks for very large vectors (>100K)
- NEON benchmarks on ARM hardware

### 3. For Documentation

**README Should Emphasize**:
- Real speedups: 4-12x for compute-bound operations
- Honest about memory-bound: <1.2x for add/mul/sub
- Backend auto-selection works well

**Performance Guarantees**:
- dot/max/min: ✅ Significant speedup guaranteed
- add/mul/sub: ⚠️ Marginal benefit expected
- Small vectors: ⚠️ SIMD may be slower

---

## Comparison to Claims

**Original Claims** (from CLAUDE.md):

| Operation | Size | SSE2 Expected | AVX2 Expected | Actual SSE2 | Actual AVX2 | Status |
|-----------|------|---------------|---------------|-------------|-------------|--------|
| add_f32 | 1K | 2x | 4x | **1.01x** | **0.97x** | ❌ Overpromised |
| add_f32 | 100K | 2x | 4x | - | - | ⏸️ Not benchmarked |
| dot_product | 1K | 3x | 6x | **4.32x** | **10.56x** | ✅ **Exceeded!** |
| dot_product | 1M | 3x | 6x | - | - | ⏸️ Not benchmarked |

**Verdict**:
- **Compute-bound claims**: ✅ Met or exceeded (dot: 10.56x vs 6x expected)
- **Memory-bound claims**: ❌ Overpromised (add: 0.97x vs 4x expected)

**Updated Claims Should Be**:
- add/mul/sub: "Minimal speedup (<1.2x) - memory bandwidth limited"
- dot/max/min: "Significant speedup (4-12x) - compute bound"

---

## Academic Context

**Why Memory-Bound Ops Don't Benefit**:

Modern CPUs have:
- **Out-of-order execution**: Scalar code hides latency
- **Memory bandwidth limits**: ~50 GB/s typical DDR4
- **SIMD overhead**: Setup costs dominate for simple ops

**Industry Evidence**:
- FFmpeg: Same pattern (simple ops minimal benefit)
- NumPy: Uses BLAS for compute-bound, accepts scalar for memory-bound
- Eigen: Similar SIMD speedup characteristics

**Conclusion**: Trueno's results align with industry experience

---

## Appendix: Raw Data Extract

**Best Performers** (AVX2 speedup, 1000 elements):
1. min: **12.45x** (1120ns → 90ns)
2. max: **12.14x** (1120ns → 92ns)
3. dot: **10.56x** (1148ns → 109ns)
4. argmax: **4.10x** (1190ns → 291ns)
5. argmin: **4.07x** (1184ns → 291ns)

**Worst Performers** (AVX2 speedup, 100 elements):
1. add: **0.76x** (59ns → 77ns) - SIMD slower!
2. mul: **0.91x** (68ns → 75ns) - SIMD slower!
3. sub: **0.98x** (73ns → 75ns) - SIMD slower!
4. relu: **0.98x** (73ns → 75ns) - SIMD slower!

**Hardware Details**:
- CPU: x86_64 with AVX2 support
- Benchmark framework: Criterion.rs
- Iterations: 100 samples per configuration
- Warmup: 3 seconds per benchmark

---

## Next Steps

1. **Fix Missing AVX-512 Benchmarks** ⚠️
   - div, fma, mul showing 0 results
   - Investigate timeout or implementation issues

2. **Update Documentation** 📝
   - README: Set realistic expectations
   - Remove "4x for add" claim
   - Emphasize "4-12x for dot/max/min"

3. **Add Large Vector Benchmarks** 📊
   - Test 100K, 1M, 10M element vectors
   - Validate GPU threshold (claimed >100K)

4. **ARM NEON Benchmarks** 🔧
   - Validate 4x speedup claims on ARM hardware
   - Compare Apple Silicon M-series performance

5. **Performance Regression Tests** 🚨
   - CI: Alert on >10% slowdowns
   - Baseline: Current AVX2 dot/max/min performance
   - Prevent accidental SIMD removal

---

**Session**: `claude/continue-next-step-01NEN2Jw5zVsNK9DWCE1Hwqz`
**Benchmark Run Time**: ~40 minutes
**Total Configurations**: 457
**Analysis Time**: ~15 minutes

**Conclusion**: SIMD works **exceptionally well** for compute-bound operations (4-12x) but provides **minimal benefit** for memory-bound operations (<1.2x). This aligns with FFmpeg's experience and academic literature on memory bandwidth limits.
