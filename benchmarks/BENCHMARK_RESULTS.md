# NumPy vs PyTorch Benchmark Results

**Date**: 2025-11-20  
**Machine**: x86_64 Linux  
**Iterations**: 100 per test  
**Vector Sizes**: 100, 1K, 10K, 100K, 1M elements

## Executive Summary

**Key Finding**: NumPy excels at small data (<10K elements), PyTorch dominates large data (1M+ elements)

### Performance Highlights

**NumPy Victories**:
- **Argmax (1M elems)**: 12.8x faster than PyTorch
- **Tanh (1K elems)**: 4.13x faster
- **Dot Product (100 elems)**: 3.02x faster

**PyTorch Victories**:
- **Sum (1M elems)**: 8.5x faster (16µs vs 135µs)
- **Relu (1M elems)**: 9.4x faster (32µs vs 301µs)
- **Most large-scale operations**: Better parallelization

### Crossover Point

**~100K elements** is where PyTorch starts winning for most operations due to better GPU/parallel scaling.

## Detailed Results

See `python_results.json` for complete data (25KB, 19 operations × 5 sizes × 2 frameworks)

## Next Steps

Compare against Trueno (Rust SIMD) to show:
1. Where hand-tuned SIMD beats NumPy
2. Zero-overhead Rust vs Python runtime cost
3. Lambda deployment advantages
