# Trueno - High-Performance Compute Library

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [First Program](./getting-started/first-program.md)
- [Core Concepts](./getting-started/core-concepts.md)

# Architecture

- [Overview](./architecture/overview.md)
- [Backend Selection](./architecture/backend-selection.md)
- [Multi-Backend Design](./architecture/multi-backend-design.md)
- [SIMD Backends](./architecture/simd-backends.md)
  - [SSE2 (x86_64 Baseline)](./architecture/sse2-backend.md)
  - [AVX/AVX2 (256-bit)](./architecture/avx-backend.md)
  - [AVX-512 (512-bit)](./architecture/avx512-backend.md)
  - [NEON (ARM64)](./architecture/neon-backend.md)
  - [WASM SIMD128](./architecture/wasm-backend.md)
- [GPU Backend](./architecture/gpu-backend.md)
- [Runtime Detection](./architecture/runtime-detection.md)

# API Reference

- [Vector Operations](./api-reference/vector-operations.md)
- [Element-wise Operations](./api-reference/element-wise.md)
- [Reduction Operations](./api-reference/reductions.md)
- [Transformation Operations](./api-reference/transformations.md)
- [Error Handling](./api-reference/error-handling.md)
- [Backend API](./api-reference/backend-api.md)

# Performance

- [Benchmarks Overview](./performance/benchmarks.md)
- [SIMD Performance Analysis](./performance/simd-performance.md)
- [GPU Performance](./performance/gpu-performance.md)
- [Optimization Guide](./performance/optimization-guide.md)
- [Profiling](./performance/profiling.md)
- [Performance Targets](./performance/targets.md)
- [Comparing Backends](./performance/backend-comparison.md)

# Safety and Correctness

- [Safety Philosophy](./safety/philosophy.md)
- [Unsafe Code Guidelines](./safety/unsafe-code.md)
- [Safety Invariants](./safety/safety-invariants.md)
- [Miri Validation](./safety/miri-validation.md)
- [Testing for Correctness](./safety/testing-correctness.md)
- [Backend Equivalence](./safety/backend-equivalence.md)

# Examples

- [Vector Math](./examples/vector-math.md)
- [Matrix Operations](./examples/matrix-operations.md)
- [Neural Networks](./examples/neural-networks.md)
- [Image Processing](./examples/image-processing.md)
- [Signal Processing](./examples/signal-processing.md)
- [Scientific Computing](./examples/scientific-computing.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [Extreme TDD](./development/extreme-tdd.md)
- [Testing](./development/testing.md)
  - [Unit Tests](./development/unit-tests.md)
  - [Property-Based Tests](./development/property-based-tests.md)
  - [Backend Equivalence Tests](./development/backend-equivalence-tests.md)
  - [Mutation Testing](./development/mutation-testing.md)
- [Benchmarking](./development/benchmarking.md)
- [Quality Gates](./development/quality-gates.md)
- [Code Review Checklist](./development/code-review.md)

# Advanced Topics

- [SIMD Intrinsics](./advanced/simd-intrinsics.md)
- [GPU Compute Shaders](./advanced/gpu-shaders.md)
- [Memory Alignment](./advanced/memory-alignment.md)
- [Vectorization Patterns](./advanced/vectorization-patterns.md)
- [Cross-Platform Portability](./advanced/portability.md)
- [FFmpeg Case Study](./advanced/ffmpeg-case-study.md)

# Ecosystem Integration

- [Ruchy Integration](./ecosystem/ruchy.md)
- [Depyler (Python → Rust)](./ecosystem/depyler.md)
- [Decy (C → Rust)](./ecosystem/decy.md)
- [AWS Lambda (ruchy-lambda)](./ecosystem/ruchy-lambda.md)
- [Docker Benchmarking](./ecosystem/ruchy-docker.md)
- [PMAT Quality Gates](./ecosystem/pmat.md)

# Specifications

- [Design Philosophy](./specifications/design-philosophy.md)
- [Initial Three-Target Spec](./specifications/three-target-spec.md)
- [PyTorch/NumPy Replacement](./specifications/pytorch-numpy-spec.md)
- [Ruchy Language Support](./specifications/ruchy-support.md)
- [Academic Foundations](./specifications/academic-foundations.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Migration Guide](./appendix/migration-guide.md)
- [Performance Comparison Tables](./appendix/performance-tables.md)
