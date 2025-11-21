#!/usr/bin/env python3
"""
Matrix multiplication benchmark comparing Trueno vs NumPy

This script benchmarks matmul performance across frameworks to validate
Issue #10 optimizations (cache-aware blocking).

Usage:
    python benchmarks/matmul_comparison.py

Requirements:
    uv pip install numpy
"""

import json
import time
import numpy as np
from typing import Dict, List
import statistics
import subprocess
import sys


def benchmark_numpy_matmul(size: int, iterations: int = 100) -> Dict:
    """Benchmark NumPy matrix multiplication"""
    # Generate random matrices
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = a @ b

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = a @ b
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "iterations": iterations,
    }


def run_trueno_benchmark(size: int) -> Dict:
    """Run Trueno matrix multiplication benchmark via cargo bench"""
    try:
        # Run cargo bench for specific size
        result = subprocess.run(
            ["cargo", "bench", "--bench", "matrix_ops", "--", f"matmul_{size}x{size}"],
            capture_output=True,
            text=True,
            cwd="/home/noah/src/trueno",
            timeout=300,
        )

        # Parse benchmark output to extract time
        # Format: "matmul_128x128   time:   [0.6234 ms 0.6289 ms 0.6351 ms]"
        for line in result.stdout.split("\n"):
            if f"matmul_{size}x{size}" in line and "time:" in line:
                # Extract the middle value (mean)
                parts = line.split("[")[1].split("]")[0].split()
                mean_time = float(parts[2])  # Middle value in ms
                return {
                    "mean_ms": mean_time,
                    "framework": "trueno",
                }

        return None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not run Trueno benchmark for size {size}: {e}")
        return None


def main():
    """Main entry point"""
    print("=" * 80)
    print("Matrix Multiplication Benchmark: Trueno vs NumPy")
    print("Issue #10: Cache-Aware Blocking Performance Validation")
    print("=" * 80)

    # Test sizes matching Issue #10 benchmarks
    sizes = [32, 64, 128, 256, 512]
    iterations = 100

    results = {"numpy": {}, "trueno": {}}

    for size in sizes:
        print(f"\n📊 Matrix Size: {size}×{size}")

        # NumPy benchmark
        print(f"   Running NumPy benchmark ({iterations} iterations)...", end=" ", flush=True)
        np_results = benchmark_numpy_matmul(size, iterations)
        print(f"✓ {np_results['mean_ms']:.4f} ms")

        results["numpy"][str(size)] = np_results

        # Trueno benchmark (if available)
        print(f"   Running Trueno benchmark...", end=" ", flush=True)
        trueno_results = run_trueno_benchmark(size)
        if trueno_results:
            print(f"✓ {trueno_results['mean_ms']:.4f} ms")
            results["trueno"][str(size)] = trueno_results

            # Calculate speedup
            speedup = np_results["mean_ms"] / trueno_results["mean_ms"]
            status = "✓" if speedup > 0.8 else "⚠️"
            print(f"   {status} Trueno vs NumPy: {speedup:.2f}x (Target: ≥1.0x)")
        else:
            print("⚠️  Skipped (benchmark not found)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Matmul Performance (Issue #10 Progress)")
    print("=" * 80)

    print("\n┌────────┬──────────────┬──────────────┬───────────┬─────────────┐")
    print("│  Size  │  NumPy (ms)  │ Trueno (ms)  │  Speedup  │   Status    │")
    print("├────────┼──────────────┼──────────────┼───────────┼─────────────┤")

    for size in sizes:
        size_str = str(size)
        if size_str in results["numpy"]:
            np_time = results["numpy"][size_str]["mean_ms"]
            if size_str in results["trueno"]:
                trueno_time = results["trueno"][size_str]["mean_ms"]
                speedup = np_time / trueno_time
                status = "✓ On Track" if speedup >= 0.8 else "⚠️  Behind"
                print(f"│ {size:>4}×{size:<2} │  {np_time:>10.4f}  │  {trueno_time:>10.4f}  │   {speedup:>5.2f}x  │ {status:^11} │")
            else:
                print(f"│ {size:>4}×{size:<2} │  {np_time:>10.4f}  │      N/A     │     -     │     N/A     │")

    print("└────────┴──────────────┴──────────────┴───────────┴─────────────┘")

    print("\n📝 Notes:")
    print("   - Target: Trueno ≥0.8× NumPy speed (accounting for pure Rust vs optimized BLAS)")
    print("   - Phase 1 Goal: 1.5-2× speedup via cache-aware blocking")
    print("   - Phase 2 Goal: Full parity via optional BLAS backend")

    # Save results
    output_file = "benchmarks/matmul_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
