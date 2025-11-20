#!/usr/bin/env python3
"""
Comprehensive benchmark comparing Trueno vs NumPy vs PyTorch

This script benchmarks 1D vector operations across three frameworks:
- NumPy (Python's standard numerical library)
- PyTorch (Deep learning framework with CPU/GPU support)
- Trueno (Rust SIMD library - benchmarked separately)

Results are saved in JSON format for comparison analysis.

Usage:
    python benchmarks/python_comparison.py

Requirements:
    uv pip install numpy torch
"""

import json
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Callable
import statistics


class BenchmarkRunner:
    """Run benchmarks across NumPy and PyTorch"""

    def __init__(self, sizes: List[int] = None, iterations: int = 100):
        """
        Args:
            sizes: List of vector sizes to benchmark
            iterations: Number of iterations per benchmark
        """
        self.sizes = sizes or [100, 1_000, 10_000, 100_000, 1_000_000]
        self.iterations = iterations
        self.results = {
            "numpy": {},
            "pytorch_cpu": {},
        }

    def benchmark_operation(
        self,
        name: str,
        numpy_fn: Callable,
        pytorch_fn: Callable,
        setup_numpy: Callable,
        setup_pytorch: Callable,
    ):
        """Benchmark a single operation across all frameworks and sizes"""
        print(f"\nBenchmarking: {name}")

        for size in self.sizes:
            print(f"  Size: {size:>10,} elements", end=" ")

            # NumPy benchmark
            np_data = setup_numpy(size)
            np_times = []
            for _ in range(self.iterations):
                start = time.perf_counter()
                result = numpy_fn(*np_data)
                end = time.perf_counter()
                np_times.append((end - start) * 1e9)  # Convert to nanoseconds

            np_mean = statistics.mean(np_times)
            np_std = statistics.stdev(np_times) if len(np_times) > 1 else 0

            # PyTorch CPU benchmark
            pt_data = setup_pytorch(size)
            pt_times = []
            for _ in range(self.iterations):
                start = time.perf_counter()
                result = pytorch_fn(*pt_data)
                end = time.perf_counter()
                pt_times.append((end - start) * 1e9)

            pt_mean = statistics.mean(pt_times)
            pt_std = statistics.stdev(pt_times) if len(pt_times) > 1 else 0

            # Store results
            if name not in self.results["numpy"]:
                self.results["numpy"][name] = {}
                self.results["pytorch_cpu"][name] = {}

            self.results["numpy"][name][str(size)] = {
                "mean_ns": np_mean,
                "std_ns": np_std,
                "iterations": self.iterations,
            }
            self.results["pytorch_cpu"][name][str(size)] = {
                "mean_ns": pt_mean,
                "std_ns": pt_std,
                "iterations": self.iterations,
            }

            print(f"| NumPy: {np_mean:>10.2f} ns | PyTorch: {pt_mean:>10.2f} ns")

    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite"""
        print("=" * 80)
        print("Trueno vs NumPy vs PyTorch - Comprehensive Benchmark Suite")
        print("=" * 80)

        # Element-wise operations
        self.benchmark_operation(
            "add",
            lambda a, b: a + b,
            lambda a, b: a + b,
            lambda size: (np.random.randn(size).astype(np.float32), np.random.randn(size).astype(np.float32)),
            lambda size: (torch.randn(size, dtype=torch.float32), torch.randn(size, dtype=torch.float32)),
        )

        self.benchmark_operation(
            "sub",
            lambda a, b: a - b,
            lambda a, b: a - b,
            lambda size: (np.random.randn(size).astype(np.float32), np.random.randn(size).astype(np.float32)),
            lambda size: (torch.randn(size, dtype=torch.float32), torch.randn(size, dtype=torch.float32)),
        )

        self.benchmark_operation(
            "mul",
            lambda a, b: a * b,
            lambda a, b: a * b,
            lambda size: (np.random.randn(size).astype(np.float32), np.random.randn(size).astype(np.float32)),
            lambda size: (torch.randn(size, dtype=torch.float32), torch.randn(size, dtype=torch.float32)),
        )

        self.benchmark_operation(
            "div",
            lambda a, b: a / b,
            lambda a, b: a / b,
            lambda size: (np.random.randn(size).astype(np.float32), np.random.randn(size).astype(np.float32) + 1.0),
            lambda size: (torch.randn(size, dtype=torch.float32), torch.randn(size, dtype=torch.float32) + 1.0),
        )

        # Dot product
        self.benchmark_operation(
            "dot",
            lambda a, b: np.dot(a, b),
            lambda a, b: torch.dot(a, b),
            lambda size: (np.random.randn(size).astype(np.float32), np.random.randn(size).astype(np.float32)),
            lambda size: (torch.randn(size, dtype=torch.float32), torch.randn(size, dtype=torch.float32)),
        )

        # Reductions
        self.benchmark_operation(
            "sum",
            lambda a: np.sum(a),
            lambda a: torch.sum(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "max",
            lambda a: np.max(a),
            lambda a: torch.max(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "min",
            lambda a: np.min(a),
            lambda a: torch.min(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "argmax",
            lambda a: np.argmax(a),
            lambda a: torch.argmax(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "argmin",
            lambda a: np.argmin(a),
            lambda a: torch.argmin(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        # Norms
        self.benchmark_operation(
            "norm_l2",
            lambda a: np.linalg.norm(a),
            lambda a: torch.norm(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "norm_l1",
            lambda a: np.linalg.norm(a, ord=1),
            lambda a: torch.norm(a, p=1),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        # Activation functions
        self.benchmark_operation(
            "relu",
            lambda a: np.maximum(0, a),
            lambda a: torch.relu(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "sigmoid",
            lambda a: 1 / (1 + np.exp(-a)),
            lambda a: torch.sigmoid(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "tanh",
            lambda a: np.tanh(a),
            lambda a: torch.tanh(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "exp",
            lambda a: np.exp(a),
            lambda a: torch.exp(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        # Additional operations
        self.benchmark_operation(
            "abs",
            lambda a: np.abs(a),
            lambda a: torch.abs(a),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

        self.benchmark_operation(
            "scale",
            lambda a, s: a * s,
            lambda a, s: a * s,
            lambda size: (np.random.randn(size).astype(np.float32), 2.5),
            lambda size: (torch.randn(size, dtype=torch.float32), 2.5),
        )

        self.benchmark_operation(
            "clamp",
            lambda a: np.clip(a, -1.0, 1.0),
            lambda a: torch.clamp(a, -1.0, 1.0),
            lambda size: (np.random.randn(size).astype(np.float32),),
            lambda size: (torch.randn(size, dtype=torch.float32),),
        )

    def save_results(self, output_file: str = "benchmarks/python_results.json"):
        """Save benchmark results to JSON file"""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to: {output_file}")

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 80)
        print("SUMMARY: NumPy vs PyTorch Performance")
        print("=" * 80)

        for op_name in self.results["numpy"].keys():
            print(f"\n{op_name.upper()}:")
            for size in self.sizes:
                size_str = str(size)
                if size_str in self.results["numpy"][op_name]:
                    np_time = self.results["numpy"][op_name][size_str]["mean_ns"]
                    pt_time = self.results["pytorch_cpu"][op_name][size_str]["mean_ns"]
                    ratio = pt_time / np_time
                    faster = "NumPy" if ratio > 1.0 else "PyTorch"
                    print(f"  {size:>10,} elements: {faster:>8} is {abs(ratio):>6.2f}x faster")


def main():
    """Main entry point"""
    runner = BenchmarkRunner(
        sizes=[100, 1_000, 10_000, 100_000, 1_000_000],
        iterations=100
    )

    runner.run_all_benchmarks()
    runner.print_summary()
    runner.save_results()


if __name__ == "__main__":
    main()
