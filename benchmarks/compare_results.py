#!/usr/bin/env python3
"""
Compare Trueno (Rust) vs NumPy vs PyTorch benchmark results

This script:
1. Parses Criterion benchmark results (JSON) from Trueno
2. Loads Python benchmark results (NumPy/PyTorch)
3. Compares performance across all three frameworks
4. Generates markdown tables and analysis

Usage:
    python benchmarks/compare_results.py

Output:
    - benchmarks/comparison_report.md (Markdown report)
    - benchmarks/comparison_summary.json (JSON data)
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import statistics


class BenchmarkComparator:
    """Compare benchmark results across Trueno, NumPy, and PyTorch"""

    def __init__(self):
        self.trueno_results = {}
        self.python_results = {}
        self.comparison = {}

    def load_criterion_results(self, criterion_dir: str = "target/criterion"):
        """Load Criterion benchmark results from Trueno"""
        print("Loading Trueno (Criterion) benchmark results...")

        # Criterion stores results in: target/criterion/<group_name>/<benchmark_name>/base/estimates.json
        criterion_path = Path(criterion_dir)

        if not criterion_path.exists():
            print(f"⚠️  Criterion results not found at {criterion_dir}")
            print("   Run: cargo bench --all-features")
            return

        # Find all estimates.json files
        for estimates_file in criterion_path.rglob("estimates.json"):
            # Parse path: <group>/<bench_name>/<backend>/base/estimates.json
            parts = estimates_file.parts
            if len(parts) < 4:
                continue

            # Extract group and benchmark name
            group_idx = parts.index("criterion") + 1
            if group_idx >= len(parts) - 2:
                continue

            group = parts[group_idx]  # e.g., "add", "dot", "relu"
            bench_name = parts[group_idx + 1]  # e.g., "Scalar/100", "AVX2/10000"

            # Parse backend and size from bench_name
            if "/" not in bench_name:
                continue

            backend, size_str = bench_name.split("/")

            try:
                with open(estimates_file, "r") as f:
                    data = json.load(f)
                    # Criterion stores mean in nanoseconds
                    mean_ns = data["mean"]["point_estimate"]
                    std_ns = data["std_dev"]["point_estimate"]

                    # Store result
                    if group not in self.trueno_results:
                        self.trueno_results[group] = {}
                    if size_str not in self.trueno_results[group]:
                        self.trueno_results[group][size_str] = {}

                    self.trueno_results[group][size_str][backend] = {
                        "mean_ns": mean_ns,
                        "std_ns": std_ns,
                    }
            except Exception as e:
                print(f"⚠️  Failed to parse {estimates_file}: {e}")
                continue

        print(f"✅ Loaded {len(self.trueno_results)} Trueno operation groups")

    def load_python_results(self, python_file: str = "benchmarks/python_results.json"):
        """Load Python (NumPy/PyTorch) benchmark results"""
        print(f"Loading Python benchmark results from {python_file}...")

        if not os.path.exists(python_file):
            print(f"⚠️  Python results not found at {python_file}")
            print("   Run: python benchmarks/python_comparison.py")
            return

        with open(python_file, "r") as f:
            self.python_results = json.load(f)

        print(f"✅ Loaded Python results for {len(self.python_results.get('numpy', {}))} operations")

    def compare_results(self):
        """Compare Trueno vs NumPy vs PyTorch"""
        print("\nComparing results...")

        # For each operation, compare Trueno (best backend) vs NumPy vs PyTorch
        for op_name in self.trueno_results.keys():
            if op_name not in self.comparison:
                self.comparison[op_name] = {}

            for size in self.trueno_results[op_name].keys():
                backends_data = self.trueno_results[op_name][size]

                # Find best Trueno backend (lowest mean time)
                best_backend = None
                best_time = float("inf")
                for backend, data in backends_data.items():
                    if data["mean_ns"] < best_time:
                        best_time = data["mean_ns"]
                        best_backend = backend

                # Get NumPy and PyTorch times
                numpy_time = None
                pytorch_time = None

                if op_name in self.python_results.get("numpy", {}):
                    if size in self.python_results["numpy"][op_name]:
                        numpy_time = self.python_results["numpy"][op_name][size]["mean_ns"]

                if op_name in self.python_results.get("pytorch_cpu", {}):
                    if size in self.python_results["pytorch_cpu"][op_name]:
                        pytorch_time = self.python_results["pytorch_cpu"][op_name][size]["mean_ns"]

                # Store comparison
                self.comparison[op_name][size] = {
                    "trueno_backend": best_backend,
                    "trueno_mean_ns": best_time,
                    "numpy_mean_ns": numpy_time,
                    "pytorch_mean_ns": pytorch_time,
                    "trueno_vs_numpy": best_time / numpy_time if numpy_time else None,
                    "trueno_vs_pytorch": best_time / pytorch_time if pytorch_time else None,
                }

        print(f"✅ Compared {len(self.comparison)} operations")

    def generate_markdown_report(self, output_file: str = "benchmarks/comparison_report.md"):
        """Generate markdown comparison report"""
        print(f"\nGenerating markdown report: {output_file}...")

        lines = []
        lines.append("# Trueno vs NumPy vs PyTorch - Performance Comparison")
        lines.append("")
        lines.append("**Goal**: Validate that Trueno is within 20% of NumPy/PyTorch for 1D operations")
        lines.append("")
        lines.append("## Summary")
        lines.append("")

        # Calculate overall statistics
        within_20_percent = 0
        faster_than_numpy = 0
        faster_than_pytorch = 0
        total_comparisons = 0

        for op_name, sizes in self.comparison.items():
            for size, data in sizes.items():
                if data["trueno_vs_numpy"] is not None:
                    total_comparisons += 1
                    ratio = data["trueno_vs_numpy"]
                    if 0.8 <= ratio <= 1.2:
                        within_20_percent += 1
                    if ratio < 1.0:
                        faster_than_numpy += 1

                if data["trueno_vs_pytorch"] is not None:
                    if data["trueno_vs_pytorch"] < 1.0:
                        faster_than_pytorch += 1

        if total_comparisons > 0:
            percent_within_20 = (within_20_percent / total_comparisons) * 100
            percent_faster_numpy = (faster_than_numpy / total_comparisons) * 100
            percent_faster_pytorch = (faster_than_pytorch / total_comparisons) * 100

            lines.append(f"- **Within 20% of NumPy**: {within_20_percent}/{total_comparisons} ({percent_within_20:.1f}%)")
            lines.append(f"- **Faster than NumPy**: {faster_than_numpy}/{total_comparisons} ({percent_faster_numpy:.1f}%)")
            lines.append(f"- **Faster than PyTorch**: {faster_than_pytorch}/{total_comparisons} ({percent_faster_pytorch:.1f}%)")
            lines.append("")

            # Determine if we pass the v0.3.0 gate
            if percent_within_20 >= 80:
                lines.append("✅ **v0.3.0 SUCCESS CRITERIA MET**: >80% of operations within 20% of NumPy")
            else:
                lines.append("❌ **v0.3.0 CRITERIA NOT MET**: Need >80% within 20% (currently {:.1f}%)".format(percent_within_20))
            lines.append("")

        # Detailed results per operation
        lines.append("## Detailed Results")
        lines.append("")

        for op_name in sorted(self.comparison.keys()):
            lines.append(f"### {op_name}")
            lines.append("")
            lines.append("| Size | Trueno (best) | NumPy | PyTorch | Trueno vs NumPy | Trueno vs PyTorch |")
            lines.append("|------|---------------|-------|---------|-----------------|-------------------|")

            for size in ["100", "1000", "10000", "100000", "1000000"]:
                if size not in self.comparison[op_name]:
                    continue

                data = self.comparison[op_name][size]
                trueno_time = data["trueno_mean_ns"]
                numpy_time = data["numpy_mean_ns"]
                pytorch_time = data["pytorch_mean_ns"]
                backend = data["trueno_backend"]

                # Format times
                def format_time(ns):
                    if ns is None:
                        return "-"
                    if ns < 1000:
                        return f"{ns:.1f} ns"
                    elif ns < 1_000_000:
                        return f"{ns/1000:.2f} µs"
                    else:
                        return f"{ns/1_000_000:.2f} ms"

                trueno_str = f"{format_time(trueno_time)} ({backend})"
                numpy_str = format_time(numpy_time)
                pytorch_str = format_time(pytorch_time)

                # Calculate ratios
                ratio_numpy = data["trueno_vs_numpy"]
                ratio_pytorch = data["trueno_vs_pytorch"]

                def format_ratio(ratio):
                    if ratio is None:
                        return "-"
                    if ratio < 1.0:
                        return f"✅ {1/ratio:.2f}x faster"
                    elif ratio <= 1.2:
                        return f"✓ {ratio:.2f}x (within 20%)"
                    else:
                        return f"⚠️ {ratio:.2f}x slower"

                ratio_numpy_str = format_ratio(ratio_numpy)
                ratio_pytorch_str = format_ratio(ratio_pytorch)

                lines.append(f"| {size:>6} | {trueno_str} | {numpy_str} | {pytorch_str} | {ratio_numpy_str} | {ratio_pytorch_str} |")

            lines.append("")

        # Write file
        with open(output_file, "w") as f:
            f.write("\n".join(lines))

        print(f"✅ Report saved to: {output_file}")

    def save_comparison_json(self, output_file: str = "benchmarks/comparison_summary.json"):
        """Save comparison data as JSON"""
        output_data = {
            "comparison": self.comparison,
            "summary": {
                "operations_compared": len(self.comparison),
                "sizes": list(set([size for op in self.comparison.values() for size in op.keys()])),
            }
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ JSON summary saved to: {output_file}")


def main():
    """Main entry point"""
    comparator = BenchmarkComparator()

    # Load results
    comparator.load_criterion_results("target/criterion")
    comparator.load_python_results("benchmarks/python_results.json")

    # Compare
    comparator.compare_results()

    # Generate reports
    comparator.generate_markdown_report()
    comparator.save_comparison_json()

    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
