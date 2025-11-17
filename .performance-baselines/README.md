# Performance Baseline Storage

This directory contains performance baselines for regression detection.

## Format

Baselines are stored in JSON format with the following structure:

```json
{
  "metadata": {
    "date": "2025-11-17",
    "commit": "573db1a",
    "cpu": "AMD Ryzen 9 5950X",
    "gpu": "NVIDIA RTX 4090",
    "platform": "linux"
  },
  "benchmarks": {
    "gpu_vec_add_1M": {
      "time_ns": 54244000,
      "throughput_elem_per_sec": 18435000,
      "comparison": "vs_scalar",
      "speedup": 1.96
    },
    ...
  }
}
```

## Usage

### Generate Baseline

```bash
# Run benchmarks and extract to baseline
cargo bench --bench gpu_ops --all-features > /tmp/bench.txt
./scripts/extract_baseline.py /tmp/bench.txt > .performance-baselines/baseline-$(date +%Y%m%d).json
```

### Check for Regressions

```bash
# Compare current performance vs baseline
make bench-compare
```

### CI Integration

The CI pipeline automatically checks for >5% performance regressions on every PR.

## Baseline Files

- `baseline-current.json` - Active baseline used for comparison
- `baseline-YYYYMMDD-COMMIT.json` - Historical baselines (archived)

## Regression Policy

**Fail CI if**:
- Any benchmark >5% slower than baseline
- GPU speedup claims not met (e.g., <10x for claimed 10-50x)

**Warning if**:
- Any benchmark 2-5% slower than baseline
- Variance >10% between runs
