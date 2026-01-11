# cbtop - Compute Block Top

**cbtop** is a real-time load testing and hardware monitoring TUI built on the Brick Architecture. It provides visibility into CPU, GPU, memory, and compute workloads with a familiar `htop`-style interface.

## Installation

```bash
# Build from source
cargo build -p cbtop --release

# Run
./target/release/cbtop
```

## Features

- **Real-time Monitoring**: CPU, GPU, memory, network, disk, thermal metrics
- **Load Generation**: SIMD, CUDA, and wgpu compute workloads
- **Compute Scoring**: BrickScore framework (0-100) with letter grades
- **Multi-Backend**: Automatic backend selection (AVX2, CUDA, wgpu)
- **Deterministic Mode**: Reproducible benchmarks for testing

## Usage

```bash
# Basic monitoring
cbtop

# With load testing
cbtop --load medium --backend simd

# Stress test with CUDA
cbtop --load stress --backend cuda

# Deterministic mode for reproducible results
cbtop --deterministic --show-fps
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --refresh <MS>` | Refresh rate in milliseconds | 100 |
| `-d, --device <N>` | GPU device index | 0 |
| `-b, --backend <TYPE>` | Backend: simd, wgpu, cuda, all | all |
| `-l, --load <LEVEL>` | Load: idle, light, medium, heavy, stress | idle |
| `-w, --workload <TYPE>` | Workload: gemm, conv, attention, bandwidth | gemm |
| `-s, --size <N>` | Problem size in elements | 1048576 |
| `--deterministic` | Enable deterministic mode | false |
| `--show-fps` | Show frame timing statistics | false |

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `Tab` | Next panel |
| `Shift+Tab` | Previous panel |
| `1-7` | Jump to panel |
| `Space` | Start/Stop load generator |
| `↑/↓` | Adjust load intensity |
| `b` | Cycle backend |
| `w` | Cycle workload type |

## TUI Layout

```
┌─────────────────────── cbtop v0.1.0 ───────────────────────┐
│ CPU: AMD Ryzen 9 5950X │ GPU: NVIDIA RTX 3080 │ Mem: 64GB  │
├────────────────────────────────────────────────────────────┤
│ [Overview] [CPU] [GPU] [Memory] [Network] [Disk] [Load]   │
├──────────────────────┬─────────────────────────────────────┤
│ CPU Usage            │ GPU Metrics                         │
│ ████████░░ 78%       │ Util: ███████░░░ 72%               │
│                      │ Mem:  ██████░░░░ 58% (6.2/10.0 GB) │
│ Core 0: ████████ 95% │ Temp: 67°C  Power: 285W            │
│ Core 1: ██████░░ 72% │                                     │
├──────────────────────┼─────────────────────────────────────┤
│ Memory               │ Load Generator                      │
│ Used: 24.5/64.0 GB   │ Backend: SIMD (AVX2)               │
│ ██████░░░░ 38%       │ GFLOP/s: 27.76                     │
│                      │ Score: 85/100 (B+)                 │
├──────────────────────┴─────────────────────────────────────┤
│ Status: Running │ 27.76 GFLOP/s │ Latency: 2.3ms │ q=quit │
└────────────────────────────────────────────────────────────┘
```

## BrickScore Framework

cbtop uses the ComputeBrick Scoring Framework to evaluate compute quality:

| Component | Weight | Description |
|-----------|--------|-------------|
| Performance | 40 pts | GFLOP/s vs theoretical peak |
| Efficiency | 25 pts | SIMD/GPU utilization |
| Correctness | 20 pts | Assertion pass rate |
| Stability | 15 pts | Coefficient of Variation |

**Grades**: A (90-100), B (80-89), C (70-79), D (60-69), F (<60)

## Brick Architecture

cbtop is built on the Brick Architecture from `presentar-terminal`:

```
Layer 4: Load Generators  → SimdLoadBrick, CudaLoadBrick, WgpuLoadBrick
Layer 3: Panels           → Overview, CPU, GPU, Memory, Network, Disk, Load
Layer 2: Analyzers        → Throughput, Bottleneck, Thermal
Layer 1: Collectors       → CPU, GPU, PCIe, Memory, Thermal, ZRAM
```

Each Brick is a falsifiable unit with:
- Assertions (correctness guarantees)
- Budget (timing constraints)
- Verification (runtime checks)

## Integration with Trueno

cbtop uses Trueno's SIMD operations for load generation:

```rust
use trueno::Vector;

// cbtop uses Trueno Vector operations for benchmarking
let a = Vector::from_slice(&data_a);
let b = Vector::from_slice(&data_b);
let result = a.dot(&b).unwrap();  // SIMD-accelerated dot product
```

## Headless Mode (AI Agent Integration)

cbtop supports headless mode for CI/CD pipelines and AI agents like Claude Code. This enables programmatic benchmarking without a TTY.

### Running Headless Benchmarks

```bash
# Basic headless benchmark with JSON output
cbtop --headless --format json --duration 5

# Using the bench subcommand
cbtop bench --backend simd --workload gemm --duration 5 --format json

# Save results to file
cbtop bench --backend simd -o results.json
```

### Example JSON Output

```json
{
  "version": "0.1.0",
  "timestamp": "2026-01-11T10:00:00Z",
  "duration_secs": 5.0,
  "system": {
    "cpu": "AMD Ryzen Threadripper 7960X",
    "cores": 48,
    "memory_gb": 128
  },
  "benchmark": {
    "backend": "Simd",
    "workload": "Gemm",
    "size": 1048576,
    "iterations": 500
  },
  "results": {
    "gflops": 25.0,
    "throughput_ops_sec": 1000.0,
    "latency_ms": {
      "mean": 1.0,
      "p50": 0.9,
      "p95": 1.5,
      "p99": 1.8,
      "cv_percent": 5.0
    }
  },
  "score": {
    "total": 85,
    "grade": "B",
    "performance": 35,
    "efficiency": 20,
    "correctness": 20,
    "stability": 10
  }
}
```

### Regression Testing

Compare against a baseline to detect performance regressions:

```bash
# Save baseline
cbtop bench --backend simd -o baseline.json

# Test against baseline (exits non-zero on >5% regression)
cbtop bench --backend simd --baseline baseline.json --fail-on-regression 5.0
```

### Backend Comparison

Compare multiple backends side-by-side:

```bash
# Compare SIMD vs all backends
cbtop bench --compare simd,cuda,wgpu --format text
```

### AI Agent Use Cases

AI coding assistants can use cbtop headless mode to:

1. **Profile before optimization**: Run benchmarks before making changes
2. **Validate improvements**: Compare results after optimization
3. **Detect regressions**: Fail CI if performance drops
4. **Generate reports**: Include benchmark data in documentation

Example workflow for an AI agent:

```bash
# 1. Baseline measurement
cbtop bench --backend simd -o /tmp/baseline.json

# 2. AI makes code changes...

# 3. Validate no regression
cbtop bench --backend simd --baseline /tmp/baseline.json --fail-on-regression 5.0
```

## Testing

```bash
# Run all cbtop tests
cargo test -p cbtop --all-features

# Run falsification tests
cargo test -p cbtop f301

# Run with ignored tests (requires isolated CPU)
cargo test -p cbtop --all-features -- --ignored
```

## Specification

See the full specification at:
- `docs/specifications/compute-block-tui-cbtop.md`

The specification includes:
- 200-point falsification protocol
- 49 peer-reviewed citations
- ComputeBrick Scoring Framework
- FKR (Falsifiable Knowledge Record) entries
