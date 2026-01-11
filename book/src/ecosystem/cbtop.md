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
