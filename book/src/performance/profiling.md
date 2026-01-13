# Profiling

## The Real Profiling Mandate

Trueno enforces a strict **"Real Profiling"** mandate. All performance metrics reported by the ecosystem MUST be measured, not derived.

> **Forbidden**: Calculating per-brick time by taking total throughput and multiplying by a budget fraction.
> **Required**: Measuring start/end times for every operation, with full synchronization.

### Why?

Simulated or derived metrics mask bottlenecks. If you assume an operation takes 10% of the time, you will never discover when it actually takes 50% due to a regression.

## BrickProfiler

The `BrickProfiler` is the core tool for this. It is built into `trueno` and used by all downstream projects.

### Usage

```rust
// 1. Get the profiler (usually from context)
let profiler = context.profiler();

// 2. Start a timer
let timer = profiler.start("MyOperation");

// 3. Perform work (and SYNC if GPU!)
my_kernel.launch();
stream.synchronize(); // CRITICAL for validity

// 4. Stop and record
profiler.stop(timer, num_elements);
```

## Falsification Protocols

To prove profiling is real, we apply Popperian Falsification:

1.  **Variance Check**: Real hardware has noise. If `cbtop` reports identical latencies for 100 runs, it is likely faking data.
2.  **Overhead Check**: Enabling profiling *should* reduce throughput (due to syncs). If throughput remains identical, profiling is likely not synchronizing correctly.

## Tools

- **cbtop**: The primary visualization tool for ComputeBrick pipelines.
- **perf / flamegraph**: For CPU-side overhead analysis.
- **nsight**: For deep GPU kernel inspection (external to the pure Rust stack).
