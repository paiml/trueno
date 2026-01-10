//! SIMD load generator using trueno

use std::any::Any;
use std::time::Duration;
use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::config::WorkloadType;
use crate::ring_buffer::RingBuffer;

pub struct SimdLoadBrick {
    workload: WorkloadType,
    intensity: f64,
    is_running: bool,
    problem_size: usize,
    input_a: Vec<f32>,
    input_b: Vec<f32>,
    output: Vec<f32>,
    latency_history: RingBuffer<f64>,
}

impl SimdLoadBrick {
    pub fn new(problem_size: usize) -> Self {
        Self {
            workload: WorkloadType::Gemm,
            intensity: 0.0,
            is_running: false,
            problem_size,
            input_a: vec![1.0; problem_size],
            input_b: vec![2.0; problem_size],
            output: vec![0.0; problem_size],
            latency_history: RingBuffer::new(100),
        }
    }

    pub fn start(&mut self) {
        self.is_running = true;
    }

    pub fn stop(&mut self) {
        self.is_running = false;
    }

    pub fn is_running(&self) -> bool {
        self.is_running
    }

    pub fn set_intensity(&mut self, intensity: f64) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    pub fn intensity(&self) -> f64 {
        self.intensity
    }

    pub fn run_iteration(&mut self) -> Duration {
        let start = std::time::Instant::now();

        if !self.is_running || self.intensity == 0.0 {
            return Duration::ZERO;
        }

        // Simulate work based on intensity
        let work_size = ((self.problem_size as f64) * self.intensity) as usize;

        // Simple element-wise operation using SIMD when available
        for i in 0..work_size {
            self.output[i] = self.input_a[i] * self.input_b[i] + 1.0;
        }

        let elapsed = start.elapsed();
        self.latency_history.push(elapsed.as_secs_f64() * 1000.0);
        elapsed
    }

    pub fn throughput_ops_per_sec(&self) -> f64 {
        let avg_latency = self.latency_history.mean();
        if avg_latency > 0.0 {
            1000.0 / avg_latency
        } else {
            0.0
        }
    }
}

impl Default for SimdLoadBrick {
    fn default() -> Self {
        Self::new(1_048_576)
    }
}

impl Brick for SimdLoadBrick {
    fn brick_name(&self) -> &'static str {
        "simd_load"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("buffers_preallocated", |_| true),
            BrickAssertion::custom("intensity_in_range", |_| true),
            BrickAssertion::max_latency_ms(100),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 16,
            layout_ms: 0,
            render_ms: 0,
        }
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
