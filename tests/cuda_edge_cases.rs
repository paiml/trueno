//! GPU Edge-Case Testing with trueno-cuda-edge
//!
//! This test suite applies trueno-cuda-edge's falsification frameworks to
//! trueno's GPU compute primitives, verifying:
//!
//! - Null pointer handling for device memory
//! - Quantization boundaries for SIMD operations
//! - PTX verification for generated kernels
//! - Shared memory constraints for GPU backends
//! - Supervision patterns for GPU worker management
//!
//! Tests run without actual GPU hardware by validating pure-Rust
//! type system guarantees and configuration.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use trueno_cuda_edge::{
    falsification::{all_claims, ClaimStatus, FalsificationReport, Framework},
    lifecycle_chaos::{
        ChaosScenario, ContextLeakDetector, DestructionOrdering, LifecycleChaosConfig,
    },
    null_fuzzer::{InjectionStrategy, NonNullDevicePtr, NullFuzzerConfig, NullSentinelFuzzer},
    ptx_poison::{default_mutators, PtxMutator, PtxVerifier, MINIMAL_VALID_PTX},
    quant_oracle::{check_values_parity, BoundaryValueGenerator, ParityConfig, QuantFormat},
    shmem_prober::{
        check_allocation, compute_sentinel_offsets, shared_memory_limit, AccessPattern,
        BankConflictInjector, ComputeCapability, SharedMemoryRegion,
    },
    supervisor::{
        GpuHealthMonitor, HealthAction, HeartbeatStatus, SupervisionStrategy, SupervisionTree,
    },
};

// ============================================================================
// F1: Null Pointer Sentinel Fuzzer — Device Memory Safety
// ============================================================================

mod null_fuzzer_tests {
    use super::*;

    /// Verify NonNullDevicePtr enforces non-null constraint at construction.
    /// Critical for trueno's GPU memory allocation wrappers.
    #[test]
    fn device_ptr_construction_safety() {
        // Null address must be rejected
        let null_result = NonNullDevicePtr::<f32>::new(0);
        assert!(null_result.is_err());

        // Valid GPU address should succeed
        let valid_result = NonNullDevicePtr::<f32>::new(0x7f00_0000_0000);
        assert!(valid_result.is_ok());

        // Verify address is preserved
        let ptr = valid_result.unwrap();
        assert_eq!(ptr.addr(), 0x7f00_0000_0000);
    }

    /// Test type-level safety for different element types.
    #[test]
    fn device_ptr_type_safety() {
        // f32 tensor buffer
        let f32_ptr = NonNullDevicePtr::<f32>::new(0x1000).unwrap();
        assert_eq!(f32_ptr.addr(), 0x1000);

        // f16 tensor buffer (different type parameter)
        let f16_ptr = NonNullDevicePtr::<u16>::new(0x2000).unwrap();
        assert_eq!(f16_ptr.addr(), 0x2000);

        // u8 quantized buffer
        let u8_ptr = NonNullDevicePtr::<u8>::new(0x3000).unwrap();
        assert_eq!(u8_ptr.addr(), 0x3000);
    }

    /// Test injection strategies for GPU kernel fuzzing.
    #[test]
    fn injection_strategies_for_kernel_calls() {
        // Periodic: inject every N kernel launches
        let periodic = InjectionStrategy::Periodic { interval: 100 };
        assert!(periodic.should_inject(0));
        assert!(periodic.should_inject(100));
        assert!(!periodic.should_inject(50));

        // Size threshold: inject for large allocations
        let size_based = InjectionStrategy::SizeThreshold {
            threshold_bytes: 1024 * 1024 * 1024, // 1 GB
        };
        // Requires context, returns false without it
        assert!(!size_based.should_inject(0));

        // Probabilistic: deterministic for reproducibility
        let prob = InjectionStrategy::Probabilistic { probability: 0.25 };
        // 25% = inject when (idx % 100) < 25
        assert!(prob.should_inject(0));
        assert!(prob.should_inject(24));
        assert!(!prob.should_inject(25));
    }

    /// Test fuzzer state machine for GPU call sequences.
    #[test]
    fn fuzzer_call_sequence() {
        let config = NullFuzzerConfig {
            strategy: InjectionStrategy::Periodic { interval: 5 },
            total_calls: 100,
            fail_fast: false,
        };

        let mut fuzzer = NullSentinelFuzzer::new(config);

        // Track injection pattern
        let mut injections = Vec::new();
        for _ in 0..20 {
            injections.push(fuzzer.next_call());
        }

        // Should inject at 0, 5, 10, 15
        assert!(injections[0]);
        assert!(!injections[1]);
        assert!(injections[5]);
        assert!(injections[10]);
        assert!(injections[15]);
    }
}

// ============================================================================
// F2: Shared Memory Boundary Prober — GPU Memory Constraints
// ============================================================================

mod shmem_prober_tests {
    use super::*;

    /// Test compute capability to shared memory mapping.
    /// Trueno targets Volta (7.x), Ampere (8.x), Hopper (9.x).
    #[test]
    fn compute_capability_shmem_limits() {
        // Volta/Turing (SM 7.x): 96 KB
        let volta = ComputeCapability::new(7, 0);
        assert_eq!(shared_memory_limit(volta), 96 * 1024);

        // Ampere (SM 8.x): 164 KB
        let ampere = ComputeCapability::new(8, 0);
        assert_eq!(shared_memory_limit(ampere), 164 * 1024);

        // Hopper (SM 9.x): 228 KB
        let hopper = ComputeCapability::new(9, 0);
        assert_eq!(shared_memory_limit(hopper), 228 * 1024);

        // Older (fallback): 48 KB
        let kepler = ComputeCapability::new(3, 5);
        assert_eq!(shared_memory_limit(kepler), 48 * 1024);
    }

    /// Test allocation validation for GPU kernels.
    #[test]
    fn allocation_bounds_checking() {
        let ampere = ComputeCapability::new(8, 0);
        let limit = shared_memory_limit(ampere);

        // At limit: OK
        assert!(check_allocation(ampere, limit).is_ok());

        // Below limit: OK
        assert!(check_allocation(ampere, limit - 1).is_ok());

        // Above limit: Error
        assert!(check_allocation(ampere, limit + 1).is_err());
    }

    /// Test bank conflict analysis for optimized memory access.
    #[test]
    fn bank_conflict_patterns() {
        let injector = BankConflictInjector::new();

        // Sequential: no conflicts (ideal)
        assert_eq!(
            injector.expected_serialization(AccessPattern::Sequential),
            1
        );

        // Stride-2: 2-way conflicts
        assert_eq!(injector.expected_serialization(AccessPattern::Stride2), 2);

        // Full conflict: 32-way serialization (worst case)
        assert_eq!(
            injector.expected_serialization(AccessPattern::FullConflict),
            32
        );

        // Padded: no conflicts (bank conflict avoidance)
        assert_eq!(injector.expected_serialization(AccessPattern::Padded), 1);

        // Stride-32 (broadcast): no conflicts
        assert_eq!(injector.expected_serialization(AccessPattern::Stride32), 1);
    }

    /// Test sentinel placement for boundary detection.
    #[test]
    fn sentinel_boundary_detection() {
        let regions = vec![
            SharedMemoryRegion::new(0, 1024),    // 1 KB region
            SharedMemoryRegion::new(1024, 2048), // 2 KB region
        ];

        let offsets = compute_sentinel_offsets(&regions);
        assert_eq!(offsets.len(), 2);

        // Region 0: before=0, after=0+4+1024=1028
        assert_eq!(offsets[0], (0, 1028));

        // Region 1: before=1024, after=1024+4+2048=3076
        assert_eq!(offsets[1], (1024, 3076));
    }

    /// Test bank index calculation for memory layout.
    #[test]
    fn bank_index_calculation() {
        let injector = BankConflictInjector::new();

        // 32 banks, 4-byte words
        // Bank = (offset / 4) % 32
        assert_eq!(injector.bank_for_offset(0), 0);
        assert_eq!(injector.bank_for_offset(4), 1);
        assert_eq!(injector.bank_for_offset(124), 31);
        assert_eq!(injector.bank_for_offset(128), 0); // wraps
    }
}

// ============================================================================
// F3: Context Lifecycle Chaos — GPU Context Management
// ============================================================================

mod lifecycle_chaos_tests {
    use super::*;

    /// Test all 8 chaos scenarios for GPU context lifecycle.
    #[test]
    fn chaos_scenario_coverage() {
        let scenarios = ChaosScenario::all();
        assert_eq!(scenarios.len(), 8);

        // Verify critical scenarios are present
        assert!(scenarios.contains(&ChaosScenario::DoubleDestroy));
        assert!(scenarios.contains(&ChaosScenario::UseAfterDestroy));
        assert!(scenarios.contains(&ChaosScenario::LeakedContext));
        assert!(scenarios.contains(&ChaosScenario::ContextExhaustion));
    }

    /// Test default chaos configuration.
    #[test]
    fn default_chaos_config() {
        let config = LifecycleChaosConfig::default();

        assert_eq!(config.scenarios.len(), 8);
        assert_eq!(config.max_contexts, 64);
        assert!(config.capture_memory_snapshots);
    }

    /// Test destruction ordering validation.
    #[test]
    fn destruction_ordering_patterns() {
        // LIFO (reverse) — correct for CUDA
        let lifo = DestructionOrdering::new(vec![2, 1, 0]);
        assert!(lifo.is_reverse());
        assert!(!lifo.is_forward());

        // FIFO (forward) — may cause issues
        let fifo = DestructionOrdering::new(vec![0, 1, 2]);
        assert!(fifo.is_forward());
        assert!(!fifo.is_reverse());

        // Random — neither
        let random = DestructionOrdering::new(vec![1, 0, 2]);
        assert!(!random.is_reverse());
        assert!(!random.is_forward());
    }

    /// Test memory leak detection with tolerance.
    #[test]
    fn leak_detection_with_tolerance() {
        let detector = ContextLeakDetector::new();

        // Within 1 MB tolerance: no leak
        let report = detector.analyze(100_000_000, 100_500_000);
        assert!(!report.has_leaks());

        // Above 1 MB tolerance: leak detected
        let report = detector.analyze(100_000_000, 102_000_000);
        assert!(report.has_leaks());
        assert!(report.total_leaked_bytes() > 0);
    }

    /// Test custom tolerance for strict leak detection.
    #[test]
    fn custom_leak_tolerance() {
        let strict = ContextLeakDetector::with_tolerance(1024); // 1 KB

        let report = strict.analyze(1000, 3000);
        assert!(report.has_leaks()); // 2000 > 1024
    }
}

// ============================================================================
// F4: Quantization Parity Oracle — SIMD/GPU Numerical Accuracy
// ============================================================================

mod quant_oracle_tests {
    use super::*;

    /// Test format-specific tolerances for trueno's quantization.
    #[test]
    fn quantization_format_tolerances() {
        // 4-bit quantization: ~5% tolerance
        assert!((QuantFormat::Q4K.tolerance() - 0.05).abs() < f64::EPSILON);

        // 5-bit quantization: ~2% tolerance
        assert!((QuantFormat::Q5K.tolerance() - 0.02).abs() < f64::EPSILON);

        // 6-bit quantization: ~1% tolerance
        assert!((QuantFormat::Q6K.tolerance() - 0.01).abs() < f64::EPSILON);

        // 8-bit quantization: ~0.5% tolerance
        assert!((QuantFormat::Q8_0.tolerance() - 0.005).abs() < f64::EPSILON);

        // F16: ~0.1% tolerance
        assert!((QuantFormat::F16.tolerance() - 0.001).abs() < f64::EPSILON);

        // F32: machine epsilon
        assert!((QuantFormat::F32.tolerance() - f64::EPSILON).abs() < f64::EPSILON);
    }

    /// Test boundary value generation for edge cases.
    #[test]
    fn boundary_value_generation() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);

        // Universal boundaries
        let universal = gen.universal_boundaries();
        assert!(universal.iter().any(|v| *v == 0.0));
        assert!(universal.iter().any(|v| v.is_nan()));
        assert!(universal.iter().any(|v| v.is_infinite()));

        // Format-specific boundaries
        let format_bounds = gen.format_boundaries();
        // Q4K has 16 levels × 2 (±) = 32 values
        assert_eq!(format_bounds.len(), 32);

        // All boundaries
        let all = gen.all_boundaries();
        assert_eq!(all.len(), universal.len() + format_bounds.len());
    }

    /// Test parity checking for CPU/GPU comparison.
    #[test]
    fn parity_check_cpu_gpu() {
        let config = ParityConfig::new(QuantFormat::Q4K);

        // Identical values: pass
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.0, 3.0];
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());

        // Small difference within tolerance: pass
        let gpu_close = vec![1.01, 2.01, 3.01];
        let report = check_values_parity(&cpu, &gpu_close, &config);
        assert!(report.passed());

        // Large difference: fail
        let gpu_far = vec![1.0, 2.5, 3.0];
        let report = check_values_parity(&cpu, &gpu_far, &config);
        assert!(!report.passed());
        assert_eq!(report.violations.len(), 1);
    }

    /// Test NaN handling in parity checks.
    #[test]
    fn parity_nan_handling() {
        let config = ParityConfig::new(QuantFormat::F32);

        // NaN vs NaN: OK (both are NaN)
        let cpu = vec![f64::NAN];
        let gpu = vec![f64::NAN];
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());

        // NaN vs number: violation
        let gpu_num = vec![1.0];
        let report = check_values_parity(&cpu, &gpu_num, &config);
        assert!(!report.passed());
    }
}

// ============================================================================
// F5: PTX Compilation Poison Trap — Kernel Verification
// ============================================================================

mod ptx_poison_tests {
    use super::*;

    /// Test PTX structural verification.
    #[test]
    fn ptx_structural_verification() {
        let verifier = PtxVerifier::new();

        // Valid PTX passes
        let result = verifier.verify(MINIMAL_VALID_PTX);
        assert!(result.is_ok());

        // Empty PTX fails
        let result = verifier.verify("");
        assert!(result.is_err());

        // Missing .version fails
        let no_version = ".target sm_80\n.address_size 64\n.entry k() { ret; }";
        let errors = verifier.check_all(no_version);
        assert!(!errors.is_empty());
    }

    /// Test mutation operators for kernel testing.
    #[test]
    fn mutation_operators() {
        let mutators = default_mutators();
        assert_eq!(mutators.len(), 8);

        // Arithmetic mutations
        assert!(mutators.contains(&PtxMutator::FlipAddSub));
        assert!(mutators.contains(&PtxMutator::FlipMulDiv));

        // Control flow mutations
        assert!(mutators.contains(&PtxMutator::InvertPredicate));
        assert!(mutators.contains(&PtxMutator::RemoveBarrier));

        // Precision mutations
        assert!(mutators.contains(&PtxMutator::WidenPrecision));
    }

    /// Test mutation application to PTX source.
    #[test]
    fn mutation_application() {
        // FlipAddSub: add → sub
        let ptx = "add.f32 %f1, %f2, %f3;";
        let mutated = PtxMutator::FlipAddSub.apply(ptx);
        assert!(mutated.is_some());
        assert!(mutated.unwrap().contains("sub.f32"));

        // FlipMulDiv: mul → div
        let ptx = "mul.f32 %f1, %f2, %f3;";
        let mutated = PtxMutator::FlipMulDiv.apply(ptx);
        assert!(mutated.is_some());
        assert!(mutated.unwrap().contains("div.f32"));

        // InvertPredicate: setp.lt → setp.ge
        let ptx = "setp.lt.f32 %p1, %f1, %f2;";
        let mutated = PtxMutator::InvertPredicate.apply(ptx);
        assert!(mutated.is_some());
        assert!(mutated.unwrap().contains("setp.ge"));
    }

    /// Test PTX verification catches common errors.
    #[test]
    fn ptx_common_errors() {
        let verifier = PtxVerifier::new();

        // Missing .target
        let no_target = ".version 7.0\n.address_size 64\n.entry k() { ret; }";
        let errors = verifier.check_all(no_target);
        assert!(!errors.is_empty());

        // Missing .address_size
        let no_addr = ".version 7.0\n.target sm_80\n.entry k() { ret; }";
        let errors = verifier.check_all(no_addr);
        assert!(!errors.is_empty());

        // Missing entry point
        let no_entry = ".version 7.0\n.target sm_80\n.address_size 64\n";
        let errors = verifier.check_all(no_entry);
        assert!(!errors.is_empty());
    }
}

// ============================================================================
// Supervisor Integration — GPU Worker Management
// ============================================================================

mod supervisor_tests {
    use super::*;

    /// Test supervision strategies for GPU workers.
    #[test]
    fn supervision_strategies() {
        // One-for-one: isolated restarts
        assert!(SupervisionStrategy::OneForOne.is_isolated());

        // One-for-all: restart all on any failure
        assert!(!SupervisionStrategy::OneForAll.is_isolated());

        // Rest-for-one: restart crashed + dependents
        assert!(!SupervisionStrategy::RestForOne.is_isolated());
    }

    /// Test supervision tree crash handling.
    #[test]
    fn supervision_tree_operations() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);

        // Crash worker 2 at time 0
        let action = tree.handle_crash(2, 0);
        match action {
            trueno_cuda_edge::supervisor::SupervisorAction::Restart(indices) => {
                assert_eq!(indices, vec![2]);
            }
            _ => panic!("Expected Restart action"),
        }
    }

    /// Test one-for-all strategy.
    #[test]
    fn one_for_all_restarts() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::OneForAll, 3);

        let action = tree.handle_crash(1, 0);
        match action {
            trueno_cuda_edge::supervisor::SupervisorAction::Restart(indices) => {
                assert_eq!(indices, vec![0, 1, 2]);
            }
            _ => panic!("Expected Restart action"),
        }
    }

    /// Test health monitoring for GPU workers.
    #[test]
    fn health_monitoring() {
        let monitor = GpuHealthMonitor::builder()
            .max_missed(3)
            .throttle_temp(85)
            .shutdown_temp(95)
            .build();

        // Alive: healthy
        assert_eq!(
            monitor.check_status(HeartbeatStatus::Alive),
            HealthAction::Healthy
        );

        // Missed beats below threshold: healthy
        assert_eq!(
            monitor.check_status(HeartbeatStatus::MissedBeats(2)),
            HealthAction::Healthy
        );

        // Missed beats at threshold: restart
        assert_eq!(
            monitor.check_status(HeartbeatStatus::MissedBeats(3)),
            HealthAction::RestartWorker
        );

        // Dead: shutdown
        assert_eq!(
            monitor.check_status(HeartbeatStatus::Dead),
            HealthAction::Shutdown
        );
    }

    /// Test thermal monitoring thresholds.
    #[test]
    fn thermal_monitoring() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);

        // Below throttle: healthy
        assert_eq!(monitor.check_temperature(70), HealthAction::Healthy);

        // At throttle threshold: throttle
        assert_eq!(monitor.check_temperature(85), HealthAction::Throttle);

        // Between throttle and shutdown: throttle
        assert_eq!(monitor.check_temperature(90), HealthAction::Throttle);

        // At shutdown threshold: shutdown
        assert_eq!(monitor.check_temperature(95), HealthAction::Shutdown);
    }
}

// ============================================================================
// Falsification Protocol — Coverage Tracking
// ============================================================================

mod falsification_tests {
    use super::*;

    /// Verify 50-point protocol completeness.
    #[test]
    fn protocol_completeness() {
        let claims = all_claims();
        assert_eq!(claims.len(), 50);
    }

    /// Test claim framework distribution.
    #[test]
    fn framework_distribution() {
        let claims = all_claims();

        let null_fuzzer = claims
            .iter()
            .filter(|c| c.framework == Framework::NullFuzzer)
            .count();
        let shmem = claims
            .iter()
            .filter(|c| c.framework == Framework::ShmemProber)
            .count();
        let lifecycle = claims
            .iter()
            .filter(|c| c.framework == Framework::LifecycleChaos)
            .count();
        let quant = claims
            .iter()
            .filter(|c| c.framework == Framework::QuantOracle)
            .count();
        let ptx = claims
            .iter()
            .filter(|c| c.framework == Framework::PtxPoison)
            .count();
        let supervisor = claims
            .iter()
            .filter(|c| c.framework == Framework::Supervisor)
            .count();

        assert_eq!(null_fuzzer, 10);
        assert_eq!(shmem, 10);
        assert_eq!(lifecycle, 8);
        assert_eq!(quant, 8);
        assert_eq!(ptx, 8);
        assert_eq!(supervisor, 6);
    }

    /// Test report status tracking.
    #[test]
    fn report_status_tracking() {
        let mut report = FalsificationReport::new();

        // All start pending
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Pending));

        // Mark verified
        report.mark_verified("NF-001");
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Verified));

        // Mark violated
        report.mark_violated("NF-002");
        assert_eq!(report.status("NF-002"), Some(ClaimStatus::Violated));

        // Coverage increases
        assert!(report.coverage() > 0.0);
    }

    /// Test framework grouping.
    #[test]
    fn framework_grouping() {
        let report = FalsificationReport::new();
        let grouped = report.by_framework();

        assert!(grouped.contains_key(&Framework::NullFuzzer));
        assert!(grouped.contains_key(&Framework::ShmemProber));
        assert!(grouped.contains_key(&Framework::LifecycleChaos));
        assert!(grouped.contains_key(&Framework::QuantOracle));
        assert!(grouped.contains_key(&Framework::PtxPoison));
        assert!(grouped.contains_key(&Framework::Supervisor));
    }
}
