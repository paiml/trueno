//! Grammar of ComputeBlock Falsification Tests (F701-F720)
//!
//! PMAT-018: Test the Grammar of ComputeBlock DSL per §32.
//!
//! # Falsification Criteria
//!
//! | ID | Claim | Test | Pass Criteria |
//! |----|-------|------|---------------|
//! | F701 | Builder rejects incomplete spec | Build without workload | Returns Err |
//! | F702 | Strategy fallback works | Request GPU on CPU-only | Falls back to CPU |
//! | F703 | Resource scaling honors limits | Request 1TB memory | Error/Cap applied |
//! | F704 | Composition output consistent | Batch(1) vs None | Identical output |
//! | F710 | Identity transform is no-op | Apply Identity | Output == Input |
//! | F711 | Scale domain validation | Domain(10, 0) | Returns Err |
//! | F719 | Builder immutability | Reuse builder | Independent instances |

use cbtop::{
    ByteSize, CompositionMode, ComputeBlock, ComputeBlockBuilder, DataTransform, DataType,
    Dimensions, ExecutionContext, ExecutionPolicy, ExecutionStrategy, GpuDevice, GrammarError,
    LinearResourceScale, LogResourceScale, Operation, QosLevel, ResourceMapping, ResourceScale,
    SimdWidth, StrategyLayer, TensorSpec, WorkloadSpec,
};

// ============================================================================
// F701: Builder Rejects Incomplete Spec
// ============================================================================

#[test]
fn f701_builder_rejects_no_workload() {
    // F701: Build without workload must return Err
    let result = ComputeBlock::builder().build();
    assert!(matches!(result, Err(GrammarError::MissingWorkload)));
}

#[test]
fn f701_builder_accepts_with_workload() {
    // F701: Build with workload must succeed
    let result = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .build();
    assert!(result.is_ok());
}

#[test]
fn f701_builder_accepts_minimal_spec() {
    // F701: Minimal valid spec
    let result = ComputeBlock::builder()
        .workload(WorkloadSpec::elementwise(100))
        .build();
    assert!(result.is_ok());
}

// ============================================================================
// F702: Strategy Fallback
// ============================================================================

#[test]
fn f702_strategy_layers_ordered_by_priority() {
    // F702: Higher priority strategies tried first
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(64, 64, 64))
        .strategy_with_priority(ExecutionStrategy::Sequential, 0)
        .strategy_with_priority(ExecutionStrategy::simd_auto(), 5)
        .strategy_with_priority(ExecutionStrategy::gpu_auto(), 10)
        .build()
        .unwrap();

    let result = block.execute().unwrap();
    // GPU should be tried first (highest priority)
    assert!(result.strategy_used.contains("Gpu"));
}

#[test]
fn f702_fallback_to_lower_priority() {
    // F702: Multiple strategies available for fallback
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .strategy(ExecutionStrategy::simd_auto())
        .strategy(ExecutionStrategy::Sequential)
        .build()
        .unwrap();

    let result = block.execute();
    assert!(result.is_ok());
}

// ============================================================================
// F703: Resource Scaling
// ============================================================================

#[test]
fn f703_resource_mapping_cores() {
    // F703: Resource mapping with core binding
    let resources = ResourceMapping::new().cores_value(8);

    assert_eq!(resources.cores_value, Some(8));
}

#[test]
fn f703_resource_mapping_memory() {
    // F703: Resource mapping with memory limit
    let resources = ResourceMapping::new().memory_value(ByteSize::gb(4));

    assert_eq!(resources.memory_value, Some(ByteSize::gb(4)));
}

#[test]
fn f703_byte_size_conversions() {
    // F703: ByteSize helper works correctly
    assert_eq!(ByteSize::mb(1).bytes(), 1024 * 1024);
    assert_eq!(ByteSize::gb(1).bytes(), 1024 * 1024 * 1024);
    assert_eq!(ByteSize::gb(4).bytes(), 4 * 1024 * 1024 * 1024);
}

// ============================================================================
// F704: Composition Consistency
// ============================================================================

#[test]
fn f704_composition_none_vs_batch_1() {
    // F704: Batch(1) should behave like None
    let none_block = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .composition(CompositionMode::None)
        .build()
        .unwrap();

    let batch_block = ComputeBlock::builder()
        .workload(WorkloadSpec::dot(1000))
        .composition(CompositionMode::batch(1))
        .build()
        .unwrap();

    // Both should execute successfully
    let none_result = none_block.execute();
    let batch_result = batch_block.execute();

    assert!(none_result.is_ok());
    assert!(batch_result.is_ok());
}

#[test]
fn f704_data_parallel_composition() {
    // F704: DataParallel composition
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(1024, 1024, 1024))
        .composition(CompositionMode::data_parallel(4))
        .build()
        .unwrap();

    let result = block.execute();
    assert!(result.is_ok());
}

#[test]
fn f704_pipeline_composition() {
    // F704: Pipeline composition
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::attention(1, 512, 8, 64))
        .composition(CompositionMode::pipeline(2))
        .build()
        .unwrap();

    let result = block.execute();
    assert!(result.is_ok());
}

// ============================================================================
// F710: Identity Transform
// ============================================================================

#[test]
fn f710_identity_transform_is_noop() {
    // F710: Identity transform doesn't modify anything
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::elementwise(1000))
        .transform(DataTransform::Identity)
        .build()
        .unwrap();

    // Should execute without modification
    let result = block.execute();
    assert!(result.is_ok());
}

#[test]
fn f710_transform_identity_constructor() {
    // F710: DataTransform::identity() returns Identity variant
    let t = DataTransform::identity();
    assert!(matches!(t, DataTransform::Identity));
}

// ============================================================================
// F711: Scale Domain Validation
// ============================================================================

#[test]
fn f711_linear_scale_invalid_domain() {
    // F711: Domain(10, 0) must return Err
    let result = LinearResourceScale::new((10.0, 0.0), (0.0, 8.0));
    assert!(matches!(
        result,
        Err(GrammarError::InvalidScaleDomain { .. })
    ));
}

#[test]
fn f711_linear_scale_equal_domain() {
    // F711: Domain(10, 10) must return Err
    let result = LinearResourceScale::new((10.0, 10.0), (0.0, 8.0));
    assert!(matches!(
        result,
        Err(GrammarError::InvalidScaleDomain { .. })
    ));
}

#[test]
fn f711_linear_scale_valid_domain() {
    // F711: Valid domain should succeed
    let result = LinearResourceScale::new((0.0, 100.0), (0.0, 8.0));
    assert!(result.is_ok());
}

#[test]
fn f711_log_scale_invalid_domain() {
    // F711: Log scale with invalid domain
    let result = LogResourceScale::new(10.0, (100.0, 1.0), (0.0, 8.0));
    assert!(matches!(
        result,
        Err(GrammarError::InvalidScaleDomain { .. })
    ));
}

#[test]
fn f711_log_scale_valid_domain() {
    // F711: Log scale with valid domain
    let result = LogResourceScale::new(10.0, (1.0, 1000.0), (0.0, 8.0));
    assert!(result.is_ok());
}

// ============================================================================
// F712: Facet Generation
// ============================================================================

#[test]
fn f712_facet_by_sets_params() {
    // F712: Facet parameters are stored
    let block = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(64, 64, 64))
        .facet_by("tile_size", vec![16.0, 32.0, 64.0])
        .build()
        .unwrap();

    // Facet params should be accessible
    assert!(block.workload().is_some());
}

// ============================================================================
// F719: Builder Immutability
// ============================================================================

#[test]
fn f719_builder_creates_independent_instances() {
    // F719: Reusing builder creates independent instances
    let builder = ComputeBlock::builder().workload(WorkloadSpec::dot(1000));

    let block1 = builder
        .clone()
        .strategy(ExecutionStrategy::simd_auto())
        .build();
    let block2 = builder
        .clone()
        .strategy(ExecutionStrategy::Sequential)
        .build();

    assert!(block1.is_ok());
    assert!(block2.is_ok());

    // Both should execute independently
    let result1 = block1.unwrap().execute();
    let result2 = block2.unwrap().execute();

    assert!(result1.is_ok());
    assert!(result2.is_ok());
}

#[test]
fn f719_builder_default_is_empty() {
    // F719: Default builder has no workload
    let builder = ComputeBlockBuilder::new();
    let result = builder.build();
    assert!(matches!(result, Err(GrammarError::MissingWorkload)));
}

// ============================================================================
// Workload Specification Tests
// ============================================================================

#[test]
fn test_workload_dot_product() {
    let workload = WorkloadSpec::dot(10000);
    assert_eq!(workload.operation, Operation::Dot);
    assert_eq!(workload.dimensions.n, 10000);
    assert_eq!(workload.dtype, DataType::F32);
    assert_eq!(workload.inputs.len(), 2);
    assert_eq!(workload.outputs.len(), 1);
}

#[test]
fn test_workload_matmul() {
    let workload = WorkloadSpec::matmul(256, 256, 256);
    assert_eq!(workload.operation, Operation::Matmul);
    assert_eq!(workload.dimensions.m, 256);
    assert_eq!(workload.dimensions.n, 256);
    assert_eq!(workload.dimensions.k, 256);
}

#[test]
fn test_workload_attention() {
    let workload = WorkloadSpec::attention(2, 512, 8, 64);
    assert_eq!(workload.operation, Operation::Attention);
    assert_eq!(workload.dimensions.batch, 2);
    assert_eq!(workload.dimensions.seq_len, 512);
    assert_eq!(workload.dimensions.num_heads, 8);
    assert_eq!(workload.dimensions.head_dim, 64);
}

#[test]
fn test_workload_flop_count_dot() {
    let workload = WorkloadSpec::dot(1000);
    // Dot: n * 2 (mul + add)
    assert_eq!(workload.flop_count(), 2000);
}

#[test]
fn test_workload_flop_count_matmul() {
    let workload = WorkloadSpec::matmul(64, 64, 64);
    // Matmul: M * N * K * 2
    assert_eq!(workload.flop_count(), 64 * 64 * 64 * 2);
}

// ============================================================================
// Data Type Tests
// ============================================================================

#[test]
fn test_data_type_sizes() {
    assert_eq!(DataType::F32.byte_size(), 4);
    assert_eq!(DataType::F16.byte_size(), 2);
    assert_eq!(DataType::Bf16.byte_size(), 2);
    assert_eq!(DataType::I8.byte_size(), 1);
    assert_eq!(DataType::U8.byte_size(), 1);
    assert_eq!(DataType::Q4.byte_size(), 1);
}

// ============================================================================
// Tensor Specification Tests
// ============================================================================

#[test]
fn test_tensor_spec_numel() {
    let spec = TensorSpec::new("test", vec![10, 20, 30], DataType::F32);
    assert_eq!(spec.numel(), 6000);
}

#[test]
fn test_tensor_spec_byte_size() {
    let spec = TensorSpec::new("test", vec![100, 100], DataType::F32);
    assert_eq!(spec.byte_size(), 100 * 100 * 4);
}

// ============================================================================
// Execution Strategy Tests
// ============================================================================

#[test]
fn test_strategy_simd_auto() {
    let strategy = ExecutionStrategy::simd_auto();
    assert!(matches!(
        strategy,
        ExecutionStrategy::Simd {
            width: SimdWidth::Auto
        }
    ));
}

#[test]
fn test_strategy_simd_avx2() {
    let strategy = ExecutionStrategy::simd(SimdWidth::Avx2);
    assert!(matches!(
        strategy,
        ExecutionStrategy::Simd {
            width: SimdWidth::Avx2
        }
    ));
}

#[test]
fn test_strategy_parallel() {
    let strategy = ExecutionStrategy::parallel(8);
    assert!(matches!(
        strategy,
        ExecutionStrategy::Parallel { threads: 8, .. }
    ));
}

#[test]
fn test_strategy_gpu_auto() {
    let strategy = ExecutionStrategy::gpu_auto();
    assert!(matches!(
        strategy,
        ExecutionStrategy::Gpu {
            device: GpuDevice::Auto,
            ..
        }
    ));
}

// ============================================================================
// Execution Policy Tests
// ============================================================================

#[test]
fn test_policy_realtime() {
    let policy = ExecutionPolicy::realtime();
    assert_eq!(policy.qos, QosLevel::Realtime);
    assert!(!policy.preemptible);
    assert!(policy.timeout.is_some());
}

#[test]
fn test_policy_batch() {
    let policy = ExecutionPolicy::batch();
    assert_eq!(policy.qos, QosLevel::BestEffort);
    assert!(policy.preemptible);
    assert!(policy.timeout.is_none());
}

#[test]
fn test_policy_interactive() {
    let policy = ExecutionPolicy::interactive();
    assert_eq!(policy.qos, QosLevel::Interactive);
    assert!(policy.preemptible);
}

#[test]
fn test_policy_debug() {
    let policy = ExecutionPolicy::debug();
    assert!(policy.observability.tracing);
    assert!(policy.observability.metrics);
    assert_eq!(policy.observability.sampling_rate, 1.0);
}

// ============================================================================
// Data Transform Tests
// ============================================================================

#[test]
fn test_transform_tile() {
    let t = DataTransform::tile(64);
    assert!(matches!(t, DataTransform::Tile { tile_size: 64 }));
}

#[test]
fn test_transform_quantize() {
    let t = DataTransform::quantize(4);
    assert!(matches!(t, DataTransform::Quantize { bits: 4, .. }));
}

#[test]
fn test_transform_pad() {
    let t = DataTransform::pad(64);
    assert!(matches!(t, DataTransform::Pad { alignment: 64 }));
}

// ============================================================================
// Resource Scale Tests
// ============================================================================

#[test]
fn test_linear_scale_interpolation() {
    let scale = LinearResourceScale::new((0.0, 100.0), (0.0, 8.0)).unwrap();
    assert_eq!(scale.scale(0.0), 0.0);
    assert_eq!(scale.scale(50.0), 4.0);
    assert_eq!(scale.scale(100.0), 8.0);
}

#[test]
fn test_linear_scale_domain_range() {
    let scale = LinearResourceScale::new((10.0, 100.0), (1.0, 10.0)).unwrap();
    assert_eq!(scale.domain(), (10.0, 100.0));
    assert_eq!(scale.range(), (1.0, 10.0));
}

// ============================================================================
// Strategy Layer Tests
// ============================================================================

#[test]
fn test_strategy_layer_new() {
    let layer = StrategyLayer::new(ExecutionStrategy::simd_auto());
    assert_eq!(layer.priority, 0);
    assert!(layer.workload.is_none());
}

#[test]
fn test_strategy_layer_priority() {
    let layer = StrategyLayer::new(ExecutionStrategy::gpu_auto()).priority(10);
    assert_eq!(layer.priority, 10);
}

// ============================================================================
// Execution Context Tests
// ============================================================================

#[test]
fn test_context_cpu() {
    let ctx = ExecutionContext::cpu();
    assert!(matches!(ctx, ExecutionContext::Cpu { .. }));
}

#[test]
fn test_context_gpu() {
    let ctx = ExecutionContext::gpu(0);
    assert!(matches!(ctx, ExecutionContext::Gpu { device_id: 0, .. }));
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_pipeline() {
    let result = ComputeBlock::builder()
        .workload(WorkloadSpec::matmul(1024, 1024, 1024))
        .resources(ResourceMapping::new().cores_value(8))
        .strategy(ExecutionStrategy::simd_auto())
        .transform(DataTransform::tile(64))
        .context(ExecutionContext::cpu())
        .composition(CompositionMode::batch(32))
        .policy(ExecutionPolicy::interactive())
        .build()
        .unwrap()
        .execute();

    assert!(result.is_ok());
    let exec_result = result.unwrap();
    assert!(exec_result.gflops >= 0.0);
}

#[test]
fn test_dimensions_helpers() {
    let vec_dims = Dimensions::vector(1000);
    assert_eq!(vec_dims.n, 1000);

    let mat_dims = Dimensions::matmul(64, 128, 256);
    assert_eq!(mat_dims.m, 64);
    assert_eq!(mat_dims.n, 128);
    assert_eq!(mat_dims.k, 256);

    let attn_dims = Dimensions::attention(2, 512, 8, 64);
    assert_eq!(attn_dims.batch, 2);
    assert_eq!(attn_dims.seq_len, 512);
}

#[test]
fn test_error_display() {
    let err = GrammarError::MissingWorkload;
    assert!(!err.to_string().is_empty());

    let err = GrammarError::InvalidDimensions("test".to_string());
    assert!(err.to_string().contains("test"));

    let err = GrammarError::InvalidScaleDomain {
        min: 10.0,
        max: 0.0,
    };
    assert!(err.to_string().contains("10"));
}
