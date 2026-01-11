//! cbtop - Compute Block Top
//!
//! Real-time load testing and hardware monitoring TUI built on the Brick Architecture.
//!
// Allow new_without_default - explicit new() is clearer for these types
#![allow(clippy::new_without_default)]
// Allow derivable_impls - explicit Default is clearer
#![allow(clippy::derivable_impls)]
// Allow missing_panics_doc - not critical for internal methods
#![allow(clippy::missing_panics_doc)]
// Allow missing_errors_doc - errors are self-explanatory
#![allow(clippy::missing_errors_doc)]
// Allow unnecessary_map_or - map_or is clearer
#![allow(clippy::unnecessary_map_or)]
// Allow collapsible_if - clarity over conciseness
#![allow(clippy::collapsible_if)]
// Allow needless_range_loop - clearer in some cases
#![allow(clippy::needless_range_loop)]
// Allow cast_precision_loss - acceptable for display values
#![allow(clippy::cast_precision_loss)]
// Allow cast_possible_truncation - handled appropriately
#![allow(clippy::cast_possible_truncation)]
// Allow dead_code - development in progress
#![allow(dead_code)]
// Allow field_reassign_with_default - clearer initialization
#![allow(clippy::field_reassign_with_default)]
// Allow manual_flatten - clearer error handling
#![allow(clippy::manual_flatten)]
//!
//! # Design Philosophy
//!
//! - **Test-as-Interface**: Every component is a falsifiable Brick (PROBAR-SPEC-009)
//! - **presentar-terminal**: All widgets and canvas from presentar-terminal (no custom reimplementation)
//! - **Toyota Way**: Jidoka, Poka-Yoke, Genchi Genbutsu principles throughout
//!
//! # Architecture
//!
//! ```text
//! Layer 4: Load Generators  → SimdLoadBrick, CudaLoadBrick, WgpuLoadBrick
//! Layer 3: Panels           → Overview, CPU, GPU, PCIe, Memory, Thermal
//! Layer 2: Analyzers        → Throughput, Bottleneck, Thermal
//! Layer 1: Collectors       → CPU, GPU, PCIe, Memory, Thermal
//! ```
//!
//! # Widget Source Policy
//!
//! All widgets and canvas implementations come from `presentar-terminal`.
//! cbtop does NOT implement its own widgets. If a widget is missing, it MUST
//! be added to presentar-terminal FIRST, then used here.

pub mod adversarial;
pub mod baseline;
pub mod brick;
pub mod bricks;
pub mod ring_buffer;
pub mod app;
pub mod cache_analysis;
pub mod config;
pub mod double_blind;
pub mod error;
pub mod grammar;
pub mod headless;
pub mod continuous_batcher;
pub mod ironman;
pub mod latency_distribution;
pub mod optimize;
pub mod paged_kv;
pub mod quantize;
pub mod roofline;
pub mod tracing_escalation;
pub mod fuzz;
pub mod statistics;
pub mod variance_analysis;
pub mod profile_persistence;
pub mod golden_trace;
pub mod thermal_prediction;
pub mod backend_regression;
pub mod correlation_analysis;
pub mod performance_prediction;
pub mod anomaly_detection;
pub mod workload_characterization;
pub mod export_reporting;
pub mod adaptive_threshold;
pub mod frequency_control;
pub mod context_regression;
pub mod alerting;
pub mod prometheus;
pub mod cost_tracker;
pub mod event_streaming;
pub mod remote_agent;
pub mod profile_compare;
pub mod observability_backend;
pub mod regression_pipeline;
pub mod federated_metrics;
pub mod adaptive_ml;
pub mod incremental_snapshot;
pub mod predictive_scheduler;

// Core brick traits (cbtop-specific)
pub use brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
// ComputeBrick Scoring Framework (§29)
pub use brick::{BrickScore, BrickGrade, Scorable};

// Application
pub use app::CbtopApp;
pub use config::Config;
pub use error::CbtopError;

// Headless benchmarking API (HL-007)
pub use headless::{
    Benchmark, BenchmarkBuilder, BenchmarkResult, BenchmarkConfig,
    BenchmarkResults, LatencyStats, RegressionResult, ComparisonResult,
    OutputFormat, SystemInfo, ScoreInfo,
};
// Re-export config types for benchmark builder
pub use config::{ComputeBackend, WorkloadType};

// Optimization tooling (OPT-001 to OPT-004)
pub use optimize::{
    CpuCapabilities,
    OptimizationSuite, BaselineReport, BaselineEntry, WorkloadConfig,
    BottleneckAnalysis, BottleneckEntry, BottleneckSeverity, AnalysisSummary,
    RegressionDetector, RegressionReport, RegressionEntry,
    OptimizationValidator, ValidationResult,
};

// Industry Baseline Validation (PMAT-016)
pub use baseline::{
    ServerBaseline, GpuClass, ThroughputGrade, SmHealth,
    BaselineComparison, SingleComparison, BaselineValidator, ValidationSummary,
    VLLM_BASELINE, TGI_BASELINE, TRITON_BASELINE, INDUSTRY_BASELINES,
};

// Quantized Weight Support (PMAT-013)
pub use quantize::{
    QuantFormat, DequantStrategy, QuantizedWeights, QuantStats, LayerQuantStats,
    GgufHeader, GgufValue, GgufTensorInfo, GgufLoader, GgufResult, GgufError,
    QuantizedBrick, ggml_type_to_format,
};

// Paged KV Cache (PMAT-014)
pub use paged_kv::{
    BlockId, SeqId, EvictionStrategy, KvBlock, SequenceInfo,
    PagedKvCache, PagedKvError, PagedKvResult, CacheStats,
};

// Continuous Batcher (PMAT-015)
pub use continuous_batcher::{
    Token, Priority, InferenceRequest, SequenceGroup, SchedulingPolicy,
    BatchSchedule, TokenOutput, BatcherStats, ContinuousBatcher,
    ExponentialMovingAverage, SpeculativeOutput, SpeculativeDecoder,
};

// Ironman Falsification Suite (PMAT-017)
pub use ironman::{
    GateResult, QualityGate, GateCategory, IronmanScorecard,
    IronmanValidator, IRONMAN_GATES, quick_validate, full_validate,
};

// Grammar of ComputeBlock (PMAT-018)
pub use grammar::{
    GrammarResult, GrammarError, Operation, DataType, Dimensions, TensorSpec,
    WorkloadSpec, ScaleBinding, ByteSize, ResourceMapping, SimdWidth, GpuDevice,
    KernelSpec, ExecutionStrategy, QuantScheme, DataTransform, CpuAffinity,
    ExecutionContext, CompositionMode, QosLevel, RetryPolicy, ResourceLimits,
    ObservabilityConfig, ExecutionPolicy, StrategyLayer, ExecutionResult,
    BuiltComputeBlock, ComputeBlock, ComputeBlockBuilder, ResourceScale,
    LinearResourceScale, LogResourceScale,
};

// Adversarial Falsification Testing (PMAT-019)
pub use adversarial::{
    AdversarialError, AdversarialResult, AdversarialTactic,
    InputValidator, BitFlipInjector, CheckedArithmetic,
    MonotonicClock, ResourceLimiter, ResourceUsage,
    ConfigValidator, CancellationToken, RecoveryHandler,
    AdversarialTestSummary,
};

// Double-Blind Verification Framework (PMAT-020)
pub use double_blind::{
    Role, VerificationResult, FalsificationCriterion,
    FalsificationClaim, BlackBoxArtifact, VerificationAttempt,
    ScorecardComponent, ScorecardV2, ReleaseDecision,
    AuditEntry, VerificationSession, SessionState,
    VerificationReport,
};

// Tracing Escalation Framework (PMAT-021)
pub use tracing_escalation::{
    EscalationReason, EscalationThresholds, SyscallBreakdown,
    TraceResult, TracingEscalation, OtlpSpanAttributes,
};

// Roofline Model Analyzer (PMAT-022)
pub use roofline::{
    BottleneckType, HardwareProfile, WorkloadMetrics,
    RooflineAnalysis, RooflinePlotPoint, RooflinePlot,
    BatchRooflineAnalysis, BatchSummary,
};

// Fuzz Testing Integration (PMAT-023)
pub use fuzz::{
    FuzzResult, FuzzFailure, FuzzInputValidator, FuzzValidationError,
    FuzzTargetConfig, FuzzSuite, FuzzSummary,
    safe_div, checked_add_u64, checked_mul_u64, bound_value, sanitize_float,
    test_float_edge_cases, test_u64_edge_cases,
};

// Statistical Analysis (PMAT-024)
pub use statistics::{
    EffectCategory, StatisticalAnalysis, EffectSize,
    ComparisonResult as StatisticalComparison,
    MannWhitneyResult, OutlierFilter,
    bootstrap_ci, percentile, trimmed_mean,
};

// Cache Efficiency Analysis (PMAT-025)
pub use cache_analysis::{
    CacheLevel, CacheConfig, WorkingSetAnalysis,
    AccessPattern, BandwidthPrediction,
    matrix_working_set, optimal_matmul_tile, elementwise_working_set,
};

// Latency Distribution Analysis (PMAT-026)
pub use latency_distribution::{
    LatencyDistribution, LatencyHistogram, HistogramBucket,
    TailSeverity, DistributionShape,
};

// Variance Source Analysis (PMAT-027)
pub use variance_analysis::{
    VarianceSource, VarianceAnalysis, VarianceInput,
};

// Profile Persistence and Rotation (PMAT-028)
pub use profile_persistence::{
    ProfileConfig, ProfileManager, ProfileOverlay, ProfileError, ProfileResult,
    BackendConfig as ProfileBackend, WorkloadConfig as ProfileWorkload,
};

// Golden Trace Comparison (PMAT-029)
pub use golden_trace::{
    GoldenTrace, GoldenTraceManager, GoldenComparator, GoldenTraceError, GoldenTraceResult,
    TraceMetrics, SyscallBreakdownDelta, TraceComparison,
    SyscallBreakdown as TraceSyscallBreakdown,
};

// Thermal Trend Prediction (PMAT-030)
pub use thermal_prediction::{
    ThermalAnalyzer, ThermalSample, ThermalPrediction, ThrottleRisk,
    CooldownRecommendation, ThermalCorrelation, ThermalVariance, RiskCategory,
    analyze_thermal, assess_throttle_risk,
    DEFAULT_THROTTLE_THRESHOLD_C, MIN_SAMPLES_FOR_ANALYSIS,
};

// Cross-Backend Regression Detector (PMAT-031)
pub use backend_regression::{
    Backend, WorkloadType as BackendWorkload, BackendMeasurement, BackendComparison,
    BackendRegressionDetector, SizeCliff, BackendRecommendation, TransferAnalysis,
    BackendSummary,
};

// Multi-Metric Correlation Analysis (PMAT-032)
pub use correlation_analysis::{
    CorrelationAnalyzer, EventType, EventSample, PerformanceSample,
    CorrelationResult, InterferenceResult, InterferenceCategory,
    IsolationRecommendation, IsolationAction, SystemSnapshot,
};

// Performance Prediction Model (PMAT-033)
pub use performance_prediction::{
    PerformancePredictor, DataPoint, ModelType, FittedModel, Prediction,
    MIN_SAMPLES_FOR_FIT,
};

// Anomaly Detection Engine (PMAT-034)
pub use anomaly_detection::{
    AnomalyDetector, Anomaly, AnomalyType, AnomalySeverity, ChangePoint,
    AnomalyReport, DEFAULT_ZSCORE_THRESHOLD, DEFAULT_IQR_MULTIPLIER,
    MIN_SAMPLES_FOR_DETECTION,
};

// Workload Characterization System (PMAT-035)
pub use workload_characterization::{
    WorkloadCharacterizer, WorkloadFeatures, WorkloadCategory,
    ClassificationResult, RecommendedBackend,
};

// Multi-Format Export System (PMAT-036)
pub use export_reporting::{
    ExportFormat, ReportType, BenchmarkMetric, BenchmarkReport,
    ComparisonEntry, ComparisonReport, ReportExporter, ReportBuilder,
};

// Adaptive Threshold Learning System (PMAT-037)
pub use adaptive_threshold::{
    ThresholdLearner, LearnedThreshold, ThresholdDirection, ThresholdCheck,
    MIN_SAMPLES_FOR_LEARNING, DEFAULT_CONFIDENCE_LEVEL, DEFAULT_OUTLIER_THRESHOLD,
};

// CPU Frequency Control Backend (PMAT-038)
pub use frequency_control::{
    FrequencyController, FrequencyReading, FrequencyVariance, FrequencyLock,
    CpuGovernor, CpuFrequencyInfo,
};

// Context-Aware Regression Predictor (PMAT-039)
pub use context_regression::{
    ContextRegressionPredictor, SystemContext,
    BaselineEntry as ContextBaselineEntry,
    RegressionThreshold, RegressionCheck, Trend,
    DEFAULT_COLD_START_MARGIN, MIN_SAMPLES_FOR_CONTEXT, DEFAULT_STALENESS_SEC,
};

// Real-Time Alert Integration System (PMAT-040)
pub use alerting::{
    AlertSeverity, AlertChannel, Alert, AlertRouter, AlertRouterConfig,
    DeliveryResult, MessageTemplate, alert_from_anomaly,
};

// Prometheus Metrics Exporter (PMAT-041)
pub use prometheus::{
    MetricType, Labels, MetricsRegistry, GaugeValue, CounterValue, HistogramValue,
    HistogramBuckets, MetricDef, validate_metric_name,
    DEFAULT_MAX_LABELS, DEFAULT_BUCKETS,
};

// Cost and Energy Efficiency Tracker (PMAT-042)
pub use cost_tracker::{
    CloudProvider, GpuPricing, EnergyMeasurement, CostResult, CostComparison,
    BudgetAlert, CostTracker, default_gpu_pricing,
    JOULES_PER_KWH, DEFAULT_CARBON_INTENSITY,
};

// Structured Event Streaming (PMAT-043)
pub use event_streaming::{
    SinkType, MetricEvent, EventBatch, SinkHealth, RetryConfig, EventStreamer,
    event_from_sample, compress_data,
    SCHEMA_VERSION, DEFAULT_BATCH_SIZE,
};

// Remote SSH/Headless Agent Integration (PMAT-044)
pub use remote_agent::{
    RemoteError, RemoteResult, AuthMethod, HostConfig, HostHealth, HostState,
    CommandResult, AggregatedResult, HostBenchmark, AggregationStrategy,
    RemoteAgentConfig, RemoteAgent,
    DEFAULT_RETRY_DELAY_MS, DEFAULT_MAX_CONCURRENT, DEFAULT_HEALTH_CHECK_INTERVAL_SEC,
};

// Profile Diffing and A/B Comparison (PMAT-045)
pub use profile_compare::{
    CompareError, CompareResult, BenchmarkProfile, MetricSamples,
    WelchTestResult, EffectMagnitude, EffectSizeResult, ChangeDirection,
    MetricComparison, ProfileComparison, ComparisonVerdict, CompareConfig,
    ProfileComparator,
    MIN_COMPARISON_SAMPLES,
    DEFAULT_CONFIDENCE_LEVEL as COMPARE_DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_REGRESSION_THRESHOLD as COMPARE_DEFAULT_REGRESSION_THRESHOLD,
};

// Observability Backend Integrations (PMAT-046)
pub use observability_backend::{
    ObservabilityError, ObservabilityResult, ObservabilityBackend,
    DatadogConfig, NewRelicConfig, HoneycombConfig, OtlpConfig, WebhookConfig,
    ExportMetric, MetricExportType, ExportResult, BackendHealth,
    ObservabilityConfig as ObsBackendConfig, ObservabilityExporter, format_dogstatsd,
    DEFAULT_BATCH_SIZE as OBS_DEFAULT_BATCH_SIZE, DEFAULT_FLUSH_INTERVAL_MS,
};

// CI/CD Regression Detection Pipeline (PMAT-047)
pub use regression_pipeline::{
    PipelineError, PipelineResult, PipelineStatus, GitRef,
    PipelineConfig, BenchmarkMetric as PipelineBenchmarkMetric, BenchmarkResults as PipelineBenchmarkResults,
    MetricRegression, RegressionAnalysis, StatusCheck, RegressionPipeline,
    DEFAULT_TIMEOUT_SEC, DEFAULT_REGRESSION_THRESHOLD as PIPELINE_REGRESSION_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
};

// Federated Metrics Aggregation (PMAT-048)
pub use federated_metrics::{
    GCounter, LwwRegister, OrSet, FederatedHost, AggregatedMetrics,
    FederationConfig, MetricsFederation,
};

// Dynamic Adaptive Thresholds with ML (PMAT-049)
pub use adaptive_ml::{
    WorkloadClass, TimeSeriesFeatures, LearnedWorkloadThreshold,
    MlThresholdConfig, AdaptiveThresholdMl, ClassificationMetrics as MlClassificationMetrics,
    MlThresholdError, AnomalyResult as MlAnomalyResult,
};

// Incremental Profile Snapshots (PMAT-050)
pub use incremental_snapshot::{
    MetricData, ProfileSnapshot, DeltaMetric, DeltaSnapshot,
    SnapshotConfig, IncrementalSnapshotStore, SnapshotIndex, SnapshotQuery,
    SnapshotError, RetentionTier,
};

// Predictive Scheduling Optimizer (PMAT-051)
pub use predictive_scheduler::{
    InstanceType, HostProfile, WorkloadSpec as SchedulerWorkloadSpec, SchedulingDecision, SchedulerMetrics,
    PredictiveSchedulerConfig, PredictiveScheduler,
};

// Re-export presentar-terminal widgets and canvas for convenience
// All widgets MUST come from presentar-terminal - DO NOT reimplement
pub use presentar_terminal::{
    BrailleGraph, GraphMode, Meter, Table,
    ColorMode,
};
pub use presentar_terminal::direct::{
    CellBuffer, DiffRenderer, DirectTerminalCanvas,
};

// Re-export presentar-core traits
pub use presentar_core::{
    Canvas, Color, Point, Rect, Size, TextStyle, Constraints,
};
