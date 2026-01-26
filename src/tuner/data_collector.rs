//! Training Data Collection
//!
//! Implements TunerDataCollector for collecting and persisting training samples.

use serde::{Deserialize, Serialize};

use crate::brick::BrickProfiler;

use super::brick_tuner::BrickTuner;
use super::error::TunerError;
use super::features::{FeatureExtractor, RunConfig, TunerFeatures};
use super::helpers::{chrono_lite_now, crc32_hash};
use super::types::{BottleneckClass, KernelType};

// ============================================================================
// TrainingSample
// ============================================================================

/// Training sample for the tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Features
    pub features: TunerFeatures,
    /// Measured throughput (label)
    pub throughput_tps: f32,
    /// Best kernel (label)
    pub best_kernel: KernelType,
    /// Bottleneck class (label)
    pub bottleneck: BottleneckClass,
    /// Timestamp
    pub timestamp: String,
    /// Hardware ID
    pub hardware_id: String,
}

// ============================================================================
// UserFeedback
// ============================================================================

/// User feedback on a recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum UserFeedback {
    /// User accepted the recommendation
    Accepted,
    /// User rejected the recommendation
    Rejected,
    /// User provided alternative (overrode recommendation)
    Alternative,
    /// No feedback (default)
    #[default]
    None,
}

// ============================================================================
// ConceptDriftStatus
// ============================================================================

/// Concept drift detection result
#[derive(Debug, Clone)]
pub struct ConceptDriftStatus {
    /// Whether drift has been detected
    pub drift_detected: bool,
    /// Estimated model staleness (0.0 = fresh, 1.0 = very stale)
    pub staleness_score: f32,
    /// Number of samples since last training
    pub samples_since_training: usize,
    /// Recommendation: should retrain?
    pub recommend_retrain: bool,
    /// Explanation of drift status
    pub explanation: String,
}

// ============================================================================
// TrainingStats
// ============================================================================

/// Training statistics summary
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total samples collected
    pub total_samples: usize,
    /// Samples since last training
    pub samples_since_training: usize,
    /// Accepted recommendations count
    pub accepted_count: usize,
    /// Rejected recommendations count
    pub rejected_count: usize,
    /// Alternative provided count
    pub alternative_count: usize,
    /// Staleness score (0.0 = fresh, 1.0 = stale)
    pub staleness_score: f32,
    /// Whether concept drift was detected
    pub drift_detected: bool,
    /// Whether online learning is enabled
    pub online_learning_enabled: bool,
}

// ============================================================================
// TunerDataCollector
// ============================================================================

/// Training data collector with online learning support (T-TUNER-005, GitHub #82)
#[derive(Debug, Default)]
pub struct TunerDataCollector {
    /// Collected samples
    pub(crate) samples: Vec<TrainingSample>,
    /// Feature extractor
    pub(crate) extractor: FeatureExtractor,
    /// Auto-retrain threshold
    pub(crate) retrain_threshold: usize,
    /// Number of samples at last training
    pub(crate) samples_at_last_train: usize,
    /// User feedback history (sample index -> feedback)
    pub(crate) feedback: Vec<UserFeedback>,
    /// Online learning enabled (privacy: opt-in only)
    pub(crate) online_learning_enabled: bool,
    /// Moving average of prediction errors (for concept drift)
    pub(crate) error_window: Vec<f32>,
    /// Error window size for drift detection
    error_window_size: usize,
}

impl TunerDataCollector {
    /// Default error window size for concept drift detection
    const DEFAULT_ERROR_WINDOW_SIZE: usize = 50;

    /// Error threshold for drift detection (mean absolute error)
    const DRIFT_ERROR_THRESHOLD: f32 = 0.15;

    /// Staleness threshold (samples since training) for recommending retrain
    const STALENESS_THRESHOLD: usize = 100;

    /// Minimum samples required before training triggers
    pub const MIN_SAMPLES_FOR_TRAINING: usize = 1000;

    /// Create a new collector
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false, // Privacy: opt-in
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        }
    }

    /// Create a collector with online learning enabled
    pub fn with_online_learning() -> Self {
        let mut collector = Self::new();
        collector.online_learning_enabled = true;
        collector
    }

    /// Enable online learning (privacy: explicit opt-in)
    pub fn enable_online_learning(&mut self) {
        self.online_learning_enabled = true;
    }

    /// Disable online learning
    pub fn disable_online_learning(&mut self) {
        self.online_learning_enabled = false;
    }

    /// Check if online learning is enabled
    pub fn is_online_learning_enabled(&self) -> bool {
        self.online_learning_enabled
    }

    /// Record a profiling run as training data
    pub fn record(
        &mut self,
        profiler: &BrickProfiler,
        config: &RunConfig,
        kernel: KernelType,
    ) -> Option<()> {
        let throughput_tps = profiler.tokens_per_sec()?;
        let features = self.extractor.extract(profiler, config);
        let bottleneck = features
            .bottleneck_class
            .unwrap_or(BottleneckClass::Unknown);

        let sample = TrainingSample {
            features,
            throughput_tps,
            best_kernel: kernel,
            bottleneck,
            timestamp: chrono_lite_now(),
            hardware_id: "unknown".to_string(),
        };

        self.samples.push(sample);
        Some(())
    }

    /// Get all samples
    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    /// Get sample count
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, TunerError> {
        serde_json::to_string_pretty(&self.samples)
            .map_err(|e| TunerError::Serialization(e.to_string()))
    }

    /// Prepare training data for model
    pub fn prepare_training_data(&self) -> Vec<(TunerFeatures, f32)> {
        self.samples
            .iter()
            .map(|s| (s.features.clone(), s.throughput_tps))
            .collect()
    }

    // ========================================================================
    // T-TUNER-003: Persistent Training Data (GitHub #80)
    // ========================================================================

    /// Training data cache path
    #[cfg(feature = "hardware-detect")]
    pub fn cache_path() -> std::path::PathBuf {
        let hw_id = Self::hardware_id();
        dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("trueno")
            .join(format!("training_data_{}.apr", hw_id))
    }

    /// Generate hardware fingerprint for hardware-specific models
    #[cfg(feature = "hardware-detect")]
    pub fn hardware_id() -> String {
        use crate::hardware::HardwareCapability;
        let hw = HardwareCapability::detect();

        // Create a stable fingerprint from hardware characteristics
        let fingerprint = format!(
            "{}-{:?}-{}-{}",
            hw.cpu.cores,
            hw.cpu.simd,
            hw.gpu.as_ref().map(|g| g.model.as_str()).unwrap_or("none"),
            hw.gpu.as_ref().map(|g| g.vram_gb as u32).unwrap_or(0),
        );

        // Hash to short hex string
        let hash = crc32_hash(fingerprint.as_bytes());
        format!("{:08x}", hash)
    }

    /// Load from cache or create empty
    #[cfg(feature = "hardware-detect")]
    pub fn load_or_create() -> Self {
        let path = Self::cache_path();
        if path.exists() {
            if let Ok(collector) = Self::load_apr(&path) {
                return collector;
            }
        }
        Self::new()
    }

    /// Save training data to APR format
    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TunerError> {
        use std::io::Write;

        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        }

        // Serialize samples to JSON
        let json = serde_json::to_string(&self.samples)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;
        let json_bytes = json.as_bytes();

        // Create APR format: MAGIC + LEN + JSON + CRC32
        let mut file = std::fs::File::create(path.as_ref())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write magic bytes: "APR2" (version 2 for training data)
        file.write_all(b"APR2")
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write length as u32 little-endian
        let len = json_bytes.len() as u32;
        file.write_all(&len.to_le_bytes())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write JSON
        file.write_all(json_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write CRC32 checksum
        let checksum = crc32_hash(json_bytes);
        file.write_all(&checksum.to_le_bytes())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        Ok(())
    }

    /// Load training data from APR format
    pub fn load_apr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TunerError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path.as_ref())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Read and verify magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        if &magic != b"APR2" {
            return Err(TunerError::InvalidFormat(format!(
                "Expected APR2 magic, got {:?}",
                magic
            )));
        }

        // Read length
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read JSON
        let mut json_bytes = vec![0u8; len];
        file.read_exact(&mut json_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Read and verify CRC32
        let mut crc_bytes = [0u8; 4];
        file.read_exact(&mut crc_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        let stored_crc = u32::from_le_bytes(crc_bytes);
        let computed_crc = crc32_hash(&json_bytes);

        if stored_crc != computed_crc {
            return Err(TunerError::InvalidFormat(format!(
                "CRC mismatch: stored={:08x}, computed={:08x}",
                stored_crc, computed_crc
            )));
        }

        // Deserialize samples
        let samples: Vec<TrainingSample> = serde_json::from_slice(&json_bytes)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;

        Ok(Self {
            samples,
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false,
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        })
    }

    /// Append a sample to the cached training data
    #[cfg(feature = "hardware-detect")]
    pub fn record_and_persist(
        &mut self,
        profiler: &BrickProfiler,
        config: &RunConfig,
        kernel: KernelType,
    ) -> Result<(), TunerError> {
        // Record the sample
        self.record(profiler, config, kernel);

        // Append to cache file
        let path = Self::cache_path();
        self.save_apr(&path)?;

        Ok(())
    }

    /// Check if we have enough samples to train
    pub fn ready_to_train(&self) -> bool {
        self.samples.len() >= Self::MIN_SAMPLES_FOR_TRAINING
    }

    /// Train a BrickTuner from collected data if we have enough samples
    pub fn train_if_ready(&self) -> Option<BrickTuner> {
        if !self.ready_to_train() {
            return None;
        }

        let training_data = self.prepare_training_data();
        let mut tuner = BrickTuner::new();

        match tuner.train(&training_data) {
            Ok(()) => Some(tuner),
            Err(_) => None,
        }
    }

    /// Get training progress as (current, required)
    pub fn training_progress(&self) -> (usize, usize) {
        (self.samples.len(), Self::MIN_SAMPLES_FOR_TRAINING)
    }

    /// Merge samples from another collector
    pub fn merge(&mut self, other: &TunerDataCollector) {
        self.samples.extend(other.samples.iter().cloned());
    }

    /// Import samples from JSON
    pub fn from_json(json: &str) -> Result<Self, TunerError> {
        let samples: Vec<TrainingSample> =
            serde_json::from_str(json).map_err(|e| TunerError::Serialization(e.to_string()))?;

        Ok(Self {
            samples,
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false,
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        })
    }

    /// Import samples from the Five-Whys archive (85 labeled iterations)
    /// Bootstrap initial training data from historical analysis
    pub fn bootstrap_from_five_whys() -> Self {
        // Five-Whys archive has 85 labeled iterations from SHOWCASE-BRICK-001
        // Each iteration has: features, throughput, kernel selection, bottleneck

        // TODO: Load actual Five-Whys data from archive
        // For now, return empty collector - data will be collected from real runs
        Self::new()
    }

    // ========================================================================
    // T-TUNER-005: Online Learning (GitHub #82)
    // ========================================================================

    /// Record user feedback on a recommendation
    pub fn record_feedback(&mut self, sample_index: usize, feedback: UserFeedback) {
        // Extend feedback vector if needed
        while self.feedback.len() <= sample_index {
            self.feedback.push(UserFeedback::None);
        }
        self.feedback[sample_index] = feedback;
    }

    /// Get feedback for a sample
    pub fn get_feedback(&self, sample_index: usize) -> UserFeedback {
        self.feedback
            .get(sample_index)
            .copied()
            .unwrap_or(UserFeedback::None)
    }

    /// Record prediction error for concept drift detection
    pub fn record_prediction_error(&mut self, predicted: f32, actual: f32) {
        if !self.online_learning_enabled {
            return;
        }

        // Compute relative error (0.0 = perfect, 1.0 = 100% error)
        let error = if actual > 0.0 {
            ((predicted - actual) / actual).abs().min(1.0)
        } else {
            1.0
        };

        // Add to sliding window
        self.error_window.push(error);

        // Trim window to max size
        if self.error_window.len() > self.error_window_size {
            self.error_window.remove(0);
        }
    }

    /// Detect concept drift based on prediction error trends
    pub fn detect_concept_drift(&self) -> ConceptDriftStatus {
        let samples_since_training = self
            .samples
            .len()
            .saturating_sub(self.samples_at_last_train);

        // Not enough data for drift detection
        if self.error_window.len() < 10 {
            return ConceptDriftStatus {
                drift_detected: false,
                staleness_score: 0.0,
                samples_since_training,
                recommend_retrain: false,
                explanation: "Insufficient data for drift detection".to_string(),
            };
        }

        // Compute mean error
        let mean_error: f32 =
            self.error_window.iter().sum::<f32>() / self.error_window.len() as f32;

        // Compute staleness score (0.0 = fresh, 1.0 = stale)
        let staleness_score =
            (samples_since_training as f32 / Self::STALENESS_THRESHOLD as f32).min(1.0);

        // Detect drift
        let drift_detected = mean_error > Self::DRIFT_ERROR_THRESHOLD;

        // Recommend retrain if drift detected OR stale
        let recommend_retrain = drift_detected || staleness_score > 0.8;

        let explanation = if drift_detected {
            format!(
                "Concept drift detected: mean error {:.1}% exceeds threshold {:.1}%",
                mean_error * 100.0,
                Self::DRIFT_ERROR_THRESHOLD * 100.0
            )
        } else if staleness_score > 0.8 {
            format!(
                "Model stale: {} samples since last training (threshold: {})",
                samples_since_training,
                Self::STALENESS_THRESHOLD
            )
        } else {
            format!(
                "Model fresh: mean error {:.1}%, {} samples since training",
                mean_error * 100.0,
                samples_since_training
            )
        };

        ConceptDriftStatus {
            drift_detected,
            staleness_score,
            samples_since_training,
            recommend_retrain,
            explanation,
        }
    }

    /// Check if auto-retrain should trigger
    pub fn should_retrain(&self) -> bool {
        if !self.online_learning_enabled {
            return false;
        }

        let samples_since = self
            .samples
            .len()
            .saturating_sub(self.samples_at_last_train);

        // Retrain if we have enough new samples
        if samples_since >= self.retrain_threshold {
            return true;
        }

        // Or if concept drift is detected
        let drift = self.detect_concept_drift();
        drift.recommend_retrain && samples_since >= 10
    }

    /// Mark that training occurred (resets drift counters)
    pub fn mark_trained(&mut self) {
        self.samples_at_last_train = self.samples.len();
        self.error_window.clear();
    }

    /// Get training statistics
    pub fn training_stats(&self) -> TrainingStats {
        let drift = self.detect_concept_drift();

        // Count feedback types
        let accepted_count = self
            .feedback
            .iter()
            .filter(|f| **f == UserFeedback::Accepted)
            .count();
        let rejected_count = self
            .feedback
            .iter()
            .filter(|f| **f == UserFeedback::Rejected)
            .count();
        let alternative_count = self
            .feedback
            .iter()
            .filter(|f| **f == UserFeedback::Alternative)
            .count();

        TrainingStats {
            total_samples: self.samples.len(),
            samples_since_training: drift.samples_since_training,
            accepted_count,
            rejected_count,
            alternative_count,
            staleness_score: drift.staleness_score,
            drift_detected: drift.drift_detected,
            online_learning_enabled: self.online_learning_enabled,
        }
    }

    /// Auto-retrain and update BrickTuner if conditions are met
    pub fn auto_retrain(&mut self, tuner: &mut BrickTuner) -> bool {
        if !self.should_retrain() {
            return false;
        }

        // Weight samples by feedback
        let training_data = self.prepare_weighted_training_data();

        if training_data.len() < 10 {
            return false;
        }

        // Train and update
        match tuner.train(&training_data) {
            Ok(()) => {
                self.mark_trained();
                true
            }
            Err(_) => false,
        }
    }

    /// Prepare training data with feedback weighting
    fn prepare_weighted_training_data(&self) -> Vec<(TunerFeatures, f32)> {
        self.samples
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let feedback = self.get_feedback(i);

                // Skip rejected samples (they had bad throughput measurements)
                if feedback == UserFeedback::Rejected {
                    return None;
                }

                // Weight accepted samples higher (duplicate them)
                let weight = match feedback {
                    UserFeedback::Accepted => 2,
                    UserFeedback::Alternative => 1, // Still use, but normal weight
                    _ => 1,
                };

                Some((0..weight).map(|_| (s.features.clone(), s.throughput_tps)))
            })
            .flatten()
            .collect()
    }
}
