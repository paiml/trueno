//! BrickTuner - ML-based ComputeBrick Tuner Ensemble
//!
//! Combines throughput regression, kernel classification, and bottleneck analysis.

use serde::{Deserialize, Serialize};

use super::error::TunerError;
use super::features::TunerFeatures;
use super::helpers::{chrono_lite_now, crc32_update, pad_right};
use super::models::{
    BottleneckClassifier, BottleneckPrediction, KernelClassifier, KernelRecommendation,
    ThroughputPrediction, ThroughputRegressor,
};
use super::types::{BottleneckClass, KernelType};

// ============================================================================
// TunerRecommendation
// ============================================================================

/// Combined tuner recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerRecommendation {
    /// Throughput prediction
    pub throughput: ThroughputPrediction,
    /// Kernel recommendation
    pub kernel: KernelRecommendation,
    /// Bottleneck analysis
    pub bottleneck: BottleneckPrediction,
    /// Model version
    pub model_version: String,
    /// Overall confidence
    pub confidence_overall: f32,
    /// Suggested experiments to try
    pub suggested_experiments: Vec<ExperimentSuggestion>,
}

/// Suggested experiment to improve performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentSuggestion {
    /// Increase batch size
    IncreaseBatchSize { from: u32, to: u32 },
    /// Enable CUDA graphs
    EnableCudaGraphs,
    /// Try a specific kernel
    TryKernel { kernel: KernelType },
    /// Reduce sequence length
    ReduceSequenceLength { factor: f32 },
    /// Enable multi-KV cache
    EnableMultiKvCache { count: u32 },
}

impl std::fmt::Display for ExperimentSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExperimentSuggestion::IncreaseBatchSize { from, to } => {
                write!(f, "Increase batch size: M={} → M={}", from, to)
            }
            ExperimentSuggestion::EnableCudaGraphs => {
                write!(f, "Enable CUDA graphs for kernel launch amortization")
            }
            ExperimentSuggestion::TryKernel { kernel } => {
                write!(f, "Try kernel: {:?}", kernel)
            }
            ExperimentSuggestion::ReduceSequenceLength { factor } => {
                write!(
                    f,
                    "Reduce sequence length by {:.0}%",
                    (1.0 - factor) * 100.0
                )
            }
            ExperimentSuggestion::EnableMultiKvCache { count } => {
                write!(
                    f,
                    "Enable {} separate KV caches for batched attention",
                    count
                )
            }
        }
    }
}

// ============================================================================
// BrickTuner
// ============================================================================

/// ML-based ComputeBrick tuner ensemble.
///
/// Combines three models for comprehensive recommendations:
/// - ThroughputRegressor: Predicts tok/s
/// - KernelClassifier: Selects best kernel
/// - BottleneckClassifier: Identifies performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickTuner {
    /// Throughput regression model
    pub(crate) throughput: ThroughputRegressor,
    /// Kernel classification model
    pub(crate) kernel: KernelClassifier,
    /// Bottleneck classification model
    pub(crate) bottleneck: BottleneckClassifier,
    /// Model version
    pub(crate) version: String,
    /// Training timestamp
    pub(crate) trained_at: String,
    /// Number of training samples
    pub(crate) sample_count: usize,
}

impl Default for BrickTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickTuner {
    /// Model version
    pub const VERSION: &'static str = "1.0.0";

    /// APR format magic bytes (APR1 = uncompressed)
    const APR_MAGIC: [u8; 4] = [b'A', b'P', b'R', b'1'];

    /// Create a new tuner with default models
    pub fn new() -> Self {
        Self {
            throughput: ThroughputRegressor::new(),
            kernel: KernelClassifier::new(),
            bottleneck: BottleneckClassifier::new(),
            version: Self::VERSION.to_string(),
            trained_at: chrono_lite_now(),
            sample_count: 0,
        }
    }

    /// Get the model version string
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get the throughput regressor's MAPE (Mean Absolute Percentage Error)
    pub fn throughput_mape(&self) -> f32 {
        self.throughput.mape
    }

    /// Get the number of training samples used
    pub fn throughput_sample_count(&self) -> usize {
        self.throughput.sample_count
    }

    /// Get comprehensive tuning recommendation
    pub fn recommend(&self, features: &TunerFeatures) -> TunerRecommendation {
        let throughput = self.throughput.predict(features);
        let kernel = self.kernel.predict(features);
        let bottleneck = self.bottleneck.predict(features);

        // Calculate overall confidence
        let confidence_overall =
            (throughput.confidence + kernel.confidence + bottleneck.confidence) / 3.0;

        // Generate experiment suggestions based on bottleneck
        let suggested_experiments = self.suggest_experiments(features, &bottleneck);

        TunerRecommendation {
            throughput,
            kernel,
            bottleneck,
            model_version: self.version.clone(),
            confidence_overall,
            suggested_experiments,
        }
    }

    /// Suggest experiments based on current bottleneck
    pub fn suggest_experiments(
        &self,
        features: &TunerFeatures,
        bottleneck: &BottleneckPrediction,
    ) -> Vec<ExperimentSuggestion> {
        let mut suggestions = Vec::new();
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;

        match bottleneck.class {
            BottleneckClass::MemoryBound => {
                if batch_size < 8 {
                    suggestions.push(ExperimentSuggestion::IncreaseBatchSize {
                        from: batch_size,
                        to: (batch_size * 2).min(8),
                    });
                }
                suggestions.push(ExperimentSuggestion::TryKernel {
                    kernel: KernelType::BatchedQ4K,
                });
                if batch_size > 1 {
                    suggestions.push(ExperimentSuggestion::EnableMultiKvCache { count: batch_size });
                }
            }
            BottleneckClass::LaunchBound => {
                if features.cuda_graphs < 0.5 {
                    suggestions.push(ExperimentSuggestion::EnableCudaGraphs);
                }
                suggestions.push(ExperimentSuggestion::TryKernel {
                    kernel: KernelType::FusedRmsNormQ4K,
                });
            }
            BottleneckClass::AttentionBound => {
                suggestions.push(ExperimentSuggestion::TryKernel {
                    kernel: KernelType::BatchedAttention,
                });
                suggestions.push(ExperimentSuggestion::ReduceSequenceLength { factor: 0.5 });
            }
            _ => {
                // Default suggestions
                if batch_size < 4 {
                    suggestions.push(ExperimentSuggestion::IncreaseBatchSize {
                        from: batch_size,
                        to: 4,
                    });
                }
            }
        }

        suggestions
    }

    /// Train all models on labeled data
    pub fn train(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        self.throughput.train(data)?;
        self.sample_count = data.len();
        self.trained_at = chrono_lite_now();
        Ok(())
    }

    /// Print recommendations to console (TUI-friendly)
    pub fn print_recommendation(&self, rec: &TunerRecommendation) {
        println!("╭─────────────────────────────────────────────────────────────╮");
        println!(
            "│           BrickTuner Recommendations v{}                 │",
            self.version
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│ Predicted throughput: {:>7.1} tok/s ({:>4.0}% confidence)     │",
            rec.throughput.predicted_tps,
            rec.throughput.confidence * 100.0
        );
        println!(
            "│ Recommended kernel:   {:>15?} ({:>4.0}% conf)       │",
            rec.kernel.top_kernel,
            rec.kernel.confidence * 100.0
        );
        println!(
            "│ Bottleneck class:     {:>15} ({:>4.0}% conf)       │",
            rec.bottleneck.class,
            rec.bottleneck.confidence * 100.0
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│ Explanation: {}│",
            pad_right(&rec.bottleneck.explanation, 47)
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Suggested experiments:                                      │");
        for (i, exp) in rec.suggested_experiments.iter().take(3).enumerate() {
            println!("│  {}. {}│", i + 1, pad_right(&exp.to_string(), 56));
        }
        println!("╰─────────────────────────────────────────────────────────────╯");
    }

    // ========================================================================
    // T-TUNER-006: cbtop TUI Integration (GitHub #83)
    // ========================================================================

    /// Render recommendation as TUI panel lines (for cbtop integration)
    ///
    /// Returns a vector of strings that can be rendered in a TUI widget.
    /// Each line is formatted for fixed-width display (width=61 chars).
    pub fn render_panel(&self, rec: &TunerRecommendation) -> Vec<String> {
        let mut lines = Vec::with_capacity(12);

        lines.push(format!(
            "│           BrickTuner Recommendations v{}                 │",
            self.version
        ));
        lines.push(
            "├─────────────────────────────────────────────────────────────┤".to_string(),
        );
        lines.push(format!(
            "│ Predicted throughput: {:>7.1} tok/s ({:>4.0}% confidence)     │",
            rec.throughput.predicted_tps,
            rec.throughput.confidence * 100.0
        ));
        lines.push(format!(
            "│ Recommended kernel:   {:>15?} ({:>4.0}% conf)       │",
            rec.kernel.top_kernel,
            rec.kernel.confidence * 100.0
        ));
        lines.push(format!(
            "│ Bottleneck class:     {:>15} ({:>4.0}% conf)       │",
            rec.bottleneck.class,
            rec.bottleneck.confidence * 100.0
        ));
        lines.push(
            "├─────────────────────────────────────────────────────────────┤".to_string(),
        );
        lines.push(format!(
            "│ Explanation: {}│",
            pad_right(&rec.bottleneck.explanation, 47)
        ));
        lines.push(
            "├─────────────────────────────────────────────────────────────┤".to_string(),
        );
        lines.push(
            "│ Suggested experiments:                                      │".to_string(),
        );

        for (i, exp) in rec.suggested_experiments.iter().take(3).enumerate() {
            lines.push(format!(
                "│  {}. {}│",
                i + 1,
                pad_right(&exp.to_string(), 56)
            ));
        }

        // Pad if fewer than 3 suggestions
        for _ in rec.suggested_experiments.len()..3 {
            lines.push(
                "│                                                             │".to_string(),
            );
        }

        lines.push(
            "├─────────────────────────────────────────────────────────────┤".to_string(),
        );
        lines.push(
            "│ [Press 'a' to apply] [Press 't' to toggle] [Press 'r' to run]│".to_string(),
        );

        lines
    }

    /// Render compact recommendation (single line for status bar)
    pub fn render_compact(&self, rec: &TunerRecommendation) -> String {
        format!(
            "Tuner: {:.0} tok/s | {:?} | {} ({:.0}%)",
            rec.throughput.predicted_tps,
            rec.kernel.top_kernel,
            rec.bottleneck.class,
            rec.confidence_overall * 100.0
        )
    }

    /// Render prediction vs actual comparison
    pub fn render_comparison(&self, rec: &TunerRecommendation, actual_tps: f32) -> Vec<String> {
        let error_pct = if actual_tps > 0.0 {
            ((rec.throughput.predicted_tps - actual_tps) / actual_tps * 100.0).abs()
        } else {
            0.0
        };

        let accuracy_indicator = if error_pct < 5.0 {
            "🎯 Excellent"
        } else if error_pct < 10.0 {
            "✓ Good"
        } else if error_pct < 20.0 {
            "△ Fair"
        } else {
            "✗ Poor"
        };

        vec![
            format!(
                "│ Predicted: {:>7.1} tok/s  Actual: {:>7.1} tok/s           │",
                rec.throughput.predicted_tps, actual_tps
            ),
            format!(
                "│ Error: {:>5.1}%  Accuracy: {:>12}                       │",
                error_pct, accuracy_indicator
            ),
        ]
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, TunerError> {
        serde_json::to_string_pretty(self).map_err(|e| TunerError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, TunerError> {
        serde_json::from_str(json).map_err(|e| TunerError::Serialization(e.to_string()))
    }

    // =========================================================================
    // APR Persistence (SOVEREIGN STACK - GH#81)
    // =========================================================================

    /// Get the default cache path for tuner models.
    ///
    /// Returns `~/.cache/trueno/tuner_model_v{VERSION}.apr`
    #[cfg(feature = "hardware-detect")]
    pub fn cache_path() -> std::path::PathBuf {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("trueno");

        // Create directory if it doesn't exist
        let _ = std::fs::create_dir_all(&cache_dir);

        cache_dir.join(format!("tuner_model_v{}.apr", Self::VERSION))
    }

    /// Load tuner from cache or create new with defaults.
    ///
    /// This is the recommended way to create a BrickTuner for production use.
    /// It will:
    /// 1. Check for cached model at `~/.cache/trueno/tuner_model_v{VERSION}.apr`
    /// 2. Load if exists and version matches
    /// 3. Create new with defaults if not found or version mismatch
    #[cfg(feature = "hardware-detect")]
    pub fn load_or_default() -> Self {
        let path = Self::cache_path();

        if path.exists() {
            match Self::load_apr(&path) {
                Ok(tuner) => {
                    // Version check
                    if tuner.version == Self::VERSION {
                        return tuner;
                    }
                    // Version mismatch - create new
                }
                Err(_) => {
                    // Load failed - create new
                }
            }
        }

        Self::new()
    }

    /// Save tuner model to .apr file.
    ///
    /// APR1 format (uncompressed):
    /// - 4-byte magic: "APR1"
    /// - 4-byte metadata_len: u32 LE
    /// - JSON metadata
    /// - 4-byte CRC32: checksum
    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TunerError> {
        use std::io::Write;

        let json = self.to_json()?;
        let json_bytes = json.as_bytes();

        let mut file =
            std::fs::File::create(path).map_err(|e| TunerError::Io(e.to_string()))?;

        // Write magic
        file.write_all(&Self::APR_MAGIC)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Write metadata length
        let len = json_bytes.len() as u32;
        file.write_all(&len.to_le_bytes())
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Write JSON metadata
        file.write_all(json_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Calculate and write CRC32
        let mut crc = 0u32;
        crc = crc32_update(crc, &Self::APR_MAGIC);
        crc = crc32_update(crc, &len.to_le_bytes());
        crc = crc32_update(crc, json_bytes);
        file.write_all(&crc.to_le_bytes())
            .map_err(|e| TunerError::Io(e.to_string()))?;

        Ok(())
    }

    /// Load tuner model from .apr file.
    pub fn load_apr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TunerError> {
        use std::io::Read;

        let mut file =
            std::fs::File::open(path).map_err(|e| TunerError::Io(e.to_string()))?;

        // Read and verify magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        if magic != Self::APR_MAGIC {
            return Err(TunerError::InvalidFormat(
                "Invalid APR magic bytes".to_string(),
            ));
        }

        // Read metadata length
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read JSON metadata
        let mut json_bytes = vec![0u8; len];
        file.read_exact(&mut json_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Read and verify CRC32
        let mut crc_bytes = [0u8; 4];
        file.read_exact(&mut crc_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;
        let stored_crc = u32::from_le_bytes(crc_bytes);

        let mut computed_crc = 0u32;
        computed_crc = crc32_update(computed_crc, &Self::APR_MAGIC);
        computed_crc = crc32_update(computed_crc, &len_bytes);
        computed_crc = crc32_update(computed_crc, &json_bytes);

        if stored_crc != computed_crc {
            return Err(TunerError::InvalidFormat(
                "CRC32 checksum mismatch".to_string(),
            ));
        }

        // Parse JSON
        let json = String::from_utf8(json_bytes)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;

        Self::from_json(&json)
    }

    /// Save to default cache path.
    #[cfg(feature = "hardware-detect")]
    pub fn save_to_cache(&self) -> Result<(), TunerError> {
        self.save_apr(Self::cache_path())
    }
}
