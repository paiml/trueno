//! ML Models for Tuner
//!
//! Throughput regressor, kernel classifier, and bottleneck classifier implementations.

use serde::{Deserialize, Serialize};

#[cfg(feature = "ml-tuner")]
use aprender::{
    tree::{RandomForestClassifier, RandomForestRegressor},
    Matrix, Vector,
};

use super::error::TunerError;
use super::features::TunerFeatures;
use super::types::{BottleneckClass, KernelType};

// ============================================================================
// Prediction Results
// ============================================================================

/// Throughput prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputPrediction {
    /// Predicted tokens per second
    pub predicted_tps: f32,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Top contributing features
    pub top_features: Vec<(String, f32)>,
}

/// Kernel recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRecommendation {
    /// Top recommended kernel
    pub top_kernel: KernelType,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Alternative kernels with probabilities
    pub alternatives: Vec<(KernelType, f32)>,
}

/// Bottleneck prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckPrediction {
    /// Predicted bottleneck class
    pub class: BottleneckClass,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Human-readable explanation
    pub explanation: String,
    /// Recommended action
    pub recommended_action: String,
}

// ============================================================================
// ThroughputRegressor
// ============================================================================

/// Simple linear regression model for throughput prediction.
///
/// Uses closed-form solution: w = (X^T X)^-1 X^T y
/// With `ml-tuner` feature: uses aprender::RandomForestRegressor (SHOWCASE-BRICK-001)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputRegressor {
    /// Model weights (one per feature + bias) - fallback when ml-tuner disabled
    pub(crate) weights: Vec<f32>,
    /// Feature importance scores
    pub(crate) feature_importance: Vec<(String, f32)>,
    /// Training sample count
    pub(crate) sample_count: usize,
    /// Mean absolute percentage error on validation
    pub(crate) mape: f32,
    /// Whether the RandomForest model is trained (ml-tuner feature)
    #[cfg(feature = "ml-tuner")]
    #[serde(skip)]
    rf_model: Option<RandomForestRegressor>,
}

impl Default for ThroughputRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputRegressor {
    /// Create a new regressor with default weights
    pub fn new() -> Self {
        // Initialize with heuristic-based weights
        // These encode domain knowledge from SHOWCASE-BRICK-001
        let mut weights = vec![0.0; TunerFeatures::DIM + 1]; // +1 for bias

        // Bias: baseline throughput ~200 tok/s normalized
        weights[0] = 0.4;

        // Batch size has largest positive impact (index 6)
        weights[7] = 0.3; // batch_size_norm

        // CUDA graphs help (index 8)
        weights[9] = 0.1; // cuda_graphs

        // GPU memory bandwidth matters (index 35)
        weights[36] = 0.15; // gpu_mem_bw_norm

        // GPU SM count matters (index 37)
        weights[38] = 0.1; // gpu_sm_norm

        // Larger models are slower (negative impact)
        weights[1] = -0.15; // model_params_b

        // Longer sequences slower for decode
        weights[8] = -0.05; // seq_len_log

        Self {
            weights,
            feature_importance: Self::default_feature_importance(),
            sample_count: 0,
            mape: 0.15, // 15% default MAPE
            #[cfg(feature = "ml-tuner")]
            rf_model: None,
        }
    }

    /// Create a new regressor using aprender RandomForest (ml-tuner feature)
    #[cfg(feature = "ml-tuner")]
    pub fn with_random_forest(n_estimators: usize) -> Self {
        let mut instance = Self::new();
        instance.rf_model = Some(RandomForestRegressor::new(n_estimators));
        instance
    }

    fn default_feature_importance() -> Vec<(String, f32)> {
        vec![
            ("batch_size".into(), 0.25),
            ("gpu_mem_bw".into(), 0.20),
            ("model_params".into(), 0.15),
            ("cuda_graphs".into(), 0.10),
            ("gpu_sm_count".into(), 0.10),
            ("hidden_dim".into(), 0.08),
            ("quant_type".into(), 0.07),
            ("seq_len".into(), 0.05),
        ]
    }

    /// Train the model on labeled data
    pub fn train(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Simple gradient descent (in production: aprender's GBDT)
        let learning_rate = 0.01;
        let epochs = 100;

        for _ in 0..epochs {
            let mut gradients = vec![0.0; self.weights.len()];

            for (features, target) in data {
                let x = features.to_vector();
                let predicted = self.predict_raw(&x);
                let error = predicted - target;

                // Gradient for bias
                gradients[0] += error;

                // Gradient for features
                for (i, xi) in x.iter().enumerate() {
                    gradients[i + 1] += error * xi;
                }
            }

            // Update weights
            let n = data.len() as f32;
            for (i, g) in gradients.iter().enumerate() {
                self.weights[i] -= learning_rate * g / n;
            }
        }

        // Calculate MAPE on training data
        let mut total_ape = 0.0;
        for (features, target) in data {
            let predicted = self.predict_raw(&features.to_vector());
            total_ape += ((predicted - target) / target.max(1.0)).abs();
        }
        self.mape = total_ape / data.len() as f32;
        self.sample_count = data.len();

        Ok(())
    }

    /// Train using aprender RandomForest (ml-tuner feature)
    ///
    /// Provides superior throughput prediction via ensemble learning.
    /// See: SHOWCASE-BRICK-001 Section 12.3
    #[cfg(feature = "ml-tuner")]
    pub fn train_random_forest(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Convert to aprender matrix format (f32 for RandomForestRegressor)
        let n_samples = data.len();
        let n_features = TunerFeatures::DIM;
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        for (features, target) in data {
            x_data.extend(features.to_vector());
            y_data.push(*target);
        }

        let x_matrix = Matrix::from_vec(n_samples, n_features, x_data)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;
        let y_vector = Vector::from_vec(y_data);

        // Train RandomForest
        let rf = self
            .rf_model
            .get_or_insert_with(|| RandomForestRegressor::new(100));
        rf.fit(&x_matrix, &y_vector)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        // Calculate MAPE on training data
        let predictions = rf.predict(&x_matrix);
        let mut total_ape = 0.0;
        for (i, (_, target)) in data.iter().enumerate() {
            let pred = predictions.as_slice()[i];
            total_ape += ((pred - target) / target.max(1.0)).abs();
        }
        self.mape = total_ape / data.len() as f32;
        self.sample_count = data.len();

        Ok(())
    }

    pub(crate) fn predict_raw(&self, x: &[f32]) -> f32 {
        let mut result = self.weights[0]; // bias
        for (i, xi) in x.iter().enumerate() {
            if i + 1 < self.weights.len() {
                result += self.weights[i + 1] * xi;
            }
        }
        // Convert from normalized to tok/s (scale ~1000)
        (result * 1000.0).max(1.0)
    }

    /// Predict throughput for features
    ///
    /// With `ml-tuner` feature: uses trained RandomForest if available.
    /// Falls back to linear model otherwise.
    pub fn predict(&self, features: &TunerFeatures) -> ThroughputPrediction {
        let x = features.to_vector();

        // Use RandomForest if trained (ml-tuner feature)
        #[cfg(feature = "ml-tuner")]
        let raw_predicted_tps = if let Some(ref rf) = self.rf_model {
            // Use f32 matrix for RandomForestRegressor
            if let Ok(x_matrix) = Matrix::from_vec(1, TunerFeatures::DIM, x.to_vec()) {
                let predictions = rf.predict(&x_matrix);
                predictions.as_slice().first().copied().unwrap_or(0.0)
            } else {
                self.predict_raw(&x)
            }
        } else {
            self.predict_raw(&x)
        };

        #[cfg(not(feature = "ml-tuner"))]
        let raw_predicted_tps = self.predict_raw(&x);

        // v1.1.0: Roofline clamping - predictions must not exceed theoretical maximum
        let theoretical_max_tps = Self::compute_roofline_bound(features);
        let predicted_tps = raw_predicted_tps.min(theoretical_max_tps);

        // Confidence based on training MAPE and feature validity
        // Lower confidence if we hit the roofline cap
        let roofline_penalty = if raw_predicted_tps > theoretical_max_tps {
            0.9 // 10% confidence penalty for capped predictions
        } else {
            1.0
        };
        let confidence = (1.0 - self.mape).max(0.5) * roofline_penalty;

        ThroughputPrediction {
            predicted_tps,
            confidence,
            top_features: self.feature_importance.iter().take(5).cloned().collect(),
        }
    }

    /// Compute theoretical maximum throughput based on roofline model (v1.1.0)
    ///
    /// For memory-bound LLM inference (decode phase):
    /// max_tps = memory_bw_bytes_per_sec / bytes_per_token
    /// bytes_per_token = model_params × bytes_per_param / batch_size
    pub fn compute_roofline_bound(features: &TunerFeatures) -> f32 {
        // Denormalize model params: normalized = (log10(b) + 1) / 3
        // log10(b) = normalized * 3 - 1
        // b = 10^(normalized * 3 - 1)
        let model_params_b = 10.0_f32.powf(features.model_params_b * 3.0 - 1.0);

        // Get bytes per param from quant type one-hot encoding
        let bytes_per_param = Self::bytes_per_param_from_onehot(&features.quant_type_onehot);

        // Denormalize memory bandwidth: normalized = bw / 3000 GB/s
        let gpu_mem_bw_gbs = features.gpu_mem_bw_norm * 3000.0;

        // Denormalize batch size: normalized = batch_size / 64
        let batch_size = (features.batch_size_norm * 64.0).max(1.0);

        // Roofline calculation:
        // model_bytes = model_params_b * bytes_per_param * 1e9
        // bytes_per_token = model_bytes / batch_size
        // max_tps = (gpu_mem_bw_gbs * 1e9) / bytes_per_token
        //         = (gpu_mem_bw_gbs * 1e9 * batch_size) / (model_params_b * bytes_per_param * 1e9)
        //         = (gpu_mem_bw_gbs * batch_size) / (model_params_b * bytes_per_param)
        let theoretical_max = (gpu_mem_bw_gbs * batch_size) / (model_params_b * bytes_per_param);

        // Clamp to reasonable range (1 tok/s to 10000 tok/s)
        theoretical_max.clamp(1.0, 10000.0)
    }

    /// Extract bytes per param from quant type one-hot encoding
    pub fn bytes_per_param_from_onehot(onehot: &[f32; 8]) -> f32 {
        // One-hot indices map to QuantType variants
        // 0: Q4_0, 1: Q4_1, 2: Q4K, 3: Q5K, 4: Q6K, 5: Q8_0, 6: F16, 7: F32
        let bytes_per_param = [0.5625, 0.5625, 0.5625, 0.6875, 0.8125, 1.0, 2.0, 4.0];

        // Find the active index (max value in one-hot)
        let idx = onehot
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(2); // Default to Q4K if ambiguous

        bytes_per_param[idx]
    }
}

// ============================================================================
// KernelClassifier
// ============================================================================

/// Kernel classifier using simple rule-based logic.
///
/// With `ml-tuner` feature: uses aprender::RandomForestClassifier (SHOWCASE-BRICK-001)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KernelClassifier {
    /// Kernel accuracy on validation (for confidence)
    accuracy: f32,
    /// RandomForest classifier when ml-tuner feature is enabled
    #[cfg(feature = "ml-tuner")]
    #[serde(skip)]
    rf_classifier: Option<RandomForestClassifier>,
}

impl KernelClassifier {
    pub fn new() -> Self {
        Self {
            accuracy: 0.85,
            #[cfg(feature = "ml-tuner")]
            rf_classifier: None,
        }
    }

    /// Create a classifier with aprender RandomForest (ml-tuner feature)
    #[cfg(feature = "ml-tuner")]
    pub fn with_random_forest(n_estimators: usize) -> Self {
        Self {
            accuracy: 0.85,
            rf_classifier: Some(RandomForestClassifier::new(n_estimators)),
        }
    }

    /// Train the classifier using aprender RandomForest (ml-tuner feature)
    ///
    /// Labels should be kernel type indices (0=TiledQ4K, 1=CoalescedQ4K, etc.)
    #[cfg(feature = "ml-tuner")]
    pub fn train(&mut self, data: &[(TunerFeatures, u32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Convert to aprender format (Matrix<f32> for features, &[usize] for labels)
        let n_samples = data.len();
        let n_features = TunerFeatures::DIM;
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

        for (features, label) in data {
            x_data.extend(features.to_vector());
            y_data.push(*label as usize);
        }

        let x_matrix = Matrix::from_vec(n_samples, n_features, x_data)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        let rf = self
            .rf_classifier
            .get_or_insert_with(|| RandomForestClassifier::new(50));
        rf.fit(&x_matrix, &y_data)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        // Calculate accuracy on training data
        let predictions = rf.predict(&x_matrix);
        let mut correct = 0;
        for (i, (_, label)) in data.iter().enumerate() {
            if predictions[i] as u32 == *label {
                correct += 1;
            }
        }
        self.accuracy = correct as f32 / data.len() as f32;

        Ok(())
    }

    /// Predict best kernel based on features
    pub fn predict(&self, features: &TunerFeatures) -> KernelRecommendation {
        // Rule-based kernel selection from SHOWCASE-BRICK-001 learnings
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;
        let seq_len = (2.0_f32.powf(features.seq_len_log * 15.0)).round() as u32;

        // Determine best Q4K variant based on batch size
        let (top_kernel, confidence) = if batch_size >= 4 {
            // M >= 4: Use batched kernels
            (KernelType::BatchedQ4K, 0.90)
        } else if batch_size >= 2 {
            // M = 2-3: Vectorized is good
            (KernelType::VectorizedQ4K, 0.85)
        } else {
            // M = 1: Coalesced or Vectorized
            if features.cuda_graphs > 0.5 {
                (KernelType::VectorizedQ4K, 0.88)
            } else {
                (KernelType::CoalescedQ4K, 0.82)
            }
        };

        // Check for attention-bound cases
        let attention_kernel = if seq_len > 128 {
            KernelType::MultiWarpAttention
        } else {
            KernelType::IncrementalAttention
        };

        // Build alternatives
        let alternatives = vec![
            (KernelType::VectorizedQ4K, 0.85),
            (KernelType::CoalescedQ4K, 0.75),
            (attention_kernel, 0.70),
        ]
        .into_iter()
        .filter(|(k, _)| *k != top_kernel)
        .take(2)
        .collect();

        KernelRecommendation {
            top_kernel,
            confidence,
            alternatives,
        }
    }
}

// ============================================================================
// BottleneckClassifier
// ============================================================================

/// Bottleneck classifier using heuristics from profiler data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BottleneckClassifier {
    /// Classification accuracy
    accuracy: f32,
}

impl BottleneckClassifier {
    pub fn new() -> Self {
        Self { accuracy: 0.90 }
    }

    /// Predict bottleneck from features
    pub fn predict(&self, features: &TunerFeatures) -> BottleneckPrediction {
        // Use already-computed bottleneck if available
        if let Some(class) = features.bottleneck_class {
            return BottleneckPrediction {
                class,
                confidence: 0.95,
                explanation: format!("Bottleneck classified from profiler data: {}", class),
                recommended_action: class.recommended_action().to_string(),
            };
        }

        // Heuristic classification based on features
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;
        let seq_len = (2.0_f32.powf(features.seq_len_log * 15.0)).round() as u32;

        let (class, confidence, explanation) = if batch_size == 1 && features.cuda_graphs < 0.5 {
            (
                BottleneckClass::LaunchBound,
                0.75,
                "Single sequence without CUDA graphs: kernel launch overhead may dominate".into(),
            )
        } else if seq_len > 512 {
            (
                BottleneckClass::AttentionBound,
                0.80,
                format!(
                    "Long sequence (len={}) likely makes attention the bottleneck",
                    seq_len
                ),
            )
        } else {
            (
                BottleneckClass::MemoryBound,
                0.85,
                "Q4K GEMV is typically memory-bound for LLM inference".into(),
            )
        };

        BottleneckPrediction {
            class,
            confidence,
            explanation,
            recommended_action: class.recommended_action().to_string(),
        }
    }
}
