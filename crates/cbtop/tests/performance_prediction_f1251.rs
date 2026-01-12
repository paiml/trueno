//! Falsification Tests for PMAT-033: Performance Prediction Model
//!
//! F1251-F1260: Performance prediction falsification tests

use cbtop::{
    PerformancePredictor, DataPoint, ModelType, FittedModel, Prediction,
    MIN_SAMPLES_FOR_FIT,
};

// =============================================================================
// F1251: Data Point Tests
// =============================================================================

/// F1251.1: Data point stores all fields correctly
#[test]
fn f1251_data_point_creation() {
    let point = DataPoint::new(1024, 100.0, 10.5);

    assert_eq!(point.size, 1024);
    assert_eq!(point.performance, 100.0);
    assert_eq!(point.latency_us, 10.5);
}

/// F1251.2: Data point with edge values
#[test]
fn f1251_data_point_edge_values() {
    let point = DataPoint::new(0, 0.0, 0.0);
    assert_eq!(point.size, 0);

    let large = DataPoint::new(usize::MAX, f64::MAX, f64::MAX);
    assert_eq!(large.size, usize::MAX);
}

// =============================================================================
// F1252: Model Type Tests
// =============================================================================

/// F1252.1: Model type names are correct
#[test]
fn f1252_model_type_names() {
    assert_eq!(ModelType::Linear.name(), "linear");
    assert_eq!(ModelType::Polynomial.name(), "polynomial");
    assert_eq!(ModelType::ExponentialDecay.name(), "exponential_decay");
    assert_eq!(ModelType::Logarithmic.name(), "logarithmic");
    assert_eq!(ModelType::Roofline.name(), "roofline");
}

/// F1252.2: Model types are comparable
#[test]
fn f1252_model_type_equality() {
    assert_eq!(ModelType::Linear, ModelType::Linear);
    assert_ne!(ModelType::Linear, ModelType::Polynomial);
}

// =============================================================================
// F1253: Predictor Creation Tests
// =============================================================================

/// F1253.1: New predictor is empty
#[test]
fn f1253_predictor_empty() {
    let predictor = PerformancePredictor::new();
    assert_eq!(predictor.point_count(), 0);
    assert!(!predictor.has_sufficient_data());
}

/// F1253.2: Predictor with custom confidence level
#[test]
fn f1253_predictor_confidence() {
    let predictor = PerformancePredictor::new().with_confidence(0.90);
    assert_eq!(predictor.point_count(), 0);
}

// =============================================================================
// F1254: Data Addition Tests
// =============================================================================

/// F1254.1: Add points tracks count
#[test]
fn f1254_add_points() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1000, 10.0, 1.0);
    assert_eq!(predictor.point_count(), 1);

    predictor.add(2000, 20.0, 2.0);
    assert_eq!(predictor.point_count(), 2);
}

/// F1254.2: Add point via struct
#[test]
fn f1254_add_point_struct() {
    let mut predictor = PerformancePredictor::new();

    let point = DataPoint::new(1024, 50.0, 5.0);
    predictor.add_point(point);

    assert_eq!(predictor.point_count(), 1);
}

/// F1254.3: Sufficient data threshold
#[test]
fn f1254_sufficient_data() {
    let mut predictor = PerformancePredictor::new();

    for i in 0..MIN_SAMPLES_FOR_FIT - 1 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }
    assert!(!predictor.has_sufficient_data());

    predictor.add(10000, 100.0, 10.0);
    assert!(predictor.has_sufficient_data());
}

// =============================================================================
// F1255: Linear Fit Tests
// =============================================================================

/// F1255.1: Linear fit on linear data has high R²
#[test]
fn f1255_linear_fit_perfect() {
    let mut predictor = PerformancePredictor::new();

    // Perfect linear: y = 0.01*x
    for i in 1..=10 {
        let size = i * 1000;
        let perf = i as f64 * 10.0;
        predictor.add(size, perf, 1.0);
    }

    let model = predictor.fit_linear().unwrap();
    assert!(model.r_squared > 0.99, "Expected R² > 0.99, got {}", model.r_squared);
    assert_eq!(model.model_type, ModelType::Linear);
}

/// F1255.2: Linear fit requires minimum samples
#[test]
fn f1255_linear_fit_insufficient() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1000, 10.0, 1.0);
    predictor.add(2000, 20.0, 2.0);

    let result = predictor.fit_linear();
    assert!(result.is_none());
}

/// F1255.3: Linear model predicts correctly
#[test]
fn f1255_linear_predict() {
    let model = FittedModel {
        model_type: ModelType::Linear,
        coefficients: vec![0.01, 0.0], // y = 0.01*x
        r_squared: 1.0,
        rss: 0.0,
        sample_count: 10,
    };

    assert!((model.predict(1000) - 10.0).abs() < 0.001);
    assert!((model.predict(5000) - 50.0).abs() < 0.001);
}

// =============================================================================
// F1256: Polynomial Fit Tests
// =============================================================================

/// F1256.1: Polynomial fit works
#[test]
fn f1256_polynomial_fit() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        let size = i * 1000;
        let perf = i as f64 * 10.0; // Linear data
        predictor.add(size, perf, 1.0);
    }

    let model = predictor.fit_polynomial().unwrap();
    assert_eq!(model.model_type, ModelType::Polynomial);
    assert!(model.r_squared > 0.9);
}

/// F1256.2: Polynomial model has correct coefficients structure
#[test]
fn f1256_polynomial_coefficients() {
    let model = FittedModel {
        model_type: ModelType::Polynomial,
        coefficients: vec![0.001, 0.1, 0.0], // y = 0.001*x² + 0.1*x
        r_squared: 0.95,
        rss: 10.0,
        sample_count: 10,
    };

    let prediction = model.predict(100);
    // 0.001*10000 + 0.1*100 = 10 + 10 = 20
    assert!((prediction - 20.0).abs() < 0.001);
}

// =============================================================================
// F1257: Logarithmic Fit Tests
// =============================================================================

/// F1257.1: Logarithmic fit works
#[test]
fn f1257_logarithmic_fit() {
    let mut predictor = PerformancePredictor::new();

    // Logarithmic-ish data
    for i in 1..=10 {
        let size = i * 1000;
        let perf = (size as f64).ln() * 10.0;
        predictor.add(size, perf, 1.0);
    }

    let model = predictor.fit_logarithmic().unwrap();
    assert_eq!(model.model_type, ModelType::Logarithmic);
    assert!(model.r_squared > 0.9);
}

/// F1257.2: Logarithmic model handles positive values
#[test]
fn f1257_logarithmic_predict() {
    let model = FittedModel {
        model_type: ModelType::Logarithmic,
        coefficients: vec![10.0, 0.0], // y = 10*ln(x)
        r_squared: 1.0,
        rss: 0.0,
        sample_count: 10,
    };

    // ln(1000) ≈ 6.907, so y ≈ 69.07
    let prediction = model.predict(1000);
    assert!((prediction - 69.07).abs() < 0.1);
}

// =============================================================================
// F1258: Fit All and Model Selection Tests
// =============================================================================

/// F1258.1: Fit all returns best model type
#[test]
fn f1258_fit_all() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    let best = predictor.fit_all();
    assert!(best.is_some());
}

/// F1258.2: Best model is accessible
#[test]
fn f1258_best_model() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_all();
    let model = predictor.best_model();
    assert!(model.is_some());
}

/// F1258.3: Compare models returns ranked list
#[test]
fn f1258_compare_models() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_all();
    let comparisons = predictor.compare_models();

    assert!(!comparisons.is_empty());
    // First should have highest R²
    if comparisons.len() >= 2 {
        assert!(comparisons[0].1 >= comparisons[1].1);
    }
}

// =============================================================================
// F1259: Prediction Tests
// =============================================================================

/// F1259.1: Interpolation prediction
#[test]
fn f1259_interpolation() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    let pred = predictor.predict_at_size(5000).unwrap();

    assert!(!pred.is_extrapolation);
    assert!(pred.is_reasonable());
    assert!(pred.predicted > 0.0);
}

/// F1259.2: Extrapolation prediction
#[test]
fn f1259_extrapolation() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    // Beyond training range
    let pred = predictor.predict_at_size(20000).unwrap();

    assert!(pred.is_extrapolation);
    // Extrapolation should have wider bounds
    assert!(pred.range_width() > 0.0);
}

/// F1259.3: Prediction has confidence bounds
#[test]
fn f1259_confidence_bounds() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    let pred = predictor.predict_at_size(5000).unwrap();

    assert!(pred.lower_bound <= pred.predicted);
    assert!(pred.upper_bound >= pred.predicted);
    assert!(pred.confidence_level > 0.0);
}

// =============================================================================
// F1260: Utility Tests
// =============================================================================

/// F1260.1: Size range calculation
#[test]
fn f1260_size_range() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1000, 10.0, 1.0);
    predictor.add(5000, 50.0, 5.0);
    predictor.add(10000, 100.0, 10.0);

    let (min, max) = predictor.size_range().unwrap();
    assert_eq!(min, 1000);
    assert_eq!(max, 10000);
}

/// F1260.2: Empty predictor size range
#[test]
fn f1260_size_range_empty() {
    let predictor = PerformancePredictor::new();
    assert!(predictor.size_range().is_none());
}

/// F1260.3: Clear predictor
#[test]
fn f1260_clear() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    assert_eq!(predictor.point_count(), 10);

    predictor.clear();
    assert_eq!(predictor.point_count(), 0);
}

/// F1260.4: Export model
#[test]
fn f1260_export() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=5 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_linear();

    let export = predictor.export_model(ModelType::Linear).unwrap();
    assert!(export.contains("linear"));
    assert!(export.contains("r_squared"));
    assert!(export.contains("coefficients"));
}

/// F1260.5: Get specific model
#[test]
fn f1260_get_model() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=5 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_linear();

    let model = predictor.get_model(ModelType::Linear);
    assert!(model.is_some());

    let missing = predictor.get_model(ModelType::Roofline);
    assert!(missing.is_none());
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test fitted model good fit threshold
#[test]
fn test_good_fit() {
    let good = FittedModel {
        model_type: ModelType::Linear,
        coefficients: vec![1.0, 0.0],
        r_squared: 0.95,
        rss: 0.1,
        sample_count: 10,
    };
    assert!(good.is_good_fit());

    let bad = FittedModel {
        model_type: ModelType::Linear,
        coefficients: vec![1.0, 0.0],
        r_squared: 0.5,
        rss: 100.0,
        sample_count: 10,
    };
    assert!(!bad.is_good_fit());
}

/// Test prediction uncertainty percent
#[test]
fn test_uncertainty_percent() {
    let pred = Prediction {
        size: 1000,
        predicted: 100.0,
        lower_bound: 80.0,
        upper_bound: 120.0,
        confidence_level: 0.95,
        model_type: ModelType::Linear,
        is_extrapolation: false,
    };

    // range_width = 40, uncertainty = 40/100*100/2 = 20%
    assert!((pred.uncertainty_percent() - 20.0).abs() < 0.001);
}

/// Test min samples constant
#[test]
fn test_min_samples_constant() {
    assert_eq!(MIN_SAMPLES_FOR_FIT, 5);
}

/// Test roofline model prediction
#[test]
fn test_roofline_predict() {
    let model = FittedModel {
        model_type: ModelType::Roofline,
        coefficients: vec![100.0, 1000.0, 0.1], // peak=100, knee=1000, slope=0.1
        r_squared: 0.95,
        rss: 10.0,
        sample_count: 10,
    };

    // Small size: linear region
    let small = model.predict(500);
    assert!((small - 50.0).abs() < 0.001); // 0.1 * 500 = 50

    // Large size: capped at peak
    let large = model.predict(2000);
    assert!((large - 100.0).abs() < 0.001); // capped at peak=100
}

/// Test exponential decay model prediction
#[test]
fn test_exponential_decay_predict() {
    let model = FittedModel {
        model_type: ModelType::ExponentialDecay,
        coefficients: vec![10.0, 0.001, 5.0], // a=10, b=0.001, c=5
        r_squared: 0.95,
        rss: 10.0,
        sample_count: 10,
    };

    let prediction = model.predict(0);
    // 10 * exp(0) + 5 = 10 + 5 = 15
    assert!((prediction - 15.0).abs() < 0.001);
}

/// Test prediction not reasonable with zero
#[test]
fn test_prediction_not_reasonable() {
    let bad_pred = Prediction {
        size: 1000,
        predicted: 0.0,
        lower_bound: 0.0,
        upper_bound: 0.0,
        confidence_level: 0.95,
        model_type: ModelType::Linear,
        is_extrapolation: false,
    };

    assert!(!bad_pred.is_reasonable());
}
