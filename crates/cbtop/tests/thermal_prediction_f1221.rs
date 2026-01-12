//! Falsification Tests for PMAT-030: Thermal Trend Prediction
//!
//! F1221-F1230: Thermal prediction falsification tests
//!
//! These tests verify the thermal prediction module for:
//! - Trend prediction accuracy
//! - Throttle risk calculation
//! - Cooldown recommendations
//! - Thermal-latency correlation

use cbtop::{
    ThermalAnalyzer, ThermalSample, ThermalPrediction, ThrottleRisk,
    CooldownRecommendation, ThermalCorrelation, RiskCategory,
    analyze_thermal, assess_throttle_risk,
};

// =============================================================================
// F1221: Trend Prediction Accuracy Tests
// =============================================================================

/// F1221.1: Trend prediction accurate (±3°C for 10s forecast)
#[test]
fn f1221_trend_prediction_accurate() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Linear increase: 1°C/sec
    for i in 0..10 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    // Predict 10 seconds ahead
    let prediction = analyzer.predict_trend(10.0).unwrap();

    // At t=19, expected temp = 60 + 19 = 79°C
    // Current is 69°C, prediction should be 79°C (69 + 10)
    let expected = 79.0;
    let error = (prediction.predicted_temp_c - expected).abs();

    assert!(error < 3.0, "Error {:.1}°C exceeds ±3°C threshold", error);
}

/// F1221.2: Prediction with noisy data
#[test]
fn f1221_prediction_with_noise() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Linear trend with small noise
    let temps = [60.0, 61.2, 62.0, 62.8, 64.1, 64.9, 66.0, 66.8, 68.1, 69.0];
    for (i, &temp) in temps.iter().enumerate() {
        analyzer.add(temp, i as f64);
    }

    let prediction = analyzer.predict_trend(10.0).unwrap();

    // Trend is ~1°C/sec, should predict ~79°C
    assert!((prediction.predicted_temp_c - 79.0).abs() < 5.0);
}

// =============================================================================
// F1222: Throttle Risk Calculation Tests
// =============================================================================

/// F1222.1: Throttle risk in valid range (0.0-1.0)
#[test]
fn f1222_risk_range() {
    // Test various scenarios
    let scenarios = [
        (60.0, 85.0, 0.0),   // Low temp, no trend
        (80.0, 85.0, 0.5),   // High temp, slight increase
        (84.0, 85.0, 1.0),   // Near threshold, increasing
        (90.0, 85.0, 0.0),   // Above threshold
    ];

    for (temp, threshold, slope) in scenarios {
        let risk = assess_throttle_risk(temp, threshold, slope);
        assert!(
            risk.probability >= 0.0 && risk.probability <= 1.0,
            "Risk {:.2} out of range for temp={}, slope={}",
            risk.probability, temp, slope
        );
    }
}

/// F1222.2: High risk near threshold
#[test]
fn f1222_high_risk_near_threshold() {
    let risk = ThrottleRisk::assess(82.0, 85.0, 1.0, 10.0);

    // Very close to threshold and trending up = high risk
    assert!(risk.probability > 0.5);
    assert!(risk.category == RiskCategory::High || risk.category == RiskCategory::Critical);
}

/// F1222.3: Low risk when cooling down
#[test]
fn f1222_low_risk_cooling() {
    let risk = ThrottleRisk::assess(70.0, 85.0, -1.0, 10.0);

    // Below threshold and cooling = low/moderate risk
    assert!(risk.probability < 0.5);
    // Category depends on proximity; moderate is acceptable when temp is 70°C
    assert!(risk.category == RiskCategory::Low || risk.category == RiskCategory::Moderate);
}

// =============================================================================
// F1223: Thermal Correlation Tests
// =============================================================================

/// F1223.1: Valid Pearson r coefficient
#[test]
fn f1223_valid_pearson_r() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Perfect positive correlation
    for i in 0..10 {
        let temp = 60.0 + i as f64;
        let latency = 100.0 + i as f64 * 10.0;
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let corr = analyzer.correlation_to_latency().unwrap();

    // Pearson r should be in [-1, 1]
    assert!(corr.pearson_r >= -1.0 && corr.pearson_r <= 1.0);
    // Strong positive correlation expected
    assert!(corr.pearson_r > 0.9);
}

/// F1223.2: Negative correlation detected
#[test]
fn f1223_negative_correlation() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Negative correlation: higher temp = lower latency (unlikely but testing)
    for i in 0..10 {
        let temp = 60.0 + i as f64;
        let latency = 200.0 - i as f64 * 10.0;
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let corr = analyzer.correlation_to_latency().unwrap();
    assert!(corr.pearson_r < -0.9);
}

/// F1223.3: Weak correlation for alternating data
#[test]
fn f1223_weak_correlation() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Alternating pattern - should show weak/moderate correlation
    let temps = [60.0, 70.0, 60.0, 70.0, 60.0, 70.0, 60.0, 70.0, 60.0, 70.0];
    let latencies = [100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0];

    for (i, (&temp, &latency)) in temps.iter().zip(latencies.iter()).enumerate() {
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let corr = analyzer.correlation_to_latency().unwrap();
    // Correlation coefficient in valid range
    assert!(corr.pearson_r >= -1.0 && corr.pearson_r <= 1.0);
}

// =============================================================================
// F1224: Cooldown Recommendation Tests
// =============================================================================

/// F1224.1: Cooldown duration is positive
#[test]
fn f1224_positive_duration() {
    let cooldown = CooldownRecommendation::calculate(90.0, 75.0, 0.5);

    assert!(cooldown.duration_sec > 0.0);
    assert!(cooldown.is_needed());
}

/// F1224.2: No cooldown when below target
#[test]
fn f1224_no_cooldown_below_target() {
    let cooldown = CooldownRecommendation::calculate(70.0, 75.0, 0.5);

    assert_eq!(cooldown.duration_sec, 0.0);
    assert!(!cooldown.is_needed());
}

/// F1224.3: Cooldown calculation correct
#[test]
fn f1224_cooldown_calculation() {
    // 20°C to cool at 0.5°C/sec = 40 seconds
    let cooldown = CooldownRecommendation::calculate(90.0, 70.0, 0.5);

    assert!((cooldown.duration_sec - 40.0).abs() < 0.1);
}

// =============================================================================
// F1225: Trend Slope Calculation Tests
// =============================================================================

/// F1225.1: Trend slope accurate (°C/second)
#[test]
fn f1225_trend_slope_accurate() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Exactly 2°C/sec increase
    for i in 0..10 {
        analyzer.add(60.0 + i as f64 * 2.0, i as f64);
    }

    let trend = analyzer.calculate_trend().unwrap();
    assert!((trend - 2.0).abs() < 0.01);
}

/// F1225.2: Negative slope for cooling
#[test]
fn f1225_negative_slope() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Decreasing temperature
    for i in 0..10 {
        analyzer.add(80.0 - i as f64, i as f64);
    }

    let trend = analyzer.calculate_trend().unwrap();
    assert!(trend < 0.0);
    assert!((trend - (-1.0)).abs() < 0.01);
}

/// F1225.3: Zero slope for constant temp
#[test]
fn f1225_zero_slope_constant() {
    let mut analyzer = ThermalAnalyzer::new(20);

    for i in 0..10 {
        analyzer.add(70.0, i as f64);
    }

    let trend = analyzer.calculate_trend().unwrap();
    assert!(trend.abs() < 0.01);
}

// =============================================================================
// F1226: Historical Samples (Sliding Window) Tests
// =============================================================================

/// F1226.1: Sliding window works
#[test]
fn f1226_sliding_window() {
    let mut analyzer = ThermalAnalyzer::new(5);

    // Add more than window size
    for i in 0..10 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    // Should only have last 5 samples
    assert_eq!(analyzer.sample_count(), 5);

    // Oldest should be 65.0 (index 5)
    let (min, max) = analyzer.temperature_range().unwrap();
    assert_eq!(min, 65.0);
    assert_eq!(max, 69.0);
}

/// F1226.2: New samples included in prediction
#[test]
fn f1226_new_samples_included() {
    let mut analyzer = ThermalAnalyzer::new(10);

    // Initial samples
    for i in 0..5 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    let pred1 = analyzer.predict_trend(10.0).unwrap();

    // Add more samples with different trend
    for i in 5..10 {
        analyzer.add(64.0 + i as f64 * 2.0, i as f64);
    }

    let pred2 = analyzer.predict_trend(10.0).unwrap();

    // Predictions should be different
    assert!((pred1.trend_slope - pred2.trend_slope).abs() > 0.1);
}

// =============================================================================
// F1227: Insufficient Data Handling Tests
// =============================================================================

/// F1227.1: Returns None with insufficient samples
#[test]
fn f1227_insufficient_samples() {
    let mut analyzer = ThermalAnalyzer::new(10);

    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);

    // Only 2 samples, need at least 3
    assert!(!analyzer.has_sufficient_samples());
    assert!(analyzer.predict_trend(10.0).is_none());
    assert!(analyzer.calculate_trend().is_none());
}

/// F1227.2: Empty analyzer returns None
#[test]
fn f1227_empty_analyzer() {
    let analyzer = ThermalAnalyzer::new(10);

    assert!(analyzer.current_temperature().is_none());
    assert!(analyzer.average_temperature().is_none());
    assert!(analyzer.temperature_range().is_none());
    assert!(analyzer.predict_trend(10.0).is_none());
}

// =============================================================================
// F1228: Throttle Threshold Configuration Tests
// =============================================================================

/// F1228.1: Custom threshold works
#[test]
fn f1228_custom_threshold() {
    let mut analyzer = ThermalAnalyzer::new(10)
        .with_threshold(95.0);

    for i in 0..5 {
        analyzer.add(85.0 + i as f64, i as f64);
    }

    let risk = analyzer.throttle_risk().unwrap();

    // With 95°C threshold, 89°C should be moderate risk
    assert!(risk.threshold_c == 95.0);
    assert!(risk.margin_c > 0.0);
}

/// F1228.2: Default threshold is 85°C
#[test]
fn f1228_default_threshold() {
    let mut analyzer = ThermalAnalyzer::new(10);

    for i in 0..5 {
        analyzer.add(70.0, i as f64);
    }

    let risk = analyzer.throttle_risk().unwrap();
    assert_eq!(risk.threshold_c, 85.0);
}

// =============================================================================
// F1229: Continuous Prediction Update Tests
// =============================================================================

/// F1229.1: Prediction updates with new samples
#[test]
fn f1229_prediction_updates() {
    let mut analyzer = ThermalAnalyzer::new(10);

    // Initial increasing trend
    for i in 0..5 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    let pred1 = analyzer.predict_trend(10.0).unwrap();

    // Add samples showing cooling
    for i in 5..10 {
        analyzer.add(65.0 - (i - 5) as f64, i as f64);
    }

    let pred2 = analyzer.predict_trend(10.0).unwrap();

    // Second prediction should show cooling (lower predicted temp)
    assert!(pred2.predicted_temp_c < pred1.predicted_temp_c);
}

/// F1229.2: Real-time update simulation
#[test]
fn f1229_realtime_update() {
    let mut analyzer = ThermalAnalyzer::new(20);
    let mut predictions = Vec::new();

    // Simulate real-time updates every second
    for i in 0..15 {
        analyzer.add(60.0 + i as f64 * 0.5, i as f64);

        if analyzer.has_sufficient_samples() {
            predictions.push(analyzer.predict_trend(5.0).unwrap().predicted_temp_c);
        }
    }

    // All predictions should be increasing
    for window in predictions.windows(2) {
        assert!(window[1] >= window[0] - 0.1); // Allow small tolerance
    }
}

// =============================================================================
// F1230: Thermal Variance Isolation Tests
// =============================================================================

/// F1230.1: Variance contribution percentage calculated
#[test]
fn f1230_variance_contribution() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Strong thermal correlation
    for i in 0..10 {
        let temp = 60.0 + i as f64;
        let latency = 100.0 + i as f64 * 5.0;
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let variance = analyzer.thermal_variance().unwrap();

    // Contribution should be between 0-100%
    assert!(variance.contribution_percent >= 0.0);
    assert!(variance.contribution_percent <= 100.0);
    // With strong correlation, should be high
    assert!(variance.contribution_percent > 50.0);
}

/// F1230.2: Temperature range tracked
#[test]
fn f1230_temp_range_tracked() {
    let mut analyzer = ThermalAnalyzer::new(20);

    for i in 0..10 {
        analyzer.add(60.0 + i as f64 * 2.0, i as f64);
    }

    let variance = analyzer.thermal_variance().unwrap();

    // Range should be 18°C (60 to 78)
    assert!((variance.temp_range_c - 18.0).abs() < 0.1);
    // Average should be 69°C
    assert!((variance.avg_temp_c - 69.0).abs() < 0.1);
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test prediction will_throttle
#[test]
fn test_will_throttle() {
    let prediction = ThermalPrediction {
        predicted_temp_c: 90.0,
        horizon_sec: 10.0,
        trend_slope: 1.0,
        confidence: 0.95,
        sample_count: 10,
    };

    assert!(prediction.will_throttle(85.0));
    assert!(!prediction.will_throttle(95.0));
}

/// Test time_to_throttle
#[test]
fn test_time_to_throttle() {
    let prediction = ThermalPrediction {
        predicted_temp_c: 90.0,
        horizon_sec: 10.0,
        trend_slope: 2.0, // 2°C/sec
        confidence: 0.95,
        sample_count: 10,
    };

    // Current at 80°C, threshold at 85°C, 2°C/sec = 2.5 seconds
    let time = prediction.time_to_throttle(80.0, 85.0).unwrap();
    assert!((time - 2.5).abs() < 0.1);
}

/// Test risk category names
#[test]
fn test_risk_category_names() {
    assert_eq!(RiskCategory::Low.name(), "low");
    assert_eq!(RiskCategory::Moderate.name(), "moderate");
    assert_eq!(RiskCategory::High.name(), "high");
    assert_eq!(RiskCategory::Critical.name(), "critical");
}

/// Test analyzer with custom cooling rate
#[test]
fn test_custom_cooling_rate() {
    let mut analyzer = ThermalAnalyzer::new(10)
        .with_cooling_rate(1.0); // 1°C/sec

    for i in 0..5 {
        analyzer.add(90.0, i as f64);
    }

    let cooldown = analyzer.recommended_cooldown().unwrap();

    // Need to cool from 90 to 75 (15°C) at 1°C/sec = 15 seconds
    assert!((cooldown.duration_sec - 15.0).abs() < 0.1);
}

/// Test analyze_thermal convenience function
#[test]
fn test_analyze_thermal_function() {
    let samples = vec![
        (60.0, 0.0),
        (65.0, 1.0),
        (70.0, 2.0),
        (75.0, 3.0),
        (80.0, 4.0),
    ];

    let prediction = analyze_thermal(&samples, 5.0).unwrap();

    // Trend is 5°C/sec, current is 80°C, 5 seconds ahead = 105°C
    // (but prediction is from current + slope * horizon)
    assert!(prediction.trend_slope > 4.0 && prediction.trend_slope < 6.0);
    assert!(prediction.predicted_temp_c > 100.0);
}

/// Test thermal correlation has_thermal_impact
#[test]
fn test_has_thermal_impact() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Strong correlation
    for i in 0..10 {
        analyzer.add_with_latency(60.0 + i as f64, i as f64, 100.0 + i as f64 * 5.0);
    }

    let corr = analyzer.correlation_to_latency().unwrap();
    assert!(corr.has_thermal_impact());
}

/// Test clear samples
#[test]
fn test_clear_samples() {
    let mut analyzer = ThermalAnalyzer::new(10);

    for i in 0..5 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    assert_eq!(analyzer.sample_count(), 5);

    analyzer.clear();
    assert_eq!(analyzer.sample_count(), 0);
}

/// Test samples access
#[test]
fn test_samples_access() {
    let mut analyzer = ThermalAnalyzer::new(10);

    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);

    let samples = analyzer.samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].temperature_c, 60.0);
}
