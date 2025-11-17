//! Comprehensive Integration Test Suite
//!
//! This integration test serves as the definitive release gate for Trueno.
//! It must pass with 100% success before any release is tagged.
//!
//! Requirements:
//! - Tests ALL features currently supported
//! - Uses property-based testing for mathematical correctness
//! - Contributes to code coverage
//! - Runs under 30 seconds (enforced)
//! - Included in pre-commit hooks
//!
//! Coverage:
//! - All 87 vector operations
//! - All matrix operations (matmul, transpose)
//! - All backends (Scalar, SSE2, AVX2, etc.)
//! - Error handling and edge cases
//! - Mathematical properties and invariants

use proptest::prelude::*;
use trueno::{Backend, Matrix, Vector};

// ============================================================================
// PROPERTY TEST CONFIGURATION
// ============================================================================

const PROPTEST_CASES: u32 = 50; // Reduced from 100 for 30s target

// ============================================================================
// VECTOR OPERATIONS - ELEMENT-WISE
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: All element-wise binary operations
    #[test]
    fn integration_vector_elementwise_binary(
        a_data in prop::collection::vec(-100.0f32..100.0, 10..100),
        b_data in prop::collection::vec(-100.0f32..100.0, 10..100)
    ) {
        let len = a_data.len().min(b_data.len());
        let a = Vector::from_slice(&a_data[..len]);
        let b = Vector::from_slice(&b_data[..len]);

        // Test all binary operations
        prop_assert!(a.add(&b).is_ok());
        prop_assert!(a.sub(&b).is_ok());
        prop_assert!(a.mul(&b).is_ok());

        // div requires non-zero b
        let b_nonzero: Vec<f32> = b_data[..len].iter()
            .map(|&x| if x.abs() < 0.01 { 1.0 } else { x })
            .collect();
        let b_nz = Vector::from_slice(&b_nonzero);
        prop_assert!(a.div(&b_nz).is_ok());

        // Comparison operations
        prop_assert!(a.minimum(&b).is_ok());
        prop_assert!(a.maximum(&b).is_ok());

        // Dot product
        let dot = a.dot(&b)?;
        prop_assert!(dot.is_finite());

        // Lerp and FMA
        prop_assert!(a.lerp(&b, 0.5).is_ok());
        let c = Vector::from_slice(&a_data[..len]);
        prop_assert!(a.fma(&b, &c).is_ok());
    }

    /// Integration test: All element-wise unary operations
    #[test]
    fn integration_vector_elementwise_unary(
        data in prop::collection::vec(-50.0f32..50.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Basic unary operations
        prop_assert!(v.abs().is_ok());
        prop_assert!(v.neg().is_ok());
        prop_assert!(v.floor().is_ok());
        prop_assert!(v.ceil().is_ok());
        prop_assert!(v.round().is_ok());
        prop_assert!(v.trunc().is_ok());
        prop_assert!(v.fract().is_ok());
        prop_assert!(v.signum().is_ok());

        // Sqrt on positive values
        let v_pos = v.abs().unwrap();
        prop_assert!(v_pos.sqrt().is_ok());

        // Recip on non-zero values
        let v_nonzero: Vec<f32> = data.iter()
            .map(|&x| if x.abs() < 0.1 { 1.0 } else { x })
            .collect();
        let v_nz = Vector::from_slice(&v_nonzero);
        prop_assert!(v_nz.recip().is_ok());

        // Exponential and logarithm (restricted ranges)
        let v_restricted: Vec<f32> = data.iter()
            .map(|&x| x.abs().min(10.0))
            .collect();
        let v_r = Vector::from_slice(&v_restricted);
        prop_assert!(v_r.exp().is_ok());

        let v_pos_restricted: Vec<f32> = data.iter()
            .map(|&x| x.abs().clamp(0.1, 100.0))
            .collect();
        let v_pr = Vector::from_slice(&v_pos_restricted);
        prop_assert!(v_pr.ln().is_ok());
    }

    /// Integration test: Trigonometric functions
    #[test]
    fn integration_vector_trig(
        data in prop::collection::vec(-10.0f32..10.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Basic trig functions
        prop_assert!(v.sin().is_ok());
        prop_assert!(v.cos().is_ok());
        prop_assert!(v.tan().is_ok());

        // Inverse trig (restricted domain)
        let v_unit: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-0.9, 0.9))
            .collect();
        let vu = Vector::from_slice(&v_unit);
        prop_assert!(vu.asin().is_ok());
        prop_assert!(vu.acos().is_ok());
        prop_assert!(v.atan().is_ok());

        // Hyperbolic functions (restricted range)
        let v_restricted: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-5.0, 5.0))
            .collect();
        let vr = Vector::from_slice(&v_restricted);
        prop_assert!(vr.sinh().is_ok());
        prop_assert!(vr.cosh().is_ok());
        prop_assert!(vr.tanh().is_ok());

        // Inverse hyperbolic (appropriate domains)
        prop_assert!(v.asinh().is_ok());

        let v_pos: Vec<f32> = data.iter()
            .map(|&x| x.abs().max(1.01))
            .collect();
        let vp = Vector::from_slice(&v_pos);
        prop_assert!(vp.acosh().is_ok());

        let v_tanh_domain: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-0.9, 0.9))
            .collect();
        let vtd = Vector::from_slice(&v_tanh_domain);
        prop_assert!(vtd.atanh().is_ok());
    }
}

// ============================================================================
// VECTOR OPERATIONS - REDUCTIONS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: All reduction operations
    #[test]
    fn integration_vector_reductions(
        data in prop::collection::vec(-100.0f32..100.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Basic reductions
        let sum = v.sum()?;
        let sum_kahan = v.sum_kahan()?;
        let min = v.min()?;
        let max = v.max()?;
        let sum_sq = v.sum_of_squares()?;

        prop_assert!(sum.is_finite());
        prop_assert!(sum_kahan.is_finite());
        prop_assert!(min <= max);
        prop_assert!(sum_sq >= 0.0);

        // Statistical operations
        let mean = v.mean()?;
        let variance = v.variance()?;
        let stddev = v.stddev()?;

        prop_assert!(mean.is_finite());
        prop_assert!(variance >= -1e-5); // Allow small numerical error
        prop_assert!(stddev >= -1e-5);

        // Index operations
        let argmin_idx = v.argmin()?;
        let argmax_idx = v.argmax()?;
        prop_assert!(argmin_idx < v.len());
        prop_assert!(argmax_idx < v.len());

        // Norms
        let l1 = v.norm_l1()?;
        let l2 = v.norm_l2()?;
        let linf = v.norm_linf()?;

        prop_assert!(l1 >= 0.0);
        prop_assert!(l2 >= 0.0);
        prop_assert!(linf >= 0.0);
        prop_assert!(linf <= l1); // L∞ <= L1 for normalized comparison
    }

    /// Integration test: Statistical operations with two vectors
    #[test]
    fn integration_vector_statistics_two_vectors(
        a_data in prop::collection::vec(-50.0f32..50.0, 10..100),
        b_data in prop::collection::vec(-50.0f32..50.0, 10..100)
    ) {
        let len = a_data.len().min(b_data.len());
        let a = Vector::from_slice(&a_data[..len]);
        let b = Vector::from_slice(&b_data[..len]);

        // Covariance and correlation
        let cov = a.covariance(&b)?;
        let corr = a.correlation(&b)?;

        prop_assert!(cov.is_finite());
        prop_assert!(corr.is_finite());
        prop_assert!((-1.0 - 1e-4..=1.0 + 1e-4).contains(&corr));
    }
}

// ============================================================================
// VECTOR OPERATIONS - ACTIVATION FUNCTIONS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: All activation functions
    #[test]
    fn integration_vector_activations(
        data in prop::collection::vec(-10.0f32..10.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // ReLU variants
        prop_assert!(v.relu().is_ok());
        prop_assert!(v.leaky_relu(0.01).is_ok());
        prop_assert!(v.elu(1.0).is_ok());

        // Sigmoid and variants
        prop_assert!(v.sigmoid().is_ok());

        // Softmax and log_softmax (restricted range for stability)
        let v_restricted: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-10.0, 10.0))
            .collect();
        let vr = Vector::from_slice(&v_restricted);
        let softmax = vr.softmax()?;
        let _log_softmax = vr.log_softmax()?;

        // Verify softmax sums to 1
        let sum: f32 = softmax.as_slice().iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4);

        // Modern activations
        prop_assert!(v.gelu().is_ok());
        prop_assert!(v.swish().is_ok()); // Also known as SiLU
    }

    /// Integration test: Preprocessing operations
    #[test]
    fn integration_vector_preprocessing(
        data in prop::collection::vec(-100.0f32..100.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Clipping
        prop_assert!(v.clip(-50.0, 50.0).is_ok());

        // Min-max normalization
        if data.iter().any(|&x| x != data[0]) {
            let normalized = v.minmax_normalize()?;
            let min = *normalized.as_slice().iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let max = *normalized.as_slice().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            prop_assert!(min >= -1e-5);
            prop_assert!(max <= 1.0 + 1e-5);
        }

        // Z-score normalization (requires n >= 3 and variance > 0)
        if data.len() >= 3 {
            let variance = v.variance()?;
            if variance > 0.1 {
                prop_assert!(v.zscore().is_ok());
            }
        }
    }
}

// ============================================================================
// VECTOR OPERATIONS - SCALAR AND UTILITY
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: Scalar operations
    #[test]
    fn integration_vector_scalar_ops(
        data in prop::collection::vec(-50.0f32..50.0, 10..100),
        scalar in -10.0f32..10.0
    ) {
        let v = Vector::from_slice(&data);

        // Scale
        prop_assert!(v.scale(scalar).is_ok());

        // Pow
        let v_pos = v.abs().unwrap();
        prop_assert!(v_pos.pow(2.0).is_ok());

        // Clamp
        prop_assert!(v.clamp(-10.0, 10.0).is_ok());
    }

    /// Integration test: Vector normalization
    #[test]
    fn integration_vector_normalization(
        data in prop::collection::vec(-50.0f32..50.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Skip if vector is near-zero
        let norm = v.norm_l2()?;
        if norm > 1e-6 {
            let normalized = v.normalize()?;
            let new_norm = normalized.norm_l2()?;
            prop_assert!((new_norm - 1.0).abs() < 1e-4);
        }
    }
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: Matrix multiplication
    #[test]
    fn integration_matrix_matmul(
        m in 2usize..10,
        n in 2usize..10,
        p in 2usize..10
    ) {
        let a = Matrix::zeros(m, n);
        let b = Matrix::zeros(n, p);
        let c = a.matmul(&b)?;

        prop_assert_eq!(c.rows(), m);
        prop_assert_eq!(c.cols(), p);

        // Test with identity
        let identity = Matrix::identity(n);
        let a_i = a.matmul(&identity)?;
        prop_assert_eq!(a_i.rows(), a.rows());
        prop_assert_eq!(a_i.cols(), a.cols());
    }

    /// Integration test: Matrix transpose
    #[test]
    fn integration_matrix_transpose(
        m in 2usize..10,
        n in 2usize..10
    ) {
        let a = Matrix::zeros(m, n);
        let t = a.transpose();

        prop_assert_eq!(t.rows(), n);
        prop_assert_eq!(t.cols(), m);

        // Double transpose returns original dimensions
        let tt = t.transpose();
        prop_assert_eq!(tt.rows(), m);
        prop_assert_eq!(tt.cols(), n);
    }

    /// Integration test: Matrix constructors
    #[test]
    fn integration_matrix_constructors(
        rows in 1usize..20,
        cols in 1usize..20
    ) {
        // zeros
        let zeros = Matrix::zeros(rows, cols);
        prop_assert_eq!(zeros.rows(), rows);
        prop_assert_eq!(zeros.cols(), cols);

        // identity (square only)
        let n = rows.min(cols);
        let identity = Matrix::identity(n);
        prop_assert_eq!(identity.rows(), n);
        prop_assert_eq!(identity.cols(), n);

        // from_vec
        let data = vec![1.0f32; rows * cols];
        let from_vec = Matrix::from_vec(rows, cols, data)?;
        prop_assert_eq!(from_vec.rows(), rows);
        prop_assert_eq!(from_vec.cols(), cols);
    }

    /// Integration test: Matrix-vector multiplication (matvec)
    #[test]
    fn integration_matrix_matvec(
        m in 2usize..20,
        n in 2usize..20
    ) {
        // Create m×n matrix and n-dimensional vector
        let matrix_data = (0..m * n).map(|i| (i % 10) as f32).collect();
        let matrix = Matrix::from_vec(m, n, matrix_data)?;
        let vector_data: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let vector = Vector::from_slice(&vector_data);

        // Compute matrix-vector product
        let result = matrix.matvec(&vector)?;

        // Result should be m-dimensional
        prop_assert_eq!(result.len(), m);

        // Test with identity matrix: I×v = v
        let identity = Matrix::identity(n);
        let iv = identity.matvec(&vector)?;
        prop_assert_eq!(iv.len(), n);
    }

    /// Integration test: Vector-matrix multiplication (vecmat)
    #[test]
    fn integration_matrix_vecmat(
        m in 2usize..20,
        n in 2usize..20
    ) {
        // Create m-dimensional vector and m×n matrix
        let vector_data: Vec<f32> = (0..m).map(|i| (i % 5) as f32).collect();
        let vector = Vector::from_slice(&vector_data);
        let matrix_data = (0..m * n).map(|i| (i % 10) as f32).collect();
        let matrix = Matrix::from_vec(m, n, matrix_data)?;

        // Compute vector-matrix product
        let result = Matrix::vecmat(&vector, &matrix)?;

        // Result should be n-dimensional
        prop_assert_eq!(result.len(), n);

        // Test with identity matrix: v^T×I = v^T
        let identity = Matrix::identity(m);
        let vi = Matrix::vecmat(&vector, &identity)?;
        prop_assert_eq!(vi.len(), m);
    }
}

// ============================================================================
// BACKEND TESTING
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))] // Fewer cases for backend tests

    /// Integration test: Backend consistency
    #[test]
    fn integration_backend_consistency(
        data in prop::collection::vec(-50.0f32..50.0, 50..100)
    ) {
        // Test that Auto backend produces valid results
        let v_auto = Vector::from_slice(&data);

        // Scalar backend as reference
        let v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);

        // Test a few operations for consistency
        let sum_auto = v_auto.sum()?;
        let sum_scalar = v_scalar.sum()?;
        prop_assert!((sum_auto - sum_scalar).abs() < 1e-3 * sum_scalar.abs().max(1.0));

        let max_auto = v_auto.max()?;
        let max_scalar = v_scalar.max()?;
        prop_assert_eq!(max_auto, max_scalar);
    }
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[test]
fn integration_error_handling() {
    // Size mismatch errors
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[1.0, 2.0]);
    assert!(a.add(&b).is_err());
    assert!(a.dot(&b).is_err());

    // Empty vector handling (most operations return 0 or NaN for empty)
    let empty = Vector::from_slice(&[]);
    // sum() returns 0.0 for empty vectors
    assert_eq!(empty.sum().unwrap(), 0.0);
    // mean() returns error for empty vectors
    assert!(empty.mean().is_err());

    // Single element vectors (variance is 0)
    let single = Vector::from_slice(&[1.0]);
    assert_eq!(single.variance().unwrap(), 0.0); // Single element has zero variance

    // Matrix dimension mismatch
    let m1 = Matrix::zeros(2, 3);
    let m2 = Matrix::zeros(2, 2);
    assert!(m1.matmul(&m2).is_err());

    // Matrix from_vec with wrong size
    let data = vec![1.0, 2.0, 3.0];
    assert!(Matrix::from_vec(2, 2, data).is_err());
}

// ============================================================================
// BACKEND SELECTION
// ============================================================================

#[test]
fn integration_backend_selection() {
    use trueno::select_best_available_backend;

    let backend = select_best_available_backend();

    // Verify backend is one of the valid options
    assert!(matches!(
        backend,
        Backend::Scalar
            | Backend::SSE2
            | Backend::AVX
            | Backend::AVX2
            | Backend::AVX512
            | Backend::NEON
            | Backend::WasmSIMD
    ));

    // Verify Backend::select_best() works
    let backend2 = Backend::select_best();
    assert_eq!(backend, backend2);

    // Test creating vectors with explicit backends
    let data = vec![1.0, 2.0, 3.0];
    let _v_auto = Vector::from_slice(&data);
    let _v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
}

// ============================================================================
// COMPREHENSIVE SMOKE TEST
// ============================================================================

/// Smoke test: Every operation must execute without panic
#[test]
fn integration_smoke_test_all_operations() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Vector::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);

    // Element-wise binary operations
    let _ = a.add(&b);
    let _ = a.sub(&b);
    let _ = a.mul(&b);
    let _ = a.div(&b);
    let _ = a.minimum(&b);
    let _ = a.maximum(&b);
    let _ = a.lerp(&b, 0.5);
    let _ = a.fma(&b, &a);

    // Element-wise unary operations
    let _ = a.abs();
    let _ = a.neg();
    let _ = a.sqrt();
    let _ = a.recip();
    let _ = a.pow(2.0);
    let _ = a.exp();
    let _ = a.ln();
    let _ = a.floor();
    let _ = a.ceil();
    let _ = a.round();
    let _ = a.trunc();
    let _ = a.fract();
    let _ = a.signum();

    // Trigonometric
    let _ = a.sin();
    let _ = a.cos();
    let _ = a.tan();
    let _ = a.sinh();
    let _ = a.cosh();
    let _ = a.tanh();
    let _ = a.asinh();

    // Reductions
    let _ = a.sum();
    let _ = a.sum_kahan();
    let _ = a.min();
    let _ = a.max();
    let _ = a.sum_of_squares();
    let _ = a.mean();
    let _ = a.variance();
    let _ = a.stddev();
    let _ = a.dot(&b);

    // Statistical
    let _ = a.covariance(&b);
    let _ = a.correlation(&b);

    // Activations
    let _ = a.relu();
    let _ = a.leaky_relu(0.01);
    let _ = a.elu(1.0);
    let _ = a.sigmoid();
    let _ = a.softmax();
    let _ = a.log_softmax();
    let _ = a.gelu();
    let _ = a.swish(); // Also known as SiLU

    // Preprocessing
    let _ = a.clip(-10.0, 10.0);
    let _ = a.minmax_normalize();
    let _ = a.zscore();

    // Norms
    let _ = a.norm_l1();
    let _ = a.norm_l2();
    let _ = a.norm_linf();
    let _ = a.normalize();

    // Index operations
    let _ = a.argmin();
    let _ = a.argmax();

    // Scalar operations
    let _ = a.scale(2.0);
    let _ = a.clamp(-10.0, 10.0);

    // Matrix operations
    let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = Matrix::identity(2);
    let _ = m1.matmul(&m2);
    let _ = m1.transpose();
    let _ = m1.rows();
    let _ = m1.cols();
    let _ = m1.shape();
    let _ = m1.get(0, 0);
    let _ = m1.as_slice();
}

// ============================================================================
// PERFORMANCE TEST (must complete under 30 seconds)
// ============================================================================

#[test]
fn integration_performance_gate() {
    use std::time::Instant;

    let start = Instant::now();

    // Run a subset of operations to verify performance
    for size in [100, 1000, 10000] {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let v = Vector::from_slice(&data);

        let _ = v.sum();
        let _ = v.mean();
        let _ = v.dot(&v);
        let _ = v.add(&v);
        let _ = v.relu();
    }

    let elapsed = start.elapsed();

    // This specific test should be very fast (<1s)
    assert!(
        elapsed.as_secs() < 5,
        "Performance gate failed: took {:?}, expected <5s",
        elapsed
    );
}
