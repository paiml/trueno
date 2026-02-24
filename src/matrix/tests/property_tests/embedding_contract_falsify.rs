//! Embedding Lookup Contract Falsification Tests
//!
//! Popperian falsification of embedding-lookup-v1.yaml claims:
//!   - FALSIFY-EM-001: Output shape = (seq_len, d_model)
//!   - FALSIFY-EM-002: OOB token IDs produce controlled error (no panic)
//!   - FALSIFY-EM-003: Deterministic output (identical W + ids → identical result)
//!   - FALSIFY-EM-004: Finite output when embedding table is finite
//!
//! Five-Whys (PMAT-354):
//!   Why #1: trueno has 9 embedding_lookup tests but zero FALSIFY-EM-* tagged tests
//!   Why #2: existing tests verify examples, not provable-contract YAML claims
//!   Why #3: trueno was built before the embedding-lookup-v1.yaml contract
//!   Why #4: no systematic link between contract YAML and Rust test names
//!   Why #5: no cross-repo falsification naming convention existed
//!
//! References:
//!   - Mikolov et al. (2013) Efficient Estimation of Word Representations
//!   - provable-contracts/contracts/embedding-lookup-v1.yaml
//!   - src/matrix/ops/ml_ops/mod.rs

use super::super::Matrix;

// ============================================================================
// FALSIFY-EM-001: Output shape correctness
// Contract: output.shape = (seq_len, d_model) for any valid seq_len
// ============================================================================

#[test]
fn falsify_em_001_output_shape_single() {
    let table = Matrix::from_vec(10, 4, vec![1.0; 40]).unwrap();
    let result = table.embedding_lookup(&[3]).unwrap();

    assert_eq!(
        result.rows(),
        1,
        "FALSIFIED EM-001: single lookup rows={}, expected 1",
        result.rows()
    );
    assert_eq!(
        result.cols(),
        4,
        "FALSIFIED EM-001: single lookup cols={}, expected 4",
        result.cols()
    );
}

#[test]
fn falsify_em_001_output_shape_batch() {
    let vocab_size = 50;
    let d_model = 16;
    let data: Vec<f32> = (0..vocab_size * d_model).map(|i| i as f32).collect();
    let table = Matrix::from_vec(vocab_size, d_model, data).unwrap();

    for seq_len in [1, 2, 5, 10, 49] {
        let indices: Vec<usize> = (0..seq_len).collect();
        let result = table.embedding_lookup(&indices).unwrap();

        assert_eq!(
            result.rows(),
            seq_len,
            "FALSIFIED EM-001: seq_len={seq_len}, got rows={}",
            result.rows()
        );
        assert_eq!(
            result.cols(),
            d_model,
            "FALSIFIED EM-001: seq_len={seq_len}, got cols={}",
            result.cols()
        );
    }
}

#[test]
fn falsify_em_001_output_shape_empty() {
    let table = Matrix::from_vec(10, 4, vec![1.0; 40]).unwrap();
    let result = table.embedding_lookup(&[]).unwrap();

    assert_eq!(
        result.rows(),
        0,
        "FALSIFIED EM-001: empty indices should produce 0 rows"
    );
    assert_eq!(
        result.cols(),
        4,
        "FALSIFIED EM-001: empty indices should preserve d_model=4"
    );
}

// ============================================================================
// FALSIFY-EM-002: Out-of-bounds safety
// Contract: OOB token IDs produce controlled error, not panic
// ============================================================================

#[test]
fn falsify_em_002_oob_returns_error() {
    let table = Matrix::from_vec(5, 3, vec![1.0; 15]).unwrap();

    // Exactly at boundary
    let result = table.embedding_lookup(&[5]);
    assert!(
        result.is_err(),
        "FALSIFIED EM-002: index=5 should error for vocab_size=5"
    );
}

#[test]
fn falsify_em_002_oob_error_message() {
    let table = Matrix::from_vec(5, 3, vec![1.0; 15]).unwrap();
    let err = table.embedding_lookup(&[100]).unwrap_err();

    assert!(
        err.to_string().contains("out of bounds"),
        "FALSIFIED EM-002: error message should mention 'out of bounds', got: {}",
        err
    );
}

#[test]
fn falsify_em_002_boundary_valid() {
    let table = Matrix::from_vec(5, 3, vec![1.0; 15]).unwrap();

    // Last valid index
    let result = table.embedding_lookup(&[4]);
    assert!(
        result.is_ok(),
        "FALSIFIED EM-002: index=4 should succeed for vocab_size=5"
    );
}

#[test]
fn falsify_em_002_mixed_valid_invalid() {
    let table = Matrix::from_vec(5, 3, vec![1.0; 15]).unwrap();

    // Valid + invalid mix: should fail (first invalid contaminates)
    let result = table.embedding_lookup(&[0, 2, 10]);
    assert!(
        result.is_err(),
        "FALSIFIED EM-002: mixed valid+invalid should error"
    );
}

// ============================================================================
// FALSIFY-EM-003: Deterministic output
// Contract: two calls with identical W and token_ids produce bit-identical output
// ============================================================================

#[test]
fn falsify_em_003_determinism() {
    let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.37).sin()).collect();
    let table = Matrix::from_vec(10, 20, data).unwrap();
    let indices = vec![3, 7, 1, 9, 0, 5];

    let r1 = table.embedding_lookup(&indices).unwrap();
    let r2 = table.embedding_lookup(&indices).unwrap();

    // Bit-identical comparison (not approximate)
    assert_eq!(
        r1.data,
        r2.data,
        "FALSIFIED EM-003: embedding lookup is non-deterministic"
    );
}

#[test]
fn falsify_em_003_repeated_index_determinism() {
    let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
    let table = Matrix::from_vec(5, 12, data).unwrap();
    let indices = vec![2, 2, 2];

    let result = table.embedding_lookup(&indices).unwrap();

    // All three rows must be identical (same index)
    for col in 0..12 {
        let v0 = result.get(0, col).unwrap();
        let v1 = result.get(1, col).unwrap();
        let v2 = result.get(2, col).unwrap();
        assert_eq!(
            v0, v1,
            "FALSIFIED EM-003: repeated index produced different rows"
        );
        assert_eq!(
            v1, v2,
            "FALSIFIED EM-003: repeated index produced different rows"
        );
    }
}

// ============================================================================
// FALSIFY-EM-004: Finite output
// Contract: all output elements are finite when W elements are finite
// ============================================================================

#[test]
fn falsify_em_004_finite_output() {
    let data: Vec<f32> = (0..500)
        .map(|i| (i as f32 * 0.123).sin() * 100.0)
        .collect();
    let table = Matrix::from_vec(25, 20, data).unwrap();
    let indices: Vec<usize> = (0..25).collect();

    let result = table.embedding_lookup(&indices).unwrap();

    for (i, val) in result.data.iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED EM-004: output[{i}] = {val} is not finite"
        );
    }
}

#[test]
fn falsify_em_004_no_nan_no_inf() {
    let data: Vec<f32> = (0..120).map(|i| (i as f32) * 0.001).collect();
    let table = Matrix::from_vec(10, 12, data).unwrap();
    let indices = vec![0, 5, 9];

    let result = table.embedding_lookup(&indices).unwrap();

    let nan_count = result.data.iter().filter(|v| v.is_nan()).count();
    let inf_count = result.data.iter().filter(|v| v.is_infinite()).count();

    assert_eq!(
        nan_count, 0,
        "FALSIFIED EM-004: output contains {} NaN values",
        nan_count
    );
    assert_eq!(
        inf_count, 0,
        "FALSIFIED EM-004: output contains {} Inf values",
        inf_count
    );
}

// ============================================================================
// FALSIFY-EM-005: Embedding value correctness (extractive)
// Contract: output[i] = W[token_ids[i]] (row copy, not transformation)
// ============================================================================

#[test]
fn falsify_em_005_value_correctness() {
    let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let table = Matrix::from_vec(10, 4, data).unwrap();
    let indices = vec![3, 7, 0];

    let result = table.embedding_lookup(&indices).unwrap();

    // Row 0 should be table row 3: [12, 13, 14, 15]
    assert_eq!(result.get(0, 0), Some(&12.0));
    assert_eq!(result.get(0, 3), Some(&15.0));

    // Row 1 should be table row 7: [28, 29, 30, 31]
    assert_eq!(result.get(1, 0), Some(&28.0));
    assert_eq!(result.get(1, 3), Some(&31.0));

    // Row 2 should be table row 0: [0, 1, 2, 3]
    assert_eq!(result.get(2, 0), Some(&0.0));
    assert_eq!(result.get(2, 3), Some(&3.0));
}
