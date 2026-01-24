use super::super::*;

// ========================================================================
// PMAT-017: AttentionOp Tests
// ========================================================================

#[test]
fn test_attention_op_basic() {
    // Simple 2x2 attention (seq_len=2, kv_seq_len=2, head_dim=2)
    let op = AttentionOp::self_attention(2, 2);

    // Q = [[1, 0], [0, 1]]
    let q = vec![1.0, 0.0, 0.0, 1.0];
    // K = [[1, 0], [0, 1]]
    let k = vec![1.0, 0.0, 0.0, 1.0];
    // V = [[1, 2], [3, 4]]
    let v = vec![1.0, 2.0, 3.0, 4.0];

    let result = op.execute((q, k, v), Backend::Scalar).unwrap();

    // Output should be [seq_len * head_dim] = 4 elements
    assert_eq!(result.len(), 4);

    // Each row should be a weighted sum of V rows
    // Q[0]·K[0] = 1, Q[0]·K[1] = 0 → softmax → [~0.73, ~0.27]
    // Output[0] ≈ 0.73 * [1,2] + 0.27 * [3,4]
    assert!(result[0] > 0.0 && result[0] < 3.0);
    assert!(result[1] > 0.0 && result[1] < 4.0);
}

#[test]
fn test_attention_op_simd_dot() {
    // Test the SIMD dot product directly
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let result = AttentionOp::simd_dot(&a, &b);
    assert!((result - 36.0).abs() < 0.001); // 1+2+3+4+5+6+7+8 = 36
}

#[test]
fn test_attention_op_softmax_row() {
    let mut scores = vec![1.0, 2.0, 3.0];
    AttentionOp::simd_softmax_row(&mut scores);

    // Sum should be 1.0
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // Values should be increasing
    assert!(scores[0] < scores[1]);
    assert!(scores[1] < scores[2]);
}

#[test]
fn test_attention_op_dimension_validation() {
    let op = AttentionOp::new(2, 3, 4);

    // Wrong Q size
    let result = op.execute(
        (vec![0.0; 4], vec![0.0; 12], vec![0.0; 12]),
        Backend::Scalar,
    );
    assert!(result.is_err());

    // Wrong K size
    let result = op.execute(
        (vec![0.0; 8], vec![0.0; 8], vec![0.0; 12]),
        Backend::Scalar,
    );
    assert!(result.is_err());
}

#[test]
fn test_attention_op_single_position() {
    // Single query position attending to 3 key positions
    let op = AttentionOp::new(1, 3, 4);

    let q = vec![1.0, 0.0, 0.0, 0.0]; // [1, 4]
    let k = vec![
        1.0, 0.0, 0.0, 0.0, // K[0]
        0.0, 1.0, 0.0, 0.0, // K[1]
        0.0, 0.0, 1.0, 0.0, // K[2]
    ];
    let v = vec![
        1.0, 0.0, 0.0, 0.0, // V[0]
        0.0, 1.0, 0.0, 0.0, // V[1]
        0.0, 0.0, 1.0, 0.0, // V[2]
    ];

    let result = op.execute((q, k, v), Backend::Scalar).unwrap();
    assert_eq!(result.len(), 4);

    // Q·K[0] = 1, Q·K[1] = 0, Q·K[2] = 0
    // After softmax: [~0.58, ~0.21, ~0.21] (approx)
    // Output ≈ 0.58*V[0] + 0.21*V[1] + 0.21*V[2]
    // Should have higher weight on first component
    assert!(result[0] > result[1]);
}

#[test]
fn test_compute_brick_run() {
    let brick = ComputeBrick::new(DotOp::new(4))
        .budget_tok_per_sec(1_000_000.0)
        .backend(Backend::Scalar);

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = brick.run((a, b)).unwrap();

    assert!((result.output - 70.0).abs() < 0.001);
    assert_eq!(result.tokens_processed, 4);
    assert!(result.tokens_per_sec > 0.0);
}

#[test]
fn test_compute_brick_verify() {
    let brick = ComputeBrick::new(DotOp::new(4))
        .assert_finite()
        .assert_bounds(-1000.0, 1000.0);

    let verification = brick.verify();
    assert!(verification.is_valid());
    assert_eq!(verification.assertion_results.len(), 2);
}

#[test]
fn test_compute_brick_no_assertions() {
    let brick = ComputeBrick::new(DotOp::new(4));
    let verification = brick.verify();
    assert!(!verification.is_valid()); // Should fail Popperian requirement
}

#[test]
fn test_brick_layer() {
    let dot_brick = ComputeBrick::new(DotOp::new(100)).budget_tok_per_sec(50_000.0);

    let add_brick = ComputeBrick::new(AddOp::new(100)).budget_tok_per_sec(30_000.0); // Bottleneck

    let layer = BrickLayer::new()
        .with_brick(&dot_brick)
        .with_brick(&add_brick);

    assert!((layer.throughput_ceiling() - 30_000.0).abs() < 1.0);
    assert_eq!(layer.bottleneck(), Some("add"));
}

#[test]
fn test_backend_display() {
    assert_eq!(format!("{}", Backend::Avx2), "AVX2");
    assert_eq!(format!("{}", Backend::Cuda), "CUDA");
    assert_eq!(format!("{}", Backend::Scalar), "Scalar");
}

#[test]
fn test_budget_utilization() {
    let budget = TokenBudget::from_latency(100.0);
    assert!((budget.utilization(50.0) - 0.5).abs() < 0.001); // 50% used
    assert!((budget.utilization(100.0) - 1.0).abs() < 0.001); // 100% used
    assert!((budget.utilization(150.0) - 1.5).abs() < 0.001); // 150% over
}
