use super::super::*;

#[test]
fn test_token_budget_from_latency() {
    let budget = TokenBudget::from_latency(50.0);
    assert!((budget.us_per_token - 50.0).abs() < 0.001);
    assert!((budget.tokens_per_sec - 20_000.0).abs() < 1.0);
}

#[test]
fn test_token_budget_from_throughput() {
    let budget = TokenBudget::from_throughput(20_000.0);
    assert!((budget.us_per_token - 50.0).abs() < 0.001);
    assert!((budget.tokens_per_sec - 20_000.0).abs() < 1.0);
}

#[test]
fn test_token_budget_is_met() {
    let budget = TokenBudget::from_latency(50.0);
    assert!(budget.is_met(40.0)); // Under budget
    assert!(budget.is_met(50.0)); // Exactly at budget
    assert!(!budget.is_met(60.0)); // Over budget
}

#[test]
fn test_dot_op() {
    let op = DotOp::new(4);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    assert!((result - 70.0).abs() < 0.001); // 1*5 + 2*6 + 3*7 + 4*8 = 70
}

#[test]
fn test_add_op() {
    let op = AddOp::new(4);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_matmul_op() {
    let op = MatmulOp::new(2, 2, 2);
    // A = [[1, 2], [3, 4]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    // B = [[5, 6], [7, 8]]
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar).unwrap();
    // C = [[19, 22], [43, 50]]
    assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_softmax_op() {
    let op = SoftmaxOp::new(3);
    let input = vec![1.0, 2.0, 3.0];
    let result = op.execute(input, Backend::Scalar).unwrap();
    // Sum should be 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
    // Values should be increasing
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
}
