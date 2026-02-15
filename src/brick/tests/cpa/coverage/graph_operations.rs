use super::super::super::super::*;

// ========================
// Coverage Tests (C021-C031): ExecutionGraph operations
// ========================

/// C021: MatmulOp::tokens() method
#[test]
fn test_c021_matmul_op_tokens() {
    let op = MatmulOp::new(4, 8, 16);
    let a = vec![0.0f32; 4 * 8];
    let b = vec![0.0f32; 8 * 16];
    // tokens = m * n = 4 * 16 = 64
    assert_eq!(op.tokens(&(a, b)), 64);
}

/// C022: ExecutionGraph::add_weighted_edge()
#[test]
fn test_c022_add_weighted_edge() {
    let mut graph = ExecutionGraph::new();
    let n1 = graph.add_node(ExecutionNode::Layer { index: 0 });
    let n2 = graph.add_node(ExecutionNode::Layer { index: 1 });

    graph.add_weighted_edge(n1, n2, EdgeType::Sequence, 2.5);

    assert_eq!(graph.num_edges(), 1);
    let edges = graph.edges();
    assert!((edges[0].weight - 2.5).abs() < 0.001);
}

/// C023: ExecutionGraph::node() lookup by ID
#[test]
fn test_c023_node_by_id() {
    let mut graph = ExecutionGraph::new();
    let id = graph.add_node(ExecutionNode::Layer { index: 42 });

    let node = graph.node(id);
    assert!(node.is_some());
    if let Some(ExecutionNode::Layer { index }) = node {
        assert_eq!(*index, 42);
    } else {
        panic!("Expected Layer node");
    }

    // Non-existent ID
    let bad_id = ExecutionNodeId(999);
    assert!(graph.node(bad_id).is_none());
}

/// C024: ExecutionGraph::node_by_name() lookup
#[test]
fn test_c024_node_by_name() {
    let mut graph = ExecutionGraph::new();

    // Add a function node with a name
    let _id = graph.add_node(ExecutionNode::Function {
        name: "test_function".into(),
        file: Some("test.rs".into()),
        line: Some(100),
    });

    let result = graph.node_by_name("test_function");
    assert!(result.is_some());

    let result = graph.node_by_name("nonexistent");
    assert!(result.is_none());
}

/// C025: record_kernel_launch_with_metrics within scope
#[test]
fn test_c025_record_kernel_with_parent() {
    let mut graph = ExecutionGraph::new();

    // Create a parent scope
    let _brick = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 100,
    });

    // Record kernel within scope
    let kernel_id = graph.record_kernel_launch_with_metrics(
        "child_kernel",
        0x1234,
        (1, 1, 1),
        (32, 1, 1),
        1024,
        500,
        10.0,
        5.0,
    );

    graph.pop_scope();

    // Should have Launches edge from brick to kernel
    let edges: Vec<_> = graph
        .edges()
        .iter()
        .filter(|e| e.dst == kernel_id && matches!(e.edge_type, EdgeType::Launches))
        .collect();
    assert_eq!(edges.len(), 1, "Should have Launches edge");
}

/// C026: record_transfer within scope
#[test]
fn test_c026_record_transfer_with_parent() {
    let mut graph = ExecutionGraph::new();

    // Create a parent scope
    let _layer = graph.push_scope(ExecutionNode::Layer { index: 0 });

    // Record transfer within scope
    let transfer_id =
        graph.record_transfer("host", "device", 1024, TransferDirection::H2D, Some(100));

    graph.pop_scope();

    // Should have Contains edge from layer to transfer
    let edges: Vec<_> = graph
        .edges()
        .iter()
        .filter(|e| e.dst == transfer_id && matches!(e.edge_type, EdgeType::Contains))
        .collect();
    assert_eq!(edges.len(), 1, "Should have Contains edge");
}

/// C027: DotOp::tokens() method
#[test]
fn test_c027_dot_op_tokens() {
    let op = DotOp::new(5);
    let input = (vec![1.0; 5], vec![1.0; 5]);
    assert_eq!(op.tokens(&input), 5);
}

/// C028: SoftmaxOp::tokens() method
#[test]
fn test_c028_softmax_op_tokens() {
    let op = SoftmaxOp::new(10);
    let input = vec![1.0f32; 10];
    assert_eq!(op.tokens(&input), 10);
}

/// C029: ExecutionGraph::current_scope()
#[test]
fn test_c029_current_scope() {
    let mut graph = ExecutionGraph::new();

    // No scope initially
    assert!(graph.current_scope().is_none());

    // Push scope
    let layer_id = graph.push_scope(ExecutionNode::Layer { index: 0 });
    assert_eq!(graph.current_scope(), Some(layer_id));

    // Push another scope
    let brick_id = graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 10,
    });
    assert_eq!(graph.current_scope(), Some(brick_id));

    // Pop back
    graph.pop_scope();
    assert_eq!(graph.current_scope(), Some(layer_id));

    graph.pop_scope();
    assert!(graph.current_scope().is_none());
}

/// C030: to_dot() with Function and Transfer nodes
#[test]
fn test_c030_to_dot_function_and_transfer() {
    let mut graph = ExecutionGraph::new();

    // Add a function node
    graph.add_node(ExecutionNode::Function {
        name: "my_function".into(),
        file: Some("src/main.rs".into()),
        line: Some(42),
    });

    // Add function without file/line
    graph.add_node(ExecutionNode::Function {
        name: "anonymous".into(),
        file: None,
        line: None,
    });

    // Add transfer nodes
    graph.add_node(ExecutionNode::Transfer {
        src: "host".into(),
        dst: "device".into(),
        bytes: 1024 * 1024,
        direction: TransferDirection::H2D,
        timing_ns: Some(100),
    });

    graph.add_node(ExecutionNode::Transfer {
        src: "dev0".into(),
        dst: "dev1".into(),
        bytes: 2 * 1024 * 1024,
        direction: TransferDirection::D2D,
        timing_ns: None,
    });

    let dot = graph.to_dot();

    // Verify DOT output contains expected elements
    assert!(dot.contains("digraph"), "Should be valid digraph");
    assert!(dot.contains("my_function"), "Should contain function name");
    assert!(dot.contains("src/main.rs:42"), "Should contain file:line");
    assert!(
        dot.contains("anonymous"),
        "Should contain anonymous function"
    );
    assert!(dot.contains("H2D"), "Should contain H2D transfer");
    assert!(dot.contains("D2D"), "Should contain D2D transfer");
    assert!(dot.contains("lightsalmon"), "Transfer should have color");
    assert!(dot.contains("lightgray"), "Function should have color");
}

/// C031: to_tree_node with Function node (presentar-tui feature)
#[cfg(feature = "presentar-tui")]
#[test]
fn test_c031_to_tree_node_function() {
    let mut graph = ExecutionGraph::new();

    graph.add_node(ExecutionNode::Function {
        name: "test_func".into(),
        file: Some("test.rs".into()),
        line: Some(10),
    });

    let tree = graph.to_tree_node();
    // Just verify it doesn't panic
    assert!(!format!("{:?}", tree).is_empty());
}
