use super::super::super::*;

// ========================
// Coverage Tests (C001-C020)
// ========================

/// C001: ComputeAssertion::equiv creates equivalence with default tolerance
#[test]
fn test_c001_compute_assertion_equiv() {
    let assertion = ComputeAssertion::equiv(Backend::Scalar);
    if let ComputeAssertion::Equivalence {
        baseline,
        tolerance,
    } = assertion
    {
        assert_eq!(baseline, Backend::Scalar);
        assert!((tolerance - 1e-5).abs() < 1e-10);
    } else {
        panic!("Expected Equivalence assertion");
    }
}

/// C002: assert_equiv builder method
#[test]
fn test_c002_compute_brick_assert_equiv() {
    let brick = ComputeBrick::new(AddOp::new(4)).assert_equiv(Backend::Scalar);
    // Verify assertion was added
    assert!(!brick.assertions.is_empty());
}

/// C003: BrickId Display trait
#[test]
fn test_c003_brick_id_display() {
    let id = BrickId::QkvProjection;
    let display = format!("{}", id);
    assert_eq!(display, "QkvProjection");

    let id2 = BrickId::RmsNorm;
    assert_eq!(format!("{}", id2), "RmsNorm");
}

/// C004: BrickCategory::name() all variants
#[test]
fn test_c004_brick_category_name() {
    assert_eq!(BrickCategory::Norm.name(), "Norm");
    assert_eq!(BrickCategory::Attention.name(), "Attention");
    assert_eq!(BrickCategory::Ffn.name(), "FFN");
    assert_eq!(BrickCategory::Other.name(), "Other");
}

/// C005: BrickCategory Display trait
#[test]
fn test_c005_brick_category_display() {
    assert_eq!(format!("{}", BrickCategory::Norm), "Norm");
    assert_eq!(format!("{}", BrickCategory::Attention), "Attention");
    assert_eq!(format!("{}", BrickCategory::Ffn), "FFN");
    assert_eq!(format!("{}", BrickCategory::Other), "Other");
}

/// C006: ExecutionNode::name() all variants
#[test]
fn test_c006_execution_node_name() {
    let layer = ExecutionNode::Layer { index: 5 };
    assert_eq!(layer.name(), "Layer5");

    let brick = ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 100,
        elements: 10,
    };
    assert_eq!(brick.name(), "GateProjection");

    let kernel = ExecutionNode::Kernel {
        name: "my_kernel".into(),
        ptx_hash: 0x123,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    };
    assert_eq!(kernel.name(), "my_kernel");

    let func = ExecutionNode::Function {
        name: "my_func".into(),
        file: Some("test.rs".into()),
        line: Some(42),
    };
    assert_eq!(func.name(), "my_func");

    // Transfer variants
    let h2d = ExecutionNode::Transfer {
        src: "host".into(),
        dst: "device".into(),
        bytes: 1024,
        direction: TransferDirection::H2D,
        timing_ns: None,
    };
    assert_eq!(h2d.name(), "H2D:host->device");

    let d2h = ExecutionNode::Transfer {
        src: "device".into(),
        dst: "host".into(),
        bytes: 1024,
        direction: TransferDirection::D2H,
        timing_ns: None,
    };
    assert_eq!(d2h.name(), "D2H:device->host");

    let d2d = ExecutionNode::Transfer {
        src: "dev0".into(),
        dst: "dev1".into(),
        bytes: 1024,
        direction: TransferDirection::D2D,
        timing_ns: None,
    };
    assert_eq!(d2d.name(), "D2D:dev0->dev1");
}

/// C007: ExecutionNode::is_transfer()
#[test]
fn test_c007_execution_node_is_transfer() {
    let transfer = ExecutionNode::Transfer {
        src: "a".into(),
        dst: "b".into(),
        bytes: 100,
        direction: TransferDirection::H2D,
        timing_ns: None,
    };
    assert!(transfer.is_transfer());

    let brick = ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 10,
    };
    assert!(!brick.is_transfer());
}

/// C008: ExecutionNode::timing_ns() all variants
#[test]
fn test_c008_execution_node_timing_ns() {
    let brick = ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 12345,
        elements: 10,
    };
    assert_eq!(brick.timing_ns(), Some(12345));

    let kernel = ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: Some(67890),
        arithmetic_intensity: None,
        achieved_tflops: None,
    };
    assert_eq!(kernel.timing_ns(), Some(67890));

    let transfer = ExecutionNode::Transfer {
        src: "a".into(),
        dst: "b".into(),
        bytes: 100,
        direction: TransferDirection::H2D,
        timing_ns: Some(11111),
    };
    assert_eq!(transfer.timing_ns(), Some(11111));

    let layer = ExecutionNode::Layer { index: 0 };
    assert_eq!(layer.timing_ns(), None);

    let func = ExecutionNode::Function {
        name: "f".into(),
        file: None,
        line: None,
    };
    assert_eq!(func.timing_ns(), None);
}

/// C009: ExecutionNode::ptx_hash()
#[test]
fn test_c009_execution_node_ptx_hash() {
    let kernel = ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0xDEADBEEF,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    };
    assert_eq!(kernel.ptx_hash(), Some(0xDEADBEEF));

    let brick = ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 10,
    };
    assert_eq!(brick.ptx_hash(), None);
}

/// C010: ExecutionNode::arithmetic_intensity() and achieved_tflops()
#[test]
fn test_c010_execution_node_roofline_accessors() {
    let kernel = ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: Some(1000),
        arithmetic_intensity: Some(50.0),
        achieved_tflops: Some(10.5),
    };
    assert_eq!(kernel.arithmetic_intensity(), Some(50.0));
    assert_eq!(kernel.achieved_tflops(), Some(10.5));

    let brick = ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 10,
    };
    assert_eq!(brick.arithmetic_intensity(), None);
    assert_eq!(brick.achieved_tflops(), None);
}

/// C011: ExecutionNode::transfer_bytes()
#[test]
fn test_c011_execution_node_transfer_bytes() {
    let transfer = ExecutionNode::Transfer {
        src: "a".into(),
        dst: "b".into(),
        bytes: 1024 * 1024,
        direction: TransferDirection::H2D,
        timing_ns: None,
    };
    assert_eq!(transfer.transfer_bytes(), Some(1024 * 1024));

    let brick = ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 10,
    };
    assert_eq!(brick.transfer_bytes(), None);
}

/// C012: AddOp::tokens() method
#[test]
fn test_c012_add_op_tokens() {
    let op = AddOp::new(3);
    let input = (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
    assert_eq!(op.tokens(&input), 3);
}

/// C013: MatmulOp::name() method
#[test]
fn test_c013_matmul_op_name() {
    let op = MatmulOp::new(4, 4, 4);
    assert_eq!(op.name(), "matmul");
}

/// C014: DotOp::name() method
#[test]
fn test_c014_dot_op_name() {
    let op = DotOp::new(4);
    assert_eq!(op.name(), "dot");
}

/// C015: SoftmaxOp::name() method
#[test]
fn test_c015_softmax_op_name() {
    let op = SoftmaxOp::new(4);
    assert_eq!(op.name(), "softmax");
}

/// C016: Zero elapsed time edge case (infinity tokens/sec)
#[test]
fn test_c016_zero_elapsed_time() {
    // This tests the f64::INFINITY case in run()
    // We can't easily trigger this without mocking time, but we verify
    // the budget calculation handles extreme cases
    let budget = TokenBudget::from_throughput(f64::MAX);
    assert!(budget.us_per_token < 1e-10);
}

/// C017: ComputeOp::clone_input() default implementation
#[test]
fn test_c017_clone_input_default() {
    let op = AddOp::new(2);
    let input = (vec![1.0, 2.0], vec![3.0, 4.0]);
    let cloned = op.clone_input(&input);
    assert!(cloned.is_some());
    let cloned = cloned.unwrap();
    assert_eq!(cloned.0, input.0);
    assert_eq!(cloned.1, input.1);
}

/// C018: EdgeType debug formatting
#[test]
fn test_c018_edge_type_debug() {
    let depends = EdgeType::DependsOn;
    let debug_str = format!("{:?}", depends);
    assert!(debug_str.contains("DependsOn"));

    let transfer = EdgeType::Transfer {
        bytes: 1024,
        direction: TransferDirection::H2D,
    };
    let debug_str = format!("{:?}", transfer);
    assert!(debug_str.contains("Transfer"));
    assert!(debug_str.contains("1024"));
}

/// C019: TransferDirection debug and clone
#[test]
fn test_c019_transfer_direction_traits() {
    let dir = TransferDirection::D2D;
    let cloned = dir;
    assert_eq!(dir, cloned);

    let debug_str = format!("{:?}", dir);
    assert!(debug_str.contains("D2D"));
}

/// C020: ExecutionNodeId hash and ordering
#[test]
fn test_c020_execution_node_id_traits() {
    use std::collections::HashSet;

    let id1 = ExecutionNodeId(1);
    let id2 = ExecutionNodeId(2);
    let id1_copy = ExecutionNodeId(1);

    assert_eq!(id1, id1_copy);
    assert_ne!(id1, id2);

    let mut set = HashSet::new();
    set.insert(id1);
    set.insert(id2);
    set.insert(id1_copy);
    assert_eq!(set.len(), 2);
}

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
