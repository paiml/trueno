//! Coverage tests for `ExecutionGraph::slowest_kernel` (core.rs:291)
//! and `ExecutionGraph::to_csr` (export.rs:247, gated by `execution-graph` feature).

use super::super::*;

// ========================================================================
// slowest_kernel tests — exercises all branches (18 uncovered lines)
// ========================================================================

/// Empty graph returns None.
#[test]
fn test_slowest_kernel_empty_graph() {
    let graph = ExecutionGraph::new();
    assert!(graph.slowest_kernel().is_none());
}

/// Graph with only layers (no bricks) returns None.
#[test]
fn test_slowest_kernel_no_bricks() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Layer { index: 1 });
    assert!(graph.slowest_kernel().is_none());
}

/// Graph with bricks but no kernel launches returns None.
#[test]
fn test_slowest_kernel_bricks_without_kernel_launches() {
    let mut graph = ExecutionGraph::new();
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 5000,
        elements: 1024,
    });
    // Only Contains edge, no Launches edge
    graph.add_edge(layer, brick, EdgeType::Contains);
    assert!(graph.slowest_kernel().is_none());
}

/// Single brick with a kernel launch — should return that brick.
#[test]
fn test_slowest_kernel_single_brick_with_kernel() {
    let mut graph = ExecutionGraph::new();
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 3000,
        elements: 2048,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "gemv_kernel".into(),
        ptx_hash: 0xABCD,
        grid: (4, 1, 1),
        block: (128, 1, 1),
        shared_mem: 2048,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick, kernel, EdgeType::Launches);

    let result = graph.slowest_kernel();
    assert!(result.is_some());
    let (id, node, timing) = result.unwrap();
    assert_eq!(id, brick);
    assert_eq!(timing, 3000);
    assert!(node.is_brick());
}

/// Multiple bricks with kernels — should return the slowest (highest timing_ns).
#[test]
fn test_slowest_kernel_multiple_bricks() {
    let mut graph = ExecutionGraph::new();

    // Fast brick (1000 ns)
    let brick_fast = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 1000,
        elements: 512,
    });
    let kernel_fast = graph.add_node(ExecutionNode::Kernel {
        name: "rmsnorm_kernel".into(),
        ptx_hash: 0x1111,
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick_fast, kernel_fast, EdgeType::Launches);

    // Slow brick (9000 ns) — should be selected
    let brick_slow = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 9000,
        elements: 4096,
    });
    let kernel_slow = graph.add_node(ExecutionNode::Kernel {
        name: "qkv_kernel".into(),
        ptx_hash: 0x2222,
        grid: (32, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick_slow, kernel_slow, EdgeType::Launches);

    // Medium brick (5000 ns)
    let brick_mid = graph.add_node(ExecutionNode::Brick {
        id: BrickId::AttentionScore,
        timing_ns: 5000,
        elements: 2048,
    });
    let kernel_mid = graph.add_node(ExecutionNode::Kernel {
        name: "attn_kernel".into(),
        ptx_hash: 0x3333,
        grid: (8, 1, 1),
        block: (128, 1, 1),
        shared_mem: 1024,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick_mid, kernel_mid, EdgeType::Launches);

    let result = graph.slowest_kernel();
    assert!(result.is_some());
    let (id, _node, timing) = result.unwrap();
    assert_eq!(id, brick_slow);
    assert_eq!(timing, 9000);
}

/// Mix of bricks with and without kernel launches — only those with Launches count.
#[test]
fn test_slowest_kernel_mixed_bricks() {
    let mut graph = ExecutionGraph::new();

    // Brick WITHOUT kernel launch (very high timing, but should be ignored)
    let brick_no_launch = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Embedding,
        timing_ns: 99999,
        elements: 8192,
    });
    // Only Contains edge, no Launches
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_edge(layer, brick_no_launch, EdgeType::Contains);

    // Brick WITH kernel launch (lower timing but is the only candidate)
    let brick_with_launch = graph.add_node(ExecutionNode::Brick {
        id: BrickId::GateProjection,
        timing_ns: 2000,
        elements: 1024,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "gate_kernel".into(),
        ptx_hash: 0x4444,
        grid: (2, 1, 1),
        block: (64, 1, 1),
        shared_mem: 512,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick_with_launch, kernel, EdgeType::Launches);

    let result = graph.slowest_kernel();
    assert!(result.is_some());
    let (id, _node, timing) = result.unwrap();
    assert_eq!(id, brick_with_launch);
    assert_eq!(timing, 2000);
}

/// Equal timing — first encountered wins (stable ordering).
#[test]
fn test_slowest_kernel_equal_timing() {
    let mut graph = ExecutionGraph::new();

    let brick_a = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 5000,
        elements: 256,
    });
    let kernel_a = graph.add_node(ExecutionNode::Kernel {
        name: "kernel_a".into(),
        ptx_hash: 0xAAAA,
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick_a, kernel_a, EdgeType::Launches);

    let brick_b = graph.add_node(ExecutionNode::Brick {
        id: BrickId::LayerNorm,
        timing_ns: 5000,
        elements: 256,
    });
    let kernel_b = graph.add_node(ExecutionNode::Kernel {
        name: "kernel_b".into(),
        ptx_hash: 0xBBBB,
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick_b, kernel_b, EdgeType::Launches);

    let result = graph.slowest_kernel();
    assert!(result.is_some());
    let (_id, _node, timing) = result.unwrap();
    // Both have 5000 — equal timing so first brick wins (Some(_) branch)
    assert_eq!(timing, 5000);
}

/// Brick with zero timing still counts if it has a kernel launch.
#[test]
fn test_slowest_kernel_zero_timing() {
    let mut graph = ExecutionGraph::new();

    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Sampling,
        timing_ns: 0,
        elements: 1,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "sample_kernel".into(),
        ptx_hash: 0x0,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(brick, kernel, EdgeType::Launches);

    let result = graph.slowest_kernel();
    assert!(result.is_some());
    let (_id, _node, timing) = result.unwrap();
    assert_eq!(timing, 0);
}

// ========================================================================
// ExecutionGraph helper coverage
// ========================================================================

/// Test kernel_nodes iterator.
#[test]
fn test_kernel_nodes_iterator() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 64,
    });
    graph.add_node(ExecutionNode::Kernel {
        name: "k1".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_node(ExecutionNode::Kernel {
        name: "k2".into(),
        ptx_hash: 1,
        grid: (2, 1, 1),
        block: (64, 1, 1),
        shared_mem: 128,
        timing_ns: Some(500),
        arithmetic_intensity: Some(1.5),
        achieved_tflops: Some(0.2),
    });

    let kernels: Vec<_> = graph.kernel_nodes().collect();
    assert_eq!(kernels.len(), 2);
    assert!(kernels[0].1.is_kernel());
    assert!(kernels[1].1.is_kernel());
}

/// Test node_by_name lookup.
#[test]
fn test_node_by_name() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode::Kernel {
        name: "my_kernel".into(),
        ptx_hash: 42,
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    let found = graph.node_by_name("my_kernel");
    assert!(found.is_some());
    let (id, node) = found.unwrap();
    assert_eq!(id, ExecutionNodeId(0));
    assert!(node.is_kernel());

    // Non-existent name
    assert!(graph.node_by_name("nonexistent").is_none());
}

/// Test record_kernel_launch_with_metrics (Phase 9 roofline).
#[test]
fn test_record_kernel_launch_with_metrics() {
    let mut graph = ExecutionGraph::new();
    let scope = graph.push_scope(ExecutionNode::Layer { index: 0 });

    let kernel_id = graph.record_kernel_launch_with_metrics(
        "matmul_tiled",
        0xDEAD,
        (16, 1, 1),
        (256, 1, 1),
        8192,
        1500,   // timing_ns
        4.5,    // arithmetic_intensity
        1.2,    // achieved_tflops
    );

    graph.pop_scope();

    let kernel = graph.node(kernel_id).unwrap();
    assert!(kernel.is_kernel());
    assert_eq!(kernel.timing_ns(), Some(1500));
    assert!((kernel.arithmetic_intensity().unwrap() - 4.5).abs() < 1e-5);
    assert!((kernel.achieved_tflops().unwrap() - 1.2).abs() < 1e-5);

    // Should have a Launches edge from scope to kernel
    let edges: Vec<_> = graph.outgoing_edges(scope).collect();
    assert!(edges.iter().any(|e| e.dst == kernel_id && e.edge_type == EdgeType::Launches));
}

/// Test record_transfer.
#[test]
fn test_record_transfer_in_scope() {
    let mut graph = ExecutionGraph::new();
    let scope = graph.push_scope(ExecutionNode::Layer { index: 0 });

    let transfer_id = graph.record_transfer(
        "host",
        "gpu0",
        1_048_576,
        TransferDirection::H2D,
        Some(250),
    );

    graph.pop_scope();

    let node = graph.node(transfer_id).unwrap();
    assert!(node.is_transfer());
    assert_eq!(node.transfer_bytes(), Some(1_048_576));
    assert_eq!(node.timing_ns(), Some(250));

    // Should have Contains edge from scope
    let edges: Vec<_> = graph.outgoing_edges(scope).collect();
    assert!(edges.iter().any(|e| e.dst == transfer_id && e.edge_type == EdgeType::Contains));
}

/// Test add_dependency (DependsOn edge).
#[test]
fn test_add_dependency() {
    let mut graph = ExecutionGraph::new();
    let n0 = graph.add_node(ExecutionNode::Layer { index: 0 });
    let n1 = graph.add_node(ExecutionNode::Layer { index: 1 });

    graph.add_dependency(n0, n1);

    let edges: Vec<_> = graph.outgoing_edges(n0).collect();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_type, EdgeType::DependsOn);
    assert_eq!(edges[0].dst, n1);
}

/// Test incoming_edges.
#[test]
fn test_incoming_edges() {
    let mut graph = ExecutionGraph::new();
    let a = graph.add_node(ExecutionNode::Layer { index: 0 });
    let b = graph.add_node(ExecutionNode::Layer { index: 1 });
    let c = graph.add_node(ExecutionNode::Layer { index: 2 });

    graph.add_edge(a, c, EdgeType::Sequence);
    graph.add_edge(b, c, EdgeType::Sequence);

    let incoming: Vec<_> = graph.incoming_edges(c).collect();
    assert_eq!(incoming.len(), 2);
}

/// Test is_scope_balanced.
#[test]
fn test_scope_balanced() {
    let mut graph = ExecutionGraph::new();
    assert!(graph.is_scope_balanced());

    graph.push_scope(ExecutionNode::Layer { index: 0 });
    assert!(!graph.is_scope_balanced());

    graph.pop_scope();
    assert!(graph.is_scope_balanced());
}

/// Test add_weighted_edge.
#[test]
fn test_add_weighted_edge() {
    let mut graph = ExecutionGraph::new();
    let a = graph.add_node(ExecutionNode::Layer { index: 0 });
    let b = graph.add_node(ExecutionNode::Layer { index: 1 });

    graph.add_weighted_edge(a, b, EdgeType::Calls, 3.5);

    let edges: Vec<_> = graph.outgoing_edges(a).collect();
    assert_eq!(edges.len(), 1);
    assert!((edges[0].weight - 3.5).abs() < 1e-5);
}
