//! Execution Graph and Brick Profiling Types
//!
//! This module contains types for execution path tracking and profiling:
//!
//! - **PAR-073**: BrickSample, BrickBottleneck - foundational profiling primitives
//! - **PAR-200**: BrickId, BrickCategory, SyncMode - O(1) hot path brick identification
//! - **PAR-201**: ExecutionGraph, ExecutionNode, etc. - full execution hierarchy tracking

mod node;
mod traversal;

pub use node::{
    BrickBottleneck, BrickCategory, BrickId, BrickSample, BrickStats, CategoryStats, EdgeType,
    ExecutionEdge, ExecutionNode, ExecutionNodeId, PtxRegistry, SyncMode, TransferDirection,
};
pub use traversal::ExecutionGraph;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brick_id_category() {
        assert_eq!(BrickId::RmsNorm.category(), BrickCategory::Norm);
        assert_eq!(BrickId::LayerNorm.category(), BrickCategory::Norm);
        assert_eq!(BrickId::QkvProjection.category(), BrickCategory::Attention);
        assert_eq!(BrickId::GateProjection.category(), BrickCategory::Ffn);
        assert_eq!(BrickId::Embedding.category(), BrickCategory::Other);
    }

    #[test]
    fn test_brick_id_name() {
        assert_eq!(BrickId::RmsNorm.name(), "RmsNorm");
        assert_eq!(BrickId::QkvProjection.name(), "QkvProjection");
    }

    #[test]
    fn test_brick_id_from_str() {
        assert_eq!(BrickId::from_str("RmsNorm"), Some(BrickId::RmsNorm));
        assert_eq!(BrickId::from_str("Qkv"), Some(BrickId::QkvProjection));
        assert_eq!(BrickId::from_str("RoPE"), Some(BrickId::RopeEmbedding));
        assert_eq!(BrickId::from_str("Unknown"), None);
    }

    #[test]
    fn test_brick_id_display() {
        assert_eq!(format!("{}", BrickId::RmsNorm), "RmsNorm");
    }

    #[test]
    fn test_brick_category_name() {
        assert_eq!(BrickCategory::Norm.name(), "Norm");
        assert_eq!(BrickCategory::Ffn.name(), "FFN");
    }

    #[test]
    fn test_brick_bottleneck_display() {
        assert_eq!(format!("{}", BrickBottleneck::Memory), "memory");
        assert_eq!(format!("{}", BrickBottleneck::Compute), "compute");
    }

    #[test]
    fn test_execution_graph_basic() {
        let mut graph = ExecutionGraph::new();
        let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
        let brick = graph.add_node(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 1000,
            elements: 4096,
        });
        graph.add_edge(layer, brick, EdgeType::Contains);

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
    }

    #[test]
    fn test_execution_node_name() {
        let brick = ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 1000,
            elements: 4096,
        };
        assert_eq!(brick.name(), "RmsNorm");

        let layer = ExecutionNode::Layer { index: 5 };
        assert_eq!(layer.name(), "Layer5");
    }

    #[test]
    fn test_execution_graph_scopes() {
        let mut graph = ExecutionGraph::new();
        let layer = graph.push_scope(ExecutionNode::Layer { index: 0 });
        let brick = graph.add_node_in_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 1000,
            elements: 4096,
        });
        graph.pop_scope();

        assert_eq!(graph.num_nodes(), 2);
        // Should have a Contains edge from layer to brick
        let edges: Vec<_> = graph.outgoing_edges(layer).collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].dst, brick);
    }

    #[test]
    fn test_brick_stats_basic() {
        let mut stats = BrickStats::new("test_brick");
        stats.add_sample(1000, 100);
        stats.add_sample(2000, 200);

        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_ns, 3000);
        assert_eq!(stats.total_elements, 300);
        assert_eq!(stats.min_ns, 1000);
        assert_eq!(stats.max_ns, 2000);
    }

    #[test]
    fn test_category_stats_percentage() {
        let stats = CategoryStats {
            total_ns: 250,
            total_elements: 1000,
            count: 10,
        };
        assert!((stats.percentage(1000) - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_ptx_registry() {
        let mut registry = PtxRegistry::new();
        registry.register("test_kernel", ".version 8.0\n.entry test {}", None);

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        let hash = PtxRegistry::hash_ptx(".version 8.0\n.entry test {}");
        assert_eq!(registry.lookup_name(hash), Some("test_kernel"));
    }

    #[test]
    fn test_transfer_direction() {
        let node = ExecutionNode::Transfer {
            src: "host".to_string(),
            dst: "device".to_string(),
            bytes: 1024,
            direction: TransferDirection::H2D,
            timing_ns: Some(100),
        };
        assert!(node.is_transfer());
        assert_eq!(node.transfer_bytes(), Some(1024));
    }

    #[test]
    fn test_execution_graph_to_dot() {
        let mut graph = ExecutionGraph::new();
        graph.add_node(ExecutionNode::Layer { index: 0 });
        let dot = graph.to_dot();
        assert!(dot.contains("digraph ExecutionGraph"));
        assert!(dot.contains("Layer 0"));
    }

    #[test]
    fn test_execution_graph_to_ascii_tree() {
        let mut graph = ExecutionGraph::new();
        graph.push_scope(ExecutionNode::Layer { index: 0 });
        graph.add_node_in_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 1000,
            elements: 4096,
        });
        graph.pop_scope();

        let tree = graph.to_ascii_tree();
        assert!(tree.contains("Layer 0"));
        assert!(tree.contains("RmsNorm"));
    }

    // ================================================================
    // Comprehensive to_dot() coverage - all node types and edge types
    // ================================================================

    #[test]
    fn test_to_dot_all_node_types() {
        let mut graph = ExecutionGraph::new();

        // Layer node
        graph.add_node(ExecutionNode::Layer { index: 0 });

        // Brick node
        graph.add_node(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 5000,
            elements: 2048,
        });

        // Kernel node
        graph.add_node(ExecutionNode::Kernel {
            name: "batched_q4k_gemv".into(),
            ptx_hash: 0xDEADBEEF,
            grid: (32, 1, 1),
            block: (256, 1, 1),
            shared_mem: 4096,
            timing_ns: Some(1500),
            arithmetic_intensity: None,
            achieved_tflops: None,
        });

        // Function node with file/line
        graph.add_node(ExecutionNode::Function {
            name: "forward_pass".into(),
            file: Some("model.rs".into()),
            line: Some(42),
        });

        // Function node without file/line
        graph.add_node(ExecutionNode::Function {
            name: "dispatch".into(),
            file: None,
            line: None,
        });

        // Transfer H2D
        graph.add_node(ExecutionNode::Transfer {
            src: "host".into(),
            dst: "gpu0".into(),
            bytes: 1_000_000,
            direction: TransferDirection::H2D,
            timing_ns: Some(500),
        });

        // Transfer D2H
        graph.add_node(ExecutionNode::Transfer {
            src: "gpu0".into(),
            dst: "host".into(),
            bytes: 2_000_000,
            direction: TransferDirection::D2H,
            timing_ns: None,
        });

        // Transfer D2D
        graph.add_node(ExecutionNode::Transfer {
            src: "gpu0".into(),
            dst: "gpu1".into(),
            bytes: 500_000,
            direction: TransferDirection::D2D,
            timing_ns: Some(200),
        });

        // AsyncTask
        graph.add_node(ExecutionNode::AsyncTask {
            name: "prefetch_weights".into(),
            poll_count: 5,
            yield_count: 3,
            total_poll_ns: 10_000,
        });

        // AsyncTask with zero polls
        graph.add_node(ExecutionNode::AsyncTask {
            name: "idle_task".into(),
            poll_count: 0,
            yield_count: 0,
            total_poll_ns: 0,
        });

        let dot = graph.to_dot();

        // Structure
        assert!(dot.starts_with("digraph ExecutionGraph {\n"));
        assert!(dot.ends_with("}\n"));
        assert!(dot.contains("rankdir=TB"));

        // Layer
        assert!(dot.contains("Layer 0"));
        assert!(dot.contains("fillcolor=lightblue"));

        // Brick
        assert!(dot.contains("RmsNorm"));
        assert!(dot.contains("fillcolor=lightgreen"));

        // Kernel
        assert!(dot.contains("batched_q4k_gemv"));
        assert!(dot.contains("<<<32,256,1>>>"));
        assert!(dot.contains("fillcolor=lightyellow"));

        // Function with location
        assert!(dot.contains("forward_pass"));
        assert!(dot.contains("model.rs:42"));
        assert!(dot.contains("fillcolor=lightgray"));

        // Function without location
        assert!(dot.contains("dispatch"));

        // Transfers
        assert!(dot.contains("H2D"));
        assert!(dot.contains("D2H"));
        assert!(dot.contains("D2D"));
        assert!(dot.contains("fillcolor=lightsalmon"));

        // AsyncTask
        assert!(dot.contains("prefetch_weights"));
        assert!(dot.contains("polls:5"));
        assert!(dot.contains("yields:3"));
        assert!(dot.contains("fillcolor=lightcyan"));
    }

    #[test]
    fn test_to_dot_all_edge_types() {
        let mut graph = ExecutionGraph::new();

        let n0 = graph.add_node(ExecutionNode::Layer { index: 0 });
        let n1 = graph.add_node(ExecutionNode::Layer { index: 1 });
        let n2 = graph.add_node(ExecutionNode::Layer { index: 2 });
        let n3 = graph.add_node(ExecutionNode::Layer { index: 3 });
        let n4 = graph.add_node(ExecutionNode::Layer { index: 4 });
        let n5 = graph.add_node(ExecutionNode::Layer { index: 5 });

        graph.add_edge(n0, n1, EdgeType::Calls);
        graph.add_edge(n0, n2, EdgeType::Contains);
        graph.add_edge(n2, n3, EdgeType::Launches);
        graph.add_edge(n3, n4, EdgeType::Sequence);
        graph.add_edge(n4, n5, EdgeType::DependsOn);
        graph.add_edge(
            n1,
            n5,
            EdgeType::Transfer {
                bytes: 1024,
                direction: TransferDirection::H2D,
            },
        );

        let dot = graph.to_dot();

        assert!(dot.contains("style=solid"));
        assert!(dot.contains("style=dashed"));
        assert!(dot.contains("style=bold,color=red"));
        assert!(dot.contains("style=dotted"));
        assert!(dot.contains("style=solid,color=blue"));
        assert!(dot.contains("style=bold,color=orange"));
    }

    #[test]
    fn test_to_dot_empty_graph() {
        let graph = ExecutionGraph::new();
        let dot = graph.to_dot();
        assert!(dot.contains("digraph ExecutionGraph"));
        assert!(dot.contains("}\n"));
        // No nodes or edges
        assert!(!dot.contains("n0"));
    }

    // ================================================================
    // Comprehensive to_ascii_tree() coverage
    // ================================================================

    #[test]
    fn test_to_ascii_tree_empty_graph() {
        let graph = ExecutionGraph::new();
        let tree = graph.to_ascii_tree();
        assert_eq!(tree, "(empty graph)");
    }

    #[test]
    fn test_to_ascii_tree_single_node() {
        let mut graph = ExecutionGraph::new();
        graph.add_node(ExecutionNode::Layer { index: 7 });
        let tree = graph.to_ascii_tree();
        assert!(tree.contains("Layer 7"));
    }

    #[test]
    fn test_to_ascii_tree_multiple_roots() {
        let mut graph = ExecutionGraph::new();
        graph.add_node(ExecutionNode::Layer { index: 0 });
        graph.add_node(ExecutionNode::Layer { index: 1 });
        graph.add_node(ExecutionNode::Layer { index: 2 });

        let tree = graph.to_ascii_tree();
        assert!(tree.contains("Execution Graph"));
        assert!(tree.contains("Layer 0"));
        assert!(tree.contains("Layer 1"));
        assert!(tree.contains("Layer 2"));
        // Should have tree connectors
        assert!(tree.contains("├──") || tree.contains("└──"));
    }

    #[test]
    fn test_to_ascii_tree_all_node_types() {
        let mut graph = ExecutionGraph::new();

        let root = graph.add_node(ExecutionNode::Layer { index: 0 });

        let brick = graph.add_node(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 3000,
            elements: 1024,
        });
        graph.add_edge(root, brick, EdgeType::Contains);

        let kernel = graph.add_node(ExecutionNode::Kernel {
            name: "gemv_kernel".into(),
            ptx_hash: 0x1234,
            grid: (4, 1, 1),
            block: (128, 1, 1),
            shared_mem: 2048,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        });
        graph.add_edge(brick, kernel, EdgeType::Launches);

        let func = graph.add_node(ExecutionNode::Function {
            name: "compute".into(),
            file: Some("ops.rs".into()),
            line: Some(100),
        });
        graph.add_edge(root, func, EdgeType::Contains);

        let func_no_loc = graph.add_node(ExecutionNode::Function {
            name: "init".into(),
            file: None,
            line: None,
        });
        graph.add_edge(root, func_no_loc, EdgeType::Contains);

        let transfer = graph.add_node(ExecutionNode::Transfer {
            src: "cpu".into(),
            dst: "gpu".into(),
            bytes: 8192,
            direction: TransferDirection::H2D,
            timing_ns: Some(250),
        });
        graph.add_edge(root, transfer, EdgeType::Contains);

        let transfer_no_timing = graph.add_node(ExecutionNode::Transfer {
            src: "gpu".into(),
            dst: "cpu".into(),
            bytes: 4096,
            direction: TransferDirection::D2H,
            timing_ns: None,
        });
        graph.add_edge(root, transfer_no_timing, EdgeType::Contains);

        let async_task = graph.add_node(ExecutionNode::AsyncTask {
            name: "prefetch".into(),
            poll_count: 3,
            yield_count: 1,
            total_poll_ns: 5000,
        });
        graph.add_edge(root, async_task, EdgeType::Contains);

        let async_zero = graph.add_node(ExecutionNode::AsyncTask {
            name: "idle".into(),
            poll_count: 0,
            yield_count: 0,
            total_poll_ns: 0,
        });
        graph.add_edge(root, async_zero, EdgeType::Contains);

        let tree = graph.to_ascii_tree();

        assert!(tree.contains("Layer 0"));
        assert!(tree.contains("QkvProjection"));
        assert!(tree.contains("gemv_kernel"));
        assert!(tree.contains("compute (ops.rs:100)"));
        assert!(tree.contains("init"));
        assert!(tree.contains("H2D: cpu"));
        assert!(tree.contains("D2H: gpu"));
        assert!(tree.contains("prefetch"));
        assert!(tree.contains("idle"));
        // Brick should show timing info
        assert!(tree.contains("µs"));
        // Kernel should show launch config
        assert!(tree.contains("<<<"));
    }

    #[test]
    fn test_to_ascii_tree_deep_hierarchy() {
        let mut graph = ExecutionGraph::new();
        let _l0 = graph.push_scope(ExecutionNode::Layer { index: 0 });
        let _b1 = graph.add_node_in_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100,
            elements: 64,
        });
        let _b2 = graph.add_node_in_scope(ExecutionNode::Brick {
            id: BrickId::AttentionScore,
            timing_ns: 200,
            elements: 128,
        });
        graph.pop_scope();

        let tree = graph.to_ascii_tree();
        assert!(tree.contains("Layer 0"));
        assert!(tree.contains("RmsNorm"));
        assert!(tree.contains("AttentionScore"));
        // Both children should have tree connectors
        assert!(tree.contains("├──") || tree.contains("└──"));
    }

    #[test]
    fn test_to_ascii_tree_nested_launches() {
        let mut graph = ExecutionGraph::new();

        let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
        let brick = graph.add_node(ExecutionNode::Brick {
            id: BrickId::GateProjection,
            timing_ns: 5000,
            elements: 2048,
        });
        graph.add_edge(layer, brick, EdgeType::Contains);

        let k1 = graph.add_node(ExecutionNode::Kernel {
            name: "kernel_a".into(),
            ptx_hash: 0,
            grid: (1, 1, 1),
            block: (64, 1, 1),
            shared_mem: 0,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        });
        graph.add_edge(brick, k1, EdgeType::Launches);

        let k2 = graph.add_node(ExecutionNode::Kernel {
            name: "kernel_b".into(),
            ptx_hash: 0,
            grid: (2, 1, 1),
            block: (32, 1, 1),
            shared_mem: 512,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        });
        graph.add_edge(brick, k2, EdgeType::Launches);

        let tree = graph.to_ascii_tree();
        assert!(tree.contains("Layer 0"));
        assert!(tree.contains("GateProjection"));
        assert!(tree.contains("kernel_a"));
        assert!(tree.contains("kernel_b"));
    }

    #[test]
    fn test_brick_stats_cycles() {
        let mut stats = BrickStats::new("test");
        stats.add_sample_with_cycles(1000, 100, 3000);

        assert_eq!(stats.total_cycles, 3000);
        assert!((stats.cycles_per_element() - 30.0).abs() < 0.001);
    }

    // FALSIFICATION TESTS

    /// FALSIFICATION TEST: BrickId round-trip via name
    #[test]
    fn test_falsify_brick_id_round_trip() {
        for brick_id in [
            BrickId::RmsNorm,
            BrickId::LayerNorm,
            BrickId::QkvProjection,
            BrickId::RopeEmbedding,
            BrickId::AttentionScore,
            BrickId::AttentionSoftmax,
            BrickId::AttentionOutput,
            BrickId::OutputProjection,
            BrickId::GateProjection,
            BrickId::UpProjection,
            BrickId::Activation,
            BrickId::DownProjection,
            BrickId::Embedding,
            BrickId::LmHead,
            BrickId::Sampling,
        ] {
            let name = brick_id.name();
            let parsed = BrickId::from_str(name);
            assert_eq!(
                parsed,
                Some(brick_id),
                "FALSIFICATION FAILED: BrickId::{:?}.name() = {:?} does not round-trip",
                brick_id,
                name
            );
        }
    }

    /// FALSIFICATION TEST: ExecutionGraph maintains node/edge count consistency
    #[test]
    fn test_falsify_graph_consistency() {
        let mut graph = ExecutionGraph::new();

        // Add nodes and edges
        let n1 = graph.add_node(ExecutionNode::Layer { index: 0 });
        let n2 = graph.add_node(ExecutionNode::Layer { index: 1 });
        graph.add_edge(n1, n2, EdgeType::Sequence);

        assert_eq!(
            graph.num_nodes(),
            2,
            "FALSIFICATION FAILED: node count mismatch"
        );
        assert_eq!(
            graph.num_edges(),
            1,
            "FALSIFICATION FAILED: edge count mismatch"
        );

        // Clear and verify
        graph.clear();
        assert_eq!(
            graph.num_nodes(),
            0,
            "FALSIFICATION FAILED: clear did not reset nodes"
        );
        assert_eq!(
            graph.num_edges(),
            0,
            "FALSIFICATION FAILED: clear did not reset edges"
        );
    }

    /// FALSIFICATION TEST: BrickStats min/max tracking
    #[test]
    fn test_falsify_brick_stats_minmax() {
        let mut stats = BrickStats::new("test");

        for ns in [1000u64, 500, 2000, 750, 1500] {
            stats.add_sample(ns, 100);
        }

        assert_eq!(
            stats.min_ns, 500,
            "FALSIFICATION FAILED: min_ns should be 500, got {}",
            stats.min_ns
        );
        assert_eq!(
            stats.max_ns, 2000,
            "FALSIFICATION FAILED: max_ns should be 2000, got {}",
            stats.max_ns
        );
    }
}
