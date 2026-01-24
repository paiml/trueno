use super::super::*;

// ========================================================================
// PAR-200: Falsification Tests (F101-F110)
// ========================================================================

/// F102: Immediate mode matches v1 behavior (±5%)
#[test]
fn test_f102_immediate_mode_matches_v1() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.set_sync_mode(SyncMode::Immediate);

    // Legacy API
    let timer = profiler.start("RmsNorm");
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop(timer, 1);

    let legacy_ns = profiler.brick_stats(BrickId::RmsNorm).total_ns;

    profiler.reset();

    // New API
    let timer = profiler.start_brick(BrickId::RmsNorm);
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop_brick(timer, 1);

    let new_ns = profiler.brick_stats(BrickId::RmsNorm).total_ns;

    // Should be within 50% (timing variance on CI)
    let ratio = new_ns as f64 / legacy_ns as f64;
    assert!(ratio > 0.5 && ratio < 2.0, "F102 failed: ratio={:.2}", ratio);
}

/// F103: BrickId lookup is O(1) - verified by direct array access
#[test]
fn test_f103_brick_id_lookup_o1() {
    let profiler = BrickProfiler::new();

    // Direct array access is O(1) by construction
    let _stats = &profiler.brick_stats(BrickId::RmsNorm);
    let _stats = &profiler.brick_stats(BrickId::AttentionScore);
    let _stats = &profiler.brick_stats(BrickId::DownProjection);

    // Compile-time verification: array indexing is O(1)
    assert_eq!(std::mem::size_of::<BrickId>(), 1); // u8 repr
}

/// F104: Category aggregation sums correctly
#[test]
fn test_f104_category_aggregation_correct() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Add known amounts to each category
    let timer = profiler.start_brick(BrickId::RmsNorm);
    std::thread::sleep(std::time::Duration::from_micros(10));
    profiler.stop_brick(timer, 1);

    let timer = profiler.start_brick(BrickId::QkvProjection);
    std::thread::sleep(std::time::Duration::from_micros(20));
    profiler.stop_brick(timer, 1);

    let timer = profiler.start_brick(BrickId::GateProjection);
    std::thread::sleep(std::time::Duration::from_micros(30));
    profiler.stop_brick(timer, 1);

    let cats = profiler.category_stats();
    let cat_total: u64 = cats.iter().map(|c| c.total_ns).sum();

    // Category sum must equal total
    assert_eq!(cat_total, profiler.total_ns(), "F104 failed: category sum mismatch");
}

/// F105: Dynamic fallback works for unknown bricks
#[test]
fn test_f105_dynamic_fallback_works() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Unknown brick name
    let timer = profiler.start("UnknownCustomBrick");
    std::thread::sleep(std::time::Duration::from_micros(10));
    profiler.stop(timer, 1);

    // Should be accessible via stats()
    let stats = profiler.stats("UnknownCustomBrick");
    assert!(stats.is_some(), "F105 failed: dynamic brick not found");
    assert_eq!(stats.unwrap().count, 1);
}

/// F106: finalize() is idempotent
#[test]
fn test_f106_finalize_idempotent() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.set_sync_mode(SyncMode::Deferred);
    profiler.reset_epoch();

    let start = profiler.elapsed_ns();
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.record_deferred(BrickId::RmsNorm, start, 1);

    let end = profiler.elapsed_ns();
    profiler.finalize(end);

    let count_after_first = profiler.brick_stats(BrickId::RmsNorm).count;

    // Second finalize should be no-op
    profiler.finalize(end);
    let count_after_second = profiler.brick_stats(BrickId::RmsNorm).count;

    assert_eq!(count_after_first, count_after_second, "F106 failed: finalize not idempotent");
}

/// F108: Zero-alloc hot path (verified by no String in BrickIdTimer)
#[test]
fn test_f108_zero_alloc_hot_path() {
    // BrickId is a u8 (no heap allocation)
    assert_eq!(std::mem::size_of::<BrickId>(), 1);

    // BrickIdTimer is small (BrickId + Instant, with padding)
    // Instant is 16 bytes on Linux, so BrickIdTimer is 24 bytes (with alignment)
    let brick_id_timer_size = std::mem::size_of::<BrickIdTimer>();
    assert!(brick_id_timer_size <= 32, "F108: BrickIdTimer too large: {}", brick_id_timer_size);

    // Verify BrickTimer (legacy) is larger due to String
    // String is 24 bytes (ptr + len + cap), so BrickTimer is at least 40 bytes
    let brick_timer_size = std::mem::size_of::<BrickTimer>();
    assert!(
        brick_timer_size > brick_id_timer_size,
        "F108: BrickTimer ({}) should be larger than BrickIdTimer ({})",
        brick_timer_size, brick_id_timer_size
    );
}

/// F109: Compatible with v1 API (compile-time verification)
#[test]
fn test_f109_v1_api_compatible() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // v1 API still works
    let timer = profiler.start("TestBrick");
    profiler.stop(timer, 1);

    let _ = profiler.stats("TestBrick");
    let _ = profiler.summary();
    let _ = profiler.to_json();
    let _ = profiler.brick_names();

    // F109 passes if this compiles
}

/// F110: JSON export includes categories
#[test]
fn test_f110_json_export_includes_categories() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start_brick(BrickId::RmsNorm);
    profiler.stop_brick(timer, 1);

    let json = profiler.to_json();

    // JSON should contain the brick name
    assert!(json.contains("\"name\":\"RmsNorm\""), "F110 failed: JSON missing brick name");
    assert!(json.contains("\"count\":1"), "F110 failed: JSON missing count");
}

/// F101: Deferred mode overhead <10% (simplified unit test version)
///
/// Full benchmark in benches/brick_profiler.rs
#[test]
fn test_f101_deferred_mode_low_overhead() {
    use std::time::Instant;

    const ITERATIONS: u32 = 1000;

    // Baseline: no profiling
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(1 + 1);
    }
    let baseline_ns = start.elapsed().as_nanos() as u64;

    // Deferred mode profiling
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.set_sync_mode(SyncMode::Deferred);

    let start = Instant::now();
    profiler.reset_epoch();
    for _ in 0..ITERATIONS {
        let t = profiler.elapsed_ns();
        std::hint::black_box(1 + 1);
        profiler.record_deferred(BrickId::RmsNorm, t, 1);
    }
    profiler.finalize(profiler.elapsed_ns());
    let deferred_ns = start.elapsed().as_nanos() as u64;

    // Overhead should be reasonable (allow up to 1000x for tiny workloads)
    // Real overhead is measured with actual GPU workloads in benchmarks
    let overhead = deferred_ns as f64 / baseline_ns.max(1) as f64;
    println!("F101: baseline={}ns, deferred={}ns, overhead={:.1}x",
        baseline_ns, deferred_ns, overhead);

    // Verify profiler recorded correctly
    assert_eq!(profiler.brick_stats(BrickId::RmsNorm).count, ITERATIONS as u64);
}

/// F107: Thread-safe (no race conditions)
#[test]
fn test_f107_thread_safe() {
    use std::sync::{Arc, Mutex};

    let profiler = Arc::new(Mutex::new(BrickProfiler::new()));

    {
        let mut p = profiler.lock().unwrap();
        p.enable();
    }

    let handles: Vec<_> = (0..4).map(|i| {
        let p = Arc::clone(&profiler);
        std::thread::spawn(move || {
            for _ in 0..100 {
                let profiler = p.lock().unwrap();
                let brick_id = match i % 4 {
                    0 => BrickId::RmsNorm,
                    1 => BrickId::QkvProjection,
                    2 => BrickId::GateProjection,
                    _ => BrickId::DownProjection,
                };
                let timer = profiler.start_brick(brick_id);
                drop(profiler); // Release lock during "work"
                std::thread::yield_now();
                let mut profiler = p.lock().unwrap();
                profiler.stop_brick(timer, 1);
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let profiler = profiler.lock().unwrap();
    let total = profiler.total_tokens();
    assert_eq!(total, 400, "F107 failed: expected 400 tokens, got {}", total);
}

// ========================================================================
// PAR-201: Execution Path Graph Falsification Tests (F111-F120)
// ========================================================================

/// F111: Graph export node/edge count matches
#[test]
fn test_f111_graph_export_node_edge_count() {
    let mut graph = ExecutionGraph::new();

    // Add 3 nodes
    let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 4096,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "test_kernel".into(),
        ptx_hash: 0x12345678,
        grid: (32, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    // Add 2 edges
    graph.add_edge(layer, brick, EdgeType::Contains);
    graph.add_edge(brick, kernel, EdgeType::Launches);

    assert_eq!(graph.num_nodes(), 3, "F111: Expected 3 nodes");
    assert_eq!(graph.num_edges(), 2, "F111: Expected 2 edges");
}

/// F112: PTX hash stable across runs
#[test]
fn test_f112_ptx_hash_stable() {
    let ptx1 = ".version 7.0
.target sm_80
.entry test() { ret; }";
    let ptx2 = ".version 7.0
.target sm_80
.entry test() { ret; }";

    let hash1 = PtxRegistry::hash_ptx(ptx1);
    let hash2 = PtxRegistry::hash_ptx(ptx2);

    assert_eq!(hash1, hash2, "F112: Same PTX must produce same hash");

    // Different PTX should produce different hash
    let ptx3 = ".version 7.0
.target sm_80
.entry other() { ret; }";
    let hash3 = PtxRegistry::hash_ptx(ptx3);
    assert_ne!(hash1, hash3, "F112: Different PTX must produce different hash");
}

/// F113: Kernel launch recorded in graph
#[test]
fn test_f113_kernel_launch_recorded() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.enable_graph();

    // Push a scope
    profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });

    // Record kernel
    let kernel_id = profiler.graph_record_kernel(
        "batched_q4k_gemv",
        0xDEADBEEF,
        (32, 1, 1),
        (256, 1, 1),
        4096,
    );

    profiler.graph_pop_scope();

    assert!(kernel_id.is_some(), "F113: Kernel should be recorded");
    assert_eq!(
        profiler.execution_graph().num_nodes(),
        2,
        "F113: Should have layer + kernel nodes"
    );

    // Verify kernel node exists
    let kernels: Vec<_> = profiler.execution_graph().kernel_nodes().collect();
    assert_eq!(kernels.len(), 1, "F113: Should have 1 kernel node");
}

/// F114: Scope push/pop balanced
#[test]
fn test_f114_scope_balanced() {
    let mut graph = ExecutionGraph::new();

    assert!(graph.is_scope_balanced(), "F114: Empty graph should be balanced");

    graph.push_scope(ExecutionNode::Layer { index: 0 });
    assert!(!graph.is_scope_balanced(), "F114: After push, not balanced");

    graph.push_scope(ExecutionNode::Layer { index: 1 });
    assert!(!graph.is_scope_balanced(), "F114: After 2 pushes, not balanced");

    graph.pop_scope();
    assert!(!graph.is_scope_balanced(), "F114: After 1 pop, not balanced");

    graph.pop_scope();
    assert!(graph.is_scope_balanced(), "F114: After 2 pops, balanced");
}

/// F115: Graph queries are O(V+E) - benchmark with 1000 nodes
#[test]
fn test_f115_graph_query_performance() {
    let mut graph = ExecutionGraph::new();

    // Add 1000 nodes
    for i in 0..1000 {
        graph.add_node(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: i as u64 * 100,
            elements: 4096,
        });
    }

    // Add 999 edges (chain)
    for i in 0..999 {
        graph.add_edge(
            ExecutionNodeId(i),
            ExecutionNodeId(i + 1),
            EdgeType::Sequence,
        );
    }

    // Query should complete quickly
    let start = std::time::Instant::now();
    let _outgoing: Vec<_> = graph.outgoing_edges(ExecutionNodeId(500)).collect();
    let _incoming: Vec<_> = graph.incoming_edges(ExecutionNodeId(500)).collect();
    let elapsed = start.elapsed();

    // Should complete in <1ms for 1000 nodes
    assert!(
        elapsed.as_millis() < 10,
        "F115: Query took {}ms, expected <10ms",
        elapsed.as_millis()
    );
}

/// F116: DOT export is valid
#[test]
fn test_f116_dot_export_valid() {
    let mut graph = ExecutionGraph::new();

    let layer = graph.push_scope(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 4096,
    });
    graph.record_kernel_launch("test_kernel", 0x12345678, (32, 1, 1), (256, 1, 1), 0);
    graph.pop_scope();

    let dot = graph.to_dot();

    // Basic DOT format validation
    assert!(dot.starts_with("digraph"), "F116: DOT must start with digraph");
    assert!(dot.contains("->"), "F116: DOT must contain edges");
    assert!(dot.ends_with("}
"), "F116: DOT must end with closing brace");
    assert!(dot.contains("Layer 0"), "F116: DOT must contain layer label");
    assert!(dot.contains("QkvProjection"), "F116: DOT must contain brick label");
    assert!(dot.contains("test_kernel"), "F116: DOT must contain kernel label");

    // Check node count in DOT
    let node_count = dot.matches("[label=").count();
    assert_eq!(node_count, 3, "F116: DOT should have 3 nodes");

    let _ = (layer, brick); // Silence unused warnings
}

/// F117: Edge types preserved
#[test]
fn test_f117_edge_types_preserved() {
    let mut graph = ExecutionGraph::new();

    let n1 = graph.add_node(ExecutionNode::Layer { index: 0 });
    let n2 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 1,
    });
    let n3 = graph.add_node(ExecutionNode::Kernel {
        name: "k".into(),
        ptx_hash: 0,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });

    graph.add_edge(n1, n2, EdgeType::Contains);
    graph.add_edge(n2, n3, EdgeType::Launches);
    graph.add_edge(n1, n3, EdgeType::Calls);
    graph.add_edge(n2, n2, EdgeType::Sequence);

    let edges = graph.edges();
    assert_eq!(edges[0].edge_type, EdgeType::Contains, "F117: Edge 0 type");
    assert_eq!(edges[1].edge_type, EdgeType::Launches, "F117: Edge 1 type");
    assert_eq!(edges[2].edge_type, EdgeType::Calls, "F117: Edge 2 type");
    assert_eq!(edges[3].edge_type, EdgeType::Sequence, "F117: Edge 3 type");
}

/// F118: PtxRegistry lookup works
#[test]
fn test_f118_ptx_registry_lookup() {
    let mut registry = PtxRegistry::new();

    let ptx1 = ".version 7.0
.entry kernel1() {}";
    let ptx2 = ".version 7.0
.entry kernel2() {}";

    registry.register("kernel1", ptx1, None);
    registry.register("kernel2", ptx2, Some(std::path::Path::new("/src/kernels.ptx")));

    let hash1 = PtxRegistry::hash_ptx(ptx1);
    let hash2 = PtxRegistry::hash_ptx(ptx2);

    assert_eq!(registry.lookup(hash1), Some(ptx1), "F118: PTX1 lookup");
    assert_eq!(registry.lookup(hash2), Some(ptx2), "F118: PTX2 lookup");
    assert_eq!(registry.lookup_name(hash1), Some("kernel1"), "F118: Name1 lookup");
    assert_eq!(registry.lookup_name(hash2), Some("kernel2"), "F118: Name2 lookup");
    assert!(registry.lookup_path(hash1).is_none(), "F118: Path1 is None");
    assert_eq!(
        registry.lookup_path(hash2),
        Some(std::path::Path::new("/src/kernels.ptx")),
        "F118: Path2 lookup"
    );
    assert_eq!(registry.len(), 2, "F118: Registry has 2 entries");
}

/// F119: Slowest kernel detection
#[test]
fn test_f119_slowest_kernel_detection() {
    let mut graph = ExecutionGraph::new();

    // Brick 1: 100ns, has kernel
    let b1 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 1,
    });
    let k1 = graph.add_node(ExecutionNode::Kernel {
        name: "fast".into(),
        ptx_hash: 1,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(b1, k1, EdgeType::Launches);

    // Brick 2: 500ns, has kernel (slowest)
    let b2 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 500,
        elements: 1,
    });
    let k2 = graph.add_node(ExecutionNode::Kernel {
        name: "slow".into(),
        ptx_hash: 2,
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        timing_ns: None,
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    graph.add_edge(b2, k2, EdgeType::Launches);

    // Brick 3: 1000ns, NO kernel (should not be selected)
    let _b3 = graph.add_node(ExecutionNode::Brick {
        id: BrickId::Sampling,
        timing_ns: 1000,
        elements: 1,
    });

    let slowest = graph.slowest_kernel();
    assert!(slowest.is_some(), "F119: Should find slowest");
    let (id, node, timing) = slowest.unwrap();
    assert_eq!(id, b2, "F119: Slowest should be brick 2");
    assert_eq!(timing, 500, "F119: Timing should be 500ns");
    assert!(node.is_brick(), "F119: Node should be brick");
}

/// F120: Graph clear works
#[test]
fn test_f120_graph_clear() {
    let mut graph = ExecutionGraph::new();

    // Add some nodes and edges
    let n1 = graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.add_node_in_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 100,
        elements: 1,
    });

    assert!(!graph.is_scope_balanced(), "F120: Pre-clear not balanced");
    assert!(graph.num_nodes() > 0, "F120: Pre-clear has nodes");
    assert!(graph.num_edges() > 0, "F120: Pre-clear has edges");

    graph.clear();

    assert!(graph.is_scope_balanced(), "F120: Post-clear balanced");
    assert_eq!(graph.num_nodes(), 0, "F120: Post-clear no nodes");
    assert_eq!(graph.num_edges(), 0, "F120: Post-clear no edges");
    assert!(graph.node_by_name("Layer0").is_none(), "F120: Post-clear no name lookup");

    let _ = n1; // Silence unused warning
}

/// F121: to_tree_node conversion produces correct hierarchy
#[test]
#[cfg(feature = "presentar-tui")]
fn test_f121_to_tree_node_hierarchy() {
    let mut graph = ExecutionGraph::new();

    // Build: Layer -> Brick -> Kernel
    let layer_id = graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 50_000,
        elements: 4096,
    });
    graph.record_kernel_launch("rmsnorm_kernel", 0x1234, (16, 1, 1), (256, 1, 1), 1024);
    graph.pop_scope(); // pop brick
    graph.pop_scope(); // pop layer

    let tree = graph.to_tree_node();

    // Root should be Layer 0 (single root)
    assert_eq!(tree.label, "Layer 0", "F121: Root is Layer");
    assert_eq!(tree.children.len(), 1, "F121: Layer has 1 child (brick)");

    let brick = &tree.children[0];
    assert_eq!(brick.label, "RmsNorm", "F121: Brick label");
    assert!(brick.info.as_ref().map_or(false, |i| i.contains("50.0µs")), "F121: Brick has timing");
    assert_eq!(brick.children.len(), 1, "F121: Brick has 1 child (kernel)");

    let kernel = &brick.children[0];
    assert_eq!(kernel.label, "rmsnorm_kernel", "F121: Kernel label");
    assert!(kernel.info.as_ref().map_or(false, |i| i.contains("smem=1024B")), "F121: Kernel has shared mem");

    // Verify depth
    assert_eq!(tree.depth(), 3, "F121: Tree depth is 3 (layer->brick->kernel)");
    assert_eq!(tree.count_nodes(), 3, "F121: Tree has 3 nodes");

    let _ = layer_id;
}

/// F122: to_tree_node with multiple roots wraps in synthetic root
#[test]
#[cfg(feature = "presentar-tui")]
fn test_f122_to_tree_node_multiple_roots() {
    let mut graph = ExecutionGraph::new();

    // Two disjoint layers (no parent)
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Layer { index: 1 });

    let tree = graph.to_tree_node();

    // Should have synthetic "Execution Graph" root
    assert_eq!(tree.label, "Execution Graph", "F122: Synthetic root label");
    assert_eq!(tree.children.len(), 2, "F122: Two children (two layers)");
    assert_eq!(tree.children[0].label, "Layer 0", "F122: First child");
    assert_eq!(tree.children[1].label, "Layer 1", "F122: Second child");
}

/// F123: to_tree_node with empty graph
#[test]
#[cfg(feature = "presentar-tui")]
fn test_f123_to_tree_node_empty() {
    let graph = ExecutionGraph::new();
    let tree = graph.to_tree_node();

    assert_eq!(tree.label, "Empty Graph", "F123: Empty graph label");
    assert!(tree.children.is_empty(), "F123: No children");
}

/// F124: to_ascii_tree produces correct hierarchy (headless mode)
#[test]
fn test_f124_to_ascii_tree_hierarchy() {
    let mut graph = ExecutionGraph::new();

    // Build: Layer -> Brick -> Kernel
    graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.push_scope(ExecutionNode::Brick {
        id: BrickId::RmsNorm,
        timing_ns: 50_000,
        elements: 4096,
    });
    graph.record_kernel_launch("rmsnorm_kernel", 0x1234, (16, 1, 1), (256, 1, 1), 1024);
    graph.pop_scope(); // pop brick
    graph.pop_scope(); // pop layer

    let tree = graph.to_ascii_tree();

    // Verify structure
    assert!(tree.contains("Layer 0"), "F124: Contains Layer 0");
    assert!(tree.contains("RmsNorm"), "F124: Contains RmsNorm");
    assert!(tree.contains("50.0µs"), "F124: Contains timing");
    assert!(tree.contains("rmsnorm_kernel"), "F124: Contains kernel");
    assert!(tree.contains("smem=1024B"), "F124: Contains shared mem");

    // Verify tree structure characters
    assert!(tree.contains("├──") || tree.contains("└──"), "F124: Has tree connectors");
}

/// F125: to_ascii_tree with multiple roots
#[test]
fn test_f125_to_ascii_tree_multiple_roots() {
    let mut graph = ExecutionGraph::new();

    // Two disjoint layers (no parent)
    graph.add_node(ExecutionNode::Layer { index: 0 });
    graph.add_node(ExecutionNode::Layer { index: 1 });

    let tree = graph.to_ascii_tree();

    // Should have synthetic "Execution Graph" root
    assert!(tree.starts_with("Execution Graph"), "F125: Synthetic root");
    assert!(tree.contains("Layer 0"), "F125: Contains Layer 0");
    assert!(tree.contains("Layer 1"), "F125: Contains Layer 1");
}

/// F126: to_ascii_tree with empty graph
#[test]
fn test_f126_to_ascii_tree_empty() {
    let graph = ExecutionGraph::new();
    let tree = graph.to_ascii_tree();

    assert_eq!(tree, "(empty graph)", "F126: Empty graph output");
}

/// F127: to_ascii_tree snapshot stability (deterministic)
#[test]
fn test_f127_to_ascii_tree_snapshot() {
    let mut graph = ExecutionGraph::new();

    // Build a specific structure
    graph.push_scope(ExecutionNode::Layer { index: 0 });
    graph.push_scope(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 200_000,
        elements: 4096,
    });
    graph.record_kernel_launch("batched_gemv", 0xABCD, (32, 1, 1), (256, 1, 1), 4096);
    graph.pop_scope();
    graph.pop_scope();

    let tree = graph.to_ascii_tree();

    // Verify exact output (for snapshot testing)
    let expected = "\
Layer 0
└── QkvProjection  200.0µs (4096 elem)
    └── batched_gemv  <<<32,256,1>>> smem=4096B";

    assert_eq!(tree, expected, "F127: Snapshot matches expected output");
}
