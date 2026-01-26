//! ExecutionGraph - Execution Path Graph for Profiling
//!
//! PAR-201: Captures the full execution hierarchy for profiling analysis.

use std::collections::HashMap;

use super::node::{EdgeType, ExecutionEdge, ExecutionNode, ExecutionNodeId, TransferDirection};

/// Execution path graph for tracking brick → kernel → PTX relationships.
///
/// PAR-201: Captures the full execution hierarchy for profiling analysis.
///
/// # Example
///
/// ```rust,ignore
/// use trueno::brick::{ExecutionGraph, ExecutionNode, EdgeType};
///
/// let mut graph = ExecutionGraph::new();
///
/// // Add layer scope
/// let layer_id = graph.add_node(ExecutionNode::Layer { index: 0 });
///
/// // Add brick within layer
/// let brick_id = graph.add_node(ExecutionNode::Brick {
///     id: BrickId::QkvProjection,
///     timing_ns: 1000,
///     elements: 4096,
/// });
/// graph.add_edge(layer_id, brick_id, EdgeType::Contains);
///
/// // Add kernel launched by brick
/// let kernel_id = graph.add_node(ExecutionNode::Kernel {
///     name: "batched_q4k_gemv".into(),
///     ptx_hash: 0x7a3b1c2d,
///     grid: (32, 1, 1),
///     block: (256, 1, 1),
///     shared_mem: 4096,
/// });
/// graph.add_edge(brick_id, kernel_id, EdgeType::Launches);
///
/// // Export to trueno-graph for analysis
/// #[cfg(feature = "execution-graph")]
/// let csr = graph.to_csr();
/// ```
#[derive(Debug, Default)]
pub struct ExecutionGraph {
    /// All nodes in the graph
    nodes: Vec<ExecutionNode>,
    /// All edges in the graph
    edges: Vec<ExecutionEdge>,
    /// Scope stack for hierarchical recording
    scope_stack: Vec<ExecutionNodeId>,
    /// Node name → ID mapping for fast lookup
    name_to_id: HashMap<String, ExecutionNodeId>,
}

impl ExecutionGraph {
    /// Create a new empty execution graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph, returning its ID.
    pub fn add_node(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = ExecutionNodeId(self.nodes.len() as u32);
        let name = node.name();
        self.name_to_id.insert(name, id);
        self.nodes.push(node);
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, src: ExecutionNodeId, dst: ExecutionNodeId, edge_type: EdgeType) {
        self.edges.push(ExecutionEdge {
            src,
            dst,
            edge_type,
            weight: 1.0,
        });
    }

    /// Add an edge with a weight.
    pub fn add_weighted_edge(
        &mut self,
        src: ExecutionNodeId,
        dst: ExecutionNodeId,
        edge_type: EdgeType,
        weight: f32,
    ) {
        self.edges.push(ExecutionEdge {
            src,
            dst,
            edge_type,
            weight,
        });
    }

    /// Push a scope for hierarchical recording.
    /// All subsequent nodes will be children of this scope.
    pub fn push_scope(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = self.add_node(node);
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, id, EdgeType::Contains);
        }
        self.scope_stack.push(id);
        id
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) -> Option<ExecutionNodeId> {
        self.scope_stack.pop()
    }

    /// Get the current scope (if any).
    pub fn current_scope(&self) -> Option<ExecutionNodeId> {
        self.scope_stack.last().copied()
    }

    /// Add a node under the current scope.
    pub fn add_node_in_scope(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = self.add_node(node);
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, id, EdgeType::Contains);
        }
        id
    }

    /// Record a kernel launch under the current scope.
    pub fn record_kernel_launch(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> ExecutionNodeId {
        let kernel = ExecutionNode::Kernel {
            name: name.to_string(),
            ptx_hash,
            grid,
            block,
            shared_mem,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        };
        let kernel_id = self.add_node(kernel);

        // Link from current scope with Launches edge
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, kernel_id, EdgeType::Launches);
        }

        kernel_id
    }

    /// Record a kernel launch with roofline metrics (Phase 9).
    #[allow(clippy::too_many_arguments)]
    pub fn record_kernel_launch_with_metrics(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        timing_ns: u64,
        arithmetic_intensity: f32,
        achieved_tflops: f32,
    ) -> ExecutionNodeId {
        let kernel = ExecutionNode::Kernel {
            name: name.to_string(),
            ptx_hash,
            grid,
            block,
            shared_mem,
            timing_ns: Some(timing_ns),
            arithmetic_intensity: Some(arithmetic_intensity),
            achieved_tflops: Some(achieved_tflops),
        };
        let kernel_id = self.add_node(kernel);

        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, kernel_id, EdgeType::Launches);
        }

        kernel_id
    }

    /// Record a memory transfer (Phase 9: data movement topology).
    pub fn record_transfer(
        &mut self,
        src: &str,
        dst: &str,
        bytes: u64,
        direction: TransferDirection,
        timing_ns: Option<u64>,
    ) -> ExecutionNodeId {
        let transfer = ExecutionNode::Transfer {
            src: src.to_string(),
            dst: dst.to_string(),
            bytes,
            direction,
            timing_ns,
        };
        let transfer_id = self.add_node(transfer);

        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, transfer_id, EdgeType::Contains);
        }

        transfer_id
    }

    /// Add a dependency edge for critical path analysis (Phase 9).
    pub fn add_dependency(&mut self, from: ExecutionNodeId, to: ExecutionNodeId) {
        self.add_edge(from, to, EdgeType::DependsOn);
    }

    /// Get a node by ID.
    pub fn node(&self, id: ExecutionNodeId) -> Option<&ExecutionNode> {
        self.nodes.get(id.0 as usize)
    }

    /// Get a node by name.
    pub fn node_by_name(&self, name: &str) -> Option<(ExecutionNodeId, &ExecutionNode)> {
        self.name_to_id
            .get(name)
            .and_then(|&id| self.nodes.get(id.0 as usize).map(|n| (id, n)))
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[ExecutionNode] {
        &self.nodes
    }

    /// Get all edges.
    pub fn edges(&self) -> &[ExecutionEdge] {
        &self.edges
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get outgoing edges for a node.
    pub fn outgoing_edges(&self, node: ExecutionNodeId) -> impl Iterator<Item = &ExecutionEdge> {
        self.edges.iter().filter(move |e| e.src == node)
    }

    /// Get incoming edges for a node.
    pub fn incoming_edges(&self, node: ExecutionNodeId) -> impl Iterator<Item = &ExecutionEdge> {
        self.edges.iter().filter(move |e| e.dst == node)
    }

    /// Find all kernel nodes.
    pub fn kernel_nodes(&self) -> impl Iterator<Item = (ExecutionNodeId, &ExecutionNode)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_kernel())
            .map(|(i, n)| (ExecutionNodeId(i as u32), n))
    }

    /// Find the slowest kernel (by parent brick timing).
    pub fn slowest_kernel(&self) -> Option<(ExecutionNodeId, &ExecutionNode, u64)> {
        let mut slowest: Option<(ExecutionNodeId, &ExecutionNode, u64)> = None;

        for (id, node) in self.nodes.iter().enumerate() {
            if let ExecutionNode::Brick { timing_ns, .. } = node {
                // Check if this brick has kernel children
                let node_id = ExecutionNodeId(id as u32);
                let has_kernel = self
                    .outgoing_edges(node_id)
                    .any(|e| e.edge_type == EdgeType::Launches);

                if has_kernel {
                    match &slowest {
                        None => slowest = Some((node_id, node, *timing_ns)),
                        Some((_, _, t)) if *timing_ns > *t => {
                            slowest = Some((node_id, node, *timing_ns))
                        }
                        _ => {}
                    }
                }
            }
        }

        slowest
    }

    /// Export to DOT format for Graphviz visualization.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph ExecutionGraph {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes with styling based on type
        for (i, node) in self.nodes.iter().enumerate() {
            let (label, style) = match node {
                ExecutionNode::Layer { index } => (
                    format!("Layer {}", index),
                    "style=filled,fillcolor=lightblue",
                ),
                ExecutionNode::Brick { id, timing_ns, .. } => (
                    format!("{}\\n{:.1}µs", id.name(), *timing_ns as f64 / 1000.0),
                    "style=filled,fillcolor=lightgreen",
                ),
                ExecutionNode::Kernel {
                    name, grid, block, ..
                } => (
                    format!("{}\\n<<<{},{},{}>>>", name, grid.0, block.0, block.1),
                    "style=filled,fillcolor=lightyellow",
                ),
                ExecutionNode::Function { name, file, line } => {
                    let loc = match (file, line) {
                        (Some(f), Some(l)) => format!("\\n{}:{}", f, l),
                        _ => String::new(),
                    };
                    (
                        format!("{}{}", name, loc),
                        "style=filled,fillcolor=lightgray",
                    )
                }
                ExecutionNode::Transfer {
                    src,
                    dst,
                    bytes,
                    direction,
                    ..
                } => {
                    let dir = match direction {
                        TransferDirection::H2D => "H2D",
                        TransferDirection::D2H => "D2H",
                        TransferDirection::D2D => "D2D",
                    };
                    (
                        format!("{}\\n{}->{}\\n{:.1}MB", dir, src, dst, *bytes as f64 / 1e6),
                        "style=filled,fillcolor=lightsalmon",
                    )
                }
                ExecutionNode::AsyncTask {
                    name,
                    poll_count,
                    yield_count,
                    total_poll_ns,
                } => {
                    let efficiency = if *poll_count > 0 {
                        100.0 / *poll_count as f64
                    } else {
                        0.0
                    };
                    (
                        format!(
                            "{}\\npolls:{} yields:{}\\n{:.1}µs ({:.0}%)",
                            name,
                            poll_count,
                            yield_count,
                            *total_poll_ns as f64 / 1000.0,
                            efficiency
                        ),
                        "style=filled,fillcolor=lightcyan",
                    )
                }
            };
            dot.push_str(&format!("  n{} [label=\"{}\",{}];\n", i, label, style));
        }

        dot.push('\n');

        // Add edges with styling based on type
        for edge in &self.edges {
            let style = match edge.edge_type {
                EdgeType::Calls => "style=solid",
                EdgeType::Contains => "style=dashed",
                EdgeType::Launches => "style=bold,color=red",
                EdgeType::Sequence => "style=dotted",
                EdgeType::DependsOn => "style=solid,color=blue",
                EdgeType::Transfer { .. } => "style=bold,color=orange",
            };
            dot.push_str(&format!(
                "  n{} -> n{} [{}];\n",
                edge.src.0, edge.dst.0, style
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Export to trueno-graph CsrGraph format.
    #[cfg(feature = "execution-graph")]
    pub fn to_csr(&self) -> trueno_graph::CsrGraph {
        use trueno_graph::{CsrGraph, NodeId};

        let edges: Vec<(NodeId, NodeId, f32)> = self
            .edges
            .iter()
            .map(|e| (NodeId(e.src.0), NodeId(e.dst.0), e.weight))
            .collect();

        let mut graph = CsrGraph::from_edge_list(&edges).unwrap_or_default();

        // Set node names for querying
        for (i, node) in self.nodes.iter().enumerate() {
            graph.set_node_name(NodeId(i as u32), node.name());
        }

        graph
    }

    /// Convert to presentar-terminal TreeNode for TUI visualization.
    ///
    /// PAR-201: Renders the execution graph as a collapsible tree in the terminal.
    #[cfg(feature = "presentar-tui")]
    pub fn to_tree_node(&self) -> presentar_terminal::TreeNode {
        use presentar_terminal::{Color, TreeNode};

        // Color scheme for node types
        let layer_color = Color::new(0.4, 0.6, 1.0, 1.0); // Light blue
        let brick_color = Color::new(0.4, 0.8, 0.4, 1.0); // Light green
        let kernel_color = Color::new(1.0, 0.8, 0.3, 1.0); // Yellow/orange
        let func_color = Color::new(0.7, 0.7, 0.7, 1.0); // Light gray

        // Build child map: parent -> [children]
        let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_parent: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for edge in &self.edges {
            if edge.edge_type == EdgeType::Contains || edge.edge_type == EdgeType::Launches {
                children_map.entry(edge.src.0).or_default().push(edge.dst.0);
                has_parent.insert(edge.dst.0);
            }
        }

        // Find root nodes (nodes with no parent)
        let root_ids: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|id| !has_parent.contains(id))
            .collect();

        // Recursive function to build TreeNode
        fn build_node(
            graph: &ExecutionGraph,
            id: u32,
            children_map: &HashMap<u32, Vec<u32>>,
            layer_color: Color,
            brick_color: Color,
            kernel_color: Color,
            func_color: Color,
        ) -> TreeNode {
            let node = &graph.nodes[id as usize];
            let (label, info, color) = match node {
                ExecutionNode::Layer { index } => (format!("Layer {}", index), None, layer_color),
                ExecutionNode::Brick {
                    id: brick_id,
                    timing_ns,
                    elements,
                } => (
                    brick_id.name().to_string(),
                    Some(format!(
                        "{:.1}µs ({} elem)",
                        *timing_ns as f64 / 1000.0,
                        elements
                    )),
                    brick_color,
                ),
                ExecutionNode::Kernel {
                    name,
                    grid,
                    block,
                    shared_mem,
                    ..
                } => (
                    name.clone(),
                    Some(format!(
                        "<<<{},{},{}>>> smem={}B",
                        grid.0, block.0, block.1, shared_mem
                    )),
                    kernel_color,
                ),
                ExecutionNode::Function { name, file, line } => {
                    let loc = match (file, line) {
                        (Some(f), Some(l)) => format!(" ({}:{})", f, l),
                        _ => String::new(),
                    };
                    (format!("{}{}", name, loc), None, func_color)
                }
                ExecutionNode::Transfer {
                    src,
                    dst,
                    bytes,
                    direction,
                    timing_ns,
                } => {
                    let timing_str = timing_ns
                        .map(|ns| format!(" {:.1}µs", ns as f64 / 1000.0))
                        .unwrap_or_default();
                    (
                        format!("{:?}: {} → {}", direction, src, dst),
                        Some(format!("{}B{}", bytes, timing_str)),
                        Color::new(0.8, 0.4, 0.8, 1.0), // Transfer color (magenta)
                    )
                }
                ExecutionNode::AsyncTask {
                    name,
                    poll_count,
                    yield_count,
                    total_poll_ns,
                } => {
                    let efficiency = if *poll_count > 0 {
                        100.0 / *poll_count as f64
                    } else {
                        0.0
                    };
                    (
                        name.clone(),
                        Some(format!(
                            "polls:{} yields:{} {:.1}µs ({:.0}% eff)",
                            poll_count,
                            yield_count,
                            *total_poll_ns as f64 / 1000.0,
                            efficiency
                        )),
                        Color::new(0.4, 0.8, 0.8, 1.0), // Async task color (cyan)
                    )
                }
            };

            let mut tree_node = TreeNode::new(id as u64, label).with_color(color);
            if let Some(info_str) = info {
                tree_node = tree_node.with_info(info_str);
            }

            // Add children
            if let Some(child_ids) = children_map.get(&id) {
                for &child_id in child_ids {
                    let child = build_node(
                        graph,
                        child_id,
                        children_map,
                        layer_color,
                        brick_color,
                        kernel_color,
                        func_color,
                    );
                    tree_node = tree_node.with_child(child);
                }
            }

            tree_node
        }

        // Build root node
        if root_ids.is_empty() {
            TreeNode::new(0, "Empty Graph")
        } else if root_ids.len() == 1 {
            build_node(
                self,
                root_ids[0],
                &children_map,
                layer_color,
                brick_color,
                kernel_color,
                func_color,
            )
        } else {
            // Multiple roots: wrap in a synthetic root
            let mut root = TreeNode::new(u64::MAX, "Execution Graph")
                .with_color(Color::new(0.9, 0.9, 0.9, 1.0));
            for &root_id in &root_ids {
                let child = build_node(
                    self,
                    root_id,
                    &children_map,
                    layer_color,
                    brick_color,
                    kernel_color,
                    func_color,
                );
                root = root.with_child(child);
            }
            root
        }
    }

    /// Render graph to ASCII tree string (headless mode for testing/automation).
    ///
    /// PAR-201: Zero-dependency tree visualization for CI/CD, logging, and snapshot tests.
    #[must_use]
    pub fn to_ascii_tree(&self) -> String {
        // Build child map: parent -> [children]
        let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_parent: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for edge in &self.edges {
            if edge.edge_type == EdgeType::Contains || edge.edge_type == EdgeType::Launches {
                children_map.entry(edge.src.0).or_default().push(edge.dst.0);
                has_parent.insert(edge.dst.0);
            }
        }

        // Find root nodes (nodes with no parent)
        let root_ids: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|id| !has_parent.contains(id))
            .collect();

        // Recursive function to build tree string
        fn build_tree(
            graph: &ExecutionGraph,
            id: u32,
            children_map: &HashMap<u32, Vec<u32>>,
            prefix: &str,
            connector: &str,
            output: &mut String,
        ) {
            let node = &graph.nodes[id as usize];
            let (label, info) = match node {
                ExecutionNode::Layer { index } => (format!("Layer {}", index), String::new()),
                ExecutionNode::Brick {
                    id: brick_id,
                    timing_ns,
                    elements,
                } => (
                    brick_id.name().to_string(),
                    format!("  {:.1}µs ({} elem)", *timing_ns as f64 / 1000.0, elements),
                ),
                ExecutionNode::Kernel {
                    name,
                    grid,
                    block,
                    shared_mem,
                    ..
                } => (
                    name.clone(),
                    format!(
                        "  <<<{},{},{}>>> smem={}B",
                        grid.0, block.0, block.1, shared_mem
                    ),
                ),
                ExecutionNode::Function { name, file, line } => {
                    let loc = match (file, line) {
                        (Some(f), Some(l)) => format!(" ({}:{})", f, l),
                        _ => String::new(),
                    };
                    (format!("{}{}", name, loc), String::new())
                }
                ExecutionNode::Transfer {
                    src,
                    dst,
                    bytes,
                    direction,
                    timing_ns,
                } => {
                    let timing_str = timing_ns
                        .map(|ns| format!(" {:.1}µs", ns as f64 / 1000.0))
                        .unwrap_or_default();
                    (
                        format!("{:?}: {} → {}", direction, src, dst),
                        format!("  {}B{}", bytes, timing_str),
                    )
                }
                ExecutionNode::AsyncTask {
                    name,
                    poll_count,
                    yield_count,
                    total_poll_ns,
                } => {
                    let efficiency = if *poll_count > 0 {
                        100.0 / *poll_count as f64
                    } else {
                        0.0
                    };
                    (
                        name.clone(),
                        format!(
                            "  polls:{} yields:{} {:.1}µs ({:.0}% eff)",
                            poll_count,
                            yield_count,
                            *total_poll_ns as f64 / 1000.0,
                            efficiency
                        ),
                    )
                }
            };

            output.push_str(&format!("{}{}{}{}\n", prefix, connector, label, info));

            if let Some(child_ids) = children_map.get(&id) {
                let child_count = child_ids.len();
                for (i, &child_id) in child_ids.iter().enumerate() {
                    let is_last = i == child_count - 1;
                    let new_connector = if is_last { "└── " } else { "├── " };
                    let new_prefix = if connector.is_empty() {
                        prefix.to_string()
                    } else if connector == "└── " {
                        format!("{}    ", prefix)
                    } else {
                        format!("{}│   ", prefix)
                    };
                    build_tree(
                        graph,
                        child_id,
                        children_map,
                        &new_prefix,
                        new_connector,
                        output,
                    );
                }
            }
        }

        let mut output = String::new();

        if root_ids.is_empty() {
            output.push_str("(empty graph)\n");
        } else if root_ids.len() == 1 {
            build_tree(self, root_ids[0], &children_map, "", "", &mut output);
        } else {
            // Multiple roots: add synthetic root
            output.push_str("Execution Graph\n");
            let root_count = root_ids.len();
            for (i, &root_id) in root_ids.iter().enumerate() {
                let is_last = i == root_count - 1;
                let connector = if is_last { "└── " } else { "├── " };
                build_tree(self, root_id, &children_map, "", connector, &mut output);
            }
        }

        // Remove trailing newline for cleaner output
        if output.ends_with('\n') {
            output.pop();
        }
        output
    }

    // ========================
    // Phase 9: Critical Path Analysis (CPA)
    // ========================

    /// Get timing for a node (ns). Returns 0 for non-timed nodes.
    fn node_timing_ns(&self, id: ExecutionNodeId) -> u64 {
        match &self.nodes[id.0 as usize] {
            ExecutionNode::Brick { timing_ns, .. } => *timing_ns,
            ExecutionNode::Kernel { timing_ns, .. } => timing_ns.unwrap_or(0),
            ExecutionNode::Transfer { timing_ns, .. } => timing_ns.unwrap_or(0),
            _ => 0,
        }
    }

    /// Compute critical path through execution graph using longest-path algorithm.
    ///
    /// Returns (critical_path_nodes, total_time_ns). The critical path represents
    /// the longest chain of dependencies that determines total execution time.
    ///
    /// Reference: Graham et al. (1979) "Scheduling Algorithms for Multi-Processor Systems"
    pub fn critical_path(&self) -> (Vec<ExecutionNodeId>, u64) {
        if self.nodes.is_empty() {
            return (vec![], 0);
        }

        // Build adjacency list for DependsOn and Sequence edges
        let mut adj: Vec<Vec<(u32, u64)>> = vec![vec![]; self.nodes.len()];
        for edge in &self.edges {
            match &edge.edge_type {
                EdgeType::DependsOn | EdgeType::Sequence => {
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
                EdgeType::Contains | EdgeType::Calls | EdgeType::Launches => {
                    // Hierarchical edges: children contribute to parent time
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
                EdgeType::Transfer { .. } => {
                    // Transfer edges carry their own timing
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
            }
        }

        // Topological sort using Kahn's algorithm
        let mut in_degree = vec![0u32; self.nodes.len()];
        for edges in &adj {
            for (dst, _) in edges {
                in_degree[*dst as usize] += 1;
            }
        }

        let mut queue: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| in_degree[i as usize] == 0)
            .collect();
        let mut topo_order = Vec::with_capacity(self.nodes.len());

        while let Some(u) = queue.pop() {
            topo_order.push(u);
            for (v, _) in &adj[u as usize] {
                in_degree[*v as usize] -= 1;
                if in_degree[*v as usize] == 0 {
                    queue.push(*v);
                }
            }
        }

        // Longest path DP
        let mut dist = vec![0u64; self.nodes.len()];
        let mut pred = vec![None::<u32>; self.nodes.len()];

        // Initialize with node's own timing for roots
        for &node in &topo_order {
            if self.edges.iter().all(|e| e.dst.0 != node) {
                dist[node as usize] = self.node_timing_ns(ExecutionNodeId(node));
            }
        }

        for &u in &topo_order {
            for (v, weight) in &adj[u as usize] {
                let new_dist = dist[u as usize] + weight;
                if new_dist > dist[*v as usize] {
                    dist[*v as usize] = new_dist;
                    pred[*v as usize] = Some(u);
                }
            }
        }

        // Find endpoint with maximum distance
        let (end_node, &total_time) = dist
            .iter()
            .enumerate()
            .max_by_key(|(_, &d)| d)
            .unwrap_or((0, &0));

        // Reconstruct path
        let mut path = vec![];
        let mut current = Some(end_node as u32);
        while let Some(node) = current {
            path.push(ExecutionNodeId(node));
            current = pred[node as usize];
        }
        path.reverse();

        (path, total_time)
    }

    /// Compute slack for each node (how much it can be delayed without affecting total time).
    ///
    /// Returns map from node ID to slack in nanoseconds. Nodes on critical path have slack = 0.
    pub fn compute_slack(&self) -> HashMap<ExecutionNodeId, u64> {
        let (critical_path, total_time) = self.critical_path();
        let critical_set: std::collections::HashSet<_> = critical_path.iter().copied().collect();

        let mut slack = HashMap::new();

        // Build reverse adjacency
        let mut reverse_adj: Vec<Vec<u32>> = vec![vec![]; self.nodes.len()];
        for edge in &self.edges {
            reverse_adj[edge.dst.0 as usize].push(edge.src.0);
        }

        // Forward pass: earliest start time
        let mut earliest = vec![0u64; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let mut max_pred = 0u64;
            for &pred in &reverse_adj[i] {
                max_pred = max_pred
                    .max(earliest[pred as usize] + self.node_timing_ns(ExecutionNodeId(pred)));
            }
            earliest[i] = max_pred;
        }

        // Backward pass: latest start time
        let mut latest = vec![total_time; self.nodes.len()];
        for i in (0..self.nodes.len()).rev() {
            let timing = self.node_timing_ns(ExecutionNodeId(i as u32));
            let mut min_succ = total_time;
            for edge in &self.edges {
                if edge.src.0 == i as u32 {
                    min_succ = min_succ.min(latest[edge.dst.0 as usize]);
                }
            }
            latest[i] = min_succ.saturating_sub(timing);
        }

        // Slack = latest - earliest
        for i in 0..self.nodes.len() {
            let node_id = ExecutionNodeId(i as u32);
            let node_slack = if critical_set.contains(&node_id) {
                0
            } else {
                latest[i].saturating_sub(earliest[i])
            };
            slack.insert(node_id, node_slack);
        }

        slack
    }

    /// Compute roofline distance for kernel nodes.
    ///
    /// Returns map from kernel node ID to distance from roofline (0.0 = optimal).
    /// Distance = 1.0 - min(achieved/peak_compute, achieved/peak_bandwidth).
    ///
    /// Reference: Williams et al. (2009) "Roofline: An Insightful Visual Performance Model"
    pub fn roofline_distance(
        &self,
        peak_tflops: f32,
        peak_bandwidth_gb_s: f32,
    ) -> HashMap<ExecutionNodeId, f32> {
        let mut distances = HashMap::new();

        for (i, node) in self.nodes.iter().enumerate() {
            if let ExecutionNode::Kernel {
                arithmetic_intensity,
                achieved_tflops,
                ..
            } = node
            {
                if let (Some(ai), Some(achieved)) = (arithmetic_intensity, achieved_tflops) {
                    // Roofline model: achievable = min(peak_compute, ai * bandwidth)
                    let bandwidth_bound = *ai * peak_bandwidth_gb_s / 1000.0; // Convert GB/s to TFLOP/s
                    let roofline_bound = peak_tflops.min(bandwidth_bound);
                    let efficiency = achieved / roofline_bound;
                    let distance = 1.0 - efficiency.min(1.0);
                    distances.insert(ExecutionNodeId(i as u32), distance);
                }
            }
        }

        distances
    }

    /// Detect ping-pong memory transfer patterns (wasteful H2D followed by D2H).
    ///
    /// Returns pairs of transfer node IDs that exhibit ping-pong behavior.
    pub fn detect_ping_pong(&self) -> Vec<(ExecutionNodeId, ExecutionNodeId)> {
        let mut patterns = Vec::new();

        // Find transfer nodes
        let transfers: Vec<(usize, &ExecutionNode)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, ExecutionNode::Transfer { .. }))
            .collect();

        // Check for H2D followed by D2H on same data
        for i in 0..transfers.len() {
            for j in (i + 1)..transfers.len() {
                if let (
                    ExecutionNode::Transfer {
                        src: src1,
                        dst: dst1,
                        direction: dir1,
                        bytes: bytes1,
                        ..
                    },
                    ExecutionNode::Transfer {
                        src: src2,
                        dst: dst2,
                        direction: dir2,
                        bytes: bytes2,
                        ..
                    },
                ) = (&transfers[i].1, &transfers[j].1)
                {
                    // Ping-pong: H2D then D2H with matching src/dst and same size
                    let is_ping_pong = (*dir1 == TransferDirection::H2D
                        && *dir2 == TransferDirection::D2H
                        && dst1 == src2
                        && bytes1 == bytes2)
                        || (*dir1 == TransferDirection::D2H
                            && *dir2 == TransferDirection::H2D
                            && src1 == dst2
                            && bytes1 == bytes2);

                    if is_ping_pong {
                        patterns.push((
                            ExecutionNodeId(transfers[i].0 as u32),
                            ExecutionNodeId(transfers[j].0 as u32),
                        ));
                    }
                }
            }
        }

        patterns
    }

    /// Get critical path analysis summary as formatted string.
    pub fn critical_path_summary(&self) -> String {
        let (path, total_ns) = self.critical_path();
        let slack = self.compute_slack();

        let mut output = String::new();
        output.push_str(&format!(
            "Critical Path: {:.2}ms ({} nodes)\n",
            total_ns as f64 / 1_000_000.0,
            path.len()
        ));
        output.push_str("─".repeat(50).as_str());
        output.push('\n');

        for (i, node_id) in path.iter().enumerate() {
            let node = &self.nodes[node_id.0 as usize];
            let timing = self.node_timing_ns(*node_id);
            let node_name = match node {
                ExecutionNode::Layer { index } => format!("Layer {}", index),
                ExecutionNode::Brick { id, .. } => id.name().to_string(),
                ExecutionNode::Kernel { name, .. } => name.clone(),
                ExecutionNode::Function { name, .. } => name.clone(),
                ExecutionNode::Transfer {
                    direction,
                    src,
                    dst,
                    ..
                } => {
                    format!("{:?} {} → {}", direction, src, dst)
                }
                ExecutionNode::AsyncTask {
                    name, poll_count, ..
                } => {
                    format!("{} ({}polls)", name, poll_count)
                }
            };

            let prefix = if i == 0 {
                "┌"
            } else if i == path.len() - 1 {
                "└"
            } else {
                "│"
            };
            output.push_str(&format!(
                "{} {} ({:.1}µs)\n",
                prefix,
                node_name,
                timing as f64 / 1000.0
            ));
        }

        // Show nodes with most slack (parallelization opportunities)
        let mut slack_vec: Vec<_> = slack.iter().collect();
        slack_vec.sort_by(|a, b| b.1.cmp(a.1));

        if slack_vec.iter().any(|(_, &s)| s > 0) {
            output.push_str("\nParallelization Opportunities (high slack):\n");
            for (node_id, &node_slack) in slack_vec.iter().take(5) {
                if node_slack > 0 {
                    let node = &self.nodes[node_id.0 as usize];
                    let node_name = match node {
                        ExecutionNode::Layer { index } => format!("Layer {}", index),
                        ExecutionNode::Brick { id, .. } => id.name().to_string(),
                        ExecutionNode::Kernel { name, .. } => name.clone(),
                        ExecutionNode::Function { name, .. } => name.clone(),
                        ExecutionNode::Transfer {
                            direction,
                            src,
                            dst,
                            ..
                        } => {
                            format!("{:?} {} → {}", direction, src, dst)
                        }
                        ExecutionNode::AsyncTask {
                            name, poll_count, ..
                        } => {
                            format!("{} ({}polls)", name, poll_count)
                        }
                    };
                    output.push_str(&format!(
                        "  {} slack={:.1}µs\n",
                        node_name,
                        node_slack as f64 / 1000.0
                    ));
                }
            }
        }

        output
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.scope_stack.clear();
        self.name_to_id.clear();
    }

    /// Check if scope stack is balanced (empty).
    pub fn is_scope_balanced(&self) -> bool {
        self.scope_stack.is_empty()
    }
}
