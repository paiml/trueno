//! Execution Graph and Brick Profiling Types
//!
//! This module contains types for execution path tracking and profiling:
//!
//! - **PAR-073**: BrickSample, BrickBottleneck - foundational profiling primitives
//! - **PAR-200**: BrickId, BrickCategory, SyncMode - O(1) hot path brick identification
//! - **PAR-201**: ExecutionGraph, ExecutionNode, etc. - full execution hierarchy tracking

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// BrickProfiler: FOUNDATIONAL Real-Time Per-Brick Timing (PAR-073)
// ============================================================================

/// Individual brick timing sample.
/// Pure Rust timing using `std::time::Instant`.
#[derive(Debug, Clone, Copy)]
pub struct BrickSample {
    /// Brick name hash (for fast lookup)
    pub brick_id: u64,
    /// Elapsed time in nanoseconds
    pub elapsed_ns: u64,
    /// Number of elements processed
    pub elements: u64,
}

/// Bottleneck classification for roofline analysis (PMAT-451)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrickBottleneck {
    /// Not classified
    #[default]
    Unknown,
    /// Limited by memory bandwidth
    Memory,
    /// Limited by compute throughput
    Compute,
}

impl fmt::Display for BrickBottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrickBottleneck::Unknown => write!(f, "unknown"),
            BrickBottleneck::Memory => write!(f, "memory"),
            BrickBottleneck::Compute => write!(f, "compute"),
        }
    }
}

// ============================================================================
// PAR-200: BrickProfiler v2 - O(1) Hot Path with BrickId Enum
// ============================================================================

/// Well-known brick types for O(1) lookup on hot path.
///
/// PAR-200: Eliminates string allocation and HashMap hashing during profiling.
/// Use `BrickId::Custom` with string fallback for unknown brick types.
///
/// # Example
/// ```rust
/// use trueno::brick::BrickId;
///
/// let brick = BrickId::RmsNorm;
/// assert_eq!(brick.category(), trueno::brick::BrickCategory::Norm);
/// assert_eq!(brick.name(), "RmsNorm");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BrickId {
    // Normalization (0-1)
    /// RMS normalization layer
    RmsNorm = 0,
    /// Layer normalization
    LayerNorm = 1,

    // Attention (2-7)
    /// Q/K/V projection (combined or separate)
    QkvProjection = 2,
    /// Rotary position embedding
    RopeEmbedding = 3,
    /// Attention score computation (Q @ K^T)
    AttentionScore = 4,
    /// Attention softmax
    AttentionSoftmax = 5,
    /// Attention output (scores @ V)
    AttentionOutput = 6,
    /// Output projection after attention
    OutputProjection = 7,

    // FFN (8-11)
    /// Gate projection (for gated FFN)
    GateProjection = 8,
    /// Up projection
    UpProjection = 9,
    /// SiLU/GELU/ReLU activation
    Activation = 10,
    /// Down projection
    DownProjection = 11,

    // Other (12-14)
    /// Token embedding lookup
    Embedding = 12,
    /// Language model head (logits)
    LmHead = 13,
    /// Token sampling
    Sampling = 14,
}

impl BrickId {
    /// Number of well-known brick types.
    pub const COUNT: usize = 15;

    /// Get the category for hierarchical aggregation.
    #[inline]
    pub fn category(self) -> BrickCategory {
        match self {
            Self::RmsNorm | Self::LayerNorm => BrickCategory::Norm,
            Self::QkvProjection
            | Self::RopeEmbedding
            | Self::AttentionScore
            | Self::AttentionSoftmax
            | Self::AttentionOutput
            | Self::OutputProjection => BrickCategory::Attention,
            Self::GateProjection | Self::UpProjection | Self::Activation | Self::DownProjection => {
                BrickCategory::Ffn
            }
            Self::Embedding | Self::LmHead | Self::Sampling => BrickCategory::Other,
        }
    }

    /// Get the string name of this brick.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RmsNorm => "RmsNorm",
            Self::LayerNorm => "LayerNorm",
            Self::QkvProjection => "QkvProjection",
            Self::RopeEmbedding => "RopeEmbedding",
            Self::AttentionScore => "AttentionScore",
            Self::AttentionSoftmax => "AttentionSoftmax",
            Self::AttentionOutput => "AttentionOutput",
            Self::OutputProjection => "OutputProjection",
            Self::GateProjection => "GateProjection",
            Self::UpProjection => "UpProjection",
            Self::Activation => "Activation",
            Self::DownProjection => "DownProjection",
            Self::Embedding => "Embedding",
            Self::LmHead => "LmHead",
            Self::Sampling => "Sampling",
        }
    }

    /// Try to parse a string into a BrickId.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "RmsNorm" => Some(Self::RmsNorm),
            "LayerNorm" => Some(Self::LayerNorm),
            "QkvProjection" | "Qkv" => Some(Self::QkvProjection),
            "RopeEmbedding" | "Rope" | "RoPE" => Some(Self::RopeEmbedding),
            "AttentionScore" => Some(Self::AttentionScore),
            "AttentionSoftmax" | "Softmax" => Some(Self::AttentionSoftmax),
            "AttentionOutput" => Some(Self::AttentionOutput),
            "OutputProjection" | "OutProj" => Some(Self::OutputProjection),
            "GateProjection" | "Gate" => Some(Self::GateProjection),
            "UpProjection" | "Up" => Some(Self::UpProjection),
            "Activation" | "SiLU" | "GELU" | "ReLU" => Some(Self::Activation),
            "DownProjection" | "Down" => Some(Self::DownProjection),
            "Embedding" | "Embed" => Some(Self::Embedding),
            "LmHead" | "Head" => Some(Self::LmHead),
            "Sampling" | "Sample" => Some(Self::Sampling),
            _ => None,
        }
    }
}

impl fmt::Display for BrickId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Category for hierarchical aggregation of brick statistics.
///
/// PAR-200: Groups related bricks for high-level performance analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum BrickCategory {
    /// Normalization layers (RmsNorm, LayerNorm)
    Norm = 0,
    /// Attention mechanism (QKV, RoPE, scores, softmax, output)
    Attention = 1,
    /// Feed-forward network (gate, up, activation, down)
    Ffn = 2,
    /// Other operations (embedding, lm_head, sampling)
    #[default]
    Other = 3,
}

impl BrickCategory {
    /// Number of categories.
    pub const COUNT: usize = 4;

    /// Get the string name of this category.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Norm => "Norm",
            Self::Attention => "Attention",
            Self::Ffn => "FFN",
            Self::Other => "Other",
        }
    }
}

impl fmt::Display for BrickCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Synchronization mode for GPU profiling.
///
/// PAR-200: Controls the trade-off between accuracy and overhead.
///
/// # Performance Characteristics
///
/// | Mode | Overhead | Accuracy | Use Case |
/// |------|----------|----------|----------|
/// | `Immediate` | ~200% | Exact per-kernel | Debugging |
/// | `PerLayer` | ~20% | Per-layer exact | Development |
/// | `Deferred` | ~5% | Approximate | Production |
/// | `None` | 0% | N/A | Disabled |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Sync after each kernel (accurate but slow).
    /// Best for debugging and detailed optimization.
    Immediate,
    /// Sync once per transformer layer.
    /// Good balance for development.
    PerLayer,
    /// Sync once per forward pass (fast, approximate).
    /// Best for production profiling.
    #[default]
    Deferred,
    /// No synchronization (profiling disabled or CPU-only).
    None,
}

// ============================================================================
// PAR-201: Execution Path Graph Types
// ============================================================================

/// Node ID in the execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionNodeId(pub u32);

/// Execution graph node types.
///
/// PAR-201: Represents different levels of the execution hierarchy.
#[derive(Debug, Clone)]
pub enum ExecutionNode {
    /// High-level brick (BrickId from v2)
    Brick {
        id: BrickId,
        timing_ns: u64,
        elements: u64,
    },
    /// GPU kernel launch
    Kernel {
        name: String,
        /// FNV-1a hash of PTX source for identity
        ptx_hash: u64,
        /// Grid dimensions (blocks)
        grid: (u32, u32, u32),
        /// Block dimensions (threads)
        block: (u32, u32, u32),
        /// Shared memory bytes
        shared_mem: u32,
        /// Kernel execution time in nanoseconds (Phase 9: for CPA)
        timing_ns: Option<u64>,
        /// Arithmetic intensity (FLOPs/byte) for roofline analysis (Phase 9)
        arithmetic_intensity: Option<f32>,
        /// Achieved throughput in TFLOP/s (Phase 9)
        achieved_tflops: Option<f32>,
    },
    /// Memory transfer operation (Phase 9: data movement topology)
    Transfer {
        /// Source location description
        src: String,
        /// Destination location description
        dst: String,
        /// Bytes transferred
        bytes: u64,
        /// Transfer direction
        direction: TransferDirection,
        /// Transfer time in nanoseconds
        timing_ns: Option<u64>,
    },
    /// Rust function (from DWARF or manual annotation)
    Function {
        name: String,
        file: Option<String>,
        line: Option<u32>,
    },
    /// Transformer layer grouping
    Layer { index: u32 },
    /// Phase 11 (E.9.4): Async task metrics for poll efficiency tracking
    AsyncTask {
        /// Task name for identification
        name: String,
        /// Number of times poll() was called
        poll_count: u64,
        /// Number of times poll() returned Pending
        yield_count: u64,
        /// Total time spent in poll() (nanoseconds)
        total_poll_ns: u64,
    },
}

impl ExecutionNode {
    /// Get the display name of this node.
    pub fn name(&self) -> String {
        match self {
            Self::Brick { id, .. } => id.name().to_string(),
            Self::Kernel { name, .. } => name.clone(),
            Self::Function { name, .. } => name.clone(),
            Self::Layer { index } => format!("Layer{}", index),
            Self::Transfer {
                src,
                dst,
                direction,
                ..
            } => {
                let dir = match direction {
                    TransferDirection::H2D => "H2D",
                    TransferDirection::D2H => "D2H",
                    TransferDirection::D2D => "D2D",
                };
                format!("{}:{}->{}", dir, src, dst)
            }
            Self::AsyncTask { name, .. } => name.clone(),
        }
    }

    /// Check if this is a kernel node.
    pub fn is_kernel(&self) -> bool {
        matches!(self, Self::Kernel { .. })
    }

    /// Check if this is a brick node.
    pub fn is_brick(&self) -> bool {
        matches!(self, Self::Brick { .. })
    }

    /// Check if this is a transfer node.
    pub fn is_transfer(&self) -> bool {
        matches!(self, Self::Transfer { .. })
    }

    /// Get timing if available (bricks, kernels, and transfers).
    pub fn timing_ns(&self) -> Option<u64> {
        match self {
            Self::Brick { timing_ns, .. } => Some(*timing_ns),
            Self::Kernel { timing_ns, .. } => *timing_ns,
            Self::Transfer { timing_ns, .. } => *timing_ns,
            _ => None,
        }
    }

    /// Get PTX hash if available (kernels only).
    pub fn ptx_hash(&self) -> Option<u64> {
        match self {
            Self::Kernel { ptx_hash, .. } => Some(*ptx_hash),
            _ => None,
        }
    }

    /// Get arithmetic intensity if available (kernels only, Phase 9).
    pub fn arithmetic_intensity(&self) -> Option<f32> {
        match self {
            Self::Kernel {
                arithmetic_intensity,
                ..
            } => *arithmetic_intensity,
            _ => None,
        }
    }

    /// Get achieved TFLOP/s if available (kernels only, Phase 9).
    pub fn achieved_tflops(&self) -> Option<f32> {
        match self {
            Self::Kernel {
                achieved_tflops, ..
            } => *achieved_tflops,
            _ => None,
        }
    }

    /// Get transfer bytes if available (transfers only, Phase 9).
    pub fn transfer_bytes(&self) -> Option<u64> {
        match self {
            Self::Transfer { bytes, .. } => Some(*bytes),
            _ => None,
        }
    }
}

/// Edge types in execution graph.
///
/// PAR-201: Describes relationships between execution nodes.
/// Phase 9 (E.7.12): Added DependsOn and Transfer for advanced profiling.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    /// Function calls function
    Calls,
    /// Brick contains sub-operations
    Contains,
    /// Function launches GPU kernel
    Launches,
    /// Temporal sequence (A happens before B)
    Sequence,
    /// Dependency edge for critical path analysis (CUDA events, stream sync)
    /// PAR-201 Phase 9: CPA requires tracking true dependencies vs containment
    DependsOn,
    /// Data transfer edge with byte count (H2D/D2H/D2D)
    /// PAR-201 Phase 9: For data movement topology and ping-pong detection
    Transfer {
        /// Bytes transferred
        bytes: u64,
        /// Transfer direction
        direction: TransferDirection,
    },
}

/// Direction of memory transfer.
///
/// PAR-201 Phase 9: Used with EdgeType::Transfer for data movement analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to Device
    H2D,
    /// Device to Host
    D2H,
    /// Device to Device
    D2D,
}

/// An edge in the execution graph.
#[derive(Debug, Clone)]
pub struct ExecutionEdge {
    /// Source node ID
    pub src: ExecutionNodeId,
    /// Destination node ID
    pub dst: ExecutionNodeId,
    /// Edge type
    pub edge_type: EdgeType,
    /// Optional weight (e.g., call count, timing)
    pub weight: f32,
}

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
                ExecutionNode::Layer { index } => {
                    (format!("Layer {}", index), "style=filled,fillcolor=lightblue")
                }
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
                children_map
                    .entry(edge.src.0)
                    .or_default()
                    .push(edge.dst.0);
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
                ExecutionNode::Layer { index } => {
                    (format!("Layer {}", index), None, layer_color)
                }
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
            let mut root =
                TreeNode::new(u64::MAX, "Execution Graph").with_color(Color::new(0.9, 0.9, 0.9, 1.0));
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
                children_map
                    .entry(edge.src.0)
                    .or_default()
                    .push(edge.dst.0);
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
                    build_tree(graph, child_id, children_map, &new_prefix, new_connector, output);
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
                max_pred =
                    max_pred.max(earliest[pred as usize] + self.node_timing_ns(ExecutionNodeId(pred)));
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
                    direction, src, dst, ..
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
                            direction, src, dst, ..
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

/// PTX kernel registry for execution graph correlation.
///
/// PAR-201: Maps PTX hashes to source code for debugging and analysis.
#[derive(Debug, Default)]
pub struct PtxRegistry {
    /// Hash → (kernel_name, ptx_source, file_path)
    kernels: HashMap<u64, (String, String, Option<std::path::PathBuf>)>,
}

impl PtxRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register PTX source code.
    ///
    /// # Arguments
    /// - `name`: Kernel name (e.g., "batched_q4k_gemv")
    /// - `ptx`: PTX source code
    /// - `path`: Optional file path for source correlation
    pub fn register(&mut self, name: &str, ptx: &str, path: Option<&std::path::Path>) {
        let hash = Self::hash_ptx(ptx);
        self.kernels.insert(
            hash,
            (
                name.to_string(),
                ptx.to_string(),
                path.map(|p| p.to_path_buf()),
            ),
        );
    }

    /// Compute FNV-1a hash of PTX source.
    #[inline]
    pub fn hash_ptx(ptx: &str) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in ptx.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Lookup PTX source by hash.
    pub fn lookup(&self, hash: u64) -> Option<&str> {
        self.kernels.get(&hash).map(|(_, ptx, _)| ptx.as_str())
    }

    /// Lookup kernel name by hash.
    pub fn lookup_name(&self, hash: u64) -> Option<&str> {
        self.kernels.get(&hash).map(|(name, _, _)| name.as_str())
    }

    /// Lookup file path by hash.
    pub fn lookup_path(&self, hash: u64) -> Option<&std::path::Path> {
        self.kernels
            .get(&hash)
            .and_then(|(_, _, path)| path.as_deref())
    }

    /// Get all registered hashes.
    pub fn hashes(&self) -> impl Iterator<Item = u64> + '_ {
        self.kernels.keys().copied()
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

/// Aggregated statistics for a brick category.
#[derive(Debug, Clone, Copy, Default)]
pub struct CategoryStats {
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// Total samples
    pub count: u64,
}

impl CategoryStats {
    /// Average time per sample in microseconds.
    #[inline]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements per second.
    #[inline]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Percentage of total time (given total_ns across all categories).
    #[inline]
    pub fn percentage(&self, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            100.0 * self.total_ns as f64 / total as f64
        }
    }
}

/// Accumulated per-brick statistics.
#[derive(Debug, Clone, Default)]
pub struct BrickStats {
    /// Brick name
    pub name: String,
    /// Total samples collected
    pub count: u64,
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Min elapsed time (nanoseconds)
    pub min_ns: u64,
    /// Max elapsed time (nanoseconds)
    pub max_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// PMAT-451: Total bytes processed (for throughput calculation)
    pub total_bytes: u64,
    /// PMAT-451: Total compressed bytes (for compression ratio)
    pub total_compressed_bytes: u64,
    /// PMAT-451: Bottleneck classification
    pub bottleneck: BrickBottleneck,
    /// Phase 11 (E.9.2): Total CPU cycles (from RDTSCP/CNTVCT)
    pub total_cycles: u64,
    /// Phase 11: Minimum CPU cycles observed
    pub min_cycles: u64,
    /// Phase 11: Maximum CPU cycles observed
    pub max_cycles: u64,
}

impl BrickStats {
    /// Create new stats for a brick.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            count: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            total_elements: 0,
            total_bytes: 0,
            total_compressed_bytes: 0,
            bottleneck: BrickBottleneck::Unknown,
            total_cycles: 0,
            min_cycles: u64::MAX,
            max_cycles: 0,
        }
    }

    /// Add a sample to statistics.
    pub fn add_sample(&mut self, elapsed_ns: u64, elements: u64) {
        self.count += 1;
        self.total_ns += elapsed_ns;
        self.min_ns = self.min_ns.min(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
        self.total_elements += elements;
    }

    /// Phase 11 (E.9.2): Add a sample with CPU cycle count.
    ///
    /// Use this for frequency-invariant performance analysis.
    /// Cycles are immune to CPU frequency scaling (turbo boost).
    pub fn add_sample_with_cycles(&mut self, elapsed_ns: u64, elements: u64, cycles: u64) {
        self.add_sample(elapsed_ns, elements);
        self.total_cycles += cycles;
        self.min_cycles = self.min_cycles.min(cycles);
        self.max_cycles = self.max_cycles.max(cycles);
    }

    /// Phase 11: Cycles per element (frequency-invariant throughput metric).
    ///
    /// Lower is better. This metric is immune to CPU frequency scaling.
    #[must_use]
    pub fn cycles_per_element(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            self.total_cycles as f64 / self.total_elements as f64
        }
    }

    /// Phase 11: Average cycles per sample.
    #[must_use]
    pub fn avg_cycles(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_cycles as f64 / self.count as f64
        }
    }

    /// Phase 11: Estimated IPC (Instructions Per Cycle).
    ///
    /// Approximation assuming ~1 instruction per element for simple ops.
    /// - Low IPC (<1.0): Memory stalls (cache misses, memory latency)
    /// - High IPC (>2.0): Compute bound (efficient execution)
    #[must_use]
    pub fn estimated_ipc(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            // Rough approximation: assume 1 instruction per element
            self.total_elements as f64 / self.total_cycles as f64
        }
    }

    /// Phase 11: Diagnose bottleneck based on cycles vs time ratio.
    ///
    /// High cycles + low time = likely cache misses
    /// Low cycles + high time = likely CPU throttling or context switches
    #[must_use]
    pub fn diagnose_from_cycles(&self) -> &'static str {
        if self.total_cycles == 0 || self.total_ns == 0 {
            return "insufficient data";
        }

        let ipc = self.estimated_ipc();
        let ns_per_cycle = self.total_ns as f64 / self.total_cycles as f64;

        // Typical CPU runs at ~3GHz, so 1 cycle ≈ 0.33ns
        // If ns_per_cycle >> 0.33, we're seeing stalls or throttling
        if ipc < 0.5 {
            "memory-bound (low IPC, likely cache misses)"
        } else if ipc > 2.0 {
            "compute-bound (efficient)"
        } else if ns_per_cycle > 1.0 {
            "throttled or context-switched"
        } else {
            "balanced"
        }
    }

    /// PMAT-451: Add a sample with byte metrics for compression workloads.
    ///
    /// # Arguments
    /// - `elapsed_ns`: Time taken in nanoseconds
    /// - `elements`: Number of elements processed (e.g., pages)
    /// - `input_bytes`: Original uncompressed size
    /// - `output_bytes`: Compressed output size
    pub fn add_sample_with_bytes(
        &mut self,
        elapsed_ns: u64,
        elements: u64,
        input_bytes: u64,
        output_bytes: u64,
    ) {
        self.add_sample(elapsed_ns, elements);
        self.total_bytes += input_bytes;
        self.total_compressed_bytes += output_bytes;
    }

    /// PMAT-451: Calculate compression ratio (input_size / output_size).
    /// Returns 1.0 if no compression data available.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes == 0 {
            1.0
        } else {
            self.total_bytes as f64 / self.total_compressed_bytes as f64
        }
    }

    /// PMAT-451: Calculate throughput in GB/s.
    /// Based on total input bytes processed.
    #[must_use]
    pub fn throughput_gbps(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            let bytes_per_ns = self.total_bytes as f64 / self.total_ns as f64;
            bytes_per_ns * 1e9 / 1e9 // Convert to GB/s (ns to sec, bytes to GB)
        }
    }

    /// PMAT-451: Set bottleneck classification.
    pub fn set_bottleneck(&mut self, bottleneck: BrickBottleneck) {
        self.bottleneck = bottleneck;
    }

    /// PMAT-451: Get bottleneck classification.
    #[must_use]
    pub fn get_bottleneck(&self) -> BrickBottleneck {
        self.bottleneck
    }

    /// Average time in microseconds.
    #[must_use]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements/second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Throughput in tokens/second (alias for throughput).
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        self.throughput()
    }

    /// Minimum time in microseconds.
    #[must_use]
    pub fn min_us(&self) -> f64 {
        if self.min_ns == u64::MAX {
            0.0
        } else {
            self.min_ns as f64 / 1000.0
        }
    }

    /// Maximum time in microseconds.
    #[must_use]
    pub fn max_us(&self) -> f64 {
        self.max_ns as f64 / 1000.0
    }
}

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
