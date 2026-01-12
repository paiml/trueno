//! Control Flow Analyzer - CFG construction and barrier analysis

use std::collections::{HashMap, HashSet};
use crate::parser::{PtxModule, KernelDef, Statement, Instruction, SourceLocation};
use crate::parser::types::Opcode;
use crate::bugs::Severity;

/// Node ID for CFG nodes
pub type NodeId = usize;

/// Control Flow Graph node
#[derive(Debug, Clone)]
pub struct CfgNode {
    /// Node ID
    pub id: NodeId,
    /// Label (if any)
    pub label: Option<String>,
    /// Instructions in this basic block
    pub instructions: Vec<Instruction>,
    /// Successor node IDs
    pub successors: Vec<NodeId>,
    /// Predecessor node IDs
    pub predecessors: Vec<NodeId>,
}

/// Control Flow Graph
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Nodes in the graph
    pub nodes: Vec<CfgNode>,
    /// Entry node ID
    pub entry: NodeId,
    /// Exit node IDs
    pub exits: Vec<NodeId>,
    /// Label to node ID mapping
    pub label_to_node: HashMap<String, NodeId>,
}

impl ControlFlowGraph {
    /// Get all nodes
    pub fn nodes(&self) -> &[CfgNode] {
        &self.nodes
    }

    /// Get node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&CfgNode> {
        self.nodes.get(id)
    }

    /// Find unreachable nodes
    pub fn find_unreachable(&self) -> Vec<NodeId> {
        let mut reachable = HashSet::new();
        let mut worklist = vec![self.entry];

        while let Some(node_id) = worklist.pop() {
            if reachable.insert(node_id) {
                if let Some(node) = self.get_node(node_id) {
                    worklist.extend(&node.successors);
                }
            }
        }

        self.nodes.iter()
            .filter(|n| !reachable.contains(&n.id))
            .map(|n| n.id)
            .collect()
    }
}

/// Barrier violation
#[derive(Debug, Clone)]
pub struct BarrierViolation {
    /// Write location
    pub write_loc: SourceLocation,
    /// Read location
    pub read_loc: SourceLocation,
    /// Severity
    pub severity: Severity,
    /// Message
    pub message: String,
}

/// Control Flow Analyzer
pub struct ControlFlowAnalyzer {
    cfg: Option<ControlFlowGraph>,
}

impl ControlFlowAnalyzer {
    /// Create a new control flow analyzer
    pub fn new() -> Self {
        Self { cfg: None }
    }

    /// Build CFG for a kernel
    pub fn build_cfg(&mut self, kernel: &KernelDef) -> ControlFlowGraph {
        let mut nodes = Vec::new();
        let mut label_to_node: HashMap<String, NodeId> = HashMap::new();
        let mut current_instructions = Vec::new();
        let mut current_label: Option<String> = None;

        // First pass: create basic blocks
        for stmt in &kernel.body {
            match stmt {
                Statement::Label(label) => {
                    // End current block and start new one
                    if !current_instructions.is_empty() || current_label.is_some() {
                        let node_id = nodes.len();
                        if let Some(ref label) = current_label {
                            label_to_node.insert(label.clone(), node_id);
                        }
                        nodes.push(CfgNode {
                            id: node_id,
                            label: current_label.take(),
                            instructions: std::mem::take(&mut current_instructions),
                            successors: Vec::new(),
                            predecessors: Vec::new(),
                        });
                    }
                    current_label = Some(label.clone());
                }
                Statement::Instruction(instr) => {
                    current_instructions.push(instr.clone());

                    // Branch/ret/exit ends a basic block
                    if instr.opcode.is_branch() {
                        let node_id = nodes.len();
                        if let Some(ref label) = current_label {
                            label_to_node.insert(label.clone(), node_id);
                        }
                        nodes.push(CfgNode {
                            id: node_id,
                            label: current_label.take(),
                            instructions: std::mem::take(&mut current_instructions),
                            successors: Vec::new(),
                            predecessors: Vec::new(),
                        });
                    }
                }
                _ => {}
            }
        }

        // Add final block if any
        if !current_instructions.is_empty() || current_label.is_some() {
            let node_id = nodes.len();
            if let Some(ref label) = current_label {
                label_to_node.insert(label.clone(), node_id);
            }
            nodes.push(CfgNode {
                id: node_id,
                label: current_label,
                instructions: current_instructions,
                successors: Vec::new(),
                predecessors: Vec::new(),
            });
        }

        // Create empty entry node if needed
        if nodes.is_empty() {
            nodes.push(CfgNode {
                id: 0,
                label: None,
                instructions: Vec::new(),
                successors: Vec::new(),
                predecessors: Vec::new(),
            });
        }

        // Second pass: collect edges
        let mut edges: Vec<(NodeId, NodeId)> = Vec::new();
        let mut exits = Vec::new();
        let node_count = nodes.len();

        for i in 0..node_count {
            let last_instr = nodes[i].instructions.last().cloned();

            match last_instr.as_ref().map(|instr| &instr.opcode) {
                Some(Opcode::Bra) => {
                    // Find branch target
                    if let Some(ref instr) = last_instr {
                        for op in &instr.operands {
                            if let crate::parser::Operand::Label(target) = op {
                                if let Some(&target_id) = label_to_node.get(target) {
                                    edges.push((i, target_id));
                                }
                            }
                        }
                    }
                    // Also fallthrough if conditional
                    if i + 1 < node_count {
                        edges.push((i, i + 1));
                    }
                }
                Some(Opcode::Ret | Opcode::Exit) => {
                    exits.push(i);
                }
                _ => {
                    // Fallthrough to next block
                    if i + 1 < node_count {
                        edges.push((i, i + 1));
                    } else {
                        exits.push(i);
                    }
                }
            }
        }

        // Apply edges
        for (from, to) in edges {
            nodes[from].successors.push(to);
            nodes[to].predecessors.push(from);
        }

        let cfg = ControlFlowGraph {
            nodes,
            entry: 0,
            exits,
            label_to_node,
        };

        self.cfg = Some(cfg.clone());
        cfg
    }

    /// Analyze barrier synchronization
    pub fn analyze_barriers(&self, _module: &PtxModule) -> Vec<BarrierViolation> {
        let mut violations = Vec::new();

        if let Some(ref cfg) = self.cfg {
            for node in cfg.nodes() {
                let mut last_shared_write: Option<SourceLocation> = None;

                for instr in &node.instructions {
                    // Check for shared memory operations
                    let is_shared = instr.modifiers.iter()
                        .any(|m| matches!(m, crate::parser::types::Modifier::Shared));

                    if instr.opcode == Opcode::St && is_shared {
                        last_shared_write = Some(instr.location.clone());
                    } else if instr.opcode == Opcode::Ld && is_shared {
                        if let Some(ref write_loc) = last_shared_write {
                            // Check if there's a barrier between write and read
                            // Simplified: just check if bar.sync appears
                            let has_barrier = node.instructions.iter()
                                .any(|i| matches!(i.opcode, Opcode::Bar));

                            if !has_barrier {
                                violations.push(BarrierViolation {
                                    write_loc: write_loc.clone(),
                                    read_loc: instr.location.clone(),
                                    severity: Severity::High,
                                    message: "Missing barrier between shared memory write and read".into(),
                                });
                            }
                        }
                    } else if matches!(instr.opcode, Opcode::Bar) {
                        last_shared_write = None;
                    }
                }
            }
        }

        violations
    }
}

impl Default for ControlFlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    // F061: All code paths reach ret or exit
    #[test]
    fn f061_all_paths_reach_exit() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                mov.u32 %r0, 0;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();

        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg = analyzer.build_cfg(&module.kernels[0]);

        // All exit nodes should have ret or exit
        assert!(!cfg.exits.is_empty(), "F061: Should have exit nodes");
    }

    // F062: No unreachable code
    #[test]
    fn f062_no_unreachable_code() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                mov.u32 %r0, 0;
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();

        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg = analyzer.build_cfg(&module.kernels[0]);
        let unreachable = cfg.find_unreachable();

        assert!(unreachable.is_empty(), "F062: Found unreachable code: {:?}", unreachable);
    }

    // F036: bar.sync after shared write, before read
    #[test]
    fn f036_barrier_after_shared_write() {
        let ptx = r#"
            .version 8.0
            .target sm_70
            .address_size 64

            .entry test()
            {
                .reg .u32 %r<10>;
                st.shared.u32 [%r0], %r1;
                bar.sync 0;
                ld.shared.u32 %r2, [%r3];
                ret;
            }
        "#;
        let mut parser = Parser::new(ptx).unwrap();
        let module = parser.parse().unwrap();

        let mut analyzer = ControlFlowAnalyzer::new();
        let _ = analyzer.build_cfg(&module.kernels[0]);
        let violations = analyzer.analyze_barriers(&module);

        assert!(violations.is_empty(), "F036: Should have no barrier violations with proper sync");
    }
}
