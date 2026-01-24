//! Warp-level operation emission
//!
//! Handles: ShflDown, ShflIdx, Vote, VoteBallot, Popc, Bfind, Clz, Bfe, Bfi

use crate::ptx::instructions::PtxOp;

/// Emit warp opcode to the output string
pub(crate) fn emit_warp_opcode(op: &PtxOp, s: &mut String) {
    match op {
        PtxOp::ShflDown => s.push_str("shfl.sync.down.b32"),
        PtxOp::ShflIdx => s.push_str("shfl.sync.idx.b32"),
        PtxOp::Vote | PtxOp::VoteBallot => s.push_str("vote.sync.ballot.b32"),
        PtxOp::Popc => s.push_str("popc"),
        PtxOp::Bfind => s.push_str("bfind"),
        PtxOp::Clz => s.push_str("clz"),
        PtxOp::Bfe => s.push_str("bfe"),
        PtxOp::Bfi => s.push_str("bfi"),
        _ => {}
    }
}

/// Check if this is a warp operation
pub(crate) fn is_warp_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::ShflDown
            | PtxOp::ShflIdx
            | PtxOp::Vote
            | PtxOp::VoteBallot
            | PtxOp::Popc
            | PtxOp::Bfind
            | PtxOp::Clz
            | PtxOp::Bfe
            | PtxOp::Bfi
    )
}

/// Check if this op requires skipping the type suffix
pub(crate) fn skip_type_for_warp_op(op: &PtxOp) -> bool {
    matches!(op, PtxOp::ShflDown | PtxOp::ShflIdx | PtxOp::Vote | PtxOp::VoteBallot)
}
