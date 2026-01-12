//! Static Analysis Passes
//!
//! This module contains the static analysis passes for PTX code:
//! - Type Checker: Validates register types match operations
//! - Control Flow Analyzer: Constructs CFG, validates barrier synchronization
//! - Data Flow Analyzer: Tracks value propagation, detects "loaded value" bug
//! - Address Space Validator: Validates correct address space usage

mod type_checker;
mod control_flow;
mod data_flow;
mod address_space;

pub use type_checker::TypeChecker;
pub use control_flow::{ControlFlowAnalyzer, ControlFlowGraph, CfgNode, BarrierViolation};
pub use data_flow::{DataFlowAnalyzer, ValueSource, UsePoint, LoadedValueBug, ComputedAddrFromLoadedBug};
pub use address_space::{AddressSpaceValidator, GenericSharedBug};
