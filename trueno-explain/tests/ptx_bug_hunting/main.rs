//! PTX Bug Hunting - Rigorous Edge Case Testing
//!
//! Inspired by bashrs/rash/tests/parser_bug_hunting.rs which found 25 bugs.
//! This module tests PTX analysis against edge cases to find bugs.
//!
//! Run: cargo test -p trueno-explain --test ptx_bug_hunting

#![allow(clippy::unwrap_used)]

use trueno_explain::{
    Analyzer, BugSeverity, PtxAnalyzer, PtxBugAnalyzer, PtxBugClass, PtxCoverageTracker,
    PtxCoverageTrackerBuilder,
};
use trueno_gpu::kernels::{
    GemmKernel, Kernel, Q5KKernel, Q6KKernel, QuantizeKernel, SoftmaxKernel,
};

mod shared_memory;
mod barrier_sync;
mod loop_branch;
mod register_and_entry;
mod real_kernels;
mod severity_and_reporting;
mod coverage_tracking;
mod extended_detectors;
