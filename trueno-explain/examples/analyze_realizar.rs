//! Analyze realizar hand-rolled PTX for bugs
//!
//! Run: `cargo run -p trueno-explain --example analyze_realizar`

use std::fs;
use trueno_explain::{BugSeverity, PtxBugAnalyzer};

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                    REALIZAR PTX BUG ANALYSIS                                 ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let ptx_dir = "/tmp/realizar_ptx_analysis";
    let files = ["bias_activation.ptx", "gemm_fp16_wmma.ptx", "fused_q4k_q8_dot.ptx", "multi_head_attention.ptx"];

    let mut total_bugs = 0;
    let mut p0_total = 0;
    let mut p1_total = 0;
    let mut p2_total = 0;

    // Use strict mode to catch all issues
    let analyzer = PtxBugAnalyzer::strict();

    for file in &files {
        let path = format!("{}/{}", ptx_dir, file);
        let ptx = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(e) => {
                println!("❌ Could not read {}: {}", file, e);
                continue;
            }
        };

        let result = analyzer.analyze(&ptx);
        let p0 = result.count_by_severity(BugSeverity::Critical);
        let p1 = result.count_by_severity(BugSeverity::High);
        let p2 = result.count_by_severity(BugSeverity::Medium);

        total_bugs += result.bugs.len();
        p0_total += p0;
        p1_total += p1;
        p2_total += p2;

        let icon = if p0 > 0 {
            "🔴"
        } else if p1 > 0 {
            "🟡"
        } else if p2 > 0 {
            "🟠"
        } else {
            "✅"
        };

        println!("{} {} - {} bugs ({} P0, {} P1, {} P2)", icon, file, result.bugs.len(), p0, p1, p2);

        for bug in &result.bugs {
            println!("   └─ [{}] {}: {}", bug.class.severity(), bug.class.code(), bug.message);
            if bug.line > 0 {
                println!("      Line {}: {}", bug.line, bug.instruction);
            }
            if let Some(fix) = &bug.fix {
                println!("      Fix: {}", fix);
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("REALIZAR PTX SUMMARY");
    println!("{}", "=".repeat(80));
    println!("  Files analyzed: {}", files.len());
    println!("  Total bugs: {}", total_bugs);
    println!("  🔴 P0 Critical: {}", p0_total);
    println!("  🟡 P1 High: {}", p1_total);
    println!("  🟠 P2 Medium: {}", p2_total);

    if p0_total > 0 {
        println!("\n⚠️  CRITICAL BUGS FOUND in realizar - these need porting to trueno-gpu!");
    }
}
