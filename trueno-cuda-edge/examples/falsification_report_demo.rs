//! Falsification Protocol Demo
//!
//! Demonstrates the 50-point falsification checklist and coverage tracking.
//!
//! Run with: cargo run --example falsification_report_demo

use trueno_cuda_edge::falsification::{
    all_claims, claims_for_framework, ClaimStatus, FalsificationReport, Framework,
};

fn main() {
    println!("=== Falsification Protocol Demo ===\n");

    // 1. Protocol Overview
    println!("1. Protocol Overview");
    println!("   ──────────────────");

    let claims = all_claims();
    println!("   Total claims: {}", claims.len());

    let frameworks = [
        Framework::NullFuzzer,
        Framework::ShmemProber,
        Framework::LifecycleChaos,
        Framework::QuantOracle,
        Framework::PtxPoison,
        Framework::Supervisor,
    ];

    println!("\n   Claims by framework:");
    for fw in &frameworks {
        let count = claims_for_framework(*fw).len();
        println!("   │ {:<20} {:>2} claims", format!("{}", fw), count);
    }

    println!();

    // 2. Sample Claims
    println!("2. Sample Claims (first 3 per framework)");
    println!("   ──────────────────────────────────────");

    for fw in &frameworks {
        println!("\n   {}:", fw);
        for claim in claims_for_framework(*fw).iter().take(3) {
            println!(
                "   │ {} [P{}]: {}",
                claim.id, claim.priority, claim.description
            );
        }
    }

    println!();

    // 3. Report Tracking
    println!("3. Report Tracking");
    println!("   ─────────────────");

    let mut report = FalsificationReport::new();

    // Simulate verification progress
    println!("\n   Initial state:");
    println!(
        "   │ Pending: {}",
        report.count_by_status(ClaimStatus::Pending)
    );
    println!("   │ Coverage: {:.1}%", report.coverage() * 100.0);

    // Mark some claims as verified
    for claim in claims_for_framework(Framework::NullFuzzer) {
        report.mark_verified(claim.id);
    }

    println!("\n   After verifying NullFuzzer (10 claims):");
    println!(
        "   │ Verified: {}",
        report.count_by_status(ClaimStatus::Verified)
    );
    println!(
        "   │ Pending: {}",
        report.count_by_status(ClaimStatus::Pending)
    );
    println!("   │ Coverage: {:.1}%", report.coverage() * 100.0);

    // Mark one as violated (bug found!)
    report.mark_violated("SP-001");

    println!("\n   After finding bug in SP-001:");
    println!(
        "   │ Violated: {}",
        report.count_by_status(ClaimStatus::Violated)
    );
    println!("   │ Coverage: {:.1}%", report.coverage() * 100.0);

    // Skip hardware-dependent claims
    for claim in ["LC-005", "LC-006", "LC-007", "LC-008"] {
        report.mark_skipped(claim);
    }

    println!("\n   After skipping hardware-dependent claims:");
    println!(
        "   │ Skipped: {}",
        report.count_by_status(ClaimStatus::Skipped)
    );
    println!("   │ Coverage: {:.1}%", report.coverage() * 100.0);

    println!();

    // 4. Completion Check
    println!("4. Completion Check");
    println!("   ──────────────────");

    println!("   Is complete: {}", report.is_complete());
    println!(
        "   Remaining pending: {}",
        report.count_by_status(ClaimStatus::Pending)
    );

    // Complete all remaining
    for claim in all_claims() {
        if report.status(claim.id) == Some(ClaimStatus::Pending) {
            report.mark_verified(claim.id);
        }
    }

    println!("\n   After verifying all remaining:");
    println!("   │ Is complete: {}", report.is_complete());
    println!("   │ Final coverage: {:.1}%", report.coverage() * 100.0);

    println!();

    // 5. Violated Claims Report
    println!("5. Violated Claims Report");
    println!("   ────────────────────────");

    let violated = report.violated_claims();
    if violated.is_empty() {
        println!("   No violations found (all claims verified).");
    } else {
        println!("   Violations found:");
        for id in &violated {
            println!("   │ {} - Bug detected!", id);
        }
    }

    println!();

    // 6. Framework Summary
    println!("6. Framework Summary");
    println!("   ───────────────────");

    let grouped = report.by_framework();
    for fw in &frameworks {
        if let Some(claims) = grouped.get(fw) {
            let verified = claims
                .iter()
                .filter(|(_, s)| *s == ClaimStatus::Verified)
                .count();
            let violated = claims
                .iter()
                .filter(|(_, s)| *s == ClaimStatus::Violated)
                .count();
            let skipped = claims
                .iter()
                .filter(|(_, s)| *s == ClaimStatus::Skipped)
                .count();

            println!(
                "   │ {:<20} ✓{:>2} verified  ✗{:>2} violated  ○{:>2} skipped",
                format!("{}", fw),
                verified,
                violated,
                skipped
            );
        }
    }

    println!("\n=== Demo Complete ===");
}
