//! Double-Blind Verification Framework (PMAT-020)
//!
//! Implements the Double-Blind Verification protocol per §36.2 of cbtop spec.
//! Separation of Dev (implementation) and QA (verification) roles with
//! black-box falsification attempts.
//!
//! # Protocol
//!
//! 1. **Group A (Dev)**: Implements feature and claims "Falsification Passed"
//! 2. **Group B (QA)**: Receives only binary + F-criteria (no source)
//! 3. **Blind Test**: Group B attempts to falsify the binary black-box
//! 4. **Confirmation**: Only if Group B fails to falsify is release approved
//!
//! # Citations
//!
//! - [Rosenthal & Fode 1963] "Psychology of the Scientist: Experimenter Bias" Psychological Bulletin
//! - [Holman et al. 2015] "A Systematic Review of Double-Blind Experiments in SE" IEEE TSE

use std::collections::HashMap;
use std::time::SystemTime;

/// Role in the double-blind verification process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Developer - implements features, makes claims
    Dev,
    /// Quality Assurance - verifies claims black-box
    Qa,
    /// System - makes release decisions
    System,
}

impl Role {
    /// Get role name
    pub fn name(&self) -> &'static str {
        match self {
            Role::Dev => "Developer",
            Role::Qa => "QA",
            Role::System => "System",
        }
    }

    /// Check if role can make claims
    pub fn can_claim(&self) -> bool {
        matches!(self, Role::Dev)
    }

    /// Check if role can verify
    pub fn can_verify(&self) -> bool {
        matches!(self, Role::Qa)
    }

    /// Check if role can approve releases
    pub fn can_approve(&self) -> bool {
        matches!(self, Role::System)
    }
}

/// Verification result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationResult {
    /// Claim was successfully falsified (QA found a bug)
    Falsified,
    /// Unable to falsify the claim (QA failed to find bugs)
    Unfalsified,
    /// Verification was inconclusive
    Inconclusive,
}

impl VerificationResult {
    /// Check if release should be approved based on this result
    pub fn should_approve(&self) -> bool {
        matches!(self, VerificationResult::Unfalsified)
    }
}

/// A falsification criterion (F-criteria)
#[derive(Debug, Clone)]
pub struct FalsificationCriterion {
    /// Criterion ID (e.g., "F1001")
    pub id: String,
    /// Description of what to test
    pub description: String,
    /// Pass condition
    pub pass_condition: String,
}

impl FalsificationCriterion {
    /// Create a new criterion
    pub fn new(id: &str, description: &str, pass_condition: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            pass_condition: pass_condition.to_string(),
        }
    }

    /// Compute hash of criterion for integrity verification
    pub fn hash(&self) -> u64 {
        // Simple hash combining id, description, and pass_condition
        let mut hash: u64 = 0;
        for byte in self.id.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        for byte in self.description.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        for byte in self.pass_condition.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }
}

/// Developer's claim that falsification passed
#[derive(Debug, Clone)]
pub struct FalsificationClaim {
    /// Claim ID
    pub id: String,
    /// Feature or component being claimed
    pub feature: String,
    /// F-criteria that were tested
    pub criteria: Vec<FalsificationCriterion>,
    /// Hash of all criteria for integrity
    pub criteria_hash: u64,
    /// Timestamp of claim
    pub timestamp: SystemTime,
    /// Developer making the claim
    pub claimant: String,
    /// Version of the software
    pub version: String,
    /// Evidence supporting the claim (test results, logs)
    pub evidence: Vec<String>,
}

impl FalsificationClaim {
    /// Create a new claim
    pub fn new(id: &str, feature: &str, claimant: &str, version: &str) -> Self {
        Self {
            id: id.to_string(),
            feature: feature.to_string(),
            criteria: Vec::new(),
            criteria_hash: 0,
            timestamp: SystemTime::now(),
            claimant: claimant.to_string(),
            version: version.to_string(),
            evidence: Vec::new(),
        }
    }

    /// Add a criterion to the claim
    pub fn add_criterion(&mut self, criterion: FalsificationCriterion) {
        self.criteria.push(criterion);
        self.update_hash();
    }

    /// Add evidence to the claim
    pub fn add_evidence(&mut self, evidence: &str) {
        self.evidence.push(evidence.to_string());
    }

    /// Update the criteria hash
    fn update_hash(&mut self) {
        let mut hash: u64 = 0;
        for criterion in &self.criteria {
            hash = hash.wrapping_add(criterion.hash());
        }
        self.criteria_hash = hash;
    }

    /// Verify the criteria hash matches
    pub fn verify_hash(&self) -> bool {
        let mut expected: u64 = 0;
        for criterion in &self.criteria {
            expected = expected.wrapping_add(criterion.hash());
        }
        expected == self.criteria_hash
    }

    /// Check if claim is valid (has required fields)
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && !self.feature.is_empty()
            && !self.claimant.is_empty()
            && !self.version.is_empty()
            && !self.criteria.is_empty()
            && self.verify_hash()
    }
}

/// Black-box artifact for QA (no source code)
#[derive(Debug, Clone)]
pub struct BlackBoxArtifact {
    /// Artifact ID
    pub id: String,
    /// Binary hash (SHA256 hex)
    pub binary_hash: String,
    /// F-criteria to test against
    pub criteria: Vec<FalsificationCriterion>,
    /// Criteria hash for integrity
    pub criteria_hash: u64,
    /// Version being tested
    pub version: String,
    /// Deadline for verification
    pub deadline: Option<SystemTime>,
}

impl BlackBoxArtifact {
    /// Create from a claim (strips source-related info)
    pub fn from_claim(claim: &FalsificationClaim, binary_hash: &str) -> Self {
        Self {
            id: format!("BB-{}", claim.id),
            binary_hash: binary_hash.to_string(),
            criteria: claim.criteria.clone(),
            criteria_hash: claim.criteria_hash,
            version: claim.version.clone(),
            deadline: None,
        }
    }

    /// Set verification deadline
    pub fn with_deadline(mut self, deadline: SystemTime) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Check if deadline has passed
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadline {
            SystemTime::now() > deadline
        } else {
            false
        }
    }

    /// Verify criteria hash matches claim
    pub fn verify_criteria_integrity(&self, claim: &FalsificationClaim) -> bool {
        self.criteria_hash == claim.criteria_hash
    }
}

/// A single verification attempt by QA
#[derive(Debug, Clone)]
pub struct VerificationAttempt {
    /// Attempt ID
    pub id: String,
    /// Artifact being tested
    pub artifact_id: String,
    /// QA engineer making the attempt
    pub verifier: String,
    /// Result of verification
    pub result: VerificationResult,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Evidence collected (logs, traces)
    pub evidence: Vec<String>,
    /// Individual criterion results
    pub criterion_results: HashMap<String, bool>,
}

impl VerificationAttempt {
    /// Create a new attempt
    pub fn new(id: &str, artifact_id: &str, verifier: &str) -> Self {
        Self {
            id: id.to_string(),
            artifact_id: artifact_id.to_string(),
            verifier: verifier.to_string(),
            result: VerificationResult::Inconclusive,
            timestamp: SystemTime::now(),
            evidence: Vec::new(),
            criterion_results: HashMap::new(),
        }
    }

    /// Record result for a criterion
    pub fn record_criterion(&mut self, criterion_id: &str, passed: bool) {
        self.criterion_results
            .insert(criterion_id.to_string(), passed);
    }

    /// Add evidence
    pub fn add_evidence(&mut self, evidence: &str) {
        self.evidence.push(evidence.to_string());
    }

    /// Finalize the attempt with result
    pub fn finalize(&mut self, result: VerificationResult) {
        self.result = result;
        self.timestamp = SystemTime::now();
    }

    /// Check if any criterion was falsified
    pub fn has_falsification(&self) -> bool {
        self.criterion_results.values().any(|&passed| !passed)
    }

    /// Get count of passed criteria
    pub fn passed_count(&self) -> usize {
        self.criterion_results.values().filter(|&&p| p).count()
    }

    /// Get count of failed criteria
    pub fn failed_count(&self) -> usize {
        self.criterion_results.values().filter(|&&p| !p).count()
    }
}

/// Scorecard component with weight
#[derive(Debug, Clone)]
pub struct ScorecardComponent {
    /// Component name
    pub name: String,
    /// Weight (0.0 to 1.0, must sum to 1.0 across all components)
    pub weight: f64,
    /// Score (0 to 100)
    pub score: u32,
}

impl ScorecardComponent {
    /// Create a new component
    pub fn new(name: &str, weight: f64, score: u32) -> Self {
        Self {
            name: name.to_string(),
            weight,
            score: score.min(100),
        }
    }

    /// Calculate weighted score
    pub fn weighted_score(&self) -> f64 {
        self.weight * f64::from(self.score)
    }
}

/// Falsification Scorecard v2 per §36.3
#[derive(Debug, Clone)]
pub struct ScorecardV2 {
    /// Components with weights
    pub components: Vec<ScorecardComponent>,
    /// Version (v1 or v2)
    pub version: u8,
}

impl Default for ScorecardV2 {
    fn default() -> Self {
        Self::new()
    }
}

impl ScorecardV2 {
    /// Create a new v2 scorecard with default components
    pub fn new() -> Self {
        Self {
            components: vec![
                ScorecardComponent::new("Core Correctness", 0.30, 0),
                ScorecardComponent::new("Performance", 0.30, 0),
                ScorecardComponent::new("Resilience", 0.20, 0),
                ScorecardComponent::new("Usability", 0.20, 0),
            ],
            version: 2,
        }
    }

    /// Set score for a component by name
    pub fn set_score(&mut self, name: &str, score: u32) -> bool {
        for component in &mut self.components {
            if component.name == name {
                component.score = score.min(100);
                return true;
            }
        }
        false
    }

    /// Calculate total weighted score
    pub fn total_score(&self) -> f64 {
        self.components.iter().map(|c| c.weighted_score()).sum()
    }

    /// Check if weights sum to 1.0
    pub fn weights_valid(&self) -> bool {
        let sum: f64 = self.components.iter().map(|c| c.weight).sum();
        (sum - 1.0).abs() < 1e-10
    }

    /// Check if scorecard passes (>= 70)
    pub fn passes(&self) -> bool {
        self.total_score() >= 70.0
    }

    /// Get grade based on score
    pub fn grade(&self) -> &'static str {
        let score = self.total_score();
        if score >= 90.0 {
            "A"
        } else if score >= 80.0 {
            "B"
        } else if score >= 70.0 {
            "C"
        } else if score >= 60.0 {
            "D"
        } else {
            "F"
        }
    }
}

/// Release decision
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseDecision {
    /// Approved for release
    Approved { reason: String },
    /// Rejected
    Rejected { reason: String },
    /// Pending more verification
    Pending { reason: String },
}

impl ReleaseDecision {
    /// Check if approved
    pub fn is_approved(&self) -> bool {
        matches!(self, ReleaseDecision::Approved { .. })
    }

    /// Get reason
    pub fn reason(&self) -> &str {
        match self {
            ReleaseDecision::Approved { reason }
            | ReleaseDecision::Rejected { reason }
            | ReleaseDecision::Pending { reason } => reason,
        }
    }
}

/// Audit trail entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Role performing action
    pub role: Role,
    /// Actor name
    pub actor: String,
    /// Action description
    pub action: String,
    /// Related artifact IDs
    pub artifacts: Vec<String>,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(id: &str, role: Role, actor: &str, action: &str) -> Self {
        Self {
            id: id.to_string(),
            timestamp: SystemTime::now(),
            role,
            actor: actor.to_string(),
            action: action.to_string(),
            artifacts: Vec::new(),
        }
    }

    /// Add related artifact
    pub fn with_artifact(mut self, artifact_id: &str) -> Self {
        self.artifacts.push(artifact_id.to_string());
        self
    }
}

/// Double-blind verification session
#[derive(Debug)]
pub struct VerificationSession {
    /// Session ID
    pub id: String,
    /// Claims submitted by Dev
    claims: Vec<FalsificationClaim>,
    /// Black-box artifacts generated
    artifacts: Vec<BlackBoxArtifact>,
    /// Verification attempts by QA
    attempts: Vec<VerificationAttempt>,
    /// Scorecard
    pub scorecard: ScorecardV2,
    /// Audit trail
    audit_trail: Vec<AuditEntry>,
    /// Current state
    state: SessionState,
}

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Awaiting claims from Dev
    AwaitingClaims,
    /// Claims received, awaiting verification
    AwaitingVerification,
    /// Verification complete, awaiting decision
    AwaitingDecision,
    /// Session completed
    Completed,
}

impl Default for VerificationSession {
    fn default() -> Self {
        Self::new("session-1")
    }
}

impl VerificationSession {
    /// Create a new session
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            claims: Vec::new(),
            artifacts: Vec::new(),
            attempts: Vec::new(),
            scorecard: ScorecardV2::new(),
            audit_trail: Vec::new(),
            state: SessionState::AwaitingClaims,
        }
    }

    /// Get current state
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Submit a claim (Dev role only)
    pub fn submit_claim(
        &mut self,
        role: Role,
        claim: FalsificationClaim,
    ) -> Result<(), &'static str> {
        if !role.can_claim() {
            return Err("Only Dev role can submit claims");
        }
        if !claim.is_valid() {
            return Err("Invalid claim structure");
        }

        // Record audit
        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            role,
            &claim.claimant,
            &format!("Submitted claim for {}", claim.feature),
        )
        .with_artifact(&claim.id);
        self.audit_trail.push(entry);

        self.claims.push(claim);
        self.state = SessionState::AwaitingVerification;
        Ok(())
    }

    /// Generate black-box artifact from claim
    pub fn generate_artifact(
        &mut self,
        claim_id: &str,
        binary_hash: &str,
    ) -> Option<BlackBoxArtifact> {
        let claim = self.claims.iter().find(|c| c.id == claim_id)?;
        let artifact = BlackBoxArtifact::from_claim(claim, binary_hash);

        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            Role::System,
            "System",
            &format!("Generated black-box artifact from claim {}", claim_id),
        )
        .with_artifact(&artifact.id);
        self.audit_trail.push(entry);

        self.artifacts.push(artifact.clone());
        Some(artifact)
    }

    /// Submit verification attempt (QA role only)
    pub fn submit_attempt(
        &mut self,
        role: Role,
        attempt: VerificationAttempt,
    ) -> Result<(), &'static str> {
        if !role.can_verify() {
            return Err("Only QA role can submit verification attempts");
        }

        // Record audit
        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            role,
            &attempt.verifier,
            &format!("Submitted verification attempt: {:?}", attempt.result),
        )
        .with_artifact(&attempt.artifact_id);
        self.audit_trail.push(entry);

        self.attempts.push(attempt);
        Ok(())
    }

    /// Get all attempts for an artifact
    pub fn get_attempts(&self, artifact_id: &str) -> Vec<&VerificationAttempt> {
        self.attempts
            .iter()
            .filter(|a| a.artifact_id == artifact_id)
            .collect()
    }

    /// Make release decision (System role only)
    pub fn make_decision(&mut self, role: Role) -> Result<ReleaseDecision, &'static str> {
        if !role.can_approve() {
            return Err("Only System role can make release decisions");
        }

        if self.attempts.is_empty() {
            return Ok(ReleaseDecision::Pending {
                reason: "No verification attempts yet".to_string(),
            });
        }

        // Check if any attempt successfully falsified
        let falsified = self
            .attempts
            .iter()
            .any(|a| a.result == VerificationResult::Falsified);

        // Check if all attempts are inconclusive
        let all_inconclusive = self
            .attempts
            .iter()
            .all(|a| a.result == VerificationResult::Inconclusive);

        let decision = if falsified {
            ReleaseDecision::Rejected {
                reason: "QA successfully falsified the claim".to_string(),
            }
        } else if all_inconclusive {
            ReleaseDecision::Pending {
                reason: "All verification attempts were inconclusive".to_string(),
            }
        } else {
            // Check scorecard
            if self.scorecard.passes() {
                ReleaseDecision::Approved {
                    reason: format!(
                        "QA failed to falsify and scorecard passes ({:.1}/100, grade {})",
                        self.scorecard.total_score(),
                        self.scorecard.grade()
                    ),
                }
            } else {
                ReleaseDecision::Rejected {
                    reason: format!(
                        "Scorecard fails ({:.1}/100 < 70, grade {})",
                        self.scorecard.total_score(),
                        self.scorecard.grade()
                    ),
                }
            }
        };

        // Record audit
        let entry = AuditEntry::new(
            &format!("audit-{}", self.audit_trail.len()),
            role,
            "System",
            &format!("Release decision: {:?}", decision),
        );
        self.audit_trail.push(entry);

        self.state = SessionState::Completed;
        Ok(decision)
    }

    /// Get audit trail
    pub fn audit_trail(&self) -> &[AuditEntry] {
        &self.audit_trail
    }

    /// Get claim count
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    /// Get attempt count
    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }

    /// Generate verification report
    pub fn generate_report(&self) -> VerificationReport {
        VerificationReport {
            session_id: self.id.clone(),
            total_claims: self.claims.len(),
            total_artifacts: self.artifacts.len(),
            total_attempts: self.attempts.len(),
            falsified_count: self
                .attempts
                .iter()
                .filter(|a| a.result == VerificationResult::Falsified)
                .count(),
            unfalsified_count: self
                .attempts
                .iter()
                .filter(|a| a.result == VerificationResult::Unfalsified)
                .count(),
            inconclusive_count: self
                .attempts
                .iter()
                .filter(|a| a.result == VerificationResult::Inconclusive)
                .count(),
            scorecard_total: self.scorecard.total_score(),
            scorecard_grade: self.scorecard.grade().to_string(),
            audit_entries: self.audit_trail.len(),
        }
    }
}

/// Summary report of verification session
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Session ID
    pub session_id: String,
    /// Total claims submitted
    pub total_claims: usize,
    /// Total artifacts generated
    pub total_artifacts: usize,
    /// Total verification attempts
    pub total_attempts: usize,
    /// Number of falsified claims
    pub falsified_count: usize,
    /// Number of unfalsified claims
    pub unfalsified_count: usize,
    /// Number of inconclusive attempts
    pub inconclusive_count: usize,
    /// Scorecard total
    pub scorecard_total: f64,
    /// Scorecard grade
    pub scorecard_grade: String,
    /// Audit trail entries
    pub audit_entries: usize,
}

impl VerificationReport {
    /// Check if report indicates success
    pub fn is_success(&self) -> bool {
        self.falsified_count == 0 && self.scorecard_total >= 70.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        assert!(Role::Dev.can_claim());
        assert!(!Role::Dev.can_verify());
        assert!(!Role::Dev.can_approve());

        assert!(!Role::Qa.can_claim());
        assert!(Role::Qa.can_verify());
        assert!(!Role::Qa.can_approve());

        assert!(!Role::System.can_claim());
        assert!(!Role::System.can_verify());
        assert!(Role::System.can_approve());
    }

    #[test]
    fn test_criterion_hash() {
        let c1 = FalsificationCriterion::new("F001", "Test", "Pass");
        let c2 = FalsificationCriterion::new("F001", "Test", "Pass");
        let c3 = FalsificationCriterion::new("F002", "Test", "Pass");

        assert_eq!(c1.hash(), c2.hash());
        assert_ne!(c1.hash(), c3.hash());
    }

    #[test]
    fn test_claim_validation() {
        let mut claim = FalsificationClaim::new("C001", "Feature X", "dev@example.com", "1.0.0");

        // Invalid without criteria
        assert!(!claim.is_valid());

        // Valid with criteria
        claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));
        assert!(claim.is_valid());
    }

    #[test]
    fn test_claim_hash_verification() {
        let mut claim = FalsificationClaim::new("C001", "Feature", "dev", "1.0");
        claim.add_criterion(FalsificationCriterion::new("F001", "Test", "Pass"));

        assert!(claim.verify_hash());

        // Tamper with hash
        claim.criteria_hash = 0;
        assert!(!claim.verify_hash());
    }

    #[test]
    fn test_scorecard_calculation() {
        let mut scorecard = ScorecardV2::new();
        scorecard.set_score("Core Correctness", 85);
        scorecard.set_score("Performance", 92);
        scorecard.set_score("Resilience", 80);
        scorecard.set_score("Usability", 95);

        // 0.30*85 + 0.30*92 + 0.20*80 + 0.20*95 = 25.5 + 27.6 + 16 + 19 = 88.1
        assert!((scorecard.total_score() - 88.1).abs() < 0.1);
        assert!(scorecard.passes());
        assert_eq!(scorecard.grade(), "B");
    }

    #[test]
    fn test_scorecard_weights_valid() {
        let scorecard = ScorecardV2::new();
        assert!(scorecard.weights_valid());
    }

    #[test]
    fn test_verification_attempt_tracking() {
        let mut attempt = VerificationAttempt::new("V001", "BB-C001", "qa@example.com");
        attempt.record_criterion("F001", true);
        attempt.record_criterion("F002", false);
        attempt.record_criterion("F003", true);

        assert_eq!(attempt.passed_count(), 2);
        assert_eq!(attempt.failed_count(), 1);
        assert!(attempt.has_falsification());
    }
}
