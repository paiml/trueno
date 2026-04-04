//! Performance contract verification (CI/CD gate).
//! Extends provable-contracts framework to performance bounds.
//! See spec section 3.4 and 7.1.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A performance contract loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceContract {
    pub kind: String,
    pub name: String,
    pub version: String,
    pub kernel: String,
    #[serde(default)]
    pub hardware: HardwareSpec,
    #[serde(default)]
    pub bounds: Vec<PerformanceBound>,
    #[serde(default)]
    pub metrics: std::collections::HashMap<String, MetricBound>,
    #[serde(default)]
    pub falsification: Vec<FalsificationCheck>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub gpu: Option<String>,
    pub cpu: Option<String>,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBound {
    pub size: Vec<u32>,
    #[serde(default)]
    pub max_time_us: Option<f64>,
    #[serde(default)]
    pub min_tflops: Option<f64>,
    #[serde(default)]
    pub max_regression_pct: Option<f64>,
    #[serde(default)]
    pub min_bandwidth_gbps: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBound {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCheck {
    pub name: String,
    pub description: String,
    pub check: String,
}

/// Result of verifying a single contract.
#[derive(Debug)]
pub struct ContractVerification {
    pub contract_name: String,
    pub passed: Vec<String>,
    pub failed: Vec<String>,
    pub skipped: Vec<String>,
}

impl ContractVerification {
    pub fn is_pass(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Load a performance contract from a YAML file.
pub fn load_contract(path: &Path) -> Result<PerformanceContract> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read contract: {}", path.display()))?;
    let contract: PerformanceContract = serde_yaml_ng::from_str(&content)
        .with_context(|| format!("Failed to parse contract: {}", path.display()))?;
    Ok(contract)
}

/// Load all contracts from a directory.
pub fn load_contracts_dir(dir: &Path) -> Result<Vec<PerformanceContract>> {
    let mut contracts = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                match load_contract(&path) {
                    Ok(c) => contracts.push(c),
                    Err(e) => eprintln!("Warning: skipping {}: {e}", path.display()),
                }
            }
        }
    }
    Ok(contracts)
}

/// Verify a contract against measured values.
/// For now, this validates the contract structure and falsification checks.
pub fn verify_contract(contract: &PerformanceContract) -> ContractVerification {
    let mut result = ContractVerification {
        contract_name: contract.name.clone(),
        passed: Vec::new(),
        failed: Vec::new(),
        skipped: Vec::new(),
    };

    // Validate structure
    if contract.kind.is_empty() {
        result.failed.push("Contract missing 'kind' field".to_string());
    } else {
        result.passed.push(format!("kind: {}", contract.kind));
    }

    if contract.kernel.is_empty() {
        result.failed.push("Contract missing 'kernel' field".to_string());
    } else {
        result.passed.push(format!("kernel: {}", contract.kernel));
    }

    // Validate bounds
    for (i, bound) in contract.bounds.iter().enumerate() {
        if bound.size.is_empty() {
            result.failed.push(format!("Bound {i}: missing size dimensions"));
        } else {
            result.passed.push(format!("Bound {i}: size {:?}", bound.size));
        }

        if bound.max_time_us.is_none()
            && bound.min_tflops.is_none()
            && bound.min_bandwidth_gbps.is_none()
        {
            result.skipped.push(format!("Bound {i}: no performance criteria specified"));
        }
    }

    // Validate falsification checks
    for check in &contract.falsification {
        if check.name.is_empty() || check.check.is_empty() {
            result
                .failed
                .push(format!("Falsification '{}': missing name or check expression", check.name));
        } else {
            // Until we have runtime profiling, skip actual verification
            result.skipped.push(format!(
                "FALSIFY {}: {} (needs runtime data)",
                check.name, check.description
            ));
        }
    }

    result
}

/// Run contract verification for a directory of contracts.
pub fn run_verify(
    contracts_dir: Option<&str>,
    contract_file: Option<&str>,
    self_verify: bool,
    fail_on_regression: bool,
) -> Result<()> {
    let contracts = if let Some(dir) = contracts_dir {
        load_contracts_dir(Path::new(dir))?
    } else if let Some(file) = contract_file {
        vec![load_contract(Path::new(file))?]
    } else if self_verify {
        let dir = Path::new("contracts/cgp");
        if dir.exists() {
            load_contracts_dir(dir)?
        } else {
            println!("No contracts found at contracts/cgp/");
            return Ok(());
        }
    } else {
        anyhow::bail!("Specify --contracts-dir, --contract, or --self");
    };

    println!("\n=== cgp Contract Verification ===\n");
    let mut total_pass = 0;
    let mut total_fail = 0;
    let mut total_skip = 0;

    for c in &contracts {
        let result = verify_contract(c);
        let status = if result.is_pass() { "\x1b[32mPASS\x1b[0m" } else { "\x1b[31mFAIL\x1b[0m" };
        println!(
            "  {} {} ({} pass, {} fail, {} skip)",
            status,
            c.name,
            result.passed.len(),
            result.failed.len(),
            result.skipped.len()
        );
        total_pass += result.passed.len();
        total_fail += result.failed.len();
        total_skip += result.skipped.len();
    }

    println!("\n  Total: {total_pass} pass, {total_fail} fail, {total_skip} skip");
    if total_fail > 0 && fail_on_regression {
        anyhow::bail!("{total_fail} contract verification(s) failed");
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_contract() -> PerformanceContract {
        PerformanceContract {
            kind: "PerformanceContract".to_string(),
            name: "test-gemm-contract".to_string(),
            version: "1.0.0".to_string(),
            kernel: "gemm_cta_wmma_fp16".to_string(),
            hardware: HardwareSpec {
                gpu: Some("NVIDIA GeForce RTX 4090".to_string()),
                cpu: None,
                compute_capability: Some("8.9".to_string()),
            },
            bounds: vec![PerformanceBound {
                size: vec![512, 512, 512],
                max_time_us: Some(30.0),
                min_tflops: Some(9.0),
                max_regression_pct: Some(10.0),
                min_bandwidth_gbps: None,
            }],
            metrics: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "warp_execution_efficiency".to_string(),
                    MetricBound { min: Some(95.0), max: None },
                );
                m
            },
            falsification: vec![FalsificationCheck {
                name: "FALSIFY-TEST-001".to_string(),
                description: "CTA WMMA must achieve >9 TFLOP/s".to_string(),
                check: "tflops > 9.0".to_string(),
            }],
        }
    }

    #[test]
    fn test_verify_valid_contract() {
        let contract = sample_contract();
        let result = verify_contract(&contract);
        assert!(result.is_pass());
        assert!(!result.passed.is_empty());
    }

    #[test]
    fn test_verify_missing_kernel() {
        let mut contract = sample_contract();
        contract.kernel = String::new();
        let result = verify_contract(&contract);
        assert!(!result.is_pass());
    }

    #[test]
    fn test_contract_yaml_roundtrip() {
        let contract = sample_contract();
        let yaml = serde_yaml_ng::to_string(&contract).unwrap();
        let parsed: PerformanceContract = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.name, contract.name);
        assert_eq!(parsed.kernel, contract.kernel);
        assert_eq!(parsed.bounds.len(), 1);
        assert_eq!(parsed.bounds[0].size, vec![512, 512, 512]);
    }

    #[test]
    fn test_contract_falsification_checks() {
        let contract = sample_contract();
        let result = verify_contract(&contract);
        // Falsification checks are skipped (need runtime data), not failed
        assert!(result.is_pass());
        assert!(!result.skipped.is_empty());
    }
}
