/// SIMD Attribute Checker - Pre-commit validation
///
/// Validates SIMD code properties to ensure correctness and performance:
/// 1. [CRITICAL] Missing #[target_feature] attributes
/// 2. [ERROR] Attribute-intrinsic mismatch
/// 3. [WARNING] Missing SAFETY comments
/// 4. [WARNING] Missing #[inline] attributes
///
/// Bug instances found: 104 functions missing #[target_feature]
/// Performance impact: 5.9x slower to missing 21x speedup potential

use anyhow::{Context, Result};
use colored::Colorize;
use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViolationLevel {
    Critical, // Blocks commit, causes severe performance bugs
    Error,    // Blocks commit, causes correctness issues
    Warning,  // Reports but doesn't block
}

#[derive(Debug)]
struct Violation {
    level: ViolationLevel,
    filepath: PathBuf,
    line_num: usize,
    function_name: String,
    message: String,
    fix_suggestion: String,
}

struct IntrinsicPattern {
    pattern: Regex,
    required_feature: &'static str,
    _name: &'static str,
}

impl IntrinsicPattern {
    fn new(pattern: &str, required_feature: &'static str, name: &'static str) -> Self {
        Self {
            pattern: Regex::new(pattern).expect("Invalid regex pattern"),
            required_feature,
            _name: name,
        }
    }
}

/// Get SIMD intrinsic patterns for a backend
fn get_intrinsic_patterns(backend: &str) -> Option<IntrinsicPattern> {
    match backend {
        "sse2" => Some(IntrinsicPattern::new(r"_mm_\w+", "sse2", "SSE2")),
        "avx2" => Some(IntrinsicPattern::new(r"_mm256_\w+", "avx2", "AVX2")),
        "avx512" => Some(IntrinsicPattern::new(r"_mm512_\w+", "avx512f", "AVX512")),
        "neon" => Some(IntrinsicPattern::new(
            r"v(?:ld|st|add|sub|mul|div)\w*q_f32",
            "neon",
            "NEON",
        )),
        _ => None,
    }
}

/// FMA intrinsics that require additional 'fma' feature
fn is_fma_intrinsic(intrinsic: &str) -> bool {
    matches!(
        intrinsic,
        "_mm256_fmadd_ps"
            | "_mm256_fmsub_ps"
            | "_mm256_fnmadd_ps"
            | "_mm256_fnmsub_ps"
            | "_mm_fmadd_ps"
            | "_mm_fmsub_ps"
            | "_mm_fnmadd_ps"
            | "_mm_fnmsub_ps"
    )
}

/// Check if #[target_feature] attribute exists within 15 lines before function
fn check_target_feature_attribute(
    lines: &[String],
    fn_line: usize,
) -> Option<String> {
    let target_feature_re = Regex::new(r#"#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]"#)
        .expect("Invalid regex");

    let start = fn_line.saturating_sub(15);
    for line in &lines[start..fn_line] {
        if let Some(caps) = target_feature_re.captures(line) {
            return Some(caps[1].to_string());
        }
    }
    None
}

/// Check if SAFETY comment exists within 10 lines before function
fn has_safety_comment(lines: &[String], fn_line: usize) -> bool {
    let safety_re = Regex::new(r"//\s*SAFETY:").expect("Invalid regex");

    let start = fn_line.saturating_sub(10);
    for line in &lines[start..fn_line] {
        if safety_re.is_match(line) {
            return true;
        }
    }
    false
}

/// Check if #[inline] attribute exists within 15 lines before function
fn has_inline_attribute(lines: &[String], fn_line: usize) -> bool {
    let inline_re = Regex::new(r"#\[inline(?:\(always\))?\]").expect("Invalid regex");

    let start = fn_line.saturating_sub(15);
    for line in &lines[start..fn_line] {
        if inline_re.is_match(line) {
            return true;
        }
    }
    false
}

/// Find all SIMD intrinsics used in a function body
fn find_intrinsics_in_function(
    lines: &[String],
    fn_start: usize,
    pattern: &Regex,
) -> (usize, HashSet<String>) {
    let mut intrinsics = HashSet::new();
    let mut brace_count = 0;
    let mut fn_end = fn_start;

    for (offset, line) in lines[fn_start..].iter().enumerate() {
        brace_count += line.matches('{').count() as i32;
        brace_count -= line.matches('}').count() as i32;

        // Collect intrinsics
        for cap in pattern.captures_iter(line) {
            intrinsics.insert(cap[0].to_string());
        }

        // Found function end
        if brace_count == 0 && offset > 0 {
            fn_end = fn_start + offset;
            break;
        }
    }

    (fn_end, intrinsics)
}

/// Check if attribute matches the intrinsics actually used
fn check_attribute_mismatch(
    feature: &str,
    intrinsics: &HashSet<String>,
) -> Option<String> {
    let has_avx512 = intrinsics.iter().any(|i| i.starts_with("_mm512_"));
    let has_avx2 = intrinsics.iter().any(|i| i.starts_with("_mm256_"));
    let _has_sse2 = intrinsics.iter().any(|i| i.starts_with("_mm_"));

    if has_avx512 && !feature.contains("avx512f") {
        return Some(format!(
            "Using AVX-512 intrinsics but attribute is '{}' (should be 'avx512f')",
            feature
        ));
    }

    if has_avx2 && feature == "sse2" {
        return Some(format!(
            "Using AVX2 intrinsics but attribute is 'sse2' (should be 'avx2')"
        ));
    }

    if !has_avx512 && feature.contains("avx512f") {
        return Some(format!(
            "Attribute is 'avx512f' but no AVX-512 intrinsics found"
        ));
    }

    None
}

/// Check if FMA intrinsics are used without 'fma' feature
fn check_fma_feature(feature: &str, intrinsics: &HashSet<String>) -> Option<String> {
    let uses_fma = intrinsics.iter().any(|i| is_fma_intrinsic(i));

    if uses_fma && !feature.contains("fma") {
        return Some(
            "Using FMA intrinsics (_mm256_fmadd_ps, etc.) but 'fma' feature not enabled"
                .to_string(),
        );
    }

    None
}

/// Check a single backend file for SIMD violations
fn check_file(filepath: &Path, backend: &str) -> Result<Vec<Violation>> {
    let pattern = match get_intrinsic_patterns(backend) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let content = std::fs::read_to_string(filepath)
        .with_context(|| format!("Failed to read {}", filepath.display()))?;

    let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let unsafe_fn_re = Regex::new(r"^\s*unsafe\s+fn\s+(\w+)").expect("Invalid regex");

    let mut violations = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        // Check for unsafe fn declaration
        if let Some(caps) = unsafe_fn_re.captures(&lines[i]) {
            let fn_name = caps[1].to_string();
            let fn_line = i;

            // Find intrinsics in function body
            let (fn_end, intrinsics) = find_intrinsics_in_function(&lines, fn_line, &pattern.pattern);

            // Skip if no intrinsics found
            if intrinsics.is_empty() {
                i = fn_end.max(i + 1);
                continue;
            }

            // Check for #[target_feature] attribute
            let target_feature = check_target_feature_attribute(&lines, fn_line);

            // [CRITICAL] Missing #[target_feature]
            if target_feature.is_none() {
                violations.push(Violation {
                    level: ViolationLevel::Critical,
                    filepath: filepath.to_path_buf(),
                    line_num: fn_line + 1, // 1-indexed
                    function_name: fn_name.clone(),
                    message: format!(
                        "Missing #[target_feature] attribute (uses {} SIMD intrinsics)",
                        intrinsics.len()
                    ),
                    fix_suggestion: format!(
                        "Add #[target_feature(enable = \"{}\")] above function",
                        pattern.required_feature
                    ),
                });
            } else {
                let feature = target_feature.unwrap();

                // [ERROR] Attribute-intrinsic mismatch
                if let Some(msg) = check_attribute_mismatch(&feature, &intrinsics) {
                    violations.push(Violation {
                        level: ViolationLevel::Error,
                        filepath: filepath.to_path_buf(),
                        line_num: fn_line + 1,
                        function_name: fn_name.clone(),
                        message: msg,
                        fix_suggestion:
                            "Correct #[target_feature] attribute to match intrinsics used"
                                .to_string(),
                    });
                }

                // [ERROR] FMA intrinsics without FMA feature
                if let Some(msg) = check_fma_feature(&feature, &intrinsics) {
                    violations.push(Violation {
                        level: ViolationLevel::Error,
                        filepath: filepath.to_path_buf(),
                        line_num: fn_line + 1,
                        function_name: fn_name.clone(),
                        message: msg,
                        fix_suggestion:
                            "Add 'fma' to target_feature: #[target_feature(enable = \"avx2,fma\")]"
                                .to_string(),
                    });
                }
            }

            // [WARNING] Missing SAFETY comment
            if !has_safety_comment(&lines, fn_line) {
                violations.push(Violation {
                    level: ViolationLevel::Warning,
                    filepath: filepath.to_path_buf(),
                    line_num: fn_line + 1,
                    function_name: fn_name.clone(),
                    message: "Missing SAFETY comment for unsafe function with SIMD".to_string(),
                    fix_suggestion:
                        "Add // SAFETY: comment explaining why unsafe code is correct".to_string(),
                });
            }

            // [WARNING] Missing #[inline] attribute
            if !has_inline_attribute(&lines, fn_line) {
                violations.push(Violation {
                    level: ViolationLevel::Warning,
                    filepath: filepath.to_path_buf(),
                    line_num: fn_line + 1,
                    function_name: fn_name,
                    message: "Missing #[inline] attribute on SIMD hot path".to_string(),
                    fix_suggestion: "Add #[inline] above function for better optimization"
                        .to_string(),
                });
            }

            i = fn_end.max(i + 1);
        } else {
            i += 1;
        }
    }

    Ok(violations)
}

/// Run SIMD attribute checker on all backend files
pub fn run() -> Result<()> {
    println!("{}", "🔍 SIMD Property Checker (Rust)".blue().bold());
    println!("{}", "━".repeat(60).blue());
    println!();

    let backend_files = vec![
        ("src/backends/sse2.rs", "sse2"),
        ("src/backends/avx2.rs", "avx2"),
        ("src/backends/avx512.rs", "avx512"),
        ("src/backends/neon.rs", "neon"),
    ];

    let mut all_violations = Vec::new();

    for (filepath, backend) in &backend_files {
        let path = Path::new(filepath);
        if path.exists() {
            let violations = check_file(path, backend)?;
            all_violations.extend(violations);
        }
    }

    // Separate by severity
    let critical: Vec<_> = all_violations
        .iter()
        .filter(|v| v.level == ViolationLevel::Critical)
        .collect();
    let errors: Vec<_> = all_violations
        .iter()
        .filter(|v| v.level == ViolationLevel::Error)
        .collect();
    let warnings: Vec<_> = all_violations
        .iter()
        .filter(|v| v.level == ViolationLevel::Warning)
        .collect();

    let should_block = !critical.is_empty() || !errors.is_empty();

    if all_violations.is_empty() {
        println!("{}", "✅ PASS: All SIMD property checks passed!".green().bold());
        println!("{}", "   • No missing #[target_feature] attributes".green());
        println!("{}", "   • All attributes match intrinsics used".green());
        println!("{}", "   • All unsafe functions have SAFETY comments".green());
        println!("{}", "   • All SIMD functions have #[inline] attributes".green());
        println!();
        return Ok(());
    }

    // Report CRITICAL violations
    if !critical.is_empty() {
        println!();
        println!("{}", "=".repeat(60).red());
        println!("{}", format!("❌ CRITICAL VIOLATIONS ({})", critical.len()).red().bold());
        println!("{}", "=".repeat(60).red());
        println!();

        for v in critical.iter().take(10) {
            println!(
                "  {} - {}",
                format!("🚨 {}:{}", v.filepath.display(), v.line_num).red(),
                format!("{}()", v.function_name).yellow()
            );
            println!("     {}: {}", "Problem".red(), v.message);
            println!("     {}: {}", "Fix".green(), v.fix_suggestion);
            println!();
        }

        if critical.len() > 10 {
            println!("  {} more critical violations...", (critical.len() - 10));
            println!();
        }
    }

    // Report ERROR violations
    if !errors.is_empty() {
        println!();
        println!("{}", "=".repeat(60).red());
        println!("{}", format!("❌ ERRORS ({})", errors.len()).red().bold());
        println!("{}", "=".repeat(60).red());
        println!();

        for v in errors.iter().take(10) {
            println!(
                "  {} - {}",
                format!("⚠️  {}:{}", v.filepath.display(), v.line_num).red(),
                format!("{}()", v.function_name).yellow()
            );
            println!("     {}: {}", "Problem".red(), v.message);
            println!("     {}: {}", "Fix".green(), v.fix_suggestion);
            println!();
        }

        if errors.len() > 10 {
            println!("  {} more errors...", (errors.len() - 10));
            println!();
        }
    }

    // Report WARNING violations (summary only)
    if !warnings.is_empty() {
        println!();
        println!("{}", "=".repeat(60).yellow());
        println!("{}", format!("⚠️  WARNINGS ({})", warnings.len()).yellow().bold());
        println!("{}", "=".repeat(60).yellow());
        println!();

        // Show first 10 warnings
        for v in warnings.iter().take(10) {
            println!(
                "  {} - {}",
                format!("⚠️  {}:{}", v.filepath.display(), v.line_num).yellow(),
                format!("{}()", v.function_name).cyan()
            );
            println!("     {}: {}", "Issue".yellow(), v.message);
            println!("     {}: {}", "Fix".green(), v.fix_suggestion);
            println!();
        }

        if warnings.len() > 10 {
            println!("  {} more warnings...", (warnings.len() - 10));
            println!();
        }
    }

    // Summary
    println!();
    println!("{}", "=".repeat(60).blue());
    println!("{}", "SUMMARY".blue().bold());
    println!("{}", "=".repeat(60).blue());
    println!();

    if !critical.is_empty() {
        println!(
            "  {} {} {} - Compiler CANNOT emit SIMD instructions",
            "🚨".red(),
            critical.len(),
            "CRITICAL".red().bold()
        );
        println!(
            "     {} Impact: 5.9x slower to missing 21x speedup potential",
            "".red()
        );
        println!();
    }

    if !errors.is_empty() {
        println!(
            "  {} {} {} - Incorrect or incompatible attributes",
            "❌".red(),
            errors.len(),
            "ERRORS".red().bold()
        );
        println!();
    }

    if !warnings.is_empty() {
        println!(
            "  {} {} {} - Best practices not followed",
            "⚠️".yellow(),
            warnings.len(),
            "WARNINGS".yellow().bold()
        );
        println!();
    }

    // Block or allow commit
    if should_block {
        println!("{}", "━".repeat(60).red());
        println!("{}", "❌ COMMIT BLOCKED - Fix CRITICAL/ERROR violations".red().bold());
        println!("{}", "━".repeat(60).red());
        println!();
        anyhow::bail!("SIMD validation failed: {} critical, {} errors", critical.len(), errors.len());
    } else {
        println!("{}", "━".repeat(60).green());
        println!("{}", "✅ COMMIT ALLOWED - Only warnings present".green().bold());
        println!("{}", "   Consider addressing warnings in follow-up commits".yellow());
        println!("{}", "━".repeat(60).green());
        println!();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_fma_intrinsic() {
        assert!(is_fma_intrinsic("_mm256_fmadd_ps"));
        assert!(is_fma_intrinsic("_mm_fmsub_ps"));
        assert!(!is_fma_intrinsic("_mm256_add_ps"));
        assert!(!is_fma_intrinsic("_mm_mul_ps"));
    }

    #[test]
    fn test_get_intrinsic_patterns() {
        assert!(get_intrinsic_patterns("sse2").is_some());
        assert!(get_intrinsic_patterns("avx2").is_some());
        assert!(get_intrinsic_patterns("avx512").is_some());
        assert!(get_intrinsic_patterns("neon").is_some());
        assert!(get_intrinsic_patterns("unknown").is_none());
    }

    #[test]
    fn test_check_attribute_mismatch() {
        let mut intrinsics = HashSet::new();
        intrinsics.insert("_mm512_add_ps".to_string());

        assert!(check_attribute_mismatch("avx2", &intrinsics).is_some());
        assert!(check_attribute_mismatch("avx512f", &intrinsics).is_none());
    }

    #[test]
    fn test_check_fma_feature() {
        let mut intrinsics = HashSet::new();
        intrinsics.insert("_mm256_fmadd_ps".to_string());

        assert!(check_fma_feature("avx2", &intrinsics).is_some());
        assert!(check_fma_feature("avx2,fma", &intrinsics).is_none());
    }
}
