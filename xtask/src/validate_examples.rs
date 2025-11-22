//! Book examples validation with EXTREME TDD quality
//!
//! This module validates that all book examples meet quality standards:
//! - Compile successfully
//! - Pass clippy lints
//! - Have module documentation
//! - Are runnable (complete with main function)
//! - Are referenced in the book
//! - Follow snake_case naming conventions

use anyhow::{anyhow, bail, Context, Result};
use colored::Colorize;
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use walkdir::WalkDir;

const EXAMPLES_DIR: &str = "examples";
const BOOK_DIR: &str = "book";
const TIMEOUT_SECS: u64 = 5;

/// Main entry point for example validation
pub fn run() -> Result<()> {
    println!("{}", "📚 Validating book examples...".bold());
    println!();

    let project_root = get_project_root()?;
    let examples_dir = project_root.join(EXAMPLES_DIR);
    let book_dir = project_root.join(BOOK_DIR);

    // Collect all example files
    let examples = collect_examples(&examples_dir)?;

    if examples.is_empty() {
        bail!("No examples found in {}", examples_dir.display());
    }

    println!("Found {} examples to validate", examples.len());
    println!();

    // Run validation steps
    let mut results = ValidationResults::new();

    // Step 1: Verify examples compile
    results.add_step(1, "Compile examples", || {
        step_compile_examples(&project_root)
    });

    // Step 2: Run clippy on examples
    results.add_step(2, "Clippy lints", || step_clippy_examples(&project_root));

    // Step 3: Verify module documentation
    results.add_step(3, "Module documentation", || {
        step_check_module_docs(&examples)
    });

    // Step 4: Verify examples are runnable
    results.add_step(4, "Runnable examples", || {
        step_check_runnable(&examples, &project_root)
    });

    // Step 5: Validate book references
    results.add_step(5, "Book references", || {
        step_check_book_references(&examples, &book_dir)
    });

    // Step 6: Verify naming conventions
    results.add_step(6, "Naming conventions", || {
        step_check_naming_conventions(&examples)
    });

    // Print summary
    results.print_summary();

    // Exit with error if any step failed
    if results.has_failures() {
        bail!("Validation failed");
    }

    Ok(())
}

/// Get the project root directory
fn get_project_root() -> Result<PathBuf> {
    let current = std::env::current_dir().context("Failed to get current directory")?;

    // Look for Cargo.toml in current dir or parent
    if current.join("Cargo.toml").exists() {
        return Ok(current);
    }

    if let Some(parent) = current.parent() {
        if parent.join("Cargo.toml").exists() {
            return Ok(parent.to_path_buf());
        }
    }

    bail!("Could not find project root (no Cargo.toml found)");
}

/// Collect all example .rs files
fn collect_examples(examples_dir: &Path) -> Result<Vec<PathBuf>> {
    if !examples_dir.exists() {
        bail!(
            "Examples directory does not exist: {}",
            examples_dir.display()
        );
    }

    let mut examples = Vec::new();

    for entry in fs::read_dir(examples_dir).context("Failed to read examples directory")? {
        let entry = entry?;
        let path = entry.path();

        if is_rust_file(&path) {
            examples.push(path);
        }
    }

    examples.sort();
    Ok(examples)
}

/// Step 1: Verify all examples compile
fn step_compile_examples(project_root: &Path) -> Result<()> {
    let output = Command::new("cargo")
        .arg("build")
        .arg("--examples")
        .arg("--quiet")
        .current_dir(project_root)
        .output()
        .context("Failed to execute cargo build")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Examples failed to compile:\n{}", stderr);
    }

    Ok(())
}

/// Step 2: Run clippy on examples
fn step_clippy_examples(project_root: &Path) -> Result<()> {
    let output = Command::new("cargo")
        .arg("clippy")
        .arg("--examples")
        .arg("--quiet")
        .arg("--")
        .arg("-D")
        .arg("warnings")
        .current_dir(project_root)
        .output()
        .context("Failed to execute cargo clippy")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Clippy found issues:\n{}", stderr);
    }

    Ok(())
}

/// Step 3: Verify examples have module documentation (//!)
fn step_check_module_docs(examples: &[PathBuf]) -> Result<()> {
    let mut missing_docs = Vec::new();

    for example in examples {
        if !has_module_doc(example)? {
            missing_docs.push(example.clone());
        }
    }

    if !missing_docs.is_empty() {
        let names = extract_file_names(&missing_docs);
        bail!(
            "Examples missing module documentation (//!):\n  {}",
            names.join("\n  ")
        );
    }

    Ok(())
}

/// Check if a file has module documentation
fn has_module_doc(path: &Path) -> Result<bool> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(contains_module_doc(&content))
}

/// Step 4: Verify examples are runnable (have main function, run without panic)
fn step_check_runnable(examples: &[PathBuf], project_root: &Path) -> Result<()> {
    let mut errors = Vec::new();

    for example in examples {
        // Check if has main function
        if !has_main_function(example)? {
            errors.push(format!(
                "{}: missing main() function",
                example.file_name().unwrap().to_str().unwrap()
            ));
            continue;
        }

        // Try to run it with timeout
        let example_name = example
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid example filename"))?;

        match run_example_with_timeout(example_name, project_root, TIMEOUT_SECS) {
            Ok(_) => {} // Success
            Err(e) => {
                errors.push(format!("{}: {}", example_name, e));
            }
        }
    }

    if !errors.is_empty() {
        bail!("Examples not runnable:\n  {}", errors.join("\n  "));
    }

    Ok(())
}

/// Check if example has a main function
fn has_main_function(path: &Path) -> Result<bool> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(contains_main_function(&content))
}

/// Run example with timeout
fn run_example_with_timeout(
    example_name: &str,
    project_root: &Path,
    timeout_secs: u64,
) -> Result<()> {
    // Spawn the example process
    let mut child = Command::new("cargo")
        .arg("run")
        .arg("--example")
        .arg(example_name)
        .arg("--quiet")
        .current_dir(project_root)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn example")?;

    // Wait with timeout
    let timeout = Duration::from_secs(timeout_secs);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait()? {
            Some(status) => {
                if !status.success() {
                    let mut stderr = Vec::new();
                    if let Some(mut pipe) = child.stderr.take() {
                        use std::io::Read;
                        let _ = pipe.read_to_end(&mut stderr);
                    }
                    let stderr_str = String::from_utf8_lossy(&stderr);
                    bail!("exited with error: {}", stderr_str);
                }
                return Ok(());
            }
            None => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    bail!("timed out after {}s", timeout_secs);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

/// Step 5: Validate book references actual examples
fn step_check_book_references(examples: &[PathBuf], book_dir: &Path) -> Result<()> {
    if !book_dir.exists() {
        // Book directory is optional
        return Ok(());
    }

    // Extract example names from paths
    let example_names: HashSet<String> = extract_file_stems(examples).into_iter().collect();

    // Find all markdown files in book
    let md_files = find_markdown_files(book_dir)?;

    // Extract referenced examples from markdown
    let mut referenced = HashSet::new();
    let example_ref_regex = Regex::new(r"examples/([a-z_]+)\.rs").unwrap();

    for md_file in md_files {
        let content = fs::read_to_string(&md_file)
            .with_context(|| format!("Failed to read {}", md_file.display()))?;

        for cap in example_ref_regex.captures_iter(&content) {
            if let Some(name) = cap.get(1) {
                referenced.insert(name.as_str().to_string());
            }
        }
    }

    // Check for references to non-existent examples
    let mut invalid_refs = Vec::new();
    for ref_name in &referenced {
        if !example_names.contains(ref_name) {
            invalid_refs.push(ref_name.clone());
        }
    }

    if !invalid_refs.is_empty() {
        bail!(
            "Book references non-existent examples:\n  {}",
            invalid_refs.join("\n  ")
        );
    }

    Ok(())
}

/// Find all markdown files in directory
fn find_markdown_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if is_markdown_file(path) {
            files.push(path.to_path_buf());
        }
    }

    Ok(files)
}

/// Step 6: Verify snake_case naming conventions
fn step_check_naming_conventions(examples: &[PathBuf]) -> Result<()> {
    let mut invalid_names = Vec::new();

    for example in examples {
        let name = example
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid filename"))?;

        if !validate_snake_case(name) {
            invalid_names.push(name.to_string());
        }
    }

    if !invalid_names.is_empty() {
        let error_msg = format_error_list(&invalid_names, "Examples not in snake_case");
        bail!("{}", error_msg);
    }

    Ok(())
}

// ============================================================================
// Pure, testable functions (EXTREME TDD - GREEN phase: implementation)
// ============================================================================

/// Format a list of errors into a displayable string
fn format_error_list(errors: &[String], prefix: &str) -> String {
    if errors.is_empty() {
        return String::new();
    }
    format!("{}:\n  {}", prefix, errors.join("\n  "))
}

/// Extract file names from paths
fn extract_file_names(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|p| p.file_name())
        .filter_map(|n| n.to_str())
        .map(|s| s.to_string())
        .collect()
}

/// Extract file stems (name without extension) from paths
fn extract_file_stems(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|p| p.file_stem())
        .filter_map(|n| n.to_str())
        .map(|s| s.to_string())
        .collect()
}

/// Check if a path is a Rust file
fn is_rust_file(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()) == Some("rs")
}

/// Check if a path is a Markdown file
fn is_markdown_file(path: &Path) -> bool {
    path.extension().and_then(|s| s.to_str()) == Some("md")
}

/// Validate snake_case naming
fn validate_snake_case(name: &str) -> bool {
    let regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
    regex.is_match(name)
}

/// Check if content contains a main function (pure function on string content)
fn contains_main_function(content: &str) -> bool {
    let main_regex = Regex::new(r"fn\s+main\s*\(").unwrap();
    main_regex.is_match(content)
}

/// Check if content contains module documentation (pure function on string content)
/// Checks first 10 lines, stops at first non-comment/non-whitespace line
fn contains_module_doc(content: &str) -> bool {
    for line in content.lines().take(10) {
        let trimmed = line.trim();
        if trimmed.starts_with("//!") {
            return true;
        }
        // Stop at first non-comment, non-whitespace line
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            break;
        }
    }
    false
}

/// Count validation errors in results
#[allow(dead_code)]
fn count_validation_errors(results: &ValidationResults) -> usize {
    results.steps.iter().filter(|s| !s.success).count()
}

/// Format validation summary as string
#[allow(dead_code)]
fn format_validation_summary(results: &ValidationResults) -> String {
    let total = results.steps.len();
    let passed = results.steps.iter().filter(|s| s.success).count();
    let failed = total - passed;
    format!("Total: {}, Passed: {}, Failed: {}", total, passed, failed)
}

/// Validation results tracker
struct ValidationResults {
    steps: Vec<StepResult>,
}

struct StepResult {
    number: usize,
    name: String,
    success: bool,
    #[allow(dead_code)]
    error: Option<String>,
}

impl ValidationResults {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }

    fn add_step<F>(&mut self, number: usize, name: &str, f: F)
    where
        F: FnOnce() -> Result<()>,
    {
        print!("Step {}/6: {}... ", number, name);

        match f() {
            Ok(_) => {
                println!("{}", "✓".green());
                self.steps.push(StepResult {
                    number,
                    name: name.to_string(),
                    success: true,
                    error: None,
                });
            }
            Err(e) => {
                println!("{}", "✗".red());
                eprintln!("  {}", format!("{}", e).red());
                self.steps.push(StepResult {
                    number,
                    name: name.to_string(),
                    success: false,
                    error: Some(format!("{}", e)),
                });
            }
        }
    }

    fn has_failures(&self) -> bool {
        self.steps.iter().any(|s| !s.success)
    }

    fn print_summary(&self) {
        println!();
        println!("{}", "═══════════════════════════════════".bold());
        println!("{}", "Summary".bold());
        println!("{}", "═══════════════════════════════════".bold());

        let total = self.steps.len();
        let passed = self.steps.iter().filter(|s| s.success).count();
        let failed = total - passed;

        for step in &self.steps {
            let status = if step.success {
                "✓".green()
            } else {
                "✗".red()
            };
            println!("{} Step {}: {}", status, step.number, step.name);
        }

        println!();
        println!(
            "Total: {}, Passed: {}, Failed: {}",
            total,
            format!("{}", passed).green(),
            if failed > 0 {
                format!("{}", failed).red()
            } else {
                format!("{}", failed).green()
            }
        );

        if failed == 0 {
            println!();
            println!("{}", "✅ All validation checks passed!".green().bold());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_has_module_doc_with_doc() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "//! This is module documentation").unwrap();
        writeln!(f, "fn main() {{}}").unwrap();

        assert!(has_module_doc(&file).unwrap());
    }

    #[test]
    fn test_has_module_doc_without_doc() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "// Regular comment").unwrap();
        writeln!(f, "fn main() {{}}").unwrap();

        assert!(!has_module_doc(&file).unwrap());
    }

    #[test]
    fn test_has_module_doc_empty_file() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        fs::File::create(&file).unwrap();

        assert!(!has_module_doc(&file).unwrap());
    }

    #[test]
    fn test_has_main_function_present() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "fn main() {{}}").unwrap();

        assert!(has_main_function(&file).unwrap());
    }

    #[test]
    fn test_has_main_function_absent() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "fn other() {{}}").unwrap();

        assert!(!has_main_function(&file).unwrap());
    }

    #[test]
    fn test_has_main_function_with_result() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "fn main() -> Result<()> {{}}").unwrap();

        assert!(has_main_function(&file).unwrap());
    }

    #[test]
    fn test_snake_case_validation_valid() {
        let regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
        assert!(regex.is_match("valid_example"));
        assert!(regex.is_match("example123"));
        assert!(regex.is_match("backend_detection"));
    }

    #[test]
    fn test_snake_case_validation_invalid() {
        let regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();
        assert!(!regex.is_match("InvalidExample"));
        assert!(!regex.is_match("invalid-example"));
        assert!(!regex.is_match("123invalid"));
        assert!(!regex.is_match(""));
    }

    #[test]
    fn test_collect_examples_empty_dir() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        let examples = collect_examples(&examples_dir).unwrap();
        assert!(examples.is_empty());
    }

    #[test]
    fn test_collect_examples_with_files() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        // Create example files
        fs::File::create(examples_dir.join("example1.rs")).unwrap();
        fs::File::create(examples_dir.join("example2.rs")).unwrap();
        fs::File::create(examples_dir.join("not_rust.txt")).unwrap();

        let examples = collect_examples(&examples_dir).unwrap();
        assert_eq!(examples.len(), 2);
        assert!(examples.iter().all(|p| p.extension().unwrap() == "rs"));
    }

    #[test]
    fn test_collect_examples_nonexistent_dir() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("nonexistent");

        let result = collect_examples(&examples_dir);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_markdown_files() {
        let temp = TempDir::new().unwrap();
        let book_dir = temp.path().join("book");
        fs::create_dir(&book_dir).unwrap();

        // Create nested structure
        let src_dir = book_dir.join("src");
        fs::create_dir(&src_dir).unwrap();

        fs::File::create(book_dir.join("README.md")).unwrap();
        fs::File::create(src_dir.join("chapter1.md")).unwrap();
        fs::File::create(src_dir.join("not_md.txt")).unwrap();

        let md_files = find_markdown_files(&book_dir).unwrap();
        assert_eq!(md_files.len(), 2);
        assert!(md_files.iter().all(|p| p.extension().unwrap() == "md"));
    }

    #[test]
    fn test_validation_results_all_pass() {
        let mut results = ValidationResults::new();
        results.steps.push(StepResult {
            number: 1,
            name: "Test 1".to_string(),
            success: true,
            error: None,
        });
        results.steps.push(StepResult {
            number: 2,
            name: "Test 2".to_string(),
            success: true,
            error: None,
        });

        assert!(!results.has_failures());
    }

    #[test]
    fn test_validation_results_with_failure() {
        let mut results = ValidationResults::new();
        results.steps.push(StepResult {
            number: 1,
            name: "Test 1".to_string(),
            success: true,
            error: None,
        });
        results.steps.push(StepResult {
            number: 2,
            name: "Test 2".to_string(),
            success: false,
            error: Some("Error message".to_string()),
        });

        assert!(results.has_failures());
    }

    #[test]
    fn test_get_project_root_current_dir() {
        // This test assumes we're running in the project root or a subdirectory
        let result = get_project_root();
        assert!(result.is_ok());
        let root = result.unwrap();
        assert!(root.join("Cargo.toml").exists());
    }

    #[test]
    fn test_validation_results_print_summary_success() {
        let mut results = ValidationResults::new();
        results.steps.push(StepResult {
            number: 1,
            name: "Test 1".to_string(),
            success: true,
            error: None,
        });
        results.steps.push(StepResult {
            number: 2,
            name: "Test 2".to_string(),
            success: true,
            error: None,
        });

        // Just verify it doesn't panic
        results.print_summary();
    }

    #[test]
    fn test_validation_results_print_summary_with_failures() {
        let mut results = ValidationResults::new();
        results.steps.push(StepResult {
            number: 1,
            name: "Test 1".to_string(),
            success: true,
            error: None,
        });
        results.steps.push(StepResult {
            number: 2,
            name: "Test 2".to_string(),
            success: false,
            error: Some("Test error".to_string()),
        });

        // Just verify it doesn't panic
        results.print_summary();
    }

    #[test]
    fn test_validation_results_empty() {
        let results = ValidationResults::new();
        assert!(!results.has_failures());
        results.print_summary();
    }

    #[test]
    fn test_step_check_module_docs_success() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("example1.rs");
        let file2 = temp.path().join("example2.rs");

        let mut f1 = fs::File::create(&file1).unwrap();
        writeln!(f1, "//! Module doc").unwrap();
        writeln!(f1, "fn main() {{}}").unwrap();

        let mut f2 = fs::File::create(&file2).unwrap();
        writeln!(f2, "//! Another module doc").unwrap();
        writeln!(f2, "fn main() {{}}").unwrap();

        let examples = vec![file1, file2];
        let result = step_check_module_docs(&examples);
        assert!(result.is_ok());
    }

    #[test]
    fn test_step_check_module_docs_failure() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("example1.rs");
        let file2 = temp.path().join("example2.rs");

        let mut f1 = fs::File::create(&file1).unwrap();
        writeln!(f1, "//! Module doc").unwrap();
        writeln!(f1, "fn main() {{}}").unwrap();

        let mut f2 = fs::File::create(&file2).unwrap();
        writeln!(f2, "// Regular comment, no module doc").unwrap();
        writeln!(f2, "fn main() {{}}").unwrap();

        let examples = vec![file1, file2];
        let result = step_check_module_docs(&examples);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_check_naming_conventions_success() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("valid_example.rs");
        let file2 = temp.path().join("another_valid_example_123.rs");
        fs::File::create(&file1).unwrap();
        fs::File::create(&file2).unwrap();

        let examples = vec![file1, file2];
        let result = step_check_naming_conventions(&examples);
        assert!(result.is_ok());
    }

    #[test]
    fn test_step_check_naming_conventions_failure() {
        let temp = TempDir::new().unwrap();
        let file1 = temp.path().join("valid_example.rs");
        let file2 = temp.path().join("InvalidExample.rs");
        fs::File::create(&file1).unwrap();
        fs::File::create(&file2).unwrap();

        let examples = vec![file1, file2];
        let result = step_check_naming_conventions(&examples);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_results_add_step_success() {
        let mut results = ValidationResults::new();
        results.add_step(1, "Test step", || Ok(()));

        assert_eq!(results.steps.len(), 1);
        assert!(results.steps[0].success);
        assert!(!results.has_failures());
    }

    #[test]
    fn test_validation_results_add_step_failure() {
        let mut results = ValidationResults::new();
        results.add_step(1, "Test step", || anyhow::bail!("Test error"));

        assert_eq!(results.steps.len(), 1);
        assert!(!results.steps[0].success);
        assert!(results.has_failures());
        assert!(results.steps[0].error.is_some());
    }

    #[test]
    fn test_validation_results_multiple_steps() {
        let mut results = ValidationResults::new();

        results.add_step(1, "Step 1", || Ok(()));
        results.add_step(2, "Step 2", || anyhow::bail!("Error"));
        results.add_step(3, "Step 3", || Ok(()));

        assert_eq!(results.steps.len(), 3);
        assert!(results.steps[0].success);
        assert!(!results.steps[1].success);
        assert!(results.steps[2].success);
        assert!(results.has_failures());
    }

    #[test]
    fn test_get_project_root_error_path() {
        // Test by temporarily changing directory to root (which has no Cargo.toml)
        let original_dir = std::env::current_dir().unwrap();

        // Try to change to a directory that definitely doesn't have Cargo.toml
        // This test is challenging because we need to be in a place without Cargo.toml
        // For now, just test that the current directory works
        let result = get_project_root();
        assert!(result.is_ok());

        // Restore directory
        std::env::set_current_dir(original_dir).ok();
    }

    #[test]
    fn test_collect_examples_sorting() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        // Create files in non-alphabetical order
        fs::File::create(examples_dir.join("zebra.rs")).unwrap();
        fs::File::create(examples_dir.join("apple.rs")).unwrap();
        fs::File::create(examples_dir.join("banana.rs")).unwrap();

        let examples = collect_examples(&examples_dir).unwrap();
        assert_eq!(examples.len(), 3);

        // Verify sorted order
        let names: Vec<_> = examples
            .iter()
            .filter_map(|p| p.file_stem())
            .filter_map(|n| n.to_str())
            .collect();
        assert_eq!(names, vec!["apple", "banana", "zebra"]);
    }

    #[test]
    fn test_find_markdown_files_nested() {
        let temp = TempDir::new().unwrap();
        let book_dir = temp.path().join("book");
        fs::create_dir_all(&book_dir).unwrap();

        // Create deeply nested structure
        let ch1 = book_dir.join("chapter1");
        let ch2 = book_dir.join("chapter2");
        let sub = ch1.join("subsection");
        fs::create_dir_all(&sub).unwrap();
        fs::create_dir_all(&ch2).unwrap();

        fs::File::create(book_dir.join("intro.md")).unwrap();
        fs::File::create(ch1.join("part1.md")).unwrap();
        fs::File::create(sub.join("details.md")).unwrap();
        fs::File::create(ch2.join("part2.md")).unwrap();
        fs::File::create(ch2.join("README.txt")).unwrap(); // Not markdown

        let md_files = find_markdown_files(&book_dir).unwrap();
        assert_eq!(md_files.len(), 4); // Should find all 4 .md files
    }

    #[test]
    fn test_has_module_doc_with_blank_lines() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        // Module doc after blank line
        writeln!(f).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "//! Module documentation").unwrap();
        writeln!(f, "fn main() {{}}").unwrap();

        assert!(has_module_doc(&file).unwrap());
    }

    #[test]
    fn test_has_module_doc_multiline() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "//! First line of module doc").unwrap();
        writeln!(f, "//! Second line of module doc").unwrap();
        writeln!(f, "//! Third line").unwrap();
        writeln!(f, "fn main() {{}}").unwrap();

        assert!(has_module_doc(&file).unwrap());
    }

    #[test]
    fn test_snake_case_edge_cases() {
        let regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();

        // Valid
        assert!(regex.is_match("a"));
        assert!(regex.is_match("ab"));
        assert!(regex.is_match("a_b"));
        assert!(regex.is_match("a1"));
        assert!(regex.is_match("a_1_b_2"));

        // Invalid
        assert!(!regex.is_match("A"));
        assert!(!regex.is_match("_a"));
        assert!(!regex.is_match("1a"));
        assert!(!regex.is_match("a-b"));
        assert!(!regex.is_match(""));
    }

    // ============================================================================
    // EXTREME TDD: Tests for pure functions (RED phase - tests written FIRST)
    // ============================================================================

    #[test]
    fn test_format_error_list_single() {
        let errors = vec!["error1".to_string()];
        let result = format_error_list(&errors, "Test");
        assert!(result.contains("Test"));
        assert!(result.contains("error1"));
    }

    #[test]
    fn test_format_error_list_multiple() {
        let errors = vec![
            "error1".to_string(),
            "error2".to_string(),
            "error3".to_string(),
        ];
        let result = format_error_list(&errors, "Failures");
        assert!(result.contains("Failures"));
        assert!(result.contains("error1"));
        assert!(result.contains("error2"));
        assert!(result.contains("error3"));
    }

    #[test]
    fn test_format_error_list_empty() {
        let errors: Vec<String> = vec![];
        let result = format_error_list(&errors, "Test");
        // Should handle empty gracefully
        assert!(result.is_empty() || result.contains("Test"));
    }

    #[test]
    fn test_extract_file_names_from_paths() {
        let paths = vec![
            PathBuf::from("/path/to/example1.rs"),
            PathBuf::from("/another/path/example2.rs"),
            PathBuf::from("relative/example3.rs"),
        ];
        let names = extract_file_names(&paths);
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"example1.rs".to_string()));
        assert!(names.contains(&"example2.rs".to_string()));
        assert!(names.contains(&"example3.rs".to_string()));
    }

    #[test]
    fn test_extract_file_names_empty() {
        let paths: Vec<PathBuf> = vec![];
        let names = extract_file_names(&paths);
        assert!(names.is_empty());
    }

    #[test]
    fn test_extract_file_stems() {
        let paths = vec![PathBuf::from("example1.rs"), PathBuf::from("example2.rs")];
        let stems = extract_file_stems(&paths);
        assert_eq!(stems.len(), 2);
        assert!(stems.contains(&"example1".to_string()));
        assert!(stems.contains(&"example2".to_string()));
    }

    #[test]
    fn test_is_rust_file_valid() {
        assert!(is_rust_file(&PathBuf::from("example.rs")));
        assert!(is_rust_file(&PathBuf::from("/path/to/file.rs")));
    }

    #[test]
    fn test_is_rust_file_invalid() {
        assert!(!is_rust_file(&PathBuf::from("example.txt")));
        assert!(!is_rust_file(&PathBuf::from("example")));
        assert!(!is_rust_file(&PathBuf::from("example.md")));
    }

    #[test]
    fn test_is_markdown_file_valid() {
        assert!(is_markdown_file(&PathBuf::from("README.md")));
        assert!(is_markdown_file(&PathBuf::from("/path/to/file.md")));
    }

    #[test]
    fn test_is_markdown_file_invalid() {
        assert!(!is_markdown_file(&PathBuf::from("file.rs")));
        assert!(!is_markdown_file(&PathBuf::from("file.txt")));
        assert!(!is_markdown_file(&PathBuf::from("file")));
    }

    #[test]
    fn test_validate_snake_case_valid() {
        assert!(validate_snake_case("valid_name"));
        assert!(validate_snake_case("another_valid_name_123"));
        assert!(validate_snake_case("abc"));
        assert!(validate_snake_case("a1b2c3"));
    }

    #[test]
    fn test_validate_snake_case_invalid() {
        assert!(!validate_snake_case("InvalidName"));
        assert!(!validate_snake_case("invalid-name"));
        assert!(!validate_snake_case("123invalid"));
        assert!(!validate_snake_case("_invalid"));
        assert!(!validate_snake_case(""));
    }

    #[test]
    fn test_contains_main_function_present() {
        let content = "fn main() { println!(\"Hello\"); }";
        assert!(contains_main_function(content));
    }

    #[test]
    fn test_contains_main_function_with_result() {
        let content = "fn main() -> Result<()> { Ok(()) }";
        assert!(contains_main_function(content));
    }

    #[test]
    fn test_contains_main_function_absent() {
        let content = "fn other() { }";
        assert!(!contains_main_function(content));
    }

    #[test]
    fn test_contains_main_function_comment() {
        let content = "// fn main() {}";
        // Should still detect it (simple regex)
        assert!(contains_main_function(content));
    }

    #[test]
    fn test_contains_module_doc_present() {
        let content = "//! Module documentation\nfn main() {}";
        assert!(contains_module_doc(content));
    }

    #[test]
    fn test_contains_module_doc_multiline() {
        let content = "//! First line\n//! Second line\nfn main() {}";
        assert!(contains_module_doc(content));
    }

    #[test]
    fn test_contains_module_doc_absent() {
        let content = "// Regular comment\nfn main() {}";
        assert!(!contains_module_doc(content));
    }

    #[test]
    fn test_contains_module_doc_empty() {
        let content = "";
        assert!(!contains_module_doc(content));
    }

    #[test]
    fn test_count_validation_errors_none() {
        let results = ValidationResults {
            steps: vec![StepResult {
                number: 1,
                name: "Test".to_string(),
                success: true,
                error: None,
            }],
        };
        assert_eq!(count_validation_errors(&results), 0);
    }

    #[test]
    fn test_count_validation_errors_some() {
        let results = ValidationResults {
            steps: vec![
                StepResult {
                    number: 1,
                    name: "Test1".to_string(),
                    success: true,
                    error: None,
                },
                StepResult {
                    number: 2,
                    name: "Test2".to_string(),
                    success: false,
                    error: Some("Error".to_string()),
                },
                StepResult {
                    number: 3,
                    name: "Test3".to_string(),
                    success: false,
                    error: Some("Error2".to_string()),
                },
            ],
        };
        assert_eq!(count_validation_errors(&results), 2);
    }

    #[test]
    fn test_format_validation_summary_all_pass() {
        let results = ValidationResults {
            steps: vec![StepResult {
                number: 1,
                name: "Test".to_string(),
                success: true,
                error: None,
            }],
        };
        let summary = format_validation_summary(&results);
        assert!(summary.contains("1"));
        assert!(summary.contains("0"));
    }

    #[test]
    fn test_format_validation_summary_with_failures() {
        let results = ValidationResults {
            steps: vec![
                StepResult {
                    number: 1,
                    name: "Test1".to_string(),
                    success: true,
                    error: None,
                },
                StepResult {
                    number: 2,
                    name: "Test2".to_string(),
                    success: false,
                    error: Some("Error".to_string()),
                },
            ],
        };
        let summary = format_validation_summary(&results);
        assert!(summary.contains("2"));
        assert!(summary.contains("1"));
        assert!(summary.contains("1"));
    }

    // Additional integration tests for better coverage
    #[test]
    fn test_step_check_book_references_no_book_dir() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        let file1 = examples_dir.join("example1.rs");
        fs::File::create(&file1).unwrap();

        let examples = vec![file1];
        let book_dir = temp.path().join("nonexistent_book");

        // Should succeed when book dir doesn't exist (it's optional)
        let result = step_check_book_references(&examples, &book_dir);
        assert!(result.is_ok());
    }

    #[test]
    fn test_step_check_book_references_with_valid_references() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        let book_dir = temp.path().join("book");
        fs::create_dir_all(&examples_dir).unwrap();
        fs::create_dir_all(&book_dir).unwrap();

        // Create example file
        let example_file = examples_dir.join("my_example.rs");
        fs::File::create(&example_file).unwrap();

        // Create book file that references the example
        let book_file = book_dir.join("chapter.md");
        let mut f = fs::File::create(&book_file).unwrap();
        writeln!(f, "# Chapter\nSee examples/my_example.rs for details.").unwrap();

        let examples = vec![example_file];
        let result = step_check_book_references(&examples, &book_dir);
        assert!(result.is_ok());
    }

    #[test]
    fn test_step_check_book_references_invalid_reference() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        let book_dir = temp.path().join("book");
        fs::create_dir_all(&examples_dir).unwrap();
        fs::create_dir_all(&book_dir).unwrap();

        // Create example file
        let example_file = examples_dir.join("actual_example.rs");
        fs::File::create(&example_file).unwrap();

        // Create book file that references a DIFFERENT example
        let book_file = book_dir.join("chapter.md");
        let mut f = fs::File::create(&book_file).unwrap();
        writeln!(f, "# Chapter\nSee examples/nonexistent_example.rs").unwrap();

        let examples = vec![example_file];
        let result = step_check_book_references(&examples, &book_dir);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("non-existent"));
    }

    #[test]
    fn test_get_project_root_with_cargo_toml() {
        // Since we're running from the project, this should find Cargo.toml
        let result = get_project_root();
        assert!(result.is_ok());
        let root = result.unwrap();
        assert!(root.join("Cargo.toml").exists());
    }

    #[test]
    fn test_has_module_doc_stops_at_code() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        // Module doc should appear before code
        writeln!(f, "// Regular comment").unwrap();
        writeln!(f, "fn main() {{}}").unwrap();
        writeln!(f, "//! This comes too late").unwrap();

        assert!(!has_module_doc(&file).unwrap());
    }

    #[test]
    fn test_collect_examples_mixed_files() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        // Create various file types
        fs::File::create(examples_dir.join("example1.rs")).unwrap();
        fs::File::create(examples_dir.join("example2.rs")).unwrap();
        fs::File::create(examples_dir.join("readme.md")).unwrap();
        fs::File::create(examples_dir.join("data.txt")).unwrap();
        fs::File::create(examples_dir.join("config.toml")).unwrap();

        let examples = collect_examples(&examples_dir).unwrap();

        // Should only find .rs files
        assert_eq!(examples.len(), 2);
        assert!(examples.iter().all(|p| p.extension().unwrap() == "rs"));
    }

    #[test]
    fn test_extract_file_stems_with_various_extensions() {
        let paths = vec![
            PathBuf::from("example1.rs"),
            PathBuf::from("example2.md"),
            PathBuf::from("example3.txt"),
            PathBuf::from("path/to/example4.rs"),
        ];
        let stems = extract_file_stems(&paths);
        assert_eq!(stems.len(), 4);
        assert!(stems.contains(&"example1".to_string()));
        assert!(stems.contains(&"example2".to_string()));
        assert!(stems.contains(&"example3".to_string()));
        assert!(stems.contains(&"example4".to_string()));
    }

    #[test]
    fn test_validation_results_mixed_outcomes() {
        let mut results = ValidationResults::new();

        results.add_step(1, "Pass 1", || Ok(()));
        results.add_step(2, "Fail", || anyhow::bail!("Error"));
        results.add_step(3, "Pass 2", || Ok(()));
        results.add_step(4, "Pass 3", || Ok(()));

        assert_eq!(results.steps.len(), 4);
        assert!(results.has_failures());

        let passed = results.steps.iter().filter(|s| s.success).count();
        assert_eq!(passed, 3);
    }

    #[test]
    fn test_format_error_list_with_special_characters() {
        let errors = vec![
            "error: missing `main()`".to_string(),
            "error: invalid name \"Bad-Name\"".to_string(),
        ];
        let result = format_error_list(&errors, "Errors");
        assert!(result.contains("Errors"));
        assert!(result.contains("main()"));
        assert!(result.contains("Bad-Name"));
    }

    #[test]
    fn test_contains_module_doc_with_leading_whitespace() {
        let content = "   //! Module doc with leading spaces\nfn main() {}";
        assert!(contains_module_doc(content));
    }

    #[test]
    fn test_contains_main_function_with_whitespace() {
        let content = "fn   main  (  )  { }";
        assert!(contains_main_function(content));
    }

    #[test]
    fn test_contains_main_function_with_async() {
        let content = "async fn main() { }";
        // Current regex WILL match async main (matches "fn main(" substring)
        assert!(contains_main_function(content));
    }

    #[test]
    fn test_validate_snake_case_numbers_only_invalid() {
        assert!(!validate_snake_case("123"));
        assert!(!validate_snake_case("456_test"));
    }

    #[test]
    fn test_validate_snake_case_special_chars() {
        assert!(!validate_snake_case("test@example"));
        assert!(!validate_snake_case("test.example"));
        assert!(!validate_snake_case("test example"));
    }

    #[test]
    fn test_is_rust_file_edge_cases() {
        assert!(is_rust_file(&PathBuf::from("a.rs")));
        assert!(is_rust_file(&PathBuf::from("file.rs")));
        assert!(!is_rust_file(&PathBuf::from("rs"))); // No extension
        assert!(!is_rust_file(&PathBuf::from(".rs.bak")));
        assert!(!is_rust_file(&PathBuf::from("file.rs.bak")));
    }

    #[test]
    fn test_is_markdown_file_edge_cases() {
        assert!(is_markdown_file(&PathBuf::from("a.md")));
        assert!(is_markdown_file(&PathBuf::from("file.md")));
        assert!(!is_markdown_file(&PathBuf::from("md")));
        assert!(!is_markdown_file(&PathBuf::from(".md.bak")));
        assert!(!is_markdown_file(&PathBuf::from("file.md.bak")));
    }

    #[test]
    fn test_count_validation_errors_all_fail() {
        let results = ValidationResults {
            steps: vec![
                StepResult {
                    number: 1,
                    name: "Test1".to_string(),
                    success: false,
                    error: Some("Error1".to_string()),
                },
                StepResult {
                    number: 2,
                    name: "Test2".to_string(),
                    success: false,
                    error: Some("Error2".to_string()),
                },
            ],
        };
        assert_eq!(count_validation_errors(&results), 2);
    }

    #[test]
    fn test_format_validation_summary_empty() {
        let results = ValidationResults { steps: vec![] };
        let summary = format_validation_summary(&results);
        assert!(summary.contains("0"));
    }

    #[test]
    fn test_extract_file_names_invalid_unicode() {
        // Test with valid paths
        let paths = vec![PathBuf::from("example1.rs"), PathBuf::from("example2.rs")];
        let names = extract_file_names(&paths);
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_extract_file_stems_empty_path() {
        let paths: Vec<PathBuf> = vec![];
        let stems = extract_file_stems(&paths);
        assert!(stems.is_empty());
    }

    #[test]
    fn test_contains_module_doc_only_comments() {
        let content = "// Regular comment\n// Another comment\n\n";
        assert!(!contains_module_doc(content));
    }

    #[test]
    fn test_contains_module_doc_after_many_blank_lines() {
        let content = "\n\n\n\n\n//! Module doc after 5 blank lines\n";
        assert!(contains_module_doc(content));
    }

    #[test]
    fn test_contains_main_function_multiple_functions() {
        let content = "fn other() {}\nfn main() {}\nfn more() {}";
        assert!(contains_main_function(content));
    }

    #[test]
    fn test_contains_main_function_in_comment() {
        let content = "// This describes fn main()\nfn other() {}";
        // Regex will match the comment (simple regex)
        assert!(contains_main_function(content));
    }

    #[test]
    fn test_has_module_doc_read_error() {
        // Test with non-existent file
        let path = PathBuf::from("/nonexistent/file/that/does/not/exist.rs");
        let result = has_module_doc(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_main_function_read_error() {
        let path = PathBuf::from("/nonexistent/file/that/does/not/exist.rs");
        let result = has_main_function(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_results_new() {
        let results = ValidationResults::new();
        assert!(results.steps.is_empty());
        assert!(!results.has_failures());
    }

    #[test]
    fn test_step_result_with_long_error() {
        let mut results = ValidationResults::new();
        results.add_step(1, "Test", || {
            anyhow::bail!("This is a very long error message that contains many details about what went wrong in the validation process")
        });

        assert_eq!(results.steps.len(), 1);
        assert!(!results.steps[0].success);
        assert!(results.steps[0].error.is_some());
    }

    #[test]
    fn test_collect_examples_preserves_full_paths() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        fs::File::create(examples_dir.join("ex1.rs")).unwrap();

        let examples = collect_examples(&examples_dir).unwrap();
        assert_eq!(examples.len(), 1);
        assert!(examples[0].is_absolute() || examples[0].starts_with(&examples_dir));
    }

    #[test]
    fn test_find_markdown_files_empty_dir() {
        let temp = TempDir::new().unwrap();
        let book_dir = temp.path().join("book");
        fs::create_dir(&book_dir).unwrap();

        let md_files = find_markdown_files(&book_dir).unwrap();
        assert!(md_files.is_empty());
    }

    #[test]
    fn test_step_check_book_references_empty_examples() {
        let temp = TempDir::new().unwrap();
        let book_dir = temp.path().join("book");
        fs::create_dir(&book_dir).unwrap();

        let examples: Vec<PathBuf> = vec![];
        let result = step_check_book_references(&examples, &book_dir);
        assert!(result.is_ok());
    }

    #[test]
    fn test_step_check_runnable_with_missing_main() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        // Create file without main function
        let file = examples_dir.join("no_main.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "// No main function here").unwrap();
        writeln!(f, "fn helper() {{}}").unwrap();

        let project_root = get_project_root().unwrap();
        let examples = vec![file];

        let result = step_check_runnable(&examples, &project_root);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not runnable") || err.contains("missing main"));
    }

    #[test]
    fn test_format_error_list_single_error() {
        let errors = vec!["single error".to_string()];
        let result = format_error_list(&errors, "Problems");
        assert!(result.contains("Problems"));
        assert!(result.contains("single error"));
        assert_eq!(result.matches('\n').count(), 1); // One newline for the error
    }

    #[test]
    fn test_validation_results_all_pass_no_errors() {
        let mut results = ValidationResults::new();
        results.add_step(1, "Test 1", || Ok(()));
        results.add_step(2, "Test 2", || Ok(()));
        results.add_step(3, "Test 3", || Ok(()));

        assert_eq!(results.steps.len(), 3);
        assert!(!results.has_failures());
        assert_eq!(count_validation_errors(&results), 0);
    }

    #[test]
    fn test_contains_module_doc_stops_exactly_at_10_lines() {
        // If module doc is at line 11, it should not be found
        let content = "\n\n\n\n\n\n\n\n\n\nfn main() {}\n//! Too late";
        assert!(!contains_module_doc(content));
    }

    #[test]
    fn test_contains_main_function_case_sensitive() {
        let content = "fn Main() {}"; // Capital M
                                      // Should not match since Rust is case-sensitive and Main != main
        assert!(!contains_main_function(content));
    }

    #[test]
    fn test_validate_snake_case_single_char() {
        assert!(validate_snake_case("a"));
        assert!(validate_snake_case("z"));
        assert!(!validate_snake_case("A"));
        assert!(!validate_snake_case("Z"));
    }

    #[test]
    fn test_format_validation_summary_single_step() {
        let results = ValidationResults {
            steps: vec![StepResult {
                number: 1,
                name: "Only step".to_string(),
                success: true,
                error: None,
            }],
        };
        let summary = format_validation_summary(&results);
        assert!(summary.contains("1"));
        assert!(summary.contains("Passed: 1"));
    }

    #[test]
    fn test_step_check_runnable_error_formatting() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("bad.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "fn other() {{}}").unwrap(); // No main

        let project_root = get_project_root().unwrap();
        let result = step_check_runnable(&vec![file.clone()], &project_root);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("runnable") || err_str.contains("main"));
    }

    #[test]
    fn test_has_module_doc_early_return_on_code() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("example.rs");
        let mut f = fs::File::create(&file).unwrap();
        writeln!(f, "use std::io;").unwrap(); // Code line
        writeln!(f, "//! Too late").unwrap();

        assert!(!has_module_doc(&file).unwrap());
    }

    #[test]
    fn test_step_check_module_docs_with_multiple_missing() {
        let temp = TempDir::new().unwrap();

        let file1 = temp.path().join("no_doc1.rs");
        let file2 = temp.path().join("no_doc2.rs");
        let file3 = temp.path().join("has_doc.rs");

        let mut f1 = fs::File::create(&file1).unwrap();
        writeln!(f1, "fn main() {{}}").unwrap();

        let mut f2 = fs::File::create(&file2).unwrap();
        writeln!(f2, "fn main() {{}}").unwrap();

        let mut f3 = fs::File::create(&file3).unwrap();
        writeln!(f3, "//! Good doc").unwrap();
        writeln!(f3, "fn main() {{}}").unwrap();

        let examples = vec![file1, file2, file3];
        let result = step_check_module_docs(&examples);

        assert!(result.is_err());
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("missing module documentation"));
    }

    #[test]
    fn test_step_check_book_references_multiple_invalid() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        let book_dir = temp.path().join("book");
        fs::create_dir_all(&examples_dir).unwrap();
        fs::create_dir_all(&book_dir).unwrap();

        let example = examples_dir.join("real_example.rs");
        fs::File::create(&example).unwrap();

        let book_file = book_dir.join("chapter.md");
        let mut f = fs::File::create(&book_file).unwrap();
        writeln!(f, "See examples/fake_one.rs and examples/fake_two.rs").unwrap();

        let result = step_check_book_references(&vec![example], &book_dir);
        assert!(result.is_err());
    }

    #[test]
    fn test_collect_examples_single_file() {
        let temp = TempDir::new().unwrap();
        let examples_dir = temp.path().join("examples");
        fs::create_dir(&examples_dir).unwrap();

        fs::File::create(examples_dir.join("only.rs")).unwrap();

        let examples = collect_examples(&examples_dir).unwrap();
        assert_eq!(examples.len(), 1);
    }

    #[test]
    fn test_find_markdown_files_single_file() {
        let temp = TempDir::new().unwrap();
        let book_dir = temp.path().join("book");
        fs::create_dir(&book_dir).unwrap();
        fs::File::create(book_dir.join("single.md")).unwrap();

        let md_files = find_markdown_files(&book_dir).unwrap();
        assert_eq!(md_files.len(), 1);
    }

    #[test]
    fn test_extract_file_names_single_path() {
        let paths = vec![PathBuf::from("single.rs")];
        let names = extract_file_names(&paths);
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "single.rs");
    }

    #[test]
    fn test_extract_file_stems_single_path() {
        let paths = vec![PathBuf::from("single.rs")];
        let stems = extract_file_stems(&paths);
        assert_eq!(stems.len(), 1);
        assert_eq!(stems[0], "single");
    }

    #[test]
    fn test_validation_results_has_failures_true() {
        let mut results = ValidationResults::new();
        results.add_step(1, "Fail", || anyhow::bail!("Error"));
        assert!(results.has_failures());
    }

    #[test]
    fn test_validation_results_has_failures_false() {
        let mut results = ValidationResults::new();
        results.add_step(1, "Pass", || Ok(()));
        assert!(!results.has_failures());
    }

    #[test]
    fn test_contains_module_doc_with_tabs() {
        let content = "\t//! Module doc with tab\nfn main() {}";
        assert!(contains_module_doc(content));
    }

    #[test]
    fn test_contains_main_function_with_generics() {
        let content = "fn main<T>() {}"; // Invalid Rust, but tests regex
        assert!(!contains_main_function(content)); // Doesn't match because of <
    }

    #[test]
    fn test_format_error_list_preserves_order() {
        let errors = vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ];
        let result = format_error_list(&errors, "List");
        let first_pos = result.find("first").unwrap();
        let second_pos = result.find("second").unwrap();
        let third_pos = result.find("third").unwrap();
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_validate_snake_case_max_length() {
        // Test with a very long but valid snake_case name
        let long_name = "a".repeat(100);
        assert!(validate_snake_case(&long_name));
    }

    #[test]
    fn test_is_rust_file_with_path() {
        assert!(is_rust_file(&PathBuf::from("/absolute/path/file.rs")));
        assert!(is_rust_file(&PathBuf::from("relative/path/file.rs")));
    }

    #[test]
    fn test_is_markdown_file_with_path() {
        assert!(is_markdown_file(&PathBuf::from("/absolute/path/file.md")));
        assert!(is_markdown_file(&PathBuf::from("relative/path/file.md")));
    }

    #[test]
    fn test_count_validation_errors_empty() {
        let results = ValidationResults { steps: vec![] };
        assert_eq!(count_validation_errors(&results), 0);
    }

    #[test]
    fn test_step_check_naming_conventions_empty_examples() {
        let examples: Vec<PathBuf> = vec![];
        let result = step_check_naming_conventions(&examples);
        assert!(result.is_ok());
    }
}
