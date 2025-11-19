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
    results.add_step(2, "Clippy lints", || {
        step_clippy_examples(&project_root)
    });

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
    let current = std::env::current_dir()
        .context("Failed to get current directory")?;

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
        bail!("Examples directory does not exist: {}", examples_dir.display());
    }

    let mut examples = Vec::new();

    for entry in fs::read_dir(examples_dir)
        .context("Failed to read examples directory")?
    {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
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
        let names: Vec<_> = missing_docs.iter()
            .filter_map(|p| p.file_name())
            .filter_map(|n| n.to_str())
            .collect();
        bail!("Examples missing module documentation (//!):\n  {}", names.join("\n  "));
    }

    Ok(())
}

/// Check if a file has module documentation
fn has_module_doc(path: &Path) -> Result<bool> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    // Look for //! comments in first 10 lines
    for line in content.lines().take(10) {
        let trimmed = line.trim();
        if trimmed.starts_with("//!") {
            return Ok(true);
        }
        // Stop at first non-comment, non-whitespace line
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            break;
        }
    }

    Ok(false)
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
        let example_name = example.file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid example filename"))?;

        match run_example_with_timeout(example_name, project_root, TIMEOUT_SECS) {
            Ok(_) => {}, // Success
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
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;

    let main_regex = Regex::new(r"fn\s+main\s*\(").unwrap();
    Ok(main_regex.is_match(&content))
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
    let example_names: HashSet<String> = examples
        .iter()
        .filter_map(|p| p.file_stem())
        .filter_map(|s| s.to_str())
        .map(|s| s.to_string())
        .collect();

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

    for entry in WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            files.push(path.to_path_buf());
        }
    }

    Ok(files)
}

/// Step 6: Verify snake_case naming conventions
fn step_check_naming_conventions(examples: &[PathBuf]) -> Result<()> {
    let mut invalid_names = Vec::new();
    let snake_case_regex = Regex::new(r"^[a-z][a-z0-9_]*$").unwrap();

    for example in examples {
        let name = example
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid filename"))?;

        if !snake_case_regex.is_match(name) {
            invalid_names.push(name.to_string());
        }
    }

    if !invalid_names.is_empty() {
        bail!(
            "Examples not in snake_case:\n  {}",
            invalid_names.join("\n  ")
        );
    }

    Ok(())
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
}
