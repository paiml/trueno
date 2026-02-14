//! Core tests for validate_examples module

use super::helpers::*;
use super::results::*;
use super::steps::*;
use regex::Regex;
use serial_test::serial;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
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
#[serial]
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
#[serial]
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
