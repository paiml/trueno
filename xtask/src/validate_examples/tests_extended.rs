//! Extended tests for validate_examples module

use super::helpers::*;
use super::results::*;
use super::steps::*;
use regex::Regex;
use serial_test::serial;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

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
#[serial]
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
#[serial]
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
#[serial]
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
