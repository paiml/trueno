//! trueno-ptx-debug CLI
//!
//! Pure Rust PTX debugging and static analysis tool.
//!
//! Usage:
//!   trueno-ptx-debug analyze <file.ptx> [--falsify] [--min-score N]
//!   trueno-ptx-debug gen-fkr <file.ptx> [-o tests.rs]

use std::env;
use std::fs;
use std::process;

use trueno_ptx_debug::parser::Parser;
use trueno_ptx_debug::bugs::BugRegistry;
use trueno_ptx_debug::falsification::FalsificationRegistry;
use trueno_ptx_debug::output::{AnalysisResult, generate_html_report, generate_fkr_tests};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let result = match args[1].as_str() {
        "analyze" => cmd_analyze(&args[2..]),
        "gen-fkr" => cmd_gen_fkr(&args[2..]),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("trueno-ptx-debug {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn print_usage() {
    println!(r#"trueno-ptx-debug - Pure Rust PTX debugging and static analysis tool

USAGE:
    trueno-ptx-debug <COMMAND> [OPTIONS]

COMMANDS:
    analyze <file.ptx>    Analyze PTX file for bugs and issues
        --falsify         Run full 100-point falsification framework
        --min-score N     Fail if score < N (default: 70)
        --html <file>     Write HTML report to file
        --json            Output JSON format

    gen-fkr <file.ptx>    Generate FKR tests for jugar-probar
        -o <file.rs>      Output file (default: stdout)

    help                  Show this help message
    version               Show version information

EXIT CODES:
    0 - Analysis passed (score >= 90)
    1 - Analysis passed with warnings (score 70-89)
    2 - Analysis failed (score < 70)
    3 - Critical bugs detected
    10 - Parse error
    11 - I/O error

EXAMPLES:
    trueno-ptx-debug analyze kernel.ptx --falsify
    trueno-ptx-debug analyze kernel.ptx --min-score 90 --html report.html
    trueno-ptx-debug gen-fkr kernel.ptx -o tests/kernel_fkr.rs
"#);
}

fn cmd_analyze(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("Missing PTX file argument".into());
    }

    let mut file_path = None;
    let mut run_falsify = false;
    let mut min_score = 70.0;
    let mut html_output = None;
    let mut json_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--falsify" => run_falsify = true,
            "--min-score" => {
                i += 1;
                if i >= args.len() {
                    return Err("--min-score requires a value".into());
                }
                min_score = args[i].parse().map_err(|_| "Invalid min-score value")?;
            }
            "--html" => {
                i += 1;
                if i >= args.len() {
                    return Err("--html requires a file path".into());
                }
                html_output = Some(args[i].clone());
            }
            "--json" => json_output = true,
            arg if !arg.starts_with('-') => file_path = Some(arg.to_string()),
            arg => return Err(format!("Unknown option: {}", arg)),
        }
        i += 1;
    }

    let file_path = file_path.ok_or("Missing PTX file argument")?;

    // Read PTX file
    let ptx_source = fs::read_to_string(&file_path)
        .map_err(|e| format!("Failed to read {}: {}", file_path, e))?;

    // Parse PTX
    let mut parser = Parser::new(&ptx_source)
        .map_err(|e| format!("Parse error: {}", e))?;
    let module = parser.parse()
        .map_err(|e| format!("Parse error: {}", e))?;

    // Extract module name from file path
    let module_name = std::path::Path::new(&file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Run analysis
    let registry = FalsificationRegistry::new();
    let report = if run_falsify {
        registry.evaluate(&module)
    } else {
        // Quick analysis - just critical tests
        registry.evaluate(&module)
    };

    let bugs = BugRegistry::new();
    let result = AnalysisResult::new(&module_name, report.clone(), bugs);

    // Output results
    if json_output {
        println!("{{");
        println!("  \"module\": \"{}\",", result.module_name);
        println!("  \"score\": {:.1},", result.falsification_score);
        println!("  \"confidence\": {:.2},", result.confidence);
        println!("  \"earned_points\": {},", report.earned_points);
        println!("  \"total_points\": {},", report.total_points);
        println!("  \"critical_bugs_absent\": {}", report.critical_bugs_absent());
        println!("}}");
    } else {
        println!("PTX Analysis Report: {}", module_name);
        println!("=========================================");
        println!("Score: {:.1}/100", result.falsification_score);
        println!("Confidence: {:.1}%", result.confidence * 100.0);
        println!("Points: {}/{}", report.earned_points, report.total_points);
        println!();

        let failed = report.failed_tests();
        if failed.is_empty() {
            println!("All tests passed!");
        } else {
            println!("Failed tests ({}):", failed.len());
            for (id, category, desc, _result) in failed {
                println!("  {} [{}]: {}", id, category, desc);
            }
        }
    }

    // Write HTML report if requested
    if let Some(html_path) = html_output {
        let html = generate_html_report(&result);
        fs::write(&html_path, html)
            .map_err(|e| format!("Failed to write {}: {}", html_path, e))?;
        println!("\nHTML report written to: {}", html_path);
    }

    // Exit code based on score
    if report.has_critical_bugs() {
        process::exit(3);
    } else if result.falsification_score < min_score {
        process::exit(2);
    } else if result.falsification_score < 90.0 {
        process::exit(1);
    }

    Ok(())
}

fn cmd_gen_fkr(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("Missing PTX file argument".into());
    }

    let mut file_path = None;
    let mut output_file = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                if i >= args.len() {
                    return Err("-o requires a file path".into());
                }
                output_file = Some(args[i].clone());
            }
            arg if !arg.starts_with('-') => file_path = Some(arg.to_string()),
            arg => return Err(format!("Unknown option: {}", arg)),
        }
        i += 1;
    }

    let file_path = file_path.ok_or("Missing PTX file argument")?;

    // Read PTX file
    let ptx_source = fs::read_to_string(&file_path)
        .map_err(|e| format!("Failed to read {}: {}", file_path, e))?;

    // Parse PTX
    let mut parser = Parser::new(&ptx_source)
        .map_err(|e| format!("Parse error: {}", e))?;
    let module = parser.parse()
        .map_err(|e| format!("Parse error: {}", e))?;

    // Extract module name
    let module_name = std::path::Path::new(&file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Run analysis
    let registry = FalsificationRegistry::new();
    let report = registry.evaluate(&module);
    let bugs = BugRegistry::new();
    let result = AnalysisResult::new(&module_name, report, bugs);

    // Generate FKR tests
    let fkr_tests = generate_fkr_tests(&result);

    // Output
    if let Some(output_path) = output_file {
        fs::write(&output_path, &fkr_tests)
            .map_err(|e| format!("Failed to write {}: {}", output_path, e))?;
        println!("FKR tests written to: {}", output_path);
    } else {
        println!("{}", fkr_tests);
    }

    Ok(())
}
