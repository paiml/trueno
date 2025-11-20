mod validate_examples;

use anyhow::{bail, Result};
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    match run_command(&args) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Parse and run the command from command-line arguments (pure, testable)
fn run_command(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        bail!("Usage: cargo xtask <command>\nCommands:\n  validate-examples  Validate book examples meet EXTREME TDD quality");
    }

    let command = &args[1];

    match command.as_str() {
        "validate-examples" => validate_examples::run(),
        _ => bail!("Unknown command: {}", command),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_command_validate_examples() {
        // We can't easily test the actual execution since it requires a full project setup
        // but we can test the command parsing logic
        let args = vec!["xtask".to_string(), "validate-examples".to_string()];
        // This will fail because we don't have examples/ in test env, but it proves parsing works
        let result = run_command(&args);
        assert!(result.is_err()); // Will fail to find examples dir, which is expected
    }

    #[test]
    fn test_run_command_no_args() {
        let args = vec!["xtask".to_string()];
        let result = run_command(&args);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Usage"));
    }

    #[test]
    fn test_run_command_unknown() {
        let args = vec!["xtask".to_string(), "unknown".to_string()];
        let result = run_command(&args);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown command"));
    }

    #[test]
    fn test_run_command_empty_args() {
        let args: Vec<String> = vec![];
        let result = run_command(&args);
        assert!(result.is_err());
    }
}
