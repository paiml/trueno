mod validate_examples;

use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo xtask <command>");
        eprintln!("Commands:");
        eprintln!("  validate-examples  Validate book examples meet EXTREME TDD quality");
        std::process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "validate-examples" => validate_examples::run(),
        _ => {
            eprintln!("Unknown command: {}", command);
            std::process::exit(1);
        }
    }
}
