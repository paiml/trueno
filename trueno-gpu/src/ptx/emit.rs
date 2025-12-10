//! PTX Emission Utilities
//!
//! Helper functions for emitting PTX text from the IR.

use super::builder::PtxModule;

/// PTX emitter configuration
#[derive(Debug, Clone, Default)]
pub struct EmitConfig {
    /// Include comments in output
    pub include_comments: bool,
    /// Pretty print with indentation
    pub pretty_print: bool,
    /// Include debug information
    pub debug_info: bool,
}

impl EmitConfig {
    /// Create a new emit configuration
    #[must_use]
    pub const fn new() -> Self {
        Self {
            include_comments: true,
            pretty_print: true,
            debug_info: false,
        }
    }

    /// Enable debug information
    #[must_use]
    pub const fn with_debug(mut self) -> Self {
        self.debug_info = true;
        self
    }

    /// Disable comments
    #[must_use]
    pub const fn without_comments(mut self) -> Self {
        self.include_comments = false;
        self
    }
}

/// Emit PTX with configuration
#[must_use]
pub fn emit_ptx(module: &PtxModule, _config: &EmitConfig) -> String {
    // For now, just use the default emit
    module.emit()
}

/// Validate emitted PTX for basic syntax errors
pub fn validate_ptx(ptx: &str) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Check for required directives
    if !ptx.contains(".version") {
        errors.push("Missing .version directive".to_string());
    }
    if !ptx.contains(".target") {
        errors.push("Missing .target directive".to_string());
    }
    if !ptx.contains(".address_size") {
        errors.push("Missing .address_size directive".to_string());
    }

    // Check for unbalanced braces
    let open_braces = ptx.matches('{').count();
    let close_braces = ptx.matches('}').count();
    if open_braces != close_braces {
        errors.push(format!(
            "Unbalanced braces: {} open, {} close",
            open_braces, close_braces
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_config_default() {
        let config = EmitConfig::new();
        assert!(config.include_comments);
        assert!(config.pretty_print);
        assert!(!config.debug_info);
    }

    #[test]
    fn test_emit_config_builder() {
        let config = EmitConfig::new().with_debug().without_comments();
        assert!(!config.include_comments);
        assert!(config.debug_info);
    }

    #[test]
    fn test_validate_ptx_valid() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry test() {
    ret;
}
"#;
        assert!(validate_ptx(ptx).is_ok());
    }

    #[test]
    fn test_validate_ptx_missing_version() {
        let ptx = r#"
.target sm_70
.address_size 64
"#;
        let result = validate_ptx(ptx);
        assert!(result.is_err());
        assert!(result.unwrap_err()[0].contains("version"));
    }

    #[test]
    fn test_validate_ptx_unbalanced_braces() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry test() {
    ret;

"#;
        let result = validate_ptx(ptx);
        assert!(result.is_err());
        assert!(result.unwrap_err()[0].contains("braces"));
    }
}
