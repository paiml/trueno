//! GPU Memory Sanitizer Wrapper
//!
//! Wraps NVIDIA compute-sanitizer to provide enhanced error reporting with
//! semantic context from trueno-gpu's knowledge of buffer allocations.
//!
//! # Features
//!
//! - **Address Registry**: Maps device addresses to buffer names/types
//! - **Semantic Errors**: Translates raw addresses to meaningful buffer offsets
//! - **PTX Source Mapping**: Maps SASS offsets to PTX source lines
//!
//! # Usage
//!
//! ```ignore
//! use trueno_gpu::driver::sanitizer::{AddressRegistry, SanitizedLaunch};
//!
//! // Register buffers before kernel launch
//! AddressRegistry::global().register("input_data", buf.as_ptr(), buf.size_bytes());
//!
//! // Run with sanitizer
//! let result = SanitizedLaunch::new(&stream, &module, "my_kernel")
//!     .with_ptx_source(ptx_source)
//!     .launch(&config, &args)?;
//! ```

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

use super::sys::CUdeviceptr;

// ============================================================================
// Address Registry - Tracks all GPU allocations
// ============================================================================

/// Information about a registered GPU buffer
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Human-readable name
    pub name: String,
    /// Device pointer (start address)
    pub ptr: CUdeviceptr,
    /// Size in bytes
    pub size: usize,
    /// Element type name (e.g., "f32", "u8")
    pub type_name: String,
    /// Element size in bytes
    pub element_size: usize,
}

impl BufferInfo {
    /// Check if an address falls within this buffer
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.ptr && addr < self.ptr + self.size as u64
    }

    /// Get the byte offset of an address within this buffer
    pub fn offset_of(&self, addr: u64) -> Option<usize> {
        if self.contains(addr) {
            Some((addr - self.ptr) as usize)
        } else {
            None
        }
    }

    /// Get the element index of an address within this buffer
    pub fn element_index_of(&self, addr: u64) -> Option<usize> {
        self.offset_of(addr).map(|off| off / self.element_size)
    }
}

/// Global registry of GPU buffer allocations
///
/// Thread-safe singleton that tracks all active GPU allocations.
/// Used by the sanitizer to translate raw addresses into meaningful names.
pub struct AddressRegistry {
    buffers: HashMap<CUdeviceptr, BufferInfo>,
}

impl AddressRegistry {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Get the global address registry
    pub fn global() -> &'static Mutex<AddressRegistry> {
        static REGISTRY: OnceLock<Mutex<AddressRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(AddressRegistry::new()))
    }

    /// Register a buffer with the registry
    pub fn register(
        &mut self,
        name: impl Into<String>,
        ptr: CUdeviceptr,
        size: usize,
        type_name: impl Into<String>,
        element_size: usize,
    ) {
        let info = BufferInfo {
            name: name.into(),
            ptr,
            size,
            type_name: type_name.into(),
            element_size,
        };
        self.buffers.insert(ptr, info);
    }

    /// Unregister a buffer when freed
    pub fn unregister(&mut self, ptr: CUdeviceptr) {
        self.buffers.remove(&ptr);
    }

    /// Look up buffer information by an address (may be within buffer)
    pub fn lookup(&self, addr: u64) -> Option<&BufferInfo> {
        // Check if address matches a buffer start
        if let Some(info) = self.buffers.get(&addr) {
            return Some(info);
        }

        // Check if address falls within any buffer
        for info in self.buffers.values() {
            if info.contains(addr) {
                return Some(info);
            }
        }

        None
    }

    /// Format an address with semantic context
    pub fn format_address(&self, addr: u64) -> String {
        if let Some(info) = self.lookup(addr) {
            if let Some(offset) = info.offset_of(addr) {
                let elem_idx = offset / info.element_size;
                let byte_in_elem = offset % info.element_size;
                if byte_in_elem == 0 {
                    format!(
                        "{}[{}] (0x{:X} + {} bytes)",
                        info.name, elem_idx, info.ptr, offset
                    )
                } else {
                    format!(
                        "{}[{}]+{} (0x{:X} + {} bytes)",
                        info.name, elem_idx, byte_in_elem, info.ptr, offset
                    )
                }
            } else {
                format!("{} @ 0x{:X}", info.name, addr)
            }
        } else {
            format!("0x{:X} (unknown buffer)", addr)
        }
    }
}

// ============================================================================
// Sanitizer Error Types
// ============================================================================

/// Type of memory violation detected by sanitizer
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryViolationType {
    /// Invalid read from global memory
    InvalidGlobalRead {
        /// Size of the invalid read in bytes
        size: usize,
    },
    /// Invalid write to global memory
    InvalidGlobalWrite {
        /// Size of the invalid write in bytes
        size: usize,
    },
    /// Invalid read from shared memory
    InvalidSharedRead {
        /// Size of the invalid read in bytes
        size: usize,
    },
    /// Invalid write to shared memory
    InvalidSharedWrite {
        /// Size of the invalid write in bytes
        size: usize,
    },
    /// Misaligned access
    MisalignedAccess {
        /// Address that was misaligned
        addr: u64,
    },
    /// Race condition detected
    RaceCondition,
    /// Other/unknown error
    Other(String),
}

/// A memory violation detected by the sanitizer
#[derive(Debug, Clone)]
pub struct MemoryViolation {
    /// Type of violation
    pub violation_type: MemoryViolationType,
    /// Kernel name
    pub kernel_name: String,
    /// SASS offset within kernel
    pub sass_offset: u64,
    /// Thread coordinates (x, y, z)
    pub thread: (u32, u32, u32),
    /// Block coordinates (x, y, z)
    pub block: (u32, u32, u32),
    /// Address that caused the violation
    pub address: u64,
    /// Raw error message from sanitizer
    pub raw_message: String,
}

impl MemoryViolation {
    /// Format with semantic address information
    pub fn format_with_registry(&self, registry: &AddressRegistry) -> String {
        let addr_info = registry.format_address(self.address);
        let violation_desc = match &self.violation_type {
            MemoryViolationType::InvalidGlobalRead { size } => {
                format!("Invalid global read of {} bytes", size)
            }
            MemoryViolationType::InvalidGlobalWrite { size } => {
                format!("Invalid global write of {} bytes", size)
            }
            MemoryViolationType::InvalidSharedRead { size } => {
                format!("Invalid shared read of {} bytes", size)
            }
            MemoryViolationType::InvalidSharedWrite { size } => {
                format!("Invalid shared write of {} bytes", size)
            }
            MemoryViolationType::MisalignedAccess { addr } => {
                format!("Misaligned access at 0x{:X}", addr)
            }
            MemoryViolationType::RaceCondition => "Race condition detected".to_string(),
            MemoryViolationType::Other(msg) => msg.clone(),
        };

        format!(
            "🛑 MEMORY VIOLATION\n\
             ├─ Kernel: {} @ SASS offset 0x{:X}\n\
             ├─ Thread: ({}, {}, {}) in Block ({}, {}, {})\n\
             ├─ Error: {}\n\
             └─ Address: {}",
            self.kernel_name,
            self.sass_offset,
            self.thread.0,
            self.thread.1,
            self.thread.2,
            self.block.0,
            self.block.1,
            self.block.2,
            violation_desc,
            addr_info
        )
    }
}

// ============================================================================
// Sanitizer Output Parser
// ============================================================================

/// Parser for compute-sanitizer output
pub struct SanitizerParser;

impl SanitizerParser {
    /// Parse compute-sanitizer output and extract violations
    pub fn parse(output: &str) -> Vec<MemoryViolation> {
        let mut violations = Vec::new();
        let lines: Vec<&str> = output.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];

            // Look for "Invalid __global__ read" or similar patterns
            if line.contains("Invalid __") {
                if let Some(violation) = Self::parse_violation(&lines[i..]) {
                    violations.push(violation);
                }
            }

            i += 1;
        }

        violations
    }

    fn parse_violation(lines: &[&str]) -> Option<MemoryViolation> {
        let first_line = lines.first()?;

        // Parse violation type from first line
        // Example: "========= Invalid __shared__ read of size 4 bytes"
        let violation_type = if first_line.contains("__shared__ read") {
            let size = Self::extract_size(first_line).unwrap_or(4);
            MemoryViolationType::InvalidSharedRead { size }
        } else if first_line.contains("__shared__ write") {
            let size = Self::extract_size(first_line).unwrap_or(4);
            MemoryViolationType::InvalidSharedWrite { size }
        } else if first_line.contains("__global__ read") {
            let size = Self::extract_size(first_line).unwrap_or(4);
            MemoryViolationType::InvalidGlobalRead { size }
        } else if first_line.contains("__global__ write") {
            let size = Self::extract_size(first_line).unwrap_or(4);
            MemoryViolationType::InvalidGlobalWrite { size }
        } else if first_line.contains("misaligned") {
            MemoryViolationType::MisalignedAccess { addr: 0 }
        } else {
            MemoryViolationType::Other(first_line.to_string())
        };

        // Parse kernel name and offset from second line
        // Example: "=========     at lz4_compress_warp+0x2160"
        let mut kernel_name = String::from("unknown");
        let mut sass_offset = 0u64;
        let mut thread = (0u32, 0u32, 0u32);
        let mut block = (0u32, 0u32, 0u32);
        let mut address = 0u64;

        for line in lines.iter().skip(1).take(10) {
            if line.contains(" at ") && line.contains("+0x") {
                // Parse "at kernel_name+0xOFFSET"
                if let Some(at_pos) = line.find(" at ") {
                    let rest = &line[at_pos + 4..];
                    if let Some(plus_pos) = rest.find("+0x") {
                        kernel_name = rest[..plus_pos].trim().to_string();
                        let hex_str = &rest[plus_pos + 3..];
                        let hex_end = hex_str
                            .find(|c: char| !c.is_ascii_hexdigit())
                            .unwrap_or(hex_str.len());
                        sass_offset = u64::from_str_radix(&hex_str[..hex_end], 16).unwrap_or(0);
                    }
                }
            }

            if line.contains("by thread") {
                // Parse "by thread (0,0,0) in block (0,0,0)"
                if let Some(parsed) = Self::parse_thread_block(line) {
                    thread = parsed.0;
                    block = parsed.1;
                }
            }

            if line.contains("Address 0x") {
                // Parse "Address 0xXXXX is misaligned"
                if let Some(addr_pos) = line.find("Address 0x") {
                    let hex_start = addr_pos + 10;
                    let rest = &line[hex_start..];
                    let hex_end = rest
                        .find(|c: char| !c.is_ascii_hexdigit())
                        .unwrap_or(rest.len());
                    address = u64::from_str_radix(&rest[..hex_end], 16).unwrap_or(0);
                }
            }
        }

        Some(MemoryViolation {
            violation_type,
            kernel_name,
            sass_offset,
            thread,
            block,
            address,
            raw_message: lines.iter().take(5).map(|s| *s).collect::<Vec<_>>().join("\n"),
        })
    }

    fn extract_size(line: &str) -> Option<usize> {
        // Parse "of size N bytes"
        if let Some(pos) = line.find("of size ") {
            let rest = &line[pos + 8..];
            let num_end = rest.find(' ').unwrap_or(rest.len());
            rest[..num_end].parse().ok()
        } else {
            None
        }
    }

    fn parse_thread_block(line: &str) -> Option<((u32, u32, u32), (u32, u32, u32))> {
        // Parse "by thread (X,Y,Z) in block (X,Y,Z)"
        let parse_triple = |s: &str| -> Option<(u32, u32, u32)> {
            let s = s.trim_start_matches('(').trim_end_matches(')');
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 3 {
                Some((
                    parts[0].parse().ok()?,
                    parts[1].parse().ok()?,
                    parts[2].parse().ok()?,
                ))
            } else {
                None
            }
        };

        let thread_pos = line.find("thread (")?;
        let thread_end = line[thread_pos..].find(')')?;
        let thread_str = &line[thread_pos + 7..thread_pos + thread_end + 1];

        let block_pos = line.find("block (")?;
        let block_end = line[block_pos..].find(')')?;
        let block_str = &line[block_pos + 6..block_pos + block_end + 1];

        Some((parse_triple(thread_str)?, parse_triple(block_str)?))
    }
}

// ============================================================================
// PTX Source Location Mapping
// ============================================================================

/// Maps PTX line numbers to source locations
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// Source file path
    pub file: String,
    /// Line number (1-based)
    pub line: u32,
    /// Column (if available)
    pub column: Option<u32>,
    /// Function/label name
    pub function: Option<String>,
}

/// PTX source map for kernel debugging
pub struct PtxSourceMap {
    /// PTX source code
    ptx_source: String,
    /// Map from PTX line number to source location
    line_map: HashMap<u32, SourceLocation>,
    /// Map from label name to PTX line number
    label_lines: HashMap<String, u32>,
}

impl PtxSourceMap {
    /// Create a new source map from PTX source
    pub fn new(ptx_source: &str) -> Self {
        let mut map = Self {
            ptx_source: ptx_source.to_string(),
            line_map: HashMap::new(),
            label_lines: HashMap::new(),
        };
        map.parse_ptx();
        map
    }

    fn parse_ptx(&mut self) {
        for (line_num, line) in self.ptx_source.lines().enumerate() {
            let line_num = (line_num + 1) as u32;

            // Look for labels
            let trimmed = line.trim();
            if trimmed.ends_with(':') && !trimmed.starts_with("//") {
                let label = trimmed.trim_end_matches(':');
                self.label_lines.insert(label.to_string(), line_num);
            }

            // Look for .loc directives (if present)
            // Format: .loc file_id line_num [column]
            if trimmed.starts_with(".loc ") {
                // Parse .loc directive
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    if let Ok(src_line) = parts[2].parse::<u32>() {
                        // For now, just store the line number
                        // A full implementation would track file_id mapping
                        self.line_map.insert(
                            line_num,
                            SourceLocation {
                                file: "kernel.rs".to_string(),
                                line: src_line,
                                column: parts.get(3).and_then(|s| s.parse().ok()),
                                function: None,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Get the label containing a given PTX line
    pub fn label_at_line(&self, target_line: u32) -> Option<&str> {
        let mut best_label = None;
        let mut best_line = 0;

        for (label, &line) in &self.label_lines {
            if line <= target_line && line > best_line {
                best_line = line;
                best_label = Some(label.as_str());
            }
        }

        best_label
    }

    /// Get PTX lines around a label
    pub fn context_around_label(&self, label: &str, context_lines: usize) -> Option<String> {
        let &label_line = self.label_lines.get(label)?;
        let lines: Vec<&str> = self.ptx_source.lines().collect();

        let start = (label_line as usize).saturating_sub(context_lines);
        let end = ((label_line as usize) + context_lines).min(lines.len());

        let mut result = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            let actual_line = start + i + 1;
            let marker = if actual_line == label_line as usize {
                ">>>"
            } else {
                "   "
            };
            result.push_str(&format!("{} {:4}: {}\n", marker, actual_line, line));
        }

        Some(result)
    }
}

// ============================================================================
// Enhanced Sanitizer Report
// ============================================================================

/// A complete sanitizer report with enhanced context
#[derive(Debug)]
pub struct SanitizerReport {
    /// All violations detected
    pub violations: Vec<MemoryViolation>,
    /// Whether the kernel completed successfully
    pub success: bool,
    /// Raw sanitizer output
    pub raw_output: String,
}

impl SanitizerReport {
    /// Format the report with full context
    pub fn format(&self) -> String {
        if self.success {
            return "✅ No memory violations detected".to_string();
        }

        let registry = AddressRegistry::global().lock().unwrap();
        let mut output = String::new();

        output.push_str(&format!(
            "🚨 SANITIZER REPORT: {} violation(s) detected\n\n",
            self.violations.len()
        ));

        for (i, violation) in self.violations.iter().enumerate() {
            output.push_str(&format!("━━━ Violation {} ━━━\n", i + 1));
            output.push_str(&violation.format_with_registry(&registry));
            output.push_str("\n\n");
        }

        output
    }

    /// Format with PTX source context
    pub fn format_with_ptx(&self, ptx_map: &PtxSourceMap) -> String {
        if self.success {
            return "✅ No memory violations detected".to_string();
        }

        let registry = AddressRegistry::global().lock().unwrap();
        let mut output = String::new();

        output.push_str(&format!(
            "🚨 SANITIZER REPORT: {} violation(s) detected\n\n",
            self.violations.len()
        ));

        for (i, violation) in self.violations.iter().enumerate() {
            output.push_str(&format!("━━━ Violation {} ━━━\n", i + 1));
            output.push_str(&violation.format_with_registry(&registry));
            output.push('\n');

            // Try to find relevant PTX context
            // Note: SASS offset doesn't directly map to PTX line, but we can show
            // the relevant labels based on the kernel name
            if let Some(label) = ptx_map.label_at_line(1) {
                if let Some(context) = ptx_map.context_around_label(label, 5) {
                    output.push_str("\n📜 PTX Context:\n");
                    output.push_str(&context);
                }
            }

            output.push_str("\n\n");
        }

        output
    }
}

// ============================================================================
// Sanitizer Runner
// ============================================================================

/// Check if compute-sanitizer is available
pub fn sanitizer_available() -> bool {
    Command::new("compute-sanitizer")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run a command under compute-sanitizer and parse the output
pub fn run_with_sanitizer(args: &[&str]) -> Result<SanitizerReport, std::io::Error> {
    let mut cmd = Command::new("compute-sanitizer");
    cmd.arg("--tool").arg("memcheck");
    cmd.args(args);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = cmd.output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", stdout, stderr);

    let violations = SanitizerParser::parse(&combined);
    let success = violations.is_empty() && output.status.success();

    Ok(SanitizerReport {
        violations,
        success,
        raw_output: combined,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_registry() {
        let mut registry = AddressRegistry::new();

        // Register a buffer
        registry.register("input_buf", 0x7f00000000, 4096, "f32", 4);

        // Test lookup at start
        let info = registry.lookup(0x7f00000000).unwrap();
        assert_eq!(info.name, "input_buf");

        // Test lookup in middle
        let info = registry.lookup(0x7f00000100).unwrap();
        assert_eq!(info.name, "input_buf");
        assert_eq!(info.element_index_of(0x7f00000100), Some(64)); // 256 bytes / 4 = 64

        // Test lookup outside
        assert!(registry.lookup(0x8000000000).is_none());
    }

    #[test]
    fn test_format_address() {
        let mut registry = AddressRegistry::new();
        registry.register("weights", 0x7f00000000, 1024 * 4, "f32", 4);

        // Element-aligned access
        let formatted = registry.format_address(0x7f00000010);
        assert!(formatted.contains("weights[4]"));

        // Non-element-aligned access
        let formatted = registry.format_address(0x7f00000011);
        assert!(formatted.contains("weights[4]+1"));

        // Unknown address
        let formatted = registry.format_address(0x1);
        assert!(formatted.contains("unknown"));
    }

    #[test]
    fn test_parse_sanitizer_output() {
        let output = r#"
========= COMPUTE-SANITIZER
========= Invalid __shared__ read of size 4 bytes
=========     at lz4_compress_warp+0x2160
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x1 is misaligned
========= ERROR SUMMARY: 1 error
"#;

        let violations = SanitizerParser::parse(output);
        assert_eq!(violations.len(), 1);

        let v = &violations[0];
        assert_eq!(v.kernel_name, "lz4_compress_warp");
        assert_eq!(v.sass_offset, 0x2160);
        assert_eq!(v.thread, (0, 0, 0));
        assert_eq!(v.block, (0, 0, 0));
        assert_eq!(v.address, 0x1);

        match &v.violation_type {
            MemoryViolationType::InvalidSharedRead { size } => assert_eq!(*size, 4),
            _ => panic!("Expected InvalidSharedRead"),
        }
    }

    #[test]
    fn test_ptx_source_map() {
        let ptx = r#"
.version 8.0
.target sm_89
.entry test_kernel() {
    mov.u32 %r0, 0;
L_loop:
    add.u32 %r0, %r0, 1;
    bra L_loop;
L_end:
    ret;
}
"#;

        let map = PtxSourceMap::new(ptx);

        // Check label parsing
        assert!(map.label_lines.contains_key("L_loop"));
        assert!(map.label_lines.contains_key("L_end"));

        // Check context retrieval
        let context = map.context_around_label("L_loop", 2);
        assert!(context.is_some());
        let ctx = context.unwrap();
        assert!(ctx.contains("L_loop:"));
    }
}
