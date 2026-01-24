//! Helper Functions for Tuner
//!
//! Utility functions: CRC32, timestamp, string formatting.

// ============================================================================
// CRC32 Implementation
// ============================================================================

/// Generate CRC32 lookup table at compile time.
const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Simple CRC32 implementation (IEEE polynomial).
/// Used for .apr file checksum verification.
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    const CRC32_TABLE: [u32; 256] = crc32_table();
    let mut crc = !crc;
    for &byte in data {
        crc = CRC32_TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Compute CRC32 hash for given data (convenience wrapper)
pub fn crc32_hash(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

// ============================================================================
// Timestamp
// ============================================================================

/// Simple timestamp (avoids chrono dependency)
pub fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

// ============================================================================
// String Formatting
// ============================================================================

/// Pad string to fixed width
pub fn pad_right(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - s.len()))
    }
}
