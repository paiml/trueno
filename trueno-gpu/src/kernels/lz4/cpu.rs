//! CPU Reference Implementation for LZ4 compression
//!
//! Used for validation and testing. These functions implement the LZ4 block
//! format specification for correctness verification.

use super::{LZ4_HASH_BITS, LZ4_HASH_MULT, LZ4_HASH_SIZE, LZ4_MAX_OFFSET, LZ4_MIN_MATCH};

/// Read 4 bytes as little-endian u32
#[inline]
#[must_use]
pub fn read_u32_le(data: &[u8], pos: usize) -> u32 {
    debug_assert!(pos + 4 <= data.len());
    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

/// LZ4 hash function: hash 4 bytes to 12-bit index
///
/// Uses Knuth multiplicative hash for good distribution.
/// Formula: hash = (val * 2654435761) >> (32 - 12)
#[inline]
#[must_use]
pub fn lz4_hash(val: u32) -> u32 {
    val.wrapping_mul(LZ4_HASH_MULT) >> (32 - LZ4_HASH_BITS)
}

/// Hash 4 bytes from a slice position
#[inline]
#[must_use]
pub fn lz4_hash_at(data: &[u8], pos: usize) -> u32 {
    lz4_hash(read_u32_le(data, pos))
}

/// Count matching bytes between two positions
///
/// Returns the number of matching bytes (minimum 0).
/// Used after finding a 4-byte hash match to extend the match.
#[inline]
#[must_use]
pub fn lz4_match_length(data: &[u8], pos1: usize, pos2: usize, limit: usize) -> usize {
    let mut len = 0;
    let max_len = limit.min(data.len() - pos1.max(pos2));

    while len < max_len && data[pos1 + len] == data[pos2 + len] {
        len += 1;
    }
    len
}

/// Encode LZ4 sequence to output buffer
///
/// Returns number of bytes written to output.
/// Format: [token] [extra_literal_len...] [literals] [offset_lo] [offset_hi] [extra_match_len...]
pub fn lz4_encode_sequence(
    output: &mut [u8],
    out_pos: &mut usize,
    literals: &[u8],
    match_offset: u16,
    match_length: usize,
) -> Result<(), &'static str> {
    let literal_len = literals.len();

    // Calculate token
    let token_lit = if literal_len >= 15 { 15 } else { literal_len as u8 };
    let token_match = if match_length == 0 {
        0
    } else if match_length - LZ4_MIN_MATCH as usize >= 15 {
        15
    } else {
        (match_length - LZ4_MIN_MATCH as usize) as u8
    };
    let token = (token_lit << 4) | token_match;

    // Check output space
    let needed = 1
        + (if literal_len >= 15 {
            1 + (literal_len - 15) / 255 + 1
        } else {
            0
        })
        + literal_len
        + if match_length > 0 { 2 } else { 0 }
        + if match_length > 0 && match_length - LZ4_MIN_MATCH as usize >= 15 {
            1 + (match_length - LZ4_MIN_MATCH as usize - 15) / 255 + 1
        } else {
            0
        };

    if *out_pos + needed > output.len() {
        return Err("Output buffer too small");
    }

    // Write token
    output[*out_pos] = token;
    *out_pos += 1;

    // Write extra literal length if >= 15
    if literal_len >= 15 {
        let mut remaining = literal_len - 15;
        while remaining >= 255 {
            output[*out_pos] = 255;
            *out_pos += 1;
            remaining -= 255;
        }
        output[*out_pos] = remaining as u8;
        *out_pos += 1;
    }

    // Write literals
    output[*out_pos..*out_pos + literal_len].copy_from_slice(literals);
    *out_pos += literal_len;

    // Write match offset and length (if match exists)
    if match_length > 0 {
        output[*out_pos] = (match_offset & 0xFF) as u8;
        output[*out_pos + 1] = (match_offset >> 8) as u8;
        *out_pos += 2;

        // Write extra match length if >= 15
        if match_length - LZ4_MIN_MATCH as usize >= 15 {
            let mut remaining = match_length - LZ4_MIN_MATCH as usize - 15;
            while remaining >= 255 {
                output[*out_pos] = 255;
                *out_pos += 1;
                remaining -= 255;
            }
            output[*out_pos] = remaining as u8;
            *out_pos += 1;
        }
    }

    Ok(())
}

/// LZ4 decompress a block (CPU reference implementation)
///
/// Returns decompressed size, or error if decompression fails.
/// Used for F001 lossless verification.
pub fn lz4_decompress_block(input: &[u8], output: &mut [u8]) -> Result<usize, &'static str> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos < input.len() {
        // Read token
        let token = input[in_pos];
        in_pos += 1;

        let mut literal_len = (token >> 4) as usize;
        let match_len_base = (token & 0x0F) as usize;

        // Read extended literal length if needed
        if literal_len == 15 {
            loop {
                if in_pos >= input.len() {
                    return Err("Truncated literal length");
                }
                let byte = input[in_pos] as usize;
                in_pos += 1;
                literal_len += byte;
                if byte != 255 {
                    break;
                }
            }
        }

        // Copy literals
        if literal_len > 0 {
            if in_pos + literal_len > input.len() {
                return Err("Truncated literals");
            }
            if out_pos + literal_len > output.len() {
                return Err("Output buffer overflow (literals)");
            }
            output[out_pos..out_pos + literal_len]
                .copy_from_slice(&input[in_pos..in_pos + literal_len]);
            in_pos += literal_len;
            out_pos += literal_len;
        }

        // Check for end of block (last sequence has no match)
        if in_pos >= input.len() {
            break;
        }

        // Read match offset (little-endian u16)
        if in_pos + 2 > input.len() {
            return Err("Truncated match offset");
        }
        let offset = (input[in_pos] as usize) | ((input[in_pos + 1] as usize) << 8);
        in_pos += 2;

        if offset == 0 {
            return Err("Invalid zero offset");
        }
        if offset > out_pos {
            return Err("Invalid offset (exceeds output)");
        }

        // Calculate match length
        let mut match_len = match_len_base + LZ4_MIN_MATCH as usize;

        // Read extended match length if needed
        if match_len_base == 15 {
            loop {
                if in_pos >= input.len() {
                    return Err("Truncated match length");
                }
                let byte = input[in_pos] as usize;
                in_pos += 1;
                match_len += byte;
                if byte != 255 {
                    break;
                }
            }
        }

        // Copy match (may overlap, so byte-by-byte)
        if out_pos + match_len > output.len() {
            return Err("Output buffer overflow (match)");
        }
        let match_start = out_pos - offset;
        for i in 0..match_len {
            output[out_pos + i] = output[match_start + i];
        }
        out_pos += match_len;
    }

    Ok(out_pos)
}

/// LZ4 compress a block (CPU reference implementation)
///
/// Returns compressed size, or error if compression fails.
pub fn lz4_compress_block(input: &[u8], output: &mut [u8]) -> Result<usize, &'static str> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut hash_table = [0u32; LZ4_HASH_SIZE as usize];
    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let mut anchor = 0usize; // Start of current literal run

    // Skip first 4 bytes (need at least 4 bytes for hash)
    if input.len() < LZ4_MIN_MATCH as usize {
        // Too small to compress, emit as literals
        lz4_encode_sequence(output, &mut out_pos, input, 0, 0)?;
        return Ok(out_pos);
    }

    // Main compression loop
    while in_pos + LZ4_MIN_MATCH as usize <= input.len() {
        let h = lz4_hash_at(input, in_pos);
        let match_pos = hash_table[h as usize] as usize;
        hash_table[h as usize] = in_pos as u32;

        // Check for match
        let offset = in_pos - match_pos;
        if offset > 0
            && offset <= LZ4_MAX_OFFSET as usize
            && match_pos + 4 <= input.len()
            && read_u32_le(input, in_pos) == read_u32_le(input, match_pos)
        {
            // Found a match! Extend it
            let match_len = lz4_match_length(
                input,
                in_pos + 4,
                match_pos + 4,
                input.len() - in_pos - 4,
            ) + 4; // Add the initial 4 bytes

            // Emit literals from anchor to in_pos, then the match
            let literals = &input[anchor..in_pos];
            lz4_encode_sequence(output, &mut out_pos, literals, offset as u16, match_len)?;

            in_pos += match_len;
            anchor = in_pos;
        } else {
            in_pos += 1;
        }

        // Safety: don't go past end minus required lookahead
        if in_pos + 5 > input.len() {
            break;
        }
    }

    // Emit remaining literals (last sequence has no match)
    if anchor < input.len() {
        let literals = &input[anchor..];
        lz4_encode_sequence(output, &mut out_pos, literals, 0, 0)?;
    }

    Ok(out_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hash Function Tests ---

    #[test]
    fn test_lz4_hash_produces_12bit_output() {
        // Hash output must always be < 4096 (12 bits)
        for val in [0u32, 1, 0x12345678, 0xFFFFFFFF, 0xDEADBEEF] {
            let h = lz4_hash(val);
            assert!(h < LZ4_HASH_SIZE, "Hash {} >= 4096 for input {}", h, val);
        }
    }

    #[test]
    fn test_lz4_hash_deterministic() {
        // Same input must produce same hash
        let val = 0x12345678u32;
        assert_eq!(lz4_hash(val), lz4_hash(val));
    }

    #[test]
    fn test_lz4_hash_distribution() {
        // Different inputs should produce different hashes (mostly)
        let h1 = lz4_hash(0x00000000);
        let h2 = lz4_hash(0x00000001);
        let h3 = lz4_hash(0x00010000);
        // Not all different, but collision rate should be low
        assert!(h1 != h2 || h2 != h3, "Too many collisions");
    }

    #[test]
    fn test_lz4_hash_at_from_slice() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9A];
        let expected_val = 0x78563412u32; // Little-endian
        assert_eq!(lz4_hash_at(&data, 0), lz4_hash(expected_val));
    }

    #[test]
    fn test_read_u32_le() {
        assert_eq!(read_u32_le(&[0x01, 0x02, 0x03, 0x04], 0), 0x04030201);
        assert_eq!(read_u32_le(&[0xFF, 0xFF, 0xFF, 0xFF], 0), 0xFFFFFFFF);
        assert_eq!(
            read_u32_le(&[0x00, 0x00, 0x01, 0x02, 0x03, 0x04], 2),
            0x04030201
        );
    }

    // --- Match Length Tests ---

    #[test]
    fn test_lz4_match_length_identical() {
        let data = b"AAAAAAAA";
        let len = lz4_match_length(data, 0, 4, 4);
        assert_eq!(len, 4, "Should match 4 bytes");
    }

    #[test]
    fn test_lz4_match_length_partial() {
        let data = b"AAABAAAC";
        let len = lz4_match_length(data, 0, 4, 8);
        assert_eq!(len, 3, "Should match 3 bytes (AAA vs AAA)");
    }

    #[test]
    fn test_lz4_match_length_no_match() {
        let data = b"ABCDWXYZ";
        let len = lz4_match_length(data, 0, 4, 4);
        assert_eq!(len, 0, "Should match 0 bytes");
    }

    #[test]
    fn test_lz4_match_length_limit_respected() {
        let data = b"AAAAAAAAAAAA";
        let len = lz4_match_length(data, 0, 4, 3);
        assert_eq!(len, 3, "Should be limited to 3 bytes");
    }

    // --- Encode Sequence Tests ---

    #[test]
    fn test_lz4_encode_literals_only() {
        let mut output = [0u8; 32];
        let mut pos = 0;
        let literals = b"HELLO";

        lz4_encode_sequence(&mut output, &mut pos, literals, 0, 0).unwrap();

        // Token: 5 literals, 0 match = 0x50
        assert_eq!(output[0], 0x50);
        assert_eq!(&output[1..6], b"HELLO");
        assert_eq!(pos, 6);
    }

    #[test]
    fn test_lz4_encode_match_only() {
        let mut output = [0u8; 32];
        let mut pos = 0;

        // Match of 4 bytes at offset 10
        lz4_encode_sequence(&mut output, &mut pos, &[], 10, 4).unwrap();

        // Token: 0 literals, 0 match (4 - 4 = 0)
        assert_eq!(output[0], 0x00);
        // Offset: 10 little-endian
        assert_eq!(output[1], 10);
        assert_eq!(output[2], 0);
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_lz4_encode_literals_and_match() {
        let mut output = [0u8; 32];
        let mut pos = 0;

        // 3 literals, match of 5 bytes at offset 20
        lz4_encode_sequence(&mut output, &mut pos, b"ABC", 20, 5).unwrap();

        // Token: 3 literals, 1 match (5 - 4 = 1)
        assert_eq!(output[0], 0x31);
        assert_eq!(&output[1..4], b"ABC");
        assert_eq!(output[4], 20); // offset low
        assert_eq!(output[5], 0); // offset high
        assert_eq!(pos, 6);
    }

    #[test]
    fn test_lz4_encode_extended_literal_length() {
        let mut output = [0u8; 64];
        let mut pos = 0;

        // 20 literals (> 15, needs extension)
        let literals = b"12345678901234567890";
        lz4_encode_sequence(&mut output, &mut pos, literals, 0, 0).unwrap();

        // Token: 15 literals (max), 0 match
        assert_eq!(output[0], 0xF0);
        // Extended length: 20 - 15 = 5
        assert_eq!(output[1], 5);
        // Literals start at output[2]
        assert_eq!(&output[2..22], literals.as_slice());
        assert_eq!(pos, 22);
    }

    // --- Compress Block Tests (F001 equivalent) ---

    #[test]
    fn test_lz4_compress_empty() {
        let mut output = [0u8; 32];
        let size = lz4_compress_block(&[], &mut output).unwrap();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_lz4_compress_small() {
        let input = b"HELLO";
        let mut output = [0u8; 32];
        let size = lz4_compress_block(input, &mut output).unwrap();

        // Small input should be stored as literals
        assert!(size > 0);
        assert_eq!(output[0] >> 4, 5); // 5 literals in token
    }

    #[test]
    fn test_lz4_compress_repeated_pattern() {
        // Pattern that should compress well
        let mut input = [0u8; 64];
        for i in 0..64 {
            input[i] = (i % 4) as u8; // Repeating 0,1,2,3,0,1,2,3...
        }
        let mut output = [0u8; 128];
        let size = lz4_compress_block(&input, &mut output).unwrap();

        // Should compress (matches found)
        assert!(size < 64, "Should compress, got {} bytes", size);
    }

    #[test]
    fn test_lz4_compress_zeros() {
        // Zero page should compress extremely well
        let input = [0u8; 256];
        let mut output = [0u8; 512];
        let size = lz4_compress_block(&input, &mut output).unwrap();

        // Should achieve good compression
        assert!(size < 128, "Zeros should compress well, got {} bytes", size);
    }

    #[test]
    fn test_lz4_compress_all_same_byte() {
        // F007: Repeated patterns compress well
        let input = [b'A'; 512];
        let mut output = [0u8; 1024];
        let size = lz4_compress_block(&input, &mut output).unwrap();

        // Should achieve >10:1 ratio
        assert!(
            size < 52,
            "Repeated pattern should achieve >10:1 ratio, got {} bytes",
            size
        );
    }

    // =========================================================================
    // F001: LZ4 Compression is Lossless (Roundtrip Tests)
    // =========================================================================

    #[test]
    fn test_f001_roundtrip_hello() {
        let input = b"HELLO WORLD";
        let mut compressed = [0u8; 64];
        let mut decompressed = [0u8; 64];

        let comp_size = lz4_compress_block(input, &mut compressed).unwrap();
        let decomp_size =
            lz4_decompress_block(&compressed[..comp_size], &mut decompressed).unwrap();

        assert_eq!(decomp_size, input.len());
        assert_eq!(&decompressed[..decomp_size], input.as_slice());
    }

    #[test]
    fn test_f001_roundtrip_zeros() {
        let input = [0u8; 256];
        let mut compressed = [0u8; 512];
        let mut decompressed = [0u8; 256];

        let comp_size = lz4_compress_block(&input, &mut compressed).unwrap();
        let decomp_size =
            lz4_decompress_block(&compressed[..comp_size], &mut decompressed).unwrap();

        assert_eq!(decomp_size, input.len());
        assert_eq!(&decompressed[..], &input[..]);
    }

    #[test]
    fn test_f001_roundtrip_repeated_pattern() {
        let mut input = [0u8; 512];
        for i in 0..512 {
            input[i] = (i % 13) as u8; // Non-power-of-2 pattern
        }
        let mut compressed = [0u8; 1024];
        let mut decompressed = [0u8; 512];

        let comp_size = lz4_compress_block(&input, &mut compressed).unwrap();
        let decomp_size =
            lz4_decompress_block(&compressed[..comp_size], &mut decompressed).unwrap();

        assert_eq!(decomp_size, input.len());
        assert_eq!(&decompressed[..], &input[..]);
    }

    #[test]
    fn test_f001_roundtrip_text() {
        let input =
            b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps again!";
        let mut compressed = [0u8; 256];
        let mut decompressed = [0u8; 256];

        let comp_size = lz4_compress_block(input, &mut compressed).unwrap();
        let decomp_size =
            lz4_decompress_block(&compressed[..comp_size], &mut decompressed).unwrap();

        assert_eq!(decomp_size, input.len());
        assert_eq!(&decompressed[..decomp_size], input.as_slice());
    }

    #[test]
    fn test_f001_roundtrip_page_size() {
        use super::super::PAGE_SIZE;
        // Test with actual 4KB page
        let mut input = [0u8; PAGE_SIZE as usize];
        for i in 0..PAGE_SIZE as usize {
            input[i] = ((i * 7) % 256) as u8;
        }
        let mut compressed = [0u8; PAGE_SIZE as usize + 1024];
        let mut decompressed = [0u8; PAGE_SIZE as usize];

        let comp_size = lz4_compress_block(&input, &mut compressed).unwrap();
        let decomp_size =
            lz4_decompress_block(&compressed[..comp_size], &mut decompressed).unwrap();

        assert_eq!(decomp_size, PAGE_SIZE as usize);
        assert_eq!(&decompressed[..], &input[..]);
    }

    #[test]
    fn test_f006_zero_page_compression_ratio() {
        use super::super::PAGE_SIZE;
        // F006: Zero page compresses to <100 bytes
        let input = [0u8; PAGE_SIZE as usize];
        let mut compressed = [0u8; PAGE_SIZE as usize];

        let comp_size = lz4_compress_block(&input, &mut compressed).unwrap();

        assert!(
            comp_size < 100,
            "Zero page should compress to <100 bytes, got {}",
            comp_size
        );
    }

    #[test]
    fn test_f007_repeated_pattern_ratio() {
        use super::super::PAGE_SIZE;
        // F007: 4KB of "AAAA..." achieves >100:1 ratio
        let input = [b'A'; PAGE_SIZE as usize];
        let mut compressed = [0u8; PAGE_SIZE as usize];

        let comp_size = lz4_compress_block(&input, &mut compressed).unwrap();
        let ratio = PAGE_SIZE as usize / comp_size;

        assert!(
            ratio >= 100,
            "Should achieve >100:1 ratio, got {}:1 ({} bytes)",
            ratio,
            comp_size
        );
    }

    #[test]
    fn test_f003_empty_page() {
        // F003: Empty pages compress correctly
        let mut compressed = [0u8; 32];
        let mut decompressed = [0u8; 32];

        let comp_size = lz4_compress_block(&[], &mut compressed).unwrap();
        let decomp_size =
            lz4_decompress_block(&compressed[..comp_size], &mut decompressed).unwrap();

        assert_eq!(comp_size, 0);
        assert_eq!(decomp_size, 0);
    }

    #[test]
    fn test_f018_deterministic_output() {
        // F018: Same input always produces same output
        let input = b"Deterministic compression test data";
        let mut compressed1 = [0u8; 128];
        let mut compressed2 = [0u8; 128];

        let size1 = lz4_compress_block(input, &mut compressed1).unwrap();
        let size2 = lz4_compress_block(input, &mut compressed2).unwrap();

        assert_eq!(size1, size2);
        assert_eq!(&compressed1[..size1], &compressed2[..size2]);
    }
}
