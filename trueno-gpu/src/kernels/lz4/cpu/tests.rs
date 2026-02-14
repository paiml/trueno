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

// =========================================================================
// Additional Coverage Tests: Error Paths and Edge Cases
// =========================================================================

#[test]
fn test_lz4_encode_output_buffer_too_small() {
    // Cover line 89: "Output buffer too small" error
    let mut output = [0u8; 2]; // Too small for any meaningful output
    let mut pos = 0;
    let literals = b"HELLO WORLD"; // 11 bytes + 1 token = needs at least 12

    let result = lz4_encode_sequence(&mut output, &mut pos, literals, 0, 0);
    assert_eq!(result, Err("Output buffer too small"));
}

#[test]
fn test_lz4_encode_extended_literal_length_over_255() {
    // Cover lines 99-103: literal length >= 15 + 255 = 270 bytes
    // This exercises the while loop that writes 255 bytes at a time
    let mut output = [0u8; 1024];
    let mut pos = 0;
    let literals = vec![b'X'; 300]; // 300 bytes (15 + 255 + 30)

    lz4_encode_sequence(&mut output, &mut pos, &literals, 0, 0).unwrap();

    // Token should have 0xF0 (15 literals, 0 match)
    assert_eq!(output[0] & 0xF0, 0xF0);
    // Extended length: first byte should be 255, second byte should be 30
    assert_eq!(output[1], 255);
    assert_eq!(output[2], 30); // 300 - 15 - 255 = 30
}

#[test]
fn test_lz4_encode_very_long_literal_multiple_255s() {
    // Cover multiple iterations of while remaining >= 255
    let mut output = [0u8; 2048];
    let mut pos = 0;
    let literals = vec![b'Y'; 600]; // 600 bytes (15 + 255 + 255 + 75)

    lz4_encode_sequence(&mut output, &mut pos, &literals, 0, 0).unwrap();

    // Extended length bytes
    assert_eq!(output[1], 255);
    assert_eq!(output[2], 255);
    assert_eq!(output[3], 75); // 600 - 15 - 255 - 255 = 75
}

#[test]
fn test_lz4_decompress_truncated_literal_length() {
    // Cover line 158: "Truncated literal length" error
    // Token 0xF0 = 15 literals, needs extended length byte but none provided
    let input = [0xF0u8]; // Token with literal_len=15, but no extension byte
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated extended length"));
}

#[test]
fn test_lz4_decompress_extended_literal_with_255() {
    // Cover line 165: byte == 255 branch in literal length loop
    // Token 0xF0 = 15 literals, extension byte 255 means continue reading
    let mut input = Vec::new();
    input.push(0xF0); // Token: 15 literals, 0 match
    input.push(255); // Extended: +255, continue reading
    input.push(10); // Extended: +10, stop (total = 15 + 255 + 10 = 280)
                    // Now we need 280 literal bytes
    input.extend(std::iter::repeat(b'A').take(280));

    let mut output = [0u8; 512];
    let result = lz4_decompress_block(&input, &mut output).unwrap();
    assert_eq!(result, 280);
    assert!(output[..280].iter().all(|&b| b == b'A'));
}

#[test]
fn test_lz4_decompress_truncated_literals() {
    // Cover line 172: "Truncated literals" error
    // Token says 5 literals, but only 3 bytes follow
    let input = [0x50u8, b'A', b'B', b'C']; // Token for 5 literals, only 3 provided
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated literals"));
}

#[test]
fn test_lz4_decompress_output_overflow_literals() {
    // Cover line 175: "Output buffer overflow (literals)" error
    let mut input = Vec::new();
    input.push(0x50); // Token: 5 literals
    input.extend(b"HELLO");

    let mut output = [0u8; 3]; // Too small for 5 literals

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Output buffer overflow (literals)"));
}

#[test]
fn test_lz4_decompress_truncated_match_offset() {
    // Cover line 190: "Truncated match offset" error
    // Token has match (non-zero lower nibble), but only 1 byte of offset provided
    let input = [0x11u8, b'A', 0x01]; // Token: 1 literal, 1 match; 1 literal byte, only 1 offset byte
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated match offset"));
}

#[test]
fn test_lz4_decompress_zero_offset() {
    // Cover line 196: "Invalid zero offset" error
    let input = [0x11u8, b'A', 0x00, 0x00]; // Token: 1 literal, 1 match; offset=0
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Invalid zero offset"));
}

#[test]
fn test_lz4_decompress_offset_exceeds_output() {
    // Cover line 199: "Invalid offset (exceeds output)" error
    // Offset points before beginning of output
    let input = [0x11u8, b'A', 0x10, 0x00]; // Token: 1 literal, 1 match; offset=16
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    // At this point, out_pos is 1 (1 literal written), but offset is 16
    assert_eq!(result, Err("Invalid offset (exceeds output)"));
}

#[test]
fn test_lz4_decompress_truncated_match_length() {
    // Cover line 209: "Truncated match length" error
    // Token has match_len_base=15, needs extension but none provided
    let mut input = Vec::new();
    input.push(0x1F); // Token: 1 literal, 15 match (needs extension)
    input.push(b'A'); // 1 literal
    input.push(0x01); // offset low
    input.push(0x00); // offset high (offset=1, valid since out_pos will be 1)
                      // No match length extension byte

    let mut output = [0u8; 64];
    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated extended length"));
}

#[test]
fn test_lz4_decompress_output_overflow_match() {
    // Cover line 222: "Output buffer overflow (match)" error
    // Valid match that would overflow output buffer
    let mut input = Vec::new();
    input.push(0x10); // Token: 1 literal, 0 match len (actual = 4)
    input.push(b'A'); // 1 literal
    input.push(0x01); // offset low
    input.push(0x00); // offset high (offset=1)
                      // match_len = 0 + 4 = 4 bytes to copy

    let mut output = [0u8; 3]; // Too small: 1 literal + 4 match = 5 needed

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Output buffer overflow (match)"));
}

#[test]
fn test_lz4_compress_input_smaller_than_minmatch() {
    // Cover lines 249-251: input.len() < LZ4_MIN_MATCH
    let input = [b'A', b'B', b'C']; // 3 bytes, less than MIN_MATCH (4)
    let mut output = [0u8; 32];

    let size = lz4_compress_block(&input, &mut output).unwrap();
    assert!(size > 0);
    // Should emit all 3 bytes as literals
    assert_eq!(output[0] >> 4, 3); // Token: 3 literals
}

#[test]
fn test_lz4_compress_single_byte() {
    // Edge case: single byte input
    let input = [b'X'];
    let mut output = [0u8; 32];

    let size = lz4_compress_block(&input, &mut output).unwrap();
    assert!(size > 0);
    assert_eq!(output[0] >> 4, 1); // Token: 1 literal
}

#[test]
fn test_lz4_compress_two_bytes() {
    // Edge case: two byte input
    let input = [b'A', b'B'];
    let mut output = [0u8; 32];

    let size = lz4_compress_block(&input, &mut output).unwrap();
    assert!(size > 0);
    assert_eq!(output[0] >> 4, 2); // Token: 2 literals
}
