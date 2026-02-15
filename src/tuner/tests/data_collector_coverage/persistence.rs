//! save_apr/load_apr round-trip and error path tests

use super::*;

// ============================================================================
// save_apr() and load_apr() - round-trip
// ============================================================================

#[test]
fn save_and_load_apr_round_trip() {
    let c = make_collector_with_samples(5);
    let dir = std::env::temp_dir().join("trueno_test_save_load_apr");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("test_data.apr");

    c.save_apr(&path).expect("save should succeed");
    assert!(path.exists());

    let loaded = TunerDataCollector::load_apr(&path).expect("load should succeed");
    assert_eq!(loaded.len(), 5);
    assert_eq!(
        loaded.samples()[0].throughput_tps,
        c.samples()[0].throughput_tps
    );
    assert_eq!(
        loaded.samples()[4].throughput_tps,
        c.samples()[4].throughput_tps
    );

    // Loaded collector should have default state for non-persisted fields
    assert!(!loaded.is_online_learning_enabled());
    assert_eq!(loaded.retrain_threshold, 100);
    assert!(loaded.feedback.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_apr_creates_parent_directories() {
    let dir = std::env::temp_dir().join("trueno_test_nested_dir/a/b/c");
    let path = dir.join("model.apr");
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("trueno_test_nested_dir"));

    let c = make_collector_with_samples(1);
    c.save_apr(&path)
        .expect("save to nested dir should succeed");
    assert!(path.exists());

    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("trueno_test_nested_dir"));
}

#[test]
fn save_apr_empty_collector() {
    let dir = std::env::temp_dir().join("trueno_test_save_empty");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("empty.apr");

    let c = TunerDataCollector::new();
    c.save_apr(&path).expect("save empty should succeed");

    let loaded = TunerDataCollector::load_apr(&path).expect("load empty should succeed");
    assert!(loaded.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// load_apr() - error paths
// ============================================================================

#[test]
fn load_apr_file_not_found() {
    let result = TunerDataCollector::load_apr("/tmp/trueno_nonexistent_file.apr");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("I/O error"));
}

#[test]
fn load_apr_bad_magic() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_bad_magic");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_magic.apr");

    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"XXXX").expect("write magic");
    file.write_all(&4u32.to_le_bytes()).expect("write len");
    file.write_all(b"test").expect("write data");
    file.write_all(&0u32.to_le_bytes()).expect("write crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("APR2"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_crc_mismatch() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_crc_mismatch");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_crc.apr");

    let json_bytes = b"[]";
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    file.write_all(&(json_bytes.len() as u32).to_le_bytes())
        .expect("write len");
    file.write_all(json_bytes).expect("write data");
    // Write wrong CRC
    file.write_all(&0xDEADBEEFu32.to_le_bytes())
        .expect("write bad crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("CRC mismatch"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_truncated_file() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_truncated");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("truncated.apr");

    // Write only magic, no length or data
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_invalid_json_in_valid_envelope() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_invalid_json_apr");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_json.apr");

    let json_bytes = b"not valid json at all";
    let crc = crate::tuner::helpers::crc32_hash(json_bytes);

    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    file.write_all(&(json_bytes.len() as u32).to_le_bytes())
        .expect("write len");
    file.write_all(json_bytes).expect("write data");
    file.write_all(&crc.to_le_bytes()).expect("write crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Serialization"));

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// save_apr error path: write to invalid path
// ============================================================================

#[test]
fn save_apr_returns_io_error_for_invalid_path() {
    let c = make_collector_with_samples(1);
    // Try to write to a directory that we can't create (root-owned)
    let result = c.save_apr("/proc/nonexistent/deep/path/file.apr");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("I/O error"));
}
