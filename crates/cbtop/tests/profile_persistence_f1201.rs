//! Falsification Tests for PMAT-028: Profile Persistence and Rotation
//!
//! F1201-F1210: Profile persistence falsification tests
//!
//! These tests verify the profile management module for:
//! - Profile save/load operations
//! - Name validation
//! - CLI overlay merging
//! - Export/import functionality

use cbtop::{
    ProfileConfig, ProfileManager, ProfileOverlay, ProfileError,
    ProfileBackend as BackendConfig, ProfileWorkload as WorkloadConfig,
    profile_persistence::templates,
};
use tempfile::TempDir;
use std::path::PathBuf;

// =============================================================================
// F1201: Profile Loading Tests
// =============================================================================

/// F1201.1: Profile loaded by name
#[test]
fn f1201_profile_loaded_by_name() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // Save a profile
    let profile = ProfileConfig::new("my_profile").unwrap()
        .backend(BackendConfig::Cuda)
        .problem_size(2048);
    manager.save_profile(&profile).unwrap();

    // Load by name
    let loaded = manager.load_profile("my_profile").unwrap();
    assert_eq!(loaded.name, "my_profile");
    assert_eq!(loaded.backend, BackendConfig::Cuda);
    assert_eq!(loaded.problem_size, 2048);
}

/// F1201.2: Profile not found returns error
#[test]
fn f1201_profile_not_found() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let result = manager.load_profile("nonexistent");
    assert!(matches!(result, Err(ProfileError::NotFound(_))));
}

// =============================================================================
// F1202: Profile Saving Tests
// =============================================================================

/// F1202.1: Profile saved to disk
#[test]
fn f1202_profile_saved_to_disk() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profile = ProfileConfig::new("saved_profile").unwrap();
    let path = manager.save_profile(&profile).unwrap();

    // Verify file exists
    assert!(path.exists());
    assert!(path.to_str().unwrap().ends_with(".toml"));

    // Verify content is valid TOML
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("name = \"saved_profile\""));
}

/// F1202.2: Profile content is valid TOML
#[test]
fn f1202_profile_content_valid_toml() {
    let profile = ProfileConfig::new("toml_test").unwrap()
        .backend(BackendConfig::Simd)
        .workload(WorkloadConfig::Attention);

    let toml = profile.to_toml().unwrap();

    // Should be parseable
    let parsed: ProfileConfig = toml::from_str(&toml).unwrap();
    assert_eq!(parsed.name, "toml_test");
    assert_eq!(parsed.backend, BackendConfig::Simd);
    assert_eq!(parsed.workload, WorkloadConfig::Attention);
}

// =============================================================================
// F1203: Profile Listing Tests
// =============================================================================

/// F1203.1: Profile listing works
#[test]
fn f1203_profile_listing() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // Save multiple profiles
    manager.save_profile(&ProfileConfig::new("alpha").unwrap()).unwrap();
    manager.save_profile(&ProfileConfig::new("beta").unwrap()).unwrap();
    manager.save_profile(&ProfileConfig::new("gamma").unwrap()).unwrap();

    let profiles = manager.list_profiles().unwrap();
    assert_eq!(profiles.len(), 3);
    assert!(profiles.contains(&"alpha".to_string()));
    assert!(profiles.contains(&"beta".to_string()));
    assert!(profiles.contains(&"gamma".to_string()));
}

/// F1203.2: Empty directory returns empty list
#[test]
fn f1203_empty_listing() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profiles = manager.list_profiles().unwrap();
    assert!(profiles.is_empty());
}

// =============================================================================
// F1204: CLI Overlay Merging Tests
// =============================================================================

/// F1204.1: CLI overlay merges correctly
#[test]
fn f1204_overlay_merges() {
    let profile = ProfileConfig::new("base").unwrap()
        .backend(BackendConfig::Simd)
        .problem_size(1024);

    let overlay = ProfileOverlay::new()
        .backend(BackendConfig::Cuda)  // Override backend
        .refresh_ms(50);  // Override refresh

    let merged = overlay.apply(profile);

    // CLI overrides should take precedence
    assert_eq!(merged.backend, BackendConfig::Cuda);
    assert_eq!(merged.refresh_ms, 50);
    // Profile value should be preserved
    assert_eq!(merged.problem_size, 1024);
}

/// F1204.2: Empty overlay preserves profile
#[test]
fn f1204_empty_overlay() {
    let profile = ProfileConfig::new("unchanged").unwrap()
        .backend(BackendConfig::Wgpu)
        .problem_size(4096);

    let overlay = ProfileOverlay::new();
    assert!(!overlay.has_overrides());

    let merged = overlay.apply(profile.clone());
    assert_eq!(merged.backend, profile.backend);
    assert_eq!(merged.problem_size, profile.problem_size);
}

// =============================================================================
// F1205: Invalid Profile Handling Tests
// =============================================================================

/// F1205.1: Invalid profile returns error
#[test]
fn f1205_invalid_profile_error() {
    // Invalid name with special characters
    let result = ProfileConfig::new("inv@lid!");
    assert!(matches!(result, Err(ProfileError::InvalidName(_))));

    // Empty name
    let result = ProfileConfig::new("");
    assert!(matches!(result, Err(ProfileError::InvalidName(_))));

    // Starting with number
    let result = ProfileConfig::new("123invalid");
    assert!(matches!(result, Err(ProfileError::InvalidName(_))));
}

/// F1205.2: Invalid TOML returns parse error
#[test]
fn f1205_invalid_toml() {
    let invalid_toml = "this is not { valid toml";
    let result = ProfileConfig::from_toml(invalid_toml);
    assert!(matches!(result, Err(ProfileError::ParseError(_))));
}

// =============================================================================
// F1206: Profile Directory Creation Tests
// =============================================================================

/// F1206.1: Profile directory created automatically
#[test]
fn f1206_directory_created() {
    let temp_dir = TempDir::new().unwrap();
    let subdir = temp_dir.path().join("nested").join("profiles");
    let mut manager = ProfileManager::new(subdir.clone());

    // Directory doesn't exist yet
    assert!(!subdir.exists());

    // Save creates directory
    manager.save_profile(&ProfileConfig::new("test").unwrap()).unwrap();

    // Now it exists
    assert!(subdir.exists());
}

/// F1206.2: ensure_directory is idempotent
#[test]
fn f1206_ensure_directory_idempotent() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().join("profiles"));

    // Call multiple times - should not fail
    manager.ensure_directory().unwrap();
    manager.ensure_directory().unwrap();
    manager.ensure_directory().unwrap();
}

// =============================================================================
// F1207: Profile Name Validation Tests
// =============================================================================

/// F1207.1: Valid names accepted
#[test]
fn f1207_valid_names() {
    assert!(ProfileConfig::new("valid_name").is_ok());
    assert!(ProfileConfig::new("valid-name").is_ok());
    assert!(ProfileConfig::new("ValidName").is_ok());
    assert!(ProfileConfig::new("valid123").is_ok());
    assert!(ProfileConfig::new("a").is_ok());
}

/// F1207.2: Invalid characters rejected
#[test]
fn f1207_invalid_chars_rejected() {
    assert!(ProfileConfig::new("no spaces").is_err());
    assert!(ProfileConfig::new("no.dots").is_err());
    assert!(ProfileConfig::new("no/slashes").is_err());
    assert!(ProfileConfig::new("no\\backslash").is_err());
    assert!(ProfileConfig::new("no:colons").is_err());
}

/// F1207.3: Name length limits enforced
#[test]
fn f1207_name_length_limits() {
    // Too long (>64 chars)
    let long_name = "a".repeat(65);
    assert!(ProfileConfig::new(&long_name).is_err());

    // Just at limit (64 chars)
    let max_name = "a".repeat(64);
    assert!(ProfileConfig::new(&max_name).is_ok());
}

// =============================================================================
// F1208: Profile Export Tests
// =============================================================================

/// F1208.1: Profile export creates file
#[test]
fn f1208_export_creates_file() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().join("profiles"));

    // Save a profile
    let profile = ProfileConfig::new("exportable").unwrap()
        .backend(BackendConfig::Cuda);
    manager.save_profile(&profile).unwrap();

    // Export to different location
    let export_path = temp_dir.path().join("exported.toml");
    manager.export_profile("exportable", &export_path).unwrap();

    // Verify exported file exists and is valid
    assert!(export_path.exists());
    let content = std::fs::read_to_string(&export_path).unwrap();
    assert!(content.contains("name = \"exportable\""));
    assert!(content.contains("backend = \"cuda\""));
}

/// F1208.2: Export non-existent profile fails
#[test]
fn f1208_export_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let export_path = temp_dir.path().join("fail.toml");
    let result = manager.export_profile("nonexistent", &export_path);
    assert!(matches!(result, Err(ProfileError::NotFound(_))));
}

// =============================================================================
// F1209: Default Profile Tests
// =============================================================================

/// F1209.1: Default profile used when none specified
#[test]
fn f1209_default_profile_fallback() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // No default set - returns default config
    let profile = manager.load_default();
    assert_eq!(profile.name, "default");
}

/// F1209.2: Named default profile loaded
#[test]
fn f1209_named_default_profile() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // Save and set as default
    let profile = ProfileConfig::new("my_default").unwrap()
        .backend(BackendConfig::Cuda);
    manager.save_profile(&profile).unwrap();
    manager.set_default("my_default");

    let loaded = manager.load_default();
    assert_eq!(loaded.name, "my_default");
    assert_eq!(loaded.backend, BackendConfig::Cuda);
}

// =============================================================================
// F1210: Profile Description/Metadata Tests
// =============================================================================

/// F1210.1: Profile description stored
#[test]
fn f1210_description_stored() {
    let profile = ProfileConfig::with_description(
        "described",
        "This is a test profile for stress testing"
    ).unwrap();

    let toml = profile.to_toml().unwrap();
    assert!(toml.contains("description = \"This is a test profile"));

    let parsed = ProfileConfig::from_toml(&toml).unwrap();
    assert_eq!(parsed.description, "This is a test profile for stress testing");
}

/// F1210.2: Metadata preserved through save/load
#[test]
fn f1210_metadata_preserved() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profile = ProfileConfig::new("with_meta").unwrap()
        .with_metadata("author", "test_user")
        .with_metadata("version", "2.0");

    manager.save_profile(&profile).unwrap();

    let loaded = manager.load_profile("with_meta").unwrap();
    assert_eq!(loaded.metadata.get("author"), Some(&"test_user".to_string()));
    assert_eq!(loaded.metadata.get("version"), Some(&"2.0".to_string()));
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test profile deletion
#[test]
fn test_profile_deletion() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    manager.save_profile(&ProfileConfig::new("deletable").unwrap()).unwrap();
    assert!(manager.profile_exists("deletable"));

    manager.delete_profile("deletable").unwrap();
    assert!(!manager.profile_exists("deletable"));
}

/// Test profile caching
#[test]
fn test_profile_caching() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profile = ProfileConfig::new("cached").unwrap();
    manager.save_profile(&profile).unwrap();

    // First load populates cache
    let _ = manager.load_profile("cached").unwrap();

    // Delete from disk but cache remains
    let path = temp_dir.path().join("cached.toml");
    std::fs::remove_file(&path).unwrap();

    // Still loads from cache
    let loaded = manager.load_profile("cached").unwrap();
    assert_eq!(loaded.name, "cached");
}

/// Test profile templates
#[test]
fn test_profile_templates() {
    let ml = templates::ml_training();
    assert_eq!(ml.name, "ml_training");
    assert_eq!(ml.backend, BackendConfig::Cuda);
    assert!(ml.metadata.contains_key("use_case"));

    let inference = templates::inference();
    assert_eq!(inference.name, "inference");
    assert!(inference.deterministic);

    let stress = templates::stress_test();
    assert_eq!(stress.load_intensity, 1.0);
    assert_eq!(stress.workload, WorkloadConfig::All);

    let simd = templates::simd_only();
    assert_eq!(simd.backend, BackendConfig::Simd);
}

/// Test profile import
#[test]
fn test_profile_import() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().join("profiles"));

    // Create an external profile file
    let external_path = temp_dir.path().join("external.toml");
    let profile = ProfileConfig::new("imported").unwrap()
        .backend(BackendConfig::Wgpu);
    std::fs::write(&external_path, profile.to_toml().unwrap()).unwrap();

    // Import it
    let imported = manager.import_profile(&external_path).unwrap();
    assert_eq!(imported.name, "imported");

    // Should now be available locally
    let loaded = manager.load_profile("imported").unwrap();
    assert_eq!(loaded.backend, BackendConfig::Wgpu);
}

/// Test profile count
#[test]
fn test_profile_count() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    assert_eq!(manager.profile_count().unwrap(), 0);

    manager.save_profile(&ProfileConfig::new("one").unwrap()).unwrap();
    assert_eq!(manager.profile_count().unwrap(), 1);

    manager.save_profile(&ProfileConfig::new("two").unwrap()).unwrap();
    assert_eq!(manager.profile_count().unwrap(), 2);
}

/// Test builder pattern
#[test]
fn test_builder_pattern() {
    let profile = ProfileConfig::new("builder_test").unwrap()
        .backend(BackendConfig::Cuda)
        .workload(WorkloadConfig::Gemm)
        .problem_size(2048)
        .load_intensity(0.75)
        .threads(8);

    assert_eq!(profile.backend, BackendConfig::Cuda);
    assert_eq!(profile.workload, WorkloadConfig::Gemm);
    assert_eq!(profile.problem_size, 2048);
    assert_eq!(profile.load_intensity, 0.75);
    assert_eq!(profile.threads, 8);
}

/// Test load intensity clamping
#[test]
fn test_load_intensity_clamping() {
    let profile = ProfileConfig::new("clamp_test").unwrap()
        .load_intensity(1.5);  // Should be clamped to 1.0
    assert_eq!(profile.load_intensity, 1.0);

    let profile2 = ProfileConfig::new("clamp_test2").unwrap()
        .load_intensity(-0.5);  // Should be clamped to 0.0
    assert_eq!(profile2.load_intensity, 0.0);
}
