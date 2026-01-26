//! Profile Persistence and Rotation (PMAT-028)
//!
//! Configuration profile management with save/load/switch/export capabilities
//! for different workload scenarios.
//!
//! # Features
//!
//! - Named profiles for different workloads (ml_training, inference, stress_test)
//! - Profile save/load/list/export operations
//! - CLI overlay merging (CLI > profile > default)
//! - TOML-based serialization
//!
//! # Falsification Criteria (F1201-F1210)
//!
//! See `tests/profile_persistence_f1201.rs` for falsification tests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Profile persistence error
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileError {
    /// Profile not found
    NotFound(String),
    /// Invalid profile name (contains invalid characters)
    InvalidName(String),
    /// IO error during read/write
    IoError(String),
    /// TOML parse error
    ParseError(String),
    /// Profile directory creation failed
    DirectoryError(String),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "Profile not found: {}", name),
            Self::InvalidName(msg) => write!(f, "Invalid profile name: {}", msg),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::DirectoryError(msg) => write!(f, "Directory error: {}", msg),
        }
    }
}

impl std::error::Error for ProfileError {}

/// Result type for profile operations
pub type ProfileResult<T> = Result<T, ProfileError>;

/// Serializable profile configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileConfig {
    /// Profile name
    pub name: String,
    /// Profile description
    #[serde(default)]
    pub description: String,
    /// Profile version
    #[serde(default = "default_version")]
    pub version: String,
    /// Refresh rate in milliseconds
    #[serde(default = "default_refresh_ms")]
    pub refresh_ms: u64,
    /// GPU device index
    #[serde(default)]
    pub device_index: u32,
    /// Compute backend
    #[serde(default)]
    pub backend: BackendConfig,
    /// Load intensity (0.0 - 1.0)
    #[serde(default = "default_load_intensity")]
    pub load_intensity: f64,
    /// Workload type
    #[serde(default)]
    pub workload: WorkloadConfig,
    /// Problem size in elements
    #[serde(default = "default_problem_size")]
    pub problem_size: usize,
    /// Thread count for SIMD
    #[serde(default = "default_threads")]
    pub threads: usize,
    /// Enable deterministic mode
    #[serde(default)]
    pub deterministic: bool,
    /// Custom metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_version() -> String {
    "1.0".to_string()
}

fn default_refresh_ms() -> u64 {
    100
}

fn default_load_intensity() -> f64 {
    0.0
}

fn default_problem_size() -> usize {
    1_048_576
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Backend configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackendConfig {
    Simd,
    Wgpu,
    Cuda,
    #[default]
    All,
}

/// Workload configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum WorkloadConfig {
    #[default]
    Gemm,
    Conv2d,
    Attention,
    Bandwidth,
    Elementwise,
    Reduction,
    All,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: String::new(),
            version: default_version(),
            refresh_ms: default_refresh_ms(),
            device_index: 0,
            backend: BackendConfig::default(),
            load_intensity: default_load_intensity(),
            workload: WorkloadConfig::default(),
            problem_size: default_problem_size(),
            threads: default_threads(),
            deterministic: false,
            metadata: HashMap::new(),
        }
    }
}

impl ProfileConfig {
    /// Create a new profile with the given name
    pub fn new(name: &str) -> ProfileResult<Self> {
        validate_profile_name(name)?;
        let mut config = Self::default();
        config.name = name.to_string();
        Ok(config)
    }

    /// Create with name and description
    pub fn with_description(name: &str, description: &str) -> ProfileResult<Self> {
        let mut config = Self::new(name)?;
        config.description = description.to_string();
        Ok(config)
    }

    /// Set backend
    pub fn backend(mut self, backend: BackendConfig) -> Self {
        self.backend = backend;
        self
    }

    /// Set workload
    pub fn workload(mut self, workload: WorkloadConfig) -> Self {
        self.workload = workload;
        self
    }

    /// Set problem size
    pub fn problem_size(mut self, size: usize) -> Self {
        self.problem_size = size;
        self
    }

    /// Set load intensity
    pub fn load_intensity(mut self, intensity: f64) -> Self {
        self.load_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set threads
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Serialize to TOML string
    pub fn to_toml(&self) -> ProfileResult<String> {
        toml::to_string_pretty(self).map_err(|e| ProfileError::ParseError(e.to_string()))
    }

    /// Parse from TOML string
    pub fn from_toml(toml_str: &str) -> ProfileResult<Self> {
        toml::from_str(toml_str).map_err(|e| ProfileError::ParseError(e.to_string()))
    }
}

/// Validate profile name
fn validate_profile_name(name: &str) -> ProfileResult<()> {
    if name.is_empty() {
        return Err(ProfileError::InvalidName(
            "name cannot be empty".to_string(),
        ));
    }

    if name.len() > 64 {
        return Err(ProfileError::InvalidName(
            "name cannot exceed 64 characters".to_string(),
        ));
    }

    // Only allow alphanumeric, underscore, hyphen
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ProfileError::InvalidName(
            "name can only contain alphanumeric, underscore, or hyphen".to_string(),
        ));
    }

    // Cannot start with hyphen or number
    if let Some(first) = name.chars().next() {
        if first == '-' || first.is_numeric() {
            return Err(ProfileError::InvalidName(
                "name cannot start with hyphen or number".to_string(),
            ));
        }
    }

    Ok(())
}

/// Profile manager for save/load/list operations
#[derive(Debug, Clone)]
pub struct ProfileManager {
    /// Profile directory
    profile_dir: PathBuf,
    /// Cached profiles
    cache: HashMap<String, ProfileConfig>,
    /// Default profile name
    default_profile: Option<String>,
}

impl ProfileManager {
    /// Create a new profile manager with the given directory
    pub fn new(profile_dir: PathBuf) -> Self {
        Self {
            profile_dir,
            cache: HashMap::new(),
            default_profile: None,
        }
    }

    /// Create with default profile directory (~/.config/cbtop/profiles)
    pub fn with_default_dir() -> Self {
        let profile_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("cbtop")
            .join("profiles");
        Self::new(profile_dir)
    }

    /// Get profile directory
    pub fn profile_dir(&self) -> &Path {
        &self.profile_dir
    }

    /// Set default profile name
    pub fn set_default(&mut self, name: &str) {
        self.default_profile = Some(name.to_string());
    }

    /// Get default profile name
    pub fn default_profile(&self) -> Option<&str> {
        self.default_profile.as_deref()
    }

    /// Ensure profile directory exists
    pub fn ensure_directory(&self) -> ProfileResult<()> {
        if !self.profile_dir.exists() {
            std::fs::create_dir_all(&self.profile_dir)
                .map_err(|e| ProfileError::DirectoryError(e.to_string()))?;
        }
        Ok(())
    }

    /// Save a profile to disk
    pub fn save_profile(&mut self, profile: &ProfileConfig) -> ProfileResult<PathBuf> {
        self.ensure_directory()?;

        let filename = format!("{}.toml", profile.name);
        let path = self.profile_dir.join(&filename);

        let toml_content = profile.to_toml()?;
        std::fs::write(&path, toml_content).map_err(|e| ProfileError::IoError(e.to_string()))?;

        // Update cache
        self.cache.insert(profile.name.clone(), profile.clone());

        Ok(path)
    }

    /// Load a profile by name
    pub fn load_profile(&mut self, name: &str) -> ProfileResult<ProfileConfig> {
        validate_profile_name(name)?;

        // Check cache first
        if let Some(profile) = self.cache.get(name) {
            return Ok(profile.clone());
        }

        let filename = format!("{}.toml", name);
        let path = self.profile_dir.join(&filename);

        if !path.exists() {
            return Err(ProfileError::NotFound(name.to_string()));
        }

        let content =
            std::fs::read_to_string(&path).map_err(|e| ProfileError::IoError(e.to_string()))?;

        let profile = ProfileConfig::from_toml(&content)?;

        // Update cache
        self.cache.insert(name.to_string(), profile.clone());

        Ok(profile)
    }

    /// Load default profile or return default config
    pub fn load_default(&mut self) -> ProfileConfig {
        if let Some(name) = &self.default_profile.clone() {
            if let Ok(profile) = self.load_profile(name) {
                return profile;
            }
        }
        ProfileConfig::default()
    }

    /// List all available profiles
    pub fn list_profiles(&self) -> ProfileResult<Vec<String>> {
        if !self.profile_dir.exists() {
            return Ok(vec![]);
        }

        let entries = std::fs::read_dir(&self.profile_dir)
            .map_err(|e| ProfileError::IoError(e.to_string()))?;

        let mut profiles = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "toml") {
                if let Some(stem) = path.file_stem() {
                    if let Some(name) = stem.to_str() {
                        profiles.push(name.to_string());
                    }
                }
            }
        }

        profiles.sort();
        Ok(profiles)
    }

    /// Delete a profile by name
    pub fn delete_profile(&mut self, name: &str) -> ProfileResult<()> {
        validate_profile_name(name)?;

        let filename = format!("{}.toml", name);
        let path = self.profile_dir.join(&filename);

        if !path.exists() {
            return Err(ProfileError::NotFound(name.to_string()));
        }

        std::fs::remove_file(&path).map_err(|e| ProfileError::IoError(e.to_string()))?;

        // Remove from cache
        self.cache.remove(name);

        Ok(())
    }

    /// Export a profile to a specific path
    pub fn export_profile(&self, name: &str, export_path: &Path) -> ProfileResult<()> {
        let profile = if let Some(cached) = self.cache.get(name) {
            cached.clone()
        } else {
            let filename = format!("{}.toml", name);
            let path = self.profile_dir.join(&filename);

            if !path.exists() {
                return Err(ProfileError::NotFound(name.to_string()));
            }

            let content =
                std::fs::read_to_string(&path).map_err(|e| ProfileError::IoError(e.to_string()))?;

            ProfileConfig::from_toml(&content)?
        };

        let toml_content = profile.to_toml()?;
        std::fs::write(export_path, toml_content).map_err(|e| ProfileError::IoError(e.to_string()))
    }

    /// Import a profile from a specific path
    pub fn import_profile(&mut self, import_path: &Path) -> ProfileResult<ProfileConfig> {
        if !import_path.exists() {
            return Err(ProfileError::NotFound(import_path.display().to_string()));
        }

        let content = std::fs::read_to_string(import_path)
            .map_err(|e| ProfileError::IoError(e.to_string()))?;

        let profile = ProfileConfig::from_toml(&content)?;

        // Save to local profile directory
        self.save_profile(&profile)?;

        Ok(profile)
    }

    /// Check if a profile exists
    pub fn profile_exists(&self, name: &str) -> bool {
        if self.cache.contains_key(name) {
            return true;
        }

        let filename = format!("{}.toml", name);
        let path = self.profile_dir.join(&filename);
        path.exists()
    }

    /// Get profile count
    pub fn profile_count(&self) -> ProfileResult<usize> {
        self.list_profiles().map(|p| p.len())
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// CLI overlay for merging profile with command-line arguments
#[derive(Debug, Clone, Default)]
pub struct ProfileOverlay {
    /// Override refresh rate
    pub refresh_ms: Option<u64>,
    /// Override device index
    pub device_index: Option<u32>,
    /// Override backend
    pub backend: Option<BackendConfig>,
    /// Override load intensity
    pub load_intensity: Option<f64>,
    /// Override workload
    pub workload: Option<WorkloadConfig>,
    /// Override problem size
    pub problem_size: Option<usize>,
    /// Override threads
    pub threads: Option<usize>,
    /// Override deterministic
    pub deterministic: Option<bool>,
}

impl ProfileOverlay {
    /// Create empty overlay
    pub fn new() -> Self {
        Self::default()
    }

    /// Set refresh rate override
    pub fn refresh_ms(mut self, ms: u64) -> Self {
        self.refresh_ms = Some(ms);
        self
    }

    /// Set backend override
    pub fn backend(mut self, backend: BackendConfig) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set workload override
    pub fn workload(mut self, workload: WorkloadConfig) -> Self {
        self.workload = Some(workload);
        self
    }

    /// Set problem size override
    pub fn problem_size(mut self, size: usize) -> Self {
        self.problem_size = Some(size);
        self
    }

    /// Apply overlay to profile (CLI > profile > default)
    pub fn apply(&self, mut profile: ProfileConfig) -> ProfileConfig {
        if let Some(v) = self.refresh_ms {
            profile.refresh_ms = v;
        }
        if let Some(v) = self.device_index {
            profile.device_index = v;
        }
        if let Some(v) = self.backend {
            profile.backend = v;
        }
        if let Some(v) = self.load_intensity {
            profile.load_intensity = v;
        }
        if let Some(v) = self.workload {
            profile.workload = v;
        }
        if let Some(v) = self.problem_size {
            profile.problem_size = v;
        }
        if let Some(v) = self.threads {
            profile.threads = v;
        }
        if let Some(v) = self.deterministic {
            profile.deterministic = v;
        }
        profile
    }

    /// Check if any overrides are set
    pub fn has_overrides(&self) -> bool {
        self.refresh_ms.is_some()
            || self.device_index.is_some()
            || self.backend.is_some()
            || self.load_intensity.is_some()
            || self.workload.is_some()
            || self.problem_size.is_some()
            || self.threads.is_some()
            || self.deterministic.is_some()
    }
}

/// Pre-defined profile templates
pub mod templates {
    use super::*;

    /// Create ML training profile
    pub fn ml_training() -> ProfileConfig {
        ProfileConfig {
            name: "ml_training".to_string(),
            description: "Optimized for ML training workloads".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 200,
            device_index: 0,
            backend: BackendConfig::Cuda,
            load_intensity: 0.75,
            workload: WorkloadConfig::Gemm,
            problem_size: 4_194_304,
            threads: default_threads(),
            deterministic: false,
            metadata: [("use_case".to_string(), "training".to_string())]
                .into_iter()
                .collect(),
        }
    }

    /// Create inference profile
    pub fn inference() -> ProfileConfig {
        ProfileConfig {
            name: "inference".to_string(),
            description: "Optimized for inference workloads".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 50,
            device_index: 0,
            backend: BackendConfig::Cuda,
            load_intensity: 0.5,
            workload: WorkloadConfig::Attention,
            problem_size: 1_048_576,
            threads: default_threads(),
            deterministic: true,
            metadata: [("use_case".to_string(), "inference".to_string())]
                .into_iter()
                .collect(),
        }
    }

    /// Create stress test profile
    pub fn stress_test() -> ProfileConfig {
        ProfileConfig {
            name: "stress_test".to_string(),
            description: "Maximum stress for stability testing".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 100,
            device_index: 0,
            backend: BackendConfig::All,
            load_intensity: 1.0,
            workload: WorkloadConfig::All,
            problem_size: 16_777_216,
            threads: default_threads(),
            deterministic: false,
            metadata: [("use_case".to_string(), "stress".to_string())]
                .into_iter()
                .collect(),
        }
    }

    /// Create SIMD-only profile
    pub fn simd_only() -> ProfileConfig {
        ProfileConfig {
            name: "simd_only".to_string(),
            description: "CPU SIMD operations only".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 100,
            device_index: 0,
            backend: BackendConfig::Simd,
            load_intensity: 0.5,
            workload: WorkloadConfig::Elementwise,
            problem_size: 1_048_576,
            threads: default_threads(),
            deterministic: false,
            metadata: [("use_case".to_string(), "cpu".to_string())]
                .into_iter()
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_profile_config_default() {
        let config = ProfileConfig::default();
        assert_eq!(config.name, "default");
        assert_eq!(config.refresh_ms, 100);
        assert_eq!(config.backend, BackendConfig::All);
    }

    #[test]
    fn test_profile_config_new() {
        let config = ProfileConfig::new("my_profile").unwrap();
        assert_eq!(config.name, "my_profile");
    }

    #[test]
    fn test_profile_name_validation() {
        assert!(validate_profile_name("valid_name").is_ok());
        assert!(validate_profile_name("valid-name").is_ok());
        assert!(validate_profile_name("valid123").is_ok());
        assert!(validate_profile_name("").is_err());
        assert!(validate_profile_name("-invalid").is_err());
        assert!(validate_profile_name("1invalid").is_err());
        assert!(validate_profile_name("inv@lid").is_err());
    }

    #[test]
    fn test_profile_toml_serialization() {
        let config = ProfileConfig::new("test").unwrap();
        let toml = config.to_toml().unwrap();
        assert!(toml.contains("name = \"test\""));

        let parsed = ProfileConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.name, "test");
    }

    #[test]
    fn test_profile_manager_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

        let profile = ProfileConfig::new("test_profile").unwrap();
        manager.save_profile(&profile).unwrap();

        let loaded = manager.load_profile("test_profile").unwrap();
        assert_eq!(loaded.name, "test_profile");
    }

    #[test]
    fn test_profile_manager_list() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

        manager
            .save_profile(&ProfileConfig::new("profile_a").unwrap())
            .unwrap();
        manager
            .save_profile(&ProfileConfig::new("profile_b").unwrap())
            .unwrap();

        let profiles = manager.list_profiles().unwrap();
        assert_eq!(profiles.len(), 2);
        assert!(profiles.contains(&"profile_a".to_string()));
        assert!(profiles.contains(&"profile_b".to_string()));
    }

    #[test]
    fn test_profile_overlay() {
        let profile = ProfileConfig::default();
        let overlay = ProfileOverlay::new()
            .refresh_ms(200)
            .backend(BackendConfig::Cuda);

        let merged = overlay.apply(profile);
        assert_eq!(merged.refresh_ms, 200);
        assert_eq!(merged.backend, BackendConfig::Cuda);
    }

    #[test]
    fn test_templates() {
        let ml = templates::ml_training();
        assert_eq!(ml.name, "ml_training");
        assert_eq!(ml.backend, BackendConfig::Cuda);

        let inference = templates::inference();
        assert_eq!(inference.name, "inference");

        let stress = templates::stress_test();
        assert_eq!(stress.load_intensity, 1.0);
    }
}
