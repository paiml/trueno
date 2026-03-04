//! Counter-based parallel random number generation.
//!
//! # Contract: rand-philox-v1.yaml
//!
//! Implements Philox 4x32-10 (Salmon et al., SC 2011), a counter-based
//! PRNG designed for massively parallel GPU generation.
//!
//! # Example
//!
//! ```
//! use trueno_rand::Philox4x32;
//!
//! let mut rng = Philox4x32::new(42);
//! let vals = rng.next_4u32();
//! assert_ne!(vals, [0, 0, 0, 0]);
//!
//! // Deterministic: same seed → same output
//! let mut rng2 = Philox4x32::new(42);
//! assert_eq!(rng2.next_4u32(), vals);
//! ```

mod error;
mod philox;
mod threefry;

#[cfg(test)]
mod tests;

pub use error::RngError;
pub use philox::Philox4x32;
pub use threefry::Threefry4x64;
