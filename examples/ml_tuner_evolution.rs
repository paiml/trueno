//! Phase 14: ML-Tuner Evolution Demo
//!
//! Demonstrates the advanced ML features added in Phase 14:
//! - MLT-10: Pre-trained weights from benchmark corpus
//! - MLT-11: First-run calibration (requires hardware-detect feature)
//! - MLT-12: Online learning with SGD
//! - MLT-13: Bandit-based kernel selection
//!
//! Run:
//!   cargo run --example ml_tuner_evolution
//!
//! With calibration:
//!   cargo run --example ml_tuner_evolution --features hardware-detect

use trueno::tuner::{pretrained, BrickTuner, KernelBandit, QuantType, TunerFeatures};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Phase 14: ML-Tuner Evolution Demo                      ║");
    println!("║       Pre-trained Weights · Online Learning · Bandits        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // MLT-10: Pre-trained Weights
    // =========================================================================
    println!("━━━ MLT-10: Pre-trained Weights ━━━\n");

    println!("Pre-trained throughput weights ({} elements):", pretrained::THROUGHPUT_WEIGHTS.len());
    println!("  Bias (baseline): {:.3}", pretrained::THROUGHPUT_WEIGHTS[0]);
    println!("  batch_size_norm weight: {:.3} (MOST IMPORTANT)", pretrained::THROUGHPUT_WEIGHTS[7]);
    println!("  gpu_mem_bw weight: {:.3}", pretrained::THROUGHPUT_WEIGHTS[36]);
    println!();

    println!("Feature Importance (top 5):");
    for (idx, name, importance) in pretrained::FEATURE_IMPORTANCE.iter().take(5) {
        println!("  [{:2}] {:20} {:.1}%", idx, name, importance * 100.0);
    }
    println!();

    // Create tuner with pre-trained weights
    let tuner = BrickTuner::with_pretrained();
    println!("Created tuner with pre-trained weights:");
    println!("  Version: {}", tuner.version());
    println!("  MAPE: {:.1}%", tuner.throughput_mape() * 100.0);
    println!("  Training samples: {}", tuner.throughput_sample_count());
    println!();

    // =========================================================================
    // MLT-12: Online Learning
    // =========================================================================
    println!("━━━ MLT-12: Online Learning with SGD ━━━\n");

    let mut learner = tuner.online_learner();
    println!("OnlineLearner initialized:");
    println!("  Weights: {} elements", learner.weights().len());
    println!("  Updates: {}", learner.num_updates());
    println!("  EMA Loss: {:.4}", learner.ema_loss());
    println!();

    // Simulate inference observations
    println!("Simulating 50 inference observations...");
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .build();

    let feature_vec = features.to_vector();

    for step in 0..50 {
        // Simulate measured throughput with noise
        let base_tps = 150.0;
        let noise = ((step * 7) % 20) as f32 - 10.0; // Deterministic "noise"
        let measured_tps = base_tps + noise;

        learner.observe(&feature_vec, measured_tps);

        if step % 10 == 9 {
            println!(
                "  Step {:2}: EMA Loss = {:.4}, Updates = {}",
                step + 1,
                learner.ema_loss(),
                learner.num_updates()
            );
        }
    }

    println!();
    println!("After training:");
    println!("  Converging: {}", if learner.is_converging() { "YES" } else { "NO" });
    println!("  Final EMA Loss: {:.4}", learner.ema_loss());
    println!();

    // Apply updates to tuner
    let mut tuner_updated = tuner.clone();
    tuner_updated.apply_online_updates(&learner);
    println!("Applied online updates to tuner:");
    println!("  New version: {}", tuner_updated.version());
    println!();

    // =========================================================================
    // MLT-13: Bandit Kernel Selection
    // =========================================================================
    println!("━━━ MLT-13: Bandit Kernel Selection ━━━\n");

    let mut bandit = KernelBandit::new();
    println!("KernelBandit (UCB1 algorithm):");
    println!("  Num kernels: {}", KernelBandit::NUM_KERNELS);
    println!("  Initial exploration rate: {:.2}", bandit.exploration_rate());
    println!();

    // Simulate kernel selection with rewards
    println!("Simulating 30 kernel selections with rewards...");
    println!("  (BatchedQ4K gets higher rewards to simulate being optimal)");
    println!();

    use trueno::tuner::KernelType;

    for trial in 0..30 {
        // Get recommendation with exploration
        let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 0.3);
        let kernel = rec.top_kernel;

        // Simulate reward (BatchedQ4K is "best" for batch_size=4)
        let reward = match kernel {
            KernelType::BatchedQ4K => 0.9 + ((trial % 5) as f32 * 0.02),
            KernelType::VectorizedQ4K => 0.7,
            KernelType::CoalescedQ4K => 0.6,
            _ => 0.5,
        };

        bandit.update(kernel, reward);

        if trial % 10 == 9 {
            println!(
                "  Trial {:2}: Selected {:?}, Reward = {:.2}, Explore Rate = {:.2}",
                trial + 1,
                kernel,
                reward,
                bandit.exploration_rate()
            );
        }
    }

    println!();
    println!("Bandit results:");
    println!("  Best kernel: {:?}", bandit.best_kernel());
    println!("  Final exploration rate: {:.2}", bandit.exploration_rate());
    println!("  Estimated regret: {:.2}", bandit.estimated_regret());
    println!();

    // =========================================================================
    // Thompson Sampling Alternative
    // =========================================================================
    println!("━━━ Thompson Sampling Variant ━━━\n");

    let mut thompson_bandit = KernelBandit::with_thompson_sampling();
    for _ in 0..20 {
        let kernel = thompson_bandit.select();
        let reward = if matches!(kernel, KernelType::BatchedQ4K) { 0.9 } else { 0.5 };
        thompson_bandit.update(kernel, reward);
    }
    println!("Thompson Sampling (20 trials):");
    println!("  Best kernel: {:?}", thompson_bandit.best_kernel());
    println!("  Exploration rate: {:.2}", thompson_bandit.exploration_rate());
    println!();

    // =========================================================================
    // Integration: Full Workflow
    // =========================================================================
    println!("━━━ Full Integration Workflow ━━━\n");

    // 1. Start with pre-trained
    let mut production_tuner = BrickTuner::with_pretrained();
    println!("1. Loaded pre-trained tuner (v{})", production_tuner.version());

    // 2. Create online learner
    let mut online = production_tuner.online_learner();
    println!("2. Created online learner");

    // 3. Create bandit
    let mut kernel_bandit = production_tuner.kernel_bandit();
    println!("3. Created kernel bandit");

    // 4. Simulate production loop
    println!("4. Running 20-step production simulation...");
    for step in 0..20 {
        // Get recommendation
        let rec = production_tuner.recommend_kernel_with_exploration(&features, &kernel_bandit, 0.2);

        // "Measure" throughput
        let measured_tps = 180.0 + (step as f32 * 1.5);

        // Update systems
        let reward = (measured_tps / 200.0).min(1.0);
        kernel_bandit.update(rec.top_kernel, reward);
        online.observe(&feature_vec, measured_tps);
    }

    // 5. Apply learnings
    production_tuner.apply_online_updates(&online);
    println!("5. Applied online updates");

    println!();
    println!("Final state:");
    println!("  Tuner version: {}", production_tuner.version());
    println!("  Online updates: {}", online.num_updates());
    println!("  Bandit best kernel: {:?}", kernel_bandit.best_kernel());
    println!("  Predicted throughput: {:.1} tok/s",
        production_tuner.recommend(&features).throughput.predicted_tps);

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 14 Demo Complete - ML-Tuner Evolution Working!        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
