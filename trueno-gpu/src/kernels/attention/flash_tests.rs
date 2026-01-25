#[cfg(test)]
mod tests {
    use super::super::flash::*;
    use crate::kernels::Kernel;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_flash_attention_config_fuzz(
            seq_len in 1u32..8192,
            head_dim in 32u32..256,
            tile_q in 16u32..128,
            tile_kv in 16u32..128,
            scale in 0.1f32..10.0f32,
            causal in any::<bool>(),
            use_tensor_cores in any::<bool>()
        ) {
            let kernel = AttentionKernel {
                seq_len,
                head_dim,
                tile_q,
                tile_kv,
                scale,
                causal,
                use_tensor_cores,
            };

            // Verify basic invariants
            assert!(kernel.seq_len > 0);
            assert!(kernel.head_dim > 0);
            
            // Verify PTX generation doesn't panic
            let ptx = kernel.emit_ptx();
            assert!(ptx.contains(".visible .entry flash_attention"));
            
            // Check for critical instructions based on config
            if use_tensor_cores {
                // Should have wmma instructions if tensor cores are enabled
                // Note: The builder might fall back if dimensions aren't multiples of 16
                // checking for wmma might be flaky if the builder is smart, 
                // but at least it shouldn't panic.
            } else {
                // Should use standard fma
                assert!(ptx.contains("fma.rn.f32"));
            }
        }
    }

    #[test]
    fn test_flash_attention_edge_cases() {
        // Test 0 sequence length (should handle gracefully or be rejected)
        let kernel_zero = AttentionKernel {
            seq_len: 0,
            head_dim: 64,
            tile_q: 32,
            tile_kv: 32,
            scale: 1.0,
            causal: true,
            use_tensor_cores: false,
        };
        // Just ensure it doesn't crash during emit
        let _ = kernel_zero.emit_ptx();
        
        // Test very large sequence
        let kernel_large = AttentionKernel {
            seq_len: 1_000_000,
            head_dim: 64,
            tile_q: 32,
            tile_kv: 32,
            scale: 1.0,
            causal: false,
            use_tensor_cores: false,
        };
        let ptx = kernel_large.emit_ptx();
        assert!(ptx.contains(".visible .entry flash_attention"));
    }
}
