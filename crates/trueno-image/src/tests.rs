//! Image processing tests — provable contracts and falsification.

use crate::conv::*;

// ============================================================================
// FALSIFY-IMG-002: Identity kernel preserves input
// ============================================================================

#[test]
fn test_identity_kernel() {
    let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0_f32];
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let out = conv2d(&image, 3, 3, &delta, 3, 3, BorderMode::Zero).expect("ok");

    // Interior pixel should be exactly preserved
    assert!(
        (out[4] - 5.0).abs() < 1e-6,
        "Identity kernel failed at center: {}",
        out[4]
    );
}

#[test]
fn test_identity_kernel_larger() {
    let w = 5;
    let h = 5;
    let image: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let out = conv2d(&image, w, h, &delta, 3, 3, BorderMode::Clamp).expect("ok");

    // Interior pixels should be preserved
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            assert!(
                (out[idx] - image[idx]).abs() < 1e-6,
                "Identity failed at ({x},{y}): out={}, expected={}",
                out[idx],
                image[idx]
            );
        }
    }
}

// ============================================================================
// FALSIFY-IMG-001: Separable equivalence
// ============================================================================

#[test]
fn test_separable_matches_2d() {
    let w = 8;
    let h = 8;
    let image: Vec<f32> = (0..w * h).map(|i| (i as f32).sin()).collect();

    // Gaussian kernel σ=1.0
    let h_kernel = [0.2742, 0.4514, 0.2742_f32]; // Approximate
    let v_kernel = h_kernel;

    // Full 2D kernel = outer product
    let mut kernel_2d = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            kernel_2d[i * 3 + j] = v_kernel[i] * h_kernel[j];
        }
    }

    let out_2d = conv2d(&image, w, h, &kernel_2d, 3, 3, BorderMode::Zero).expect("ok");
    let out_sep =
        separable_conv2d(&image, w, h, &h_kernel, &v_kernel, BorderMode::Zero).expect("ok");

    for i in 0..w * h {
        assert!(
            (out_2d[i] - out_sep[i]).abs() < 1e-4,
            "Separable mismatch at {i}: 2d={}, sep={}",
            out_2d[i],
            out_sep[i]
        );
    }
}

// ============================================================================
// Gaussian blur tests
// ============================================================================

#[test]
fn test_gaussian_blur_constant_image() {
    let w = 10;
    let h = 10;
    let image = vec![5.0f32; w * h];
    let blurred = gaussian_blur(&image, w, h, 1.0).expect("ok");

    // Blurring a constant image should give the same constant
    for (i, &v) in blurred.iter().enumerate() {
        assert!(
            (v - 5.0).abs() < 1e-3,
            "Gaussian blur changed constant at {i}: {v}"
        );
    }
}

#[test]
fn test_gaussian_blur_reduces_range() {
    let w = 10;
    let h = 10;
    let mut image = vec![0.0f32; w * h];
    image[5 * w + 5] = 100.0; // Single bright pixel

    let blurred = gaussian_blur(&image, w, h, 1.0).expect("ok");
    let max_blurred = blurred.iter().copied().fold(0.0f32, f32::max);

    // Blurring should reduce the peak
    assert!(
        max_blurred < 100.0,
        "Gaussian blur didn't reduce peak: {max_blurred}"
    );
    // Total energy should be approximately conserved
    let sum_orig: f32 = image.iter().sum();
    let sum_blurred: f32 = blurred.iter().sum();
    assert!(
        (sum_orig - sum_blurred).abs() / sum_orig < 0.05,
        "Energy not conserved: orig={sum_orig}, blurred={sum_blurred}"
    );
}

// ============================================================================
// Sobel tests
// ============================================================================

#[test]
fn test_sobel_uniform_image() {
    let w = 5;
    let h = 5;
    let image = vec![3.0f32; w * h]; // Uniform → zero gradient

    let (gx, gy) = sobel(&image, w, h).expect("ok");

    // Interior gradients should be ~0
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            assert!(
                gx[idx].abs() < 1e-5 && gy[idx].abs() < 1e-5,
                "Non-zero gradient on uniform at ({x},{y}): gx={}, gy={}",
                gx[idx],
                gy[idx]
            );
        }
    }
}

#[test]
fn test_sobel_horizontal_edge() {
    let w = 5;
    let h = 5;
    let mut image = vec![0.0f32; w * h];
    // Top half = 0, bottom half = 1 → vertical gradient
    for y in h / 2 + 1..h {
        for x in 0..w {
            image[y * w + x] = 1.0;
        }
    }

    let (_gx, gy) = sobel(&image, w, h).expect("ok");

    // Should have non-zero vertical gradient near the edge
    let edge_y = h / 2;
    let center_x = w / 2;
    let idx = edge_y * w + center_x;
    assert!(
        gy[idx].abs() > 0.1,
        "Expected vertical gradient at edge: gy={}",
        gy[idx]
    );
}

// ============================================================================
// Canny tests
// ============================================================================

#[test]
fn test_canny_uniform_no_edges() {
    let w = 20;
    let h = 20;
    let image = vec![0.5f32; w * h];

    let edges = canny(&image, w, h, 1.0, 0.1, 0.3).expect("ok");
    let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
    assert_eq!(edge_count, 0, "Uniform image should have no edges");
}

#[test]
fn test_canny_strong_edge() {
    let w = 20;
    let h = 20;
    let mut image = vec![0.0f32; w * h];
    // Left half = 0, right half = 1 → strong vertical edge
    for y in 0..h {
        for x in w / 2..w {
            image[y * w + x] = 1.0;
        }
    }

    let edges = canny(&image, w, h, 1.0, 0.05, 0.15).expect("ok");
    let edge_count: usize = edges.iter().filter(|&&v| v > 0.5).count();
    assert!(edge_count > 0, "Strong edge should be detected");
}

#[test]
fn test_canny_invalid_thresholds() {
    let image = vec![0.0f32; 100];
    assert!(canny(&image, 10, 10, 1.0, 0.5, 0.3).is_err()); // low > high
    assert!(canny(&image, 10, 10, 1.0, -0.1, 0.3).is_err()); // negative
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn test_conv2d_zero_dimensions() {
    assert!(conv2d(&[], 0, 0, &[1.0], 1, 1, BorderMode::Zero).is_err());
}

#[test]
fn test_conv2d_even_kernel() {
    let image = vec![1.0f32; 4];
    assert!(conv2d(&image, 2, 2, &[1.0; 4], 2, 2, BorderMode::Zero).is_err());
}

#[test]
fn test_conv2d_buffer_mismatch() {
    let image = vec![1.0f32; 3]; // 3 pixels but claiming 2x2
    assert!(conv2d(&image, 2, 2, &[1.0], 1, 1, BorderMode::Zero).is_err());
}

// ============================================================================
// Border mode tests
// ============================================================================

#[test]
fn test_border_clamp() {
    let image = vec![1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let k = [0.0, 1.0, 0.0_f32]; // 3×1 horizontal kernel (just right neighbor)
    // Actually let's use a 3x1 → need it as 1x3 for conv2d (kw=3, kh=1)
    let out = conv2d(&image, 2, 2, &k, 3, 1, BorderMode::Clamp).expect("ok");

    // At (0,0): neighbors are clamp(-1,0)=image[0]=1, image[0]=1, image[1]=2
    // kernel [0, 1, 0] → picks center = image[0] = 1.0
    assert!((out[0] - 1.0).abs() < 1e-5, "Clamp border: {}", out[0]);
}

// ============================================================================
// Property-based tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_identity_kernel_preserves(
            w in 3usize..20,
            h in 3usize..20,
        ) {
            let image: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
            let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
            let out = conv2d(&image, w, h, &delta, 3, 3, BorderMode::Clamp).expect("ok");

            // Interior pixels preserved
            for y in 1..h-1 {
                for x in 1..w-1 {
                    let idx = y * w + x;
                    prop_assert!((out[idx] - image[idx]).abs() < 1e-5);
                }
            }
        }

        #[test]
        fn prop_gaussian_constant_preserves(
            w in 5usize..15,
            h in 5usize..15,
            val in -100.0f32..100.0,
        ) {
            let image = vec![val; w * h];
            let blurred = gaussian_blur(&image, w, h, 1.0).expect("ok");

            for (i, &v) in blurred.iter().enumerate() {
                prop_assert!(
                    (v - val).abs() < 0.1,
                    "Gaussian changed constant at {i}: {v} vs {val}"
                );
            }
        }
    }
}
