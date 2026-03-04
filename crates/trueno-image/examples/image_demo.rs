//! Image processing demonstration: convolution, Gaussian blur, Sobel, Canny.
//!
//! ```sh
//! cargo run --example image_demo -p trueno-image
//! ```

use trueno_image::{canny, conv2d, gaussian_blur, gradient_magnitude, sobel, BorderMode};

fn main() {
    println!("=== trueno-image: Image Processing Demo ===\n");

    let w = 16;
    let h = 16;

    // Create a test image: left half = 0, right half = 1 (sharp vertical edge)
    let mut image = vec![0.0f32; w * h];
    for y in 0..h {
        for x in w / 2..w {
            image[y * w + x] = 1.0;
        }
    }

    // 1. Identity convolution
    let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
    let identity_out = conv2d(&image, w, h, &delta, 3, 3, BorderMode::Zero).expect("ok");
    let max_err: f32 = image
        .iter()
        .zip(identity_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Identity convolution max error: {max_err:.2e}");

    // 2. Gaussian blur
    let blurred = gaussian_blur(&image, w, h, 1.5).expect("ok");
    let blur_min = blurred.iter().copied().fold(f32::MAX, f32::min);
    let blur_max = blurred.iter().copied().fold(f32::MIN, f32::max);
    println!("Gaussian blur (σ=1.5): range [{blur_min:.3}, {blur_max:.3}]");

    // 3. Sobel edge detection
    let (gx, gy) = sobel(&image, w, h).expect("ok");
    let mag = gradient_magnitude(&gx, &gy);
    let mag_max = mag.iter().copied().fold(0.0f32, f32::max);
    let edge_pixels = mag.iter().filter(|&&m| m > 0.5).count();
    println!("Sobel: max gradient = {mag_max:.3}, {edge_pixels} edge pixels");

    // 4. Canny edge detection
    let edges = canny(&image, w, h, 1.0, 0.05, 0.15).expect("ok");
    let canny_edges = edges.iter().filter(|&&v| v > 0.5).count();
    println!("Canny: {canny_edges} edge pixels detected");

    // Print edge map (small visualization)
    println!("\nEdge map (Canny):");
    for y in 0..h {
        let row: String = (0..w)
            .map(|x| if edges[y * w + x] > 0.5 { '#' } else { '.' })
            .collect();
        println!("  {row}");
    }

    println!("\n=== All image demos passed ===");
}
