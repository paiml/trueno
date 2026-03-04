//! Image resize: bilinear and nearest-neighbor interpolation (NPP parity).

use crate::error::ImageError;

/// Interpolation method for resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Nearest-neighbor (fastest, blocky).
    Nearest,
    /// Bilinear interpolation (smooth).
    Bilinear,
}

/// Resize a grayscale image.
///
/// # Errors
///
/// Returns error if dimensions don't match or output size is zero.
pub fn resize(
    image: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    method: Interpolation,
) -> Result<Vec<f32>, ImageError> {
    if image.len() != src_w * src_h {
        return Err(ImageError::BufferLengthMismatch {
            expected: src_w * src_h,
            got: image.len(),
            width: src_w,
            height: src_h,
        });
    }
    if dst_w == 0 || dst_h == 0 {
        return Err(ImageError::ZeroDimension {
            width: dst_w,
            height: dst_h,
        });
    }

    let mut output = vec![0.0f32; dst_w * dst_h];

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * scale_x - 0.5;
            let sy = (dy as f32 + 0.5) * scale_y - 0.5;

            output[dy * dst_w + dx] = match method {
                Interpolation::Nearest => {
                    let ix = (sx + 0.5) as usize;
                    let iy = (sy + 0.5) as usize;
                    let ix = ix.min(src_w - 1);
                    let iy = iy.min(src_h - 1);
                    image[iy * src_w + ix]
                }
                Interpolation::Bilinear => {
                    bilinear_sample(image, src_w, src_h, sx, sy)
                }
            };
        }
    }

    Ok(output)
}

/// Bilinear interpolation at fractional coordinates.
fn bilinear_sample(image: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = (x.floor() as isize).max(0) as usize;
    let y0 = (y.floor() as isize).max(0) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);

    let p00 = image[y0 * w + x0];
    let p10 = image[y0 * w + x1];
    let p01 = image[y1 * w + x0];
    let p11 = image[y1 * w + x1];

    p00 * (1.0 - fx) * (1.0 - fy) + p10 * fx * (1.0 - fy) + p01 * (1.0 - fx) * fy + p11 * fx * fy
}
