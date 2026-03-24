//! Auto-generated contract assertions from YAML.
//! Zero cost in release builds (debug_assert!).
//! Regenerate with: pv codegen

#![allow(dead_code, unused_variables)]

// Auto-generated from contracts/blas-level3-v1.yaml — DO NOT EDIT
// Contract: blas-level3-v1

/// Preconditions for equation `symm`.
/// Call at function entry: `contract_pre_symm!(var1, var2, ...)`
macro_rules! contract_pre_symm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0 && m > 0, "Pre-condition violated: n > 0 && m > 0");
        debug_assert!(a.len() == n * n, "Pre-condition violated: a.len() == n * n");
        debug_assert!(b.len() == n * m, "Pre-condition violated: b.len() == n * m");
        debug_assert!(c.len() == n * m, "Pre-condition violated: c.len() == n * m");
    }};
}

/// Postconditions for equation `symm`.
/// Call before return: `contract_post_symm!(ret, var1, ...)`
macro_rules! contract_post_symm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n * m, "Post-condition violated: ret.len() == n * m");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

/// Preconditions for equation `syrk`.
/// Call at function entry: `contract_pre_syrk!(var1, var2, ...)`
macro_rules! contract_pre_syrk {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0 && k > 0, "Pre-condition violated: n > 0 && k > 0");
        debug_assert!(a.len() == n * k, "Pre-condition violated: a.len() == n * k");
        debug_assert!(c.len() == n * n, "Pre-condition violated: c.len() == n * n");
    }};
}

/// Postconditions for equation `syrk`.
/// Call before return: `contract_post_syrk!(ret, var1, ...)`
macro_rules! contract_post_syrk {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n * n, "Post-condition violated: ret.len() == n * n");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
        debug_assert!((0..n).all(|i| (0..n).all(|j| (ret[i * n + j] - ret[j * n + i]).abs() < 1e-6)), "Post-condition violated: (0..n).all(|i| (0..n).all(|j| (ret[i * n + j] - ret[j * n + i]).abs() < 1e-6))");
    }};
}

/// Preconditions for equation `trmm`.
/// Call at function entry: `contract_pre_trmm!(var1, var2, ...)`
macro_rules! contract_pre_trmm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0 && m > 0, "Pre-condition violated: n > 0 && m > 0");
        debug_assert!(a.len() == n * n, "Pre-condition violated: a.len() == n * n");
        debug_assert!(b.len() == n * m, "Pre-condition violated: b.len() == n * m");
    }};
}

/// Postconditions for equation `trmm`.
/// Call before return: `contract_post_trmm!(ret, var1, ...)`
macro_rules! contract_post_trmm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n * m, "Post-condition violated: ret.len() == n * m");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/blas-trsm-v1.yaml — DO NOT EDIT
// Contract: blas-trsm-v1

/// Preconditions for equation `trsm`.
/// Call at function entry: `contract_pre_trsm!(var1, var2, ...)`
macro_rules! contract_pre_trsm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0 && nrhs > 0, "Pre-condition violated: n > 0 && nrhs > 0");
        debug_assert!(a.len() == n * n, "Pre-condition violated: a.len() == n * n");
        debug_assert!(b.len() == n * nrhs, "Pre-condition violated: b.len() == n * nrhs");
        debug_assert!((0..n).all(|i| a[i * n + i] != 0.0), "Pre-condition violated: (0..n).all(|i| a[i * n + i] != 0.0)");
    }};
}

/// Postconditions for equation `trsm`.
/// Call before return: `contract_post_trsm!(ret, var1, ...)`
macro_rules! contract_post_trsm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n * nrhs, "Post-condition violated: ret.len() == n * nrhs");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/elementwise-kernel-v1.yaml — DO NOT EDIT
// Contract: elementwise-kernel-v1

/// Preconditions for equation `add`.
/// Call at function entry: `contract_pre_add!(var1, var2, ...)`
macro_rules! contract_pre_add {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(!a.is_empty(), "Pre-condition violated: !a.is_empty()");
        debug_assert!(a.len() == b.len(), "Pre-condition violated: a.len() == b.len()");
        debug_assert!(out.len() == a.len(), "Pre-condition violated: out.len() == a.len()");
    }};
}

/// Postconditions for equation `add`.
/// Call before return: `contract_post_add!(ret, var1, ...)`
macro_rules! contract_post_add {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == a.len(), "Post-condition violated: ret.len() == a.len()");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

/// Preconditions for equation `mul_scalar`.
/// Call at function entry: `contract_pre_mul_scalar!(var1, var2, ...)`
macro_rules! contract_pre_mul_scalar {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(!x.is_empty(), "Pre-condition violated: !x.is_empty()");
        debug_assert!(out.len() == x.len(), "Pre-condition violated: out.len() == x.len()");
        debug_assert!(s.is_finite(), "Pre-condition violated: s.is_finite()");
    }};
}

/// Postconditions for equation `mul_scalar`.
/// Call before return: `contract_post_mul_scalar!(ret, var1, ...)`
macro_rules! contract_post_mul_scalar {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == x.len(), "Post-condition violated: ret.len() == x.len()");
        debug_assert!(ret.iter().all(|y| y.is_finite()), "Post-condition violated: ret.iter().all(|y| y.is_finite())");
    }};
}

/// Preconditions for equation `relu`.
/// Call at function entry: `contract_pre_relu!(var1, var2, ...)`
macro_rules! contract_pre_relu {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(!x.is_empty(), "Pre-condition violated: !x.is_empty()");
        debug_assert!(out.len() == x.len(), "Pre-condition violated: out.len() == x.len()");
    }};
}

/// Postconditions for equation `relu`.
/// Call before return: `contract_post_relu!(ret, var1, ...)`
macro_rules! contract_post_relu {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.iter().all(|y| *y >= 0.0), "Post-condition violated: ret.iter().all(|y| *y >= 0.0)");
        debug_assert!(ret.len() == x.len(), "Post-condition violated: ret.len() == x.len()");
        debug_assert!(ret.iter().all(|y| y.is_finite() || y.is_nan()), "Post-condition violated: ret.iter().all(|y| y.is_finite() || y.is_nan())");
    }};
}

// Auto-generated from contracts/fft-2d-v1.yaml — DO NOT EDIT
// Contract: fft-2d-v1

/// Preconditions for equation `fft_2d`.
/// Call at function entry: `contract_pre_fft_2d!(var1, var2, ...)`
macro_rules! contract_pre_fft_2d {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n1 > 0 && n2 > 0, "Pre-condition violated: n1 > 0 && n2 > 0");
        debug_assert!(n1.is_power_of_two() && n2.is_power_of_two(), "Pre-condition violated: n1.is_power_of_two() && n2.is_power_of_two()");
        debug_assert!(x.len() == n1 * n2, "Pre-condition violated: x.len() == n1 * n2");
    }};
}

/// Postconditions for equation `fft_2d`.
/// Call before return: `contract_post_fft_2d!(ret, var1, ...)`
macro_rules! contract_post_fft_2d {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n1 * n2, "Post-condition violated: ret.len() == n1 * n2");
        debug_assert!(ret.iter().all(|c| c.re.is_finite() && c.im.is_finite()), "Post-condition violated: ret.iter().all(|c| c.re.is_finite() && c.im.is_finite())");
    }};
}

// Auto-generated from contracts/fft-3d-v1.yaml — DO NOT EDIT
// Contract: fft-3d-v1

/// Preconditions for equation `fft_3d`.
/// Call at function entry: `contract_pre_fft_3d!(var1, var2, ...)`
macro_rules! contract_pre_fft_3d {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n1 > 0 && n2 > 0 && n3 > 0, "Pre-condition violated: n1 > 0 && n2 > 0 && n3 > 0");
        debug_assert!(n1.is_power_of_two() && n2.is_power_of_two() && n3.is_power_of_two(), "Pre-condition violated: n1.is_power_of_two() && n2.is_power_of_two() && n3.is_power_of_two()");
        debug_assert!(x.len() == n1 * n2 * n3, "Pre-condition violated: x.len() == n1 * n2 * n3");
    }};
}

/// Postconditions for equation `fft_3d`.
/// Call before return: `contract_post_fft_3d!(ret, var1, ...)`
macro_rules! contract_post_fft_3d {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n1 * n2 * n3, "Post-condition violated: ret.len() == n1 * n2 * n3");
        debug_assert!(ret.iter().all(|c| c.re.is_finite() && c.im.is_finite()), "Post-condition violated: ret.iter().all(|c| c.re.is_finite() && c.im.is_finite())");
    }};
}

/// Preconditions for equation `fft_batched`.
/// Call at function entry: `contract_pre_fft_batched!(var1, var2, ...)`
macro_rules! contract_pre_fft_batched {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(batch_count > 0 && n > 0, "Pre-condition violated: batch_count > 0 && n > 0");
        debug_assert!(n.is_power_of_two(), "Pre-condition violated: n.is_power_of_two()");
        debug_assert!(x.len() == batch_count * n, "Pre-condition violated: x.len() == batch_count * n");
    }};
}

/// Postconditions for equation `fft_batched`.
/// Call before return: `contract_post_fft_batched!(ret, var1, ...)`
macro_rules! contract_post_fft_batched {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == batch_count * n, "Post-condition violated: ret.len() == batch_count * n");
        debug_assert!(ret.iter().all(|c| c.re.is_finite() && c.im.is_finite()), "Post-condition violated: ret.iter().all(|c| c.re.is_finite() && c.im.is_finite())");
    }};
}

// Auto-generated from contracts/fft-bluestein-v1.yaml — DO NOT EDIT
// Contract: fft-bluestein-v1

/// Preconditions for equation `bluestein`.
/// Call at function entry: `contract_pre_bluestein!(var1, var2, ...)`
macro_rules! contract_pre_bluestein {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0, "Pre-condition violated: n > 0");
        debug_assert!(x.len() == n, "Pre-condition violated: x.len() == n");
    }};
}

/// Postconditions for equation `bluestein`.
/// Call before return: `contract_post_bluestein!(ret, var1, ...)`
macro_rules! contract_post_bluestein {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n, "Post-condition violated: ret.len() == n");
        debug_assert!(ret.iter().all(|c| c.re.is_finite() && c.im.is_finite()), "Post-condition violated: ret.iter().all(|c| c.re.is_finite() && c.im.is_finite())");
    }};
}

// Auto-generated from contracts/fft-stockham-v1.yaml — DO NOT EDIT
// Contract: fft-stockham-v1

/// Preconditions for equation `dft`.
/// Call at function entry: `contract_pre_dft!(var1, var2, ...)`
macro_rules! contract_pre_dft {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0, "Pre-condition violated: n > 0");
        debug_assert!(n.is_power_of_two(), "Pre-condition violated: n.is_power_of_two()");
        debug_assert!(x.len() == n, "Pre-condition violated: x.len() == n");
    }};
}

/// Postconditions for equation `dft`.
/// Call before return: `contract_post_dft!(ret, var1, ...)`
macro_rules! contract_post_dft {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n, "Post-condition violated: ret.len() == n");
        debug_assert!(ret.iter().all(|c| c.re.is_finite() && c.im.is_finite()), "Post-condition violated: ret.iter().all(|c| c.re.is_finite() && c.im.is_finite())");
    }};
}

// Auto-generated from contracts/gemv-kernel-v1.yaml — DO NOT EDIT
// Contract: gemv-kernel-v1

/// Preconditions for equation `gemv`.
/// Call at function entry: `contract_pre_gemv!(var1, var2, ...)`
macro_rules! contract_pre_gemv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(k > 0 && n > 0, "Pre-condition violated: k > 0 && n > 0");
        debug_assert!(a.len() == k, "Pre-condition violated: a.len() == k");
        debug_assert!(b.len() == k * n, "Pre-condition violated: b.len() == k * n");
        debug_assert!(c.len() == n, "Pre-condition violated: c.len() == n");
    }};
}

/// Postconditions for equation `gemv`.
/// Call before return: `contract_post_gemv!(ret, var1, ...)`
macro_rules! contract_post_gemv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(c.iter().all(|x| x.is_finite()), "Post-condition violated: c.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/image-canny-v1.yaml — DO NOT EDIT
// Contract: image-canny-v1

/// Preconditions for equation `canny`.
/// Call at function entry: `contract_pre_canny!(var1, var2, ...)`
macro_rules! contract_pre_canny {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(w > 0 && h > 0, "Pre-condition violated: w > 0 && h > 0");
        debug_assert!(image.len() == w * h, "Pre-condition violated: image.len() == w * h");
        debug_assert!(sigma > 0.0, "Pre-condition violated: sigma > 0.0");
        debug_assert!(low >= 0.0 && high <= 1.0 && low <= high, "Pre-condition violated: low >= 0.0 && high <= 1.0 && low <= high");
    }};
}

/// Postconditions for equation `canny`.
/// Call before return: `contract_post_canny!(ret, var1, ...)`
macro_rules! contract_post_canny {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == w * h, "Post-condition violated: ret.len() == w * h");
        debug_assert!(ret.iter().all(|v| *v == 0.0 || *v == 1.0), "Post-condition violated: ret.iter().all(|v| *v == 0.0 || *v == 1.0)");
    }};
}

/// Preconditions for equation `sobel`.
/// Call at function entry: `contract_pre_sobel!(var1, var2, ...)`
macro_rules! contract_pre_sobel {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(w >= 3 && h >= 3, "Pre-condition violated: w >= 3 && h >= 3");
        debug_assert!(image.len() == w * h, "Pre-condition violated: image.len() == w * h");
    }};
}

/// Postconditions for equation `sobel`.
/// Call before return: `contract_post_sobel!(ret, var1, ...)`
macro_rules! contract_post_sobel {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(gx.len() == w * h && gy.len() == w * h, "Post-condition violated: gx.len() == w * h && gy.len() == w * h");
        debug_assert!(gx.iter().all(|x| x.is_finite()) && gy.iter().all(|x| x.is_finite()), "Post-condition violated: gx.iter().all(|x| x.is_finite()) && gy.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/image-color-v1.yaml — DO NOT EDIT
// Contract: image-color-v1

/// Preconditions for equation `connected_components`.
/// Call at function entry: `contract_pre_connected_components!(var1, var2, ...)`
macro_rules! contract_pre_connected_components {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(w > 0 && h > 0, "Pre-condition violated: w > 0 && h > 0");
        debug_assert!(image.len() == w * h, "Pre-condition violated: image.len() == w * h");
    }};
}

/// Postconditions for equation `connected_components`.
/// Call before return: `contract_post_connected_components!(ret, var1, ...)`
macro_rules! contract_post_connected_components {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == w * h, "Post-condition violated: ret.len() == w * h");
        debug_assert!(ret.iter().all(|label| *label >= 0), "Post-condition violated: ret.iter().all(|label| *label >= 0)");
    }};
}

/// Preconditions for equation `hsv_roundtrip`.
/// Call at function entry: `contract_pre_hsv_roundtrip!(var1, var2, ...)`
macro_rules! contract_pre_hsv_roundtrip {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(h >= 0.0 && h < 360.0, "Pre-condition violated: h >= 0.0 && h < 360.0");
        debug_assert!(s >= 0.0 && s <= 1.0, "Pre-condition violated: s >= 0.0 && s <= 1.0");
        debug_assert!(v >= 0.0 && v <= 1.0, "Pre-condition violated: v >= 0.0 && v <= 1.0");
    }};
}

/// Postconditions for equation `hsv_roundtrip`.
/// Call before return: `contract_post_hsv_roundtrip!(ret, var1, ...)`
macro_rules! contract_post_hsv_roundtrip {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.0 >= 0.0 && ret.0 < 360.0, "Post-condition violated: ret.0 >= 0.0 && ret.0 < 360.0");
        debug_assert!(ret.1 >= 0.0 && ret.1 <= 1.0, "Post-condition violated: ret.1 >= 0.0 && ret.1 <= 1.0");
        debug_assert!(ret.2 >= 0.0 && ret.2 <= 1.0, "Post-condition violated: ret.2 >= 0.0 && ret.2 <= 1.0");
    }};
}

/// Preconditions for equation `rgb_to_gray`.
/// Call at function entry: `contract_pre_rgb_to_gray!(var1, var2, ...)`
macro_rules! contract_pre_rgb_to_gray {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(r >= 0.0 && r <= 1.0, "Pre-condition violated: r >= 0.0 && r <= 1.0");
        debug_assert!(g >= 0.0 && g <= 1.0, "Pre-condition violated: g >= 0.0 && g <= 1.0");
        debug_assert!(b >= 0.0 && b <= 1.0, "Pre-condition violated: b >= 0.0 && b <= 1.0");
    }};
}

/// Postconditions for equation `rgb_to_gray`.
/// Call before return: `contract_post_rgb_to_gray!(ret, var1, ...)`
macro_rules! contract_post_rgb_to_gray {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret >= 0.0 && ret <= 1.0, "Post-condition violated: ret >= 0.0 && ret <= 1.0");
        debug_assert!(ret.is_finite(), "Post-condition violated: ret.is_finite()");
    }};
}

// Auto-generated from contracts/image-conv2d-v1.yaml — DO NOT EDIT
// Contract: image-conv2d-v1

/// Preconditions for equation `conv2d`.
/// Call at function entry: `contract_pre_conv2d!(var1, var2, ...)`
macro_rules! contract_pre_conv2d {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(h > 0 && w > 0, "Pre-condition violated: h > 0 && w > 0");
        debug_assert!(kh > 0 && kw > 0 && kh % 2 == 1 && kw % 2 == 1, "Pre-condition violated: kh > 0 && kw > 0 && kh % 2 == 1 && kw % 2 == 1");
        debug_assert!(image.len() == h * w, "Pre-condition violated: image.len() == h * w");
        debug_assert!(kernel.len() == kh * kw, "Pre-condition violated: kernel.len() == kh * kw");
    }};
}

/// Postconditions for equation `conv2d`.
/// Call before return: `contract_post_conv2d!(ret, var1, ...)`
macro_rules! contract_post_conv2d {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == h * w, "Post-condition violated: ret.len() == h * w");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/image-histogram-v1.yaml — DO NOT EDIT
// Contract: image-histogram-v1

/// Preconditions for equation `histogram`.
/// Call at function entry: `contract_pre_histogram!(var1, var2, ...)`
macro_rules! contract_pre_histogram {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(w > 0 && h > 0, "Pre-condition violated: w > 0 && h > 0");
        debug_assert!(bins > 0, "Pre-condition violated: bins > 0");
        debug_assert!(image.len() == w * h, "Pre-condition violated: image.len() == w * h");
        debug_assert!(image.iter().all(|v| *v >= 0.0 && *v <= 1.0), "Pre-condition violated: image.iter().all(|v| *v >= 0.0 && *v <= 1.0)");
    }};
}

/// Postconditions for equation `histogram`.
/// Call before return: `contract_post_histogram!(ret, var1, ...)`
macro_rules! contract_post_histogram {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == bins, "Post-condition violated: ret.len() == bins");
        debug_assert!(ret.iter().sum::<usize>() == w * h, "Post-condition violated: ret.iter().sum::<usize>() == w * h");
        debug_assert!(ret.iter().all(|c| *c >= 0), "Post-condition violated: ret.iter().all(|c| *c >= 0)");
    }};
}

/// Preconditions for equation `morphology`.
/// Call at function entry: `contract_pre_morphology!(var1, var2, ...)`
macro_rules! contract_pre_morphology {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(w > 0 && h > 0, "Pre-condition violated: w > 0 && h > 0");
        debug_assert!(sw > 0 && sh > 0, "Pre-condition violated: sw > 0 && sh > 0");
        debug_assert!(image.len() == w * h, "Pre-condition violated: image.len() == w * h");
        debug_assert!(structuring_element.len() == sw * sh, "Pre-condition violated: structuring_element.len() == sw * sh");
    }};
}

/// Postconditions for equation `morphology`.
/// Call before return: `contract_post_morphology!(ret, var1, ...)`
macro_rules! contract_post_morphology {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == w * h, "Post-condition violated: ret.len() == w * h");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

/// Preconditions for equation `resize`.
/// Call at function entry: `contract_pre_resize!(var1, var2, ...)`
macro_rules! contract_pre_resize {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(sw > 0 && sh > 0, "Pre-condition violated: sw > 0 && sh > 0");
        debug_assert!(dw > 0 && dh > 0, "Pre-condition violated: dw > 0 && dh > 0");
        debug_assert!(image.len() == sw * sh, "Pre-condition violated: image.len() == sw * sh");
    }};
}

/// Postconditions for equation `resize`.
/// Call before return: `contract_post_resize!(ret, var1, ...)`
macro_rules! contract_post_resize {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == dw * dh, "Post-condition violated: ret.len() == dw * dh");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/image-resize-v1.yaml — DO NOT EDIT
// Contract: image-resize-v1

/// Preconditions for equation `resize`.
/// Call at function entry: `contract_pre_resize!(var1, var2, ...)`
macro_rules! contract_pre_resize {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(w_src > 0 && h_src > 0, "Pre-condition violated: w_src > 0 && h_src > 0");
        debug_assert!(w_dst > 0 && h_dst > 0, "Pre-condition violated: w_dst > 0 && h_dst > 0");
        debug_assert!(image.len() == w_src * h_src, "Pre-condition violated: image.len() == w_src * h_src");
    }};
}

/// Postconditions for equation `resize`.
/// Call before return: `contract_post_resize!(ret, var1, ...)`
macro_rules! contract_post_resize {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == w_dst * h_dst, "Post-condition violated: ret.len() == w_dst * h_dst");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/rand-philox-v1.yaml — DO NOT EDIT
// Contract: rand-philox-v1

/// Preconditions for equation `philox`.
/// Call at function entry: `contract_pre_philox!(var1, var2, ...)`
macro_rules! contract_pre_philox {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(true, "Pre-condition violated: true");
    }};
}

/// Postconditions for equation `philox`.
/// Call before return: `contract_post_philox!(ret, var1, ...)`
macro_rules! contract_post_philox {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == 4, "Post-condition violated: ret.len() == 4");
        debug_assert!(ret == Philox4x32_10(counter, key), "Post-condition violated: ret == Philox4x32_10(counter, key)");
    }};
}

// Auto-generated from contracts/rand-threefry-v1.yaml — DO NOT EDIT
// Contract: rand-threefry-v1

/// Preconditions for equation `threefry`.
/// Call at function entry: `contract_pre_threefry!(var1, var2, ...)`
macro_rules! contract_pre_threefry {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(key.len() == 4, "Pre-condition violated: key.len() == 4");
        debug_assert!(counter.len() == 4, "Pre-condition violated: counter.len() == 4");
    }};
}

/// Postconditions for equation `threefry`.
/// Call before return: `contract_post_threefry!(ret, var1, ...)`
macro_rules! contract_post_threefry {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == 4, "Post-condition violated: ret.len() == 4");
        debug_assert!(ret == Threefry_20(counter, key), "Post-condition violated: ret == Threefry_20(counter, key)");
    }};
}

// Auto-generated from contracts/softmax-kernel-v1.yaml — DO NOT EDIT
// Contract: softmax-kernel-v1

/// Preconditions for equation `softmax`.
/// Call at function entry: `contract_pre_softmax!(var1, var2, ...)`
macro_rules! contract_pre_softmax {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(!x.is_empty(), "Pre-condition violated: !x.is_empty()");
        debug_assert!(x.iter().all(|v| v.is_finite()), "Pre-condition violated: x.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `softmax`.
/// Call before return: `contract_post_softmax!(ret, var1, ...)`
macro_rules! contract_post_softmax {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == x.len(), "Post-condition violated: ret.len() == x.len()");
        debug_assert!(ret.iter().all(|y| *y >= 0.0 && *y <= 1.0), "Post-condition violated: ret.iter().all(|y| *y >= 0.0 && *y <= 1.0)");
        debug_assert!((ret.iter().sum::<f32>() - 1.0).abs() < 1e-5, "Post-condition violated: (ret.iter().sum::<f32>() - 1.0).abs() < 1e-5");
    }};
}

// Auto-generated from contracts/solve-cholesky-v1.yaml — DO NOT EDIT
// Contract: solve-cholesky-v1

/// Preconditions for equation `cholesky`.
/// Call at function entry: `contract_pre_cholesky!(var1, var2, ...)`
macro_rules! contract_pre_cholesky {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0, "Pre-condition violated: n > 0");
        debug_assert!(a.len() == n * n, "Pre-condition violated: a.len() == n * n");
        debug_assert!((0..n).all(|i| (0..n).all(|j| (a[i * n + j] - a[j * n + i]).abs() < 1e-10)), "Pre-condition violated: (0..n).all(|i| (0..n).all(|j| (a[i * n + j] - a[j * n + i]).abs() < 1e-10))");
    }};
}

/// Postconditions for equation `cholesky`.
/// Call before return: `contract_post_cholesky!(ret, var1, ...)`
macro_rules! contract_post_cholesky {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == n * n, "Post-condition violated: ret.len() == n * n");
        debug_assert!((0..n).all(|i| ret[i * n + i] > 0.0), "Post-condition violated: (0..n).all(|i| ret[i * n + i] > 0.0)");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/solve-lu-v1.yaml — DO NOT EDIT
// Contract: solve-lu-v1

/// Preconditions for equation `lu_factorization`.
/// Call at function entry: `contract_pre_lu_factorization!(var1, var2, ...)`
macro_rules! contract_pre_lu_factorization {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(n > 0, "Pre-condition violated: n > 0");
        debug_assert!(a.len() == n * n, "Pre-condition violated: a.len() == n * n");
    }};
}

/// Postconditions for equation `lu_factorization`.
/// Call before return: `contract_post_lu_factorization!(ret, var1, ...)`
macro_rules! contract_post_lu_factorization {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(l.len() == n * n && u.len() == n * n, "Post-condition violated: l.len() == n * n && u.len() == n * n");
        debug_assert!(perm.len() == n, "Post-condition violated: perm.len() == n");
        debug_assert!(l.iter().all(|x| x.is_finite()) && u.iter().all(|x| x.is_finite()), "Post-condition violated: l.iter().all(|x| x.is_finite()) && u.iter().all(|x| x.is_finite())");
        debug_assert!((0..n).all(|i| (l[i * n + i] - 1.0).abs() < 1e-10), "Post-condition violated: (0..n).all(|i| (l[i * n + i] - 1.0).abs() < 1e-10)");
    }};
}

// Auto-generated from contracts/solve-qr-v1.yaml — DO NOT EDIT
// Contract: solve-qr-v1

/// Preconditions for equation `qr_factorization`.
/// Call at function entry: `contract_pre_qr_factorization!(var1, var2, ...)`
macro_rules! contract_pre_qr_factorization {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(m > 0 && n > 0 && m >= n, "Pre-condition violated: m > 0 && n > 0 && m >= n");
        debug_assert!(a.len() == m * n, "Pre-condition violated: a.len() == m * n");
    }};
}

/// Postconditions for equation `qr_factorization`.
/// Call before return: `contract_post_qr_factorization!(ret, var1, ...)`
macro_rules! contract_post_qr_factorization {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(q.len() == m * m, "Post-condition violated: q.len() == m * m");
        debug_assert!(r.len() == m * n, "Post-condition violated: r.len() == m * n");
        debug_assert!(q.iter().all(|x| x.is_finite()) && r.iter().all(|x| x.is_finite()), "Post-condition violated: q.iter().all(|x| x.is_finite()) && r.iter().all(|x| x.is_finite())");
    }};
}

// Auto-generated from contracts/solve-svd-v1.yaml — DO NOT EDIT
// Contract: solve-svd-v1

/// Preconditions for equation `svd`.
/// Call at function entry: `contract_pre_svd!(var1, var2, ...)`
macro_rules! contract_pre_svd {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(m > 0 && n > 0, "Pre-condition violated: m > 0 && n > 0");
        debug_assert!(a.len() == m * n, "Pre-condition violated: a.len() == m * n");
    }};
}

/// Postconditions for equation `svd`.
/// Call before return: `contract_post_svd!(ret, var1, ...)`
macro_rules! contract_post_svd {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(u.len() == m * m, "Post-condition violated: u.len() == m * m");
        debug_assert!(sigma.len() == m.min(n), "Post-condition violated: sigma.len() == m.min(n)");
        debug_assert!(vt.len() == n * n, "Post-condition violated: vt.len() == n * n");
        debug_assert!(sigma.iter().all(|s| *s >= 0.0 && s.is_finite()), "Post-condition violated: sigma.iter().all(|s| *s >= 0.0 && s.is_finite())");
        debug_assert!(sigma.windows(2).all(|w| w[0] >= w[1]), "Post-condition violated: sigma.windows(2).all(|w| w[0] >= w[1])");
    }};
}

// Auto-generated from contracts/sparse-bsr-v1.yaml — DO NOT EDIT
// Contract: sparse-bsr-v1

/// Preconditions for equation `bsr_spmv`.
/// Call at function entry: `contract_pre_bsr_spmv!(var1, var2, ...)`
macro_rules! contract_pre_bsr_spmv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(m > 0 && n > 0, "Pre-condition violated: m > 0 && n > 0");
        debug_assert!(x.len() == n, "Pre-condition violated: x.len() == n");
        debug_assert!(y.len() == m, "Pre-condition violated: y.len() == m");
        debug_assert!(!a.row_ptr.is_empty(), "Pre-condition violated: !a.row_ptr.is_empty()");
    }};
}

/// Postconditions for equation `bsr_spmv`.
/// Call before return: `contract_post_bsr_spmv!(ret, var1, ...)`
macro_rules! contract_post_bsr_spmv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == m, "Post-condition violated: ret.len() == m");
        debug_assert!(ret.iter().all(|v| v.is_finite()), "Post-condition violated: ret.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/sparse-formats-v1.yaml — DO NOT EDIT
// Contract: sparse-formats-v1

/// Preconditions for equation `sell_spmv`.
/// Call at function entry: `contract_pre_sell_spmv!(var1, var2, ...)`
macro_rules! contract_pre_sell_spmv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(m > 0 && n > 0, "Pre-condition violated: m > 0 && n > 0");
        debug_assert!(x.len() == n, "Pre-condition violated: x.len() == n");
        debug_assert!(y.len() == m, "Pre-condition violated: y.len() == m");
    }};
}

/// Postconditions for equation `sell_spmv`.
/// Call before return: `contract_post_sell_spmv!(ret, var1, ...)`
macro_rules! contract_post_sell_spmv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == m, "Post-condition violated: ret.len() == m");
        debug_assert!(ret.iter().all(|v| v.is_finite()), "Post-condition violated: ret.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/sparse-spgemm-v1.yaml — DO NOT EDIT
// Contract: sparse-spgemm-v1

/// Preconditions for equation `spgemm`.
/// Call at function entry: `contract_pre_spgemm!(var1, var2, ...)`
macro_rules! contract_pre_spgemm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(a.cols() == b.rows(), "Pre-condition violated: a.cols() == b.rows()");
        debug_assert!(a.rows() > 0 && b.cols() > 0, "Pre-condition violated: a.rows() > 0 && b.cols() > 0");
    }};
}

/// Postconditions for equation `spgemm`.
/// Call before return: `contract_post_spgemm!(ret, var1, ...)`
macro_rules! contract_post_spgemm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.rows() == a.rows(), "Post-condition violated: ret.rows() == a.rows()");
        debug_assert!(ret.cols() == b.cols(), "Post-condition violated: ret.cols() == b.cols()");
        debug_assert!(ret.values().iter().all(|v| v.is_finite()), "Post-condition violated: ret.values().iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/sparse-spmm-v1.yaml — DO NOT EDIT
// Contract: sparse-spmm-v1

/// Preconditions for equation `spmm`.
/// Call at function entry: `contract_pre_spmm!(var1, var2, ...)`
macro_rules! contract_pre_spmm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(a.cols() == k && b.len() == k * n, "Pre-condition violated: a.cols() == k && b.len() == k * n");
        debug_assert!(c.len() == m * n, "Pre-condition violated: c.len() == m * n");
        debug_assert!(m > 0 && k > 0 && n > 0, "Pre-condition violated: m > 0 && k > 0 && n > 0");
    }};
}

/// Postconditions for equation `spmm`.
/// Call before return: `contract_post_spmm!(ret, var1, ...)`
macro_rules! contract_post_spmm {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == m * n, "Post-condition violated: ret.len() == m * n");
        debug_assert!(ret.iter().all(|v| v.is_finite()), "Post-condition violated: ret.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/sparse-spmv-v1.yaml — DO NOT EDIT
// Contract: sparse-spmv-v1

/// Preconditions for equation `spmv`.
/// Call at function entry: `contract_pre_spmv!(var1, var2, ...)`
macro_rules! contract_pre_spmv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(m > 0 && n > 0, "Pre-condition violated: m > 0 && n > 0");
        debug_assert!(x.len() == n, "Pre-condition violated: x.len() == n");
        debug_assert!(y.len() == m, "Pre-condition violated: y.len() == m");
        debug_assert!(alpha.is_finite() && beta.is_finite(), "Pre-condition violated: alpha.is_finite() && beta.is_finite()");
    }};
}

/// Postconditions for equation `spmv`.
/// Call before return: `contract_post_spmv!(ret, var1, ...)`
macro_rules! contract_post_spmv {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == m, "Post-condition violated: ret.len() == m");
        debug_assert!(ret.iter().all(|v| v.is_finite()), "Post-condition violated: ret.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/tensor-contraction-v1.yaml — DO NOT EDIT
// Contract: tensor-contraction-v1

/// Preconditions for equation `einsum`.
/// Call at function entry: `contract_pre_einsum!(var1, var2, ...)`
macro_rules! contract_pre_einsum {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(!subscript.is_empty(), "Pre-condition violated: !subscript.is_empty()");
        debug_assert!(subscript.contains("->"), "Pre-condition violated: subscript.contains(\"->\")");
        debug_assert!(!a.is_empty() && !b.is_empty(), "Pre-condition violated: !a.is_empty() && !b.is_empty()");
    }};
}

/// Postconditions for equation `einsum`.
/// Call before return: `contract_post_einsum!(ret, var1, ...)`
macro_rules! contract_post_einsum {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(!ret.is_empty(), "Post-condition violated: !ret.is_empty()");
        debug_assert!(ret.iter().all(|v| v.is_finite()), "Post-condition violated: ret.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/transpose-kernel-v1.yaml — DO NOT EDIT
// Contract: transpose-kernel-v1

/// Preconditions for equation `transpose`.
/// Call at function entry: `contract_pre_transpose!(var1, var2, ...)`
macro_rules! contract_pre_transpose {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(rows > 0 && cols > 0, "Pre-condition violated: rows > 0 && cols > 0");
        debug_assert!(a.len() == rows * cols, "Pre-condition violated: a.len() == rows * cols");
    }};
}

/// Postconditions for equation `transpose`.
/// Call before return: `contract_post_transpose!(ret, var1, ...)`
macro_rules! contract_post_transpose {
    ($($arg:ident),* $(,)?) => {{
        debug_assert!(ret.len() == rows * cols, "Post-condition violated: ret.len() == rows * cols");
        debug_assert!(ret.iter().all(|x| x.is_finite()), "Post-condition violated: ret.iter().all(|x| x.is_finite())");
    }};
}

// Total: 107 preconditions, 86 postconditions from 27 contracts
