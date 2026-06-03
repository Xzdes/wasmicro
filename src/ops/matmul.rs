//! Matrix multiplication (forward inference).
//!
//! Two variants:
//!
//! * [`matmul`] — `C = A @ B`, with `A: [m, k]`, `B: [k, n]`, `C: [m, n]`.
//! * [`matmul_t_b`] — `C = A @ B^T`, with `A: [m, k]`, `B: [n, k]`, `C: [m, n]`.
//!   This matches PyTorch's `nn.Linear` weight layout (`[out, in]`), so it is
//!   the hot path for transformer inference.
//!
//! The hot inner kernels are factored into [`kernels`] so that a WASM SIMD128
//! path can replace them transparently. When the crate is compiled for
//! `wasm32` with `-C target-feature=+simd128` (the default in
//! `.cargo/config.toml`), the SIMD path runs; otherwise the scalar fallback
//! is used. Native CPU builds always take the scalar path, which keeps tests
//! and benchmarks deterministic across platforms.

use crate::tensor::Tensor;

/// Computes `C = A @ B` for 2D tensors.
///
/// - `a` shape: `[m, k]`
/// - `b` shape: `[k, n]`
/// - returned tensor shape: `[m, n]`
///
/// Panics on shape mismatch or non-2D inputs.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape().as_slice();
    let b_shape = b.shape().as_slice();

    assert_eq!(
        a_shape.len(),
        2,
        "matmul: `a` must be 2D, got {:?}",
        a_shape
    );
    assert_eq!(
        b_shape.len(),
        2,
        "matmul: `b` must be 2D, got {:?}",
        b_shape
    );
    assert_eq!(
        a_shape[1], b_shape[0],
        "matmul: inner dimensions must match: {:?} @ {:?}",
        a_shape, b_shape
    );

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let a_data = a.data();
    let b_data = b.data();
    let mut out = vec![0.0f32; m * n];

    // ikj loop ordering: best cache behavior for row-major data.
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a_data[i * k + kk];
            let b_row = &b_data[kk * n..(kk + 1) * n];
            let out_row = &mut out[i * n..(i + 1) * n];
            kernels::axpy(a_ik, b_row, out_row);
        }
    }

    Tensor::from_vec(out, &[m, n])
}

/// Computes `C = A @ B^T` (B treated as transposed without materializing the transpose).
///
/// - `a` shape: `[m, k]`
/// - `b` shape: `[n, k]`  (rows of `b` are columns of `B^T`)
/// - returned tensor shape: `[m, n]`
///
/// This is the operation behind a `Linear` layer when weights are stored in
/// PyTorch convention `[out_features, in_features]`.
pub fn matmul_t_b(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape().as_slice();
    let b_shape = b.shape().as_slice();

    assert_eq!(
        a_shape.len(),
        2,
        "matmul_t_b: `a` must be 2D, got {:?}",
        a_shape
    );
    assert_eq!(
        b_shape.len(),
        2,
        "matmul_t_b: `b` must be 2D, got {:?}",
        b_shape
    );
    assert_eq!(
        a_shape[1], b_shape[1],
        "matmul_t_b: inner dimensions must match: {:?} @ {:?}^T",
        a_shape, b_shape
    );

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];

    let a_data = a.data();
    let b_data = b.data();
    let mut out = vec![0.0f32; m * n];

    // For each output element (i, j): dot(a[i, :], b[j, :]).
    for i in 0..m {
        let a_row = &a_data[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for (j, out_cell) in out_row.iter_mut().enumerate() {
            let b_row = &b_data[j * k..(j + 1) * k];
            *out_cell = kernels::dot(a_row, b_row);
        }
    }

    Tensor::from_vec(out, &[m, n])
}

/// Inner kernels with WASM SIMD128 specializations.
///
/// On `wasm32` with `target_feature = "simd128"` the kernels load four `f32`
/// lanes at a time through `core::arch::wasm32`. On any other build (native
/// CPUs, WASM without SIMD) the scalar fallback runs. The signatures and
/// numerical results are identical across paths.
mod kernels {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use core::arch::wasm32::{
        f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128, v128_load, v128_store,
    };

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    pub fn axpy(a_ik: f32, b_row: &[f32], out_row: &mut [f32]) {
        debug_assert_eq!(b_row.len(), out_row.len());
        let n = b_row.len();
        let chunks = n / 4;
        let a_vec = f32x4_splat(a_ik);

        for c in 0..chunks {
            let offset = c * 4;
            unsafe {
                let b_ptr = b_row.as_ptr().add(offset) as *const v128;
                let out_ptr = out_row.as_mut_ptr().add(offset) as *mut v128;
                let b_vec = v128_load(b_ptr);
                let out_vec = v128_load(out_ptr);
                let result = f32x4_add(out_vec, f32x4_mul(a_vec, b_vec));
                v128_store(out_ptr, result);
            }
        }

        for j in (chunks * 4)..n {
            out_row[j] += a_ik * b_row[j];
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 4;
        let mut acc_vec = f32x4_splat(0.0);

        for c in 0..chunks {
            let offset = c * 4;
            unsafe {
                let a_ptr = a.as_ptr().add(offset) as *const v128;
                let b_ptr = b.as_ptr().add(offset) as *const v128;
                let a_vec = v128_load(a_ptr);
                let b_vec = v128_load(b_ptr);
                acc_vec = f32x4_add(acc_vec, f32x4_mul(a_vec, b_vec));
            }
        }

        let mut sum = f32x4_extract_lane::<0>(acc_vec)
            + f32x4_extract_lane::<1>(acc_vec)
            + f32x4_extract_lane::<2>(acc_vec)
            + f32x4_extract_lane::<3>(acc_vec);
        for i in (chunks * 4)..n {
            sum += a[i] * b[i];
        }
        sum
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    #[inline]
    pub fn axpy(a_ik: f32, b_row: &[f32], out_row: &mut [f32]) {
        debug_assert_eq!(b_row.len(), out_row.len());
        for (b, o) in b_row.iter().zip(out_row.iter_mut()) {
            *o += a_ik * b;
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2() {
        // [[1, 2],     [[5, 6],     [[19, 22],
        //  [3, 4]]  @   [7, 8]]  =   [43, 50]]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape().as_slice(), &[2, 2]);
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_rect() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape().as_slice(), &[1, 1]);
        assert_eq!(c.data(), &[14.0]);
    }

    #[test]
    #[should_panic]
    fn matmul_dim_mismatch_panics() {
        let a = Tensor::from_vec(vec![1.0; 6], &[2, 3]);
        let b = Tensor::from_vec(vec![1.0; 8], &[4, 2]);
        let _ = matmul(&a, &b);
    }

    #[test]
    fn matmul_t_b_matches_explicit_transpose() {
        // a @ b.T  where a: [2, 3], b: [4, 3].
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = Tensor::from_vec(
            vec![
                1., 0., 0., // row 0
                0., 1., 0., // row 1
                0., 0., 1., // row 2
                1., 1., 1., // row 3
            ],
            &[4, 3],
        );
        let c = matmul_t_b(&a, &b);
        assert_eq!(c.shape().as_slice(), &[2, 4]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }

    /// Exercises kernel paths with sizes that are not multiples of 4 to make
    /// sure the tail loops match the vector body in both SIMD and scalar
    /// builds.
    #[test]
    fn matmul_handles_non_multiple_of_4_dims() {
        // a: [3, 5], b: [5, 7]. None of 3, 5, 7 are multiples of 4.
        let m = 3usize;
        let k = 5usize;
        let n = 7usize;
        let a: Vec<f32> = (0..m * k).map(|i| 0.1 + i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| -0.2 + i as f32 * 0.013).collect();

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for kk in 0..k {
                for j in 0..n {
                    expected[i * n + j] += a[i * k + kk] * b[kk * n + j];
                }
            }
        }

        let ta = Tensor::from_vec(a, &[m, k]);
        let tb = Tensor::from_vec(b, &[k, n]);
        let tc = matmul(&ta, &tb);
        for (got, want) in tc.data().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn matmul_t_b_handles_non_multiple_of_4_dims() {
        // a: [3, 5], b: [7, 5] -> out [3, 7].
        let m = 3usize;
        let k = 5usize;
        let n = 7usize;
        let a: Vec<f32> = (0..m * k).map(|i| 0.05 + i as f32 * 0.011).collect();
        let b: Vec<f32> = (0..n * k).map(|i| -0.3 + i as f32 * 0.017).collect();

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[j * k + kk];
                }
                expected[i * n + j] = acc;
            }
        }

        let ta = Tensor::from_vec(a, &[m, k]);
        let tb = Tensor::from_vec(b, &[n, k]);
        let tc = matmul_t_b(&ta, &tb);
        for (got, want) in tc.data().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }
}
