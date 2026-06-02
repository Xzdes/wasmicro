//! Matrix multiplication (forward inference).
//!
//! Reference implementations: naive triple loop with cache-friendly `ikj`
//! ordering. Two variants:
//!
//! * [`matmul`] — `C = A @ B`, with `A: [m, k]`, `B: [k, n]`, `C: [m, n]`.
//! * [`matmul_t_b`] — `C = A @ B^T`, with `A: [m, k]`, `B: [n, k]`, `C: [m, n]`.
//!   This matches PyTorch's `nn.Linear` weight layout (`[out, in]`), so it is
//!   the hot path for transformer inference.

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
            for j in 0..n {
                out_row[j] += a_ik * b_row[j];
            }
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
    // Row-major access on both `a` and `b` — cache-friendly.
    for i in 0..m {
        let a_row = &a_data[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = &b_data[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_row[kk] * b_row[kk];
            }
            out_row[j] = acc;
        }
    }

    Tensor::from_vec(out, &[m, n])
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
        // Expected:
        //   row 0 of a = [1, 2, 3]   dot [1,0,0]=1  [0,1,0]=2  [0,0,1]=3  [1,1,1]=6
        //   row 1 of a = [4, 5, 6]   dot [1,0,0]=4  [0,1,0]=5  [0,0,1]=6  [1,1,1]=15
        assert_eq!(c.shape().as_slice(), &[2, 4]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }
}
