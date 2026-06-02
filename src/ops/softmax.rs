//! Numerically stable softmax along the last dimension.

use crate::tensor::Tensor;

/// Computes softmax along the last dimension of `x`.
///
/// Numerically stable: subtracts the row max before exponentiation.
/// Output has the same shape as input.
pub fn softmax_last_dim(x: &Tensor) -> Tensor {
    let shape = x.shape().as_slice();
    let n = *shape
        .last()
        .expect("softmax_last_dim: x must be non-scalar");
    let data = x.data();
    let mut out = vec![0.0f32; data.len()];

    for (in_row, out_row) in data.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        // 1. Row maximum.
        let mut max_v = f32::NEG_INFINITY;
        for &v in in_row {
            if v > max_v {
                max_v = v;
            }
        }
        // 2. Exponentiate shifted values, sum.
        let mut sum = 0.0f32;
        for (o, &v) in out_row.iter_mut().zip(in_row) {
            let e = (v - max_v).exp();
            *o = e;
            sum += e;
        }
        // 3. Normalize. Guard against degenerate sums.
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for o in out_row.iter_mut() {
            *o *= inv;
        }
    }

    Tensor::from_vec(out, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0], &[2, 3]);
        let y = softmax_last_dim(&x);
        for row in y.data().chunks_exact(3) {
            let s: f32 = row.iter().sum();
            assert!(approx_eq(s, 1.0, 1e-6), "row sum {} != 1", s);
        }
    }

    #[test]
    fn softmax_uniform_for_equal_inputs() {
        let x = Tensor::from_vec(vec![5.0; 4], &[1, 4]);
        let y = softmax_last_dim(&x);
        for &v in y.data() {
            assert!(approx_eq(v, 0.25, 1e-6));
        }
    }

    #[test]
    fn softmax_handles_large_values() {
        // Without the max-subtraction trick, exp(1000) would overflow to inf.
        let x = Tensor::from_vec(vec![1000.0, 1000.0, 1000.0], &[1, 3]);
        let y = softmax_last_dim(&x);
        for &v in y.data() {
            assert!(approx_eq(v, 1.0 / 3.0, 1e-6));
        }
    }
}
