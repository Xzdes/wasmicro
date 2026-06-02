//! Linear (fully-connected) layer.
//!
//! Convention matches PyTorch: weights are stored as `[out_features, in_features]`,
//! so the operation is `y = x @ W^T + b`.

use crate::ops::elementwise::add_bias;
use crate::ops::matmul::matmul_t_b;
use crate::tensor::Tensor;

/// Applies a linear transformation: `y = x @ W^T + b`.
///
/// - `x`: shape `[m, in_features]`
/// - `weight`: shape `[out_features, in_features]` (PyTorch layout)
/// - `bias`: optional, shape `[out_features]`
/// - returned tensor shape: `[m, out_features]`
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let y = matmul_t_b(x, weight);
    match bias {
        Some(b) => add_bias(&y, b),
        None => y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_with_bias() {
        // x: [1, 2] = [[1, 2]]
        // W: [3, 2] = [[1, 0], [0, 1], [1, 1]]  (out=3, in=2)
        // b: [3]   = [10, 20, 30]
        // expected y = x @ W^T + b
        //   = [1*1 + 2*0, 1*0 + 2*1, 1*1 + 2*1] + [10, 20, 30]
        //   = [1, 2, 3] + [10, 20, 30] = [11, 22, 33]
        let x = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        let w = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1.], &[3, 2]);
        let b = Tensor::from_vec(vec![10., 20., 30.], &[3]);
        let y = linear(&x, &w, Some(&b));
        assert_eq!(y.shape().as_slice(), &[1, 3]);
        assert_eq!(y.data(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn linear_without_bias() {
        let x = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        let w = Tensor::from_vec(vec![1., 0., 0., 1.], &[2, 2]);
        let y = linear(&x, &w, None);
        assert_eq!(y.data(), &[1.0, 2.0]);
    }
}
