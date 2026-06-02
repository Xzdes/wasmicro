//! Elementwise tensor operations.
//!
//! All operations support tensors of identical shape. A single broadcast
//! pattern is supported: adding/multiplying a 1D `[n]` tensor against the
//! last dimension of a `[.., n]` tensor — this covers transformer biases.

use crate::tensor::Tensor;

/// Returns `a + b` for tensors of identical shape.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.shape().as_slice(),
        b.shape().as_slice(),
        "add: shapes must match"
    );
    let mut out = vec![0.0f32; a.numel()];
    for ((o, &x), &y) in out.iter_mut().zip(a.data()).zip(b.data()) {
        *o = x + y;
    }
    Tensor::from_vec(out, a.shape().as_slice())
}

/// Returns `a - b` for tensors of identical shape.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.shape().as_slice(),
        b.shape().as_slice(),
        "sub: shapes must match"
    );
    let mut out = vec![0.0f32; a.numel()];
    for ((o, &x), &y) in out.iter_mut().zip(a.data()).zip(b.data()) {
        *o = x - y;
    }
    Tensor::from_vec(out, a.shape().as_slice())
}

/// Returns elementwise `a * b` for tensors of identical shape.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.shape().as_slice(),
        b.shape().as_slice(),
        "mul: shapes must match"
    );
    let mut out = vec![0.0f32; a.numel()];
    for ((o, &x), &y) in out.iter_mut().zip(a.data()).zip(b.data()) {
        *o = x * y;
    }
    Tensor::from_vec(out, a.shape().as_slice())
}

/// Returns `a * scalar`.
pub fn scale(a: &Tensor, scalar: f32) -> Tensor {
    let mut out = vec![0.0f32; a.numel()];
    for (o, &x) in out.iter_mut().zip(a.data()) {
        *o = x * scalar;
    }
    Tensor::from_vec(out, a.shape().as_slice())
}

/// Adds a 1D bias `[n]` to the last dimension of `x: [.., n]`.
///
/// This is the only broadcast pattern supported — it covers the bias term in
/// `Linear` layers and the gamma/beta channels in normalization.
pub fn add_bias(x: &Tensor, bias: &Tensor) -> Tensor {
    let x_shape = x.shape().as_slice();
    let b_shape = bias.shape().as_slice();
    assert_eq!(
        b_shape.len(),
        1,
        "add_bias: bias must be 1D, got {:?}",
        b_shape
    );
    let n = b_shape[0];
    assert_eq!(
        *x_shape.last().expect("add_bias: x must be non-scalar"),
        n,
        "add_bias: last dim of x must equal bias length"
    );

    let mut out = x.data().to_vec();
    let bias_data = bias.data();
    for chunk in out.chunks_exact_mut(n) {
        for (o, &b) in chunk.iter_mut().zip(bias_data) {
            *o += b;
        }
    }
    Tensor::from_vec(out, x_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_basic() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]);
        let b = Tensor::from_vec(vec![10., 20., 30., 40.], &[2, 2]);
        let c = add(&a, &b);
        assert_eq!(c.data(), &[11., 22., 33., 44.]);
    }

    #[test]
    fn add_bias_broadcasts_last_dim() {
        // x = [[1, 2, 3], [4, 5, 6]], bias = [10, 20, 30]
        let x = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = Tensor::from_vec(vec![10., 20., 30.], &[3]);
        let y = add_bias(&x, &b);
        assert_eq!(y.shape().as_slice(), &[2, 3]);
        assert_eq!(y.data(), &[11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn scale_basic() {
        let a = Tensor::from_vec(vec![1., -2., 3.], &[3]);
        let b = scale(&a, 0.5);
        assert_eq!(b.data(), &[0.5, -1.0, 1.5]);
    }
}
