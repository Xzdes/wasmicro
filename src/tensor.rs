//! Plain, forward-only tensor.
//!
//! No `Rc`, no `RefCell`, no autograd state. Data is owned (`Vec<f32>`).
//! Shape is stored inline (up to 4 dimensions, which covers every transformer
//! op without a heap allocation for the shape).

use core::fmt;

/// Maximum number of dimensions stored inline. Transformer ops never exceed 4.
pub const MAX_DIMS: usize = 4;

/// Inline shape: up to `MAX_DIMS` dimensions, no heap allocation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    ndim: u8,
}

impl Shape {
    /// Creates a shape from a slice of dimensions. Panics if `dims.len() > MAX_DIMS`.
    pub fn new(dims: &[usize]) -> Self {
        assert!(
            dims.len() <= MAX_DIMS,
            "wasmicro::Shape supports up to {} dimensions, got {}",
            MAX_DIMS,
            dims.len()
        );
        let mut out = [0usize; MAX_DIMS];
        out[..dims.len()].copy_from_slice(dims);
        Self {
            dims: out,
            ndim: dims.len() as u8,
        }
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim as usize
    }

    /// Slice view of the dimensions.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.dims[..self.ndim()]
    }

    /// Total number of elements (product of dimensions).
    #[inline]
    pub fn numel(&self) -> usize {
        self.as_slice().iter().product()
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_slice())
    }
}

/// Owned 32-bit float tensor.
///
/// Designed for inference: no gradient tracking, no reference counting.
/// Cheap to move, explicit to clone.
#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    /// Creates a tensor from an owned `Vec<f32>` and shape. Panics on length mismatch.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let shape = Shape::new(shape);
        assert_eq!(
            data.len(),
            shape.numel(),
            "data length {} does not match shape {:?} (expected {} elements)",
            data.len(),
            shape.as_slice(),
            shape.numel()
        );
        Self { data, shape }
    }

    /// Allocates a tensor of zeros with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let shape = Shape::new(shape);
        Self {
            data: vec![0.0; shape.numel()],
            shape,
        }
    }

    /// Borrows the underlying data as a slice.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutably borrows the underlying data as a slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("numel", &self.numel())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_basic() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.as_slice(), &[2, 3, 4]);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn tensor_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.data(), &[0.0; 6]);
    }

    #[test]
    fn tensor_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(t.shape().as_slice(), &[2, 2]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn tensor_shape_mismatch_panics() {
        let _ = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 2]);
    }
}
