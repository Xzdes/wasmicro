//! Embedding lookup: index a weight matrix by token ids.

use crate::tensor::Tensor;

/// Looks up rows of the `weight` matrix for each id in `ids`.
///
/// - `ids`: flat slice of token indices.
/// - `weight`: shape `[vocab_size, embed_dim]`.
/// - returned tensor shape: `[ids.len(), embed_dim]`.
///
/// Panics if any id is out of range or if `weight` is not 2D.
pub fn embedding(ids: &[u32], weight: &Tensor) -> Tensor {
    let w_shape = weight.shape().as_slice();
    assert_eq!(
        w_shape.len(),
        2,
        "embedding: weight must be 2D, got {:?}",
        w_shape
    );
    let vocab_size = w_shape[0];
    let embed_dim = w_shape[1];

    let w_data = weight.data();
    let mut out = vec![0.0f32; ids.len() * embed_dim];

    for (out_row, &id) in out.chunks_exact_mut(embed_dim).zip(ids.iter()) {
        let id = id as usize;
        assert!(
            id < vocab_size,
            "embedding: id {} out of range [0, {})",
            id,
            vocab_size
        );
        let src = &w_data[id * embed_dim..(id + 1) * embed_dim];
        out_row.copy_from_slice(src);
    }

    Tensor::from_vec(out, &[ids.len(), embed_dim])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_picks_correct_rows() {
        // weight = [[1, 2], [3, 4], [5, 6]] (vocab=3, embed=2)
        let w = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2]);
        let ids = [2u32, 0, 1, 1];
        let out = embedding(&ids, &w);
        assert_eq!(out.shape().as_slice(), &[4, 2]);
        assert_eq!(out.data(), &[5., 6., 1., 2., 3., 4., 3., 4.]);
    }

    #[test]
    #[should_panic]
    fn embedding_panics_on_out_of_range() {
        let w = Tensor::from_vec(vec![1., 2.], &[1, 2]);
        let _ = embedding(&[5], &w);
    }
}
