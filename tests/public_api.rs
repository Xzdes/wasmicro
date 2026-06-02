//! Integration tests that exercise only the public API.
//!
//! These tests simulate how a downstream crate consumes `wasmicro`. They use
//! `use wasmicro::...` exclusively — no `crate::` access. If a test here
//! breaks, downstream users will also break.

use wasmicro::{ops, Dtype, ModelFile, Tensor};

/// Builds a tiny safetensors blob with two named tensors.
fn build_blob() -> Vec<u8> {
    let weight: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
    let bias: [f32; 3] = [10.0, 20.0, 30.0];

    let mut data = Vec::new();
    for v in weight.iter() {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let w_end = data.len();
    for v in bias.iter() {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let b_end = data.len();

    let header = format!(
        r#"{{"linear.weight":{{"dtype":"F32","shape":[3,2],"data_offsets":[0,{w_end}]}},"linear.bias":{{"dtype":"F32","shape":[3],"data_offsets":[{w_end},{b_end}]}}}}"#,
    );
    let header = header.into_bytes();

    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

#[test]
fn end_to_end_linear_layer() {
    let bytes = build_blob();
    let model = ModelFile::parse(&bytes).expect("parse");

    let names: Vec<&str> = model.names().collect();
    assert!(names.contains(&"linear.weight"));
    assert!(names.contains(&"linear.bias"));

    let weight = model.get("linear.weight").unwrap();
    assert_eq!(weight.dtype, Dtype::F32);
    assert_eq!(weight.shape, &[3, 2]);

    let weight = weight.to_tensor().unwrap();
    let bias = model.get("linear.bias").unwrap().to_tensor().unwrap();

    // x = [[1, 2]] -> linear(x) = x @ W^T + b
    //   = [1*1+2*0, 1*0+2*1, 1*1+2*1] + [10, 20, 30]
    //   = [1, 2, 3] + [10, 20, 30] = [11, 22, 33]
    let x = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
    let y = ops::linear::linear(&x, &weight, Some(&bias));
    assert_eq!(y.shape().as_slice(), &[1, 3]);
    assert_eq!(y.data(), &[11.0, 22.0, 33.0]);
}

#[test]
fn missing_tensor_is_a_clean_error() {
    let bytes = build_blob();
    let model = ModelFile::parse(&bytes).unwrap();
    let err = model.get("does.not.exist").unwrap_err();
    assert_eq!(err, wasmicro::Error::TensorNotFound);
}

#[test]
fn softmax_then_argmax_pipeline() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 0.5, 3.5, -1.0], &[1, 5]);
    let probs = ops::softmax::softmax_last_dim(&logits);

    // Argmax = index of largest value (3.5 at index 3).
    let (argmax, _) = probs
        .data()
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
            if v > best_v { (i, v) } else { (best_i, best_v) }
        });
    assert_eq!(argmax, 3);

    let sum: f32 = probs.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}
