//! Integration test: build a synthetic small BERT in safetensors format,
//! load it via `BertModel::from_safetensors`, run a forward pass.
//!
//! This exercises the full public path: byte buffer -> ModelFile -> BertModel
//! -> embedding. If this test breaks, every downstream user breaks.

use wasmicro::{
    models::bert::{BertConfig, BertModel},
    ModelFile,
};

fn f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Builds a safetensors file containing every tensor BERT expects.
/// `prefix` is prepended to every tensor name (joined with a `.`); pass `""`
/// for sentence-transformers-style saves with no prefix.
///
/// Weights are tiny deterministic patterns — the goal is correctness of the
/// loader + forward plumbing, not numerical accuracy.
fn synthetic_bert_safetensors(config: &BertConfig, prefix: &str) -> Vec<u8> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let p = if prefix.is_empty() {
        String::new()
    } else {
        format!("{prefix}.")
    };

    // (name, shape)
    let mut entries: Vec<(String, Vec<usize>)> = Vec::new();
    let push = |entries: &mut Vec<(String, Vec<usize>)>, name: &str, shape: Vec<usize>| {
        entries.push((format!("{p}{name}"), shape));
    };

    push(
        &mut entries,
        "embeddings.word_embeddings.weight",
        vec![config.vocab_size, h],
    );
    push(
        &mut entries,
        "embeddings.position_embeddings.weight",
        vec![config.max_position_embeddings, h],
    );
    push(
        &mut entries,
        "embeddings.token_type_embeddings.weight",
        vec![config.type_vocab_size, h],
    );
    push(&mut entries, "embeddings.LayerNorm.weight", vec![h]);
    push(&mut entries, "embeddings.LayerNorm.bias", vec![h]);

    for i in 0..config.num_hidden_layers {
        let layer_entries = [
            ("attention.self.query.weight", vec![h, h]),
            ("attention.self.query.bias", vec![h]),
            ("attention.self.key.weight", vec![h, h]),
            ("attention.self.key.bias", vec![h]),
            ("attention.self.value.weight", vec![h, h]),
            ("attention.self.value.bias", vec![h]),
            ("attention.output.dense.weight", vec![h, h]),
            ("attention.output.dense.bias", vec![h]),
            ("attention.output.LayerNorm.weight", vec![h]),
            ("attention.output.LayerNorm.bias", vec![h]),
            ("intermediate.dense.weight", vec![inter, h]),
            ("intermediate.dense.bias", vec![inter]),
            ("output.dense.weight", vec![h, inter]),
            ("output.dense.bias", vec![h]),
            ("output.LayerNorm.weight", vec![h]),
            ("output.LayerNorm.bias", vec![h]),
        ];
        for (suffix, shape) in layer_entries {
            push(&mut entries, &format!("encoder.layer.{i}.{suffix}"), shape);
        }
    }

    let mut header = String::from("{");
    let mut offset = 0usize;
    let mut payload = Vec::new();
    for (i, (name, shape)) in entries.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let n: usize = shape.iter().product();
        let shape_csv = shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // Deterministic per-role values: keep LayerNorm.weight at 1, biases
        // at 0, everything else small and positive so forward stays finite.
        let values: Vec<f32> = if name.ends_with("LayerNorm.weight") {
            vec![1.0; n]
        } else if name.ends_with(".bias") {
            vec![0.0; n]
        } else {
            vec![0.01; n]
        };
        let bytes = f32_le(&values);
        let len = bytes.len();

        header.push_str(&format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape_csv}],"data_offsets":[{offset},{end}]}}"#,
            end = offset + len,
        ));
        offset += len;
        payload.extend_from_slice(&bytes);
    }
    header.push('}');

    let header_bytes = header.into_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&payload);
    out
}

#[test]
fn bert_from_safetensors_runs_forward() {
    let config = BertConfig {
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        intermediate_size: 16,
        vocab_size: 16,
        max_position_embeddings: 32,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
    };

    let bytes = synthetic_bert_safetensors(&config, "");
    let file = ModelFile::parse(&bytes).expect("parse safetensors");
    let model = BertModel::from_safetensors(&file, config, "").expect("build BertModel");

    let ids = vec![1u32, 2, 3, 4];
    let hidden = model.forward(&ids, None);
    assert_eq!(hidden.shape().as_slice(), &[ids.len(), config.hidden_size]);
    for &v in hidden.data() {
        assert!(v.is_finite(), "non-finite hidden state: {v}");
    }

    let emb = model.embed_sentence(&ids, None, None);
    assert_eq!(emb.shape().as_slice(), &[config.hidden_size]);
    for &v in emb.data() {
        assert!(v.is_finite(), "non-finite embedding: {v}");
    }
}

#[test]
fn bert_with_prefix_loads_correctly() {
    let config = BertConfig {
        hidden_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 8,
        vocab_size: 8,
        max_position_embeddings: 16,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
    };

    // Save with `bert.` prefix on every tensor (the `transformers.BertModel`
    // save layout), then load with the same prefix.
    let bytes = synthetic_bert_safetensors(&config, "bert");
    let file = ModelFile::parse(&bytes).expect("parse prefixed safetensors");
    let model = BertModel::from_safetensors(&file, config, "bert").expect("load with prefix");

    let ids = vec![1u32, 2];
    let out = model.forward(&ids, None);
    assert_eq!(out.shape().as_slice(), &[2, config.hidden_size]);
}

#[test]
fn missing_prefix_gives_clear_error() {
    let config = BertConfig {
        hidden_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 8,
        vocab_size: 8,
        max_position_embeddings: 16,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
    };
    // File written with prefix `bert.`, but caller forgets it.
    let bytes = synthetic_bert_safetensors(&config, "bert");
    let file = ModelFile::parse(&bytes).unwrap();
    match BertModel::from_safetensors(&file, config, "") {
        Ok(_) => panic!("expected TensorNotFound"),
        Err(e) => assert_eq!(e, wasmicro::Error::TensorNotFound),
    }
}
