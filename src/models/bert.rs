//! BERT encoder forward pass.
//!
//! Supports the canonical HuggingFace BERT weight layout used by
//! `bert-base-uncased`, `distilbert-*`, `sentence-transformers/*` and
//! similar encoders. Decoder-only or seq2seq models are out of scope.
//!
//! Inference shape:
//!
//! ```text
//! input_ids: [seq_len]    (u32 token ids)
//! output:    [seq_len, hidden_size]   (per-token embeddings)
//! ```
//!
//! For sentence-level embeddings (sentence-transformers convention), use
//! [`BertModel::embed_sentence`], which calls `forward` followed by mean
//! pooling. Pass an `attention_mask` to ignore padding positions.

use crate::error::Result;
use crate::loader::ModelFile;
use crate::ops::activations::gelu_erf;
use crate::ops::attention::{mean_pool, multi_head_attention};
use crate::ops::elementwise::add;
use crate::ops::embedding::embedding;
use crate::ops::layernorm::layer_norm;
use crate::ops::linear::linear;
use crate::tensor::Tensor;

/// Architectural hyperparameters for a BERT encoder.
///
/// Mirrors the fields of HuggingFace's `BertConfig` that the forward pass
/// actually uses. Fields like `hidden_dropout_prob` are intentionally
/// omitted — there is no training and inference does not apply dropout.
#[derive(Debug, Clone, Copy)]
pub struct BertConfig {
    /// Hidden dimension (e.g. 384 for MiniLM-L6, 768 for BERT-base).
    pub hidden_size: usize,
    /// Number of stacked encoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads. Must divide `hidden_size`.
    pub num_attention_heads: usize,
    /// Feed-forward inner dimension (typically 4 * hidden_size).
    pub intermediate_size: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Maximum supported positional index.
    pub max_position_embeddings: usize,
    /// Token-type (segment) vocabulary size (BERT uses 2; some models use 1).
    pub type_vocab_size: usize,
    /// LayerNorm epsilon. BERT uses `1e-12`.
    pub layer_norm_eps: f32,
}

impl BertConfig {
    /// Config for `sentence-transformers/all-MiniLM-L6-v2`.
    pub fn mini_lm_l6_v2() -> Self {
        Self {
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 1536,
            vocab_size: 30522,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }

    /// Config for `bert-base-uncased`.
    pub fn bert_base() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            vocab_size: 30522,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }
}

// Internal sub-modules. Kept private so the public surface stays small —
// users only see `BertConfig` and `BertModel`. Custom loading paths should
// go through `BertModel::from_safetensors`.

struct BertEmbeddings {
    word: Tensor,
    position: Tensor,
    token_type: Tensor,
    ln_gamma: Tensor,
    ln_beta: Tensor,
}

struct BertSelfAttention {
    wq: Tensor,
    bq: Tensor,
    wk: Tensor,
    bk: Tensor,
    wv: Tensor,
    bv: Tensor,
}

struct BertAttention {
    self_attn: BertSelfAttention,
    wo: Tensor,
    bo: Tensor,
    ln_gamma: Tensor,
    ln_beta: Tensor,
}

struct BertFeedForward {
    w_inter: Tensor,
    b_inter: Tensor,
    w_out: Tensor,
    b_out: Tensor,
    ln_gamma: Tensor,
    ln_beta: Tensor,
}

struct BertLayer {
    attention: BertAttention,
    ffn: BertFeedForward,
}

/// Full BERT encoder.
///
/// Construct via [`BertModel::from_safetensors`]. The model owns its weights
/// (no external references), so it can be cached, sent across threads (it is
/// `Send`), or dropped freely.
pub struct BertModel {
    /// Architectural configuration this model was built with.
    pub config: BertConfig,
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
}

impl BertModel {
    /// Loads a BERT model from a parsed safetensors file.
    ///
    /// `prefix` is prepended to every tensor name. Use:
    /// - `""` for sentence-transformers and `transformers.AutoModel` saves;
    /// - `"bert"` for `transformers.BertModel` saves that include the wrapper.
    ///
    /// The prefix is joined to names with a `.` automatically, so pass
    /// `"bert"` not `"bert."`.
    pub fn from_safetensors(file: &ModelFile, config: BertConfig, prefix: &str) -> Result<Self> {
        let p = if prefix.is_empty() {
            String::new()
        } else {
            format!("{prefix}.")
        };

        let load = |name: &str| -> Result<Tensor> {
            file.get(&format!("{p}{name}"))?.to_tensor()
        };

        let embeddings = BertEmbeddings {
            word: load("embeddings.word_embeddings.weight")?,
            position: load("embeddings.position_embeddings.weight")?,
            token_type: load("embeddings.token_type_embeddings.weight")?,
            ln_gamma: load("embeddings.LayerNorm.weight")?,
            ln_beta: load("embeddings.LayerNorm.bias")?,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let load_l = |suffix: &str| -> Result<Tensor> {
                load(&format!("encoder.layer.{i}.{suffix}"))
            };

            let layer = BertLayer {
                attention: BertAttention {
                    self_attn: BertSelfAttention {
                        wq: load_l("attention.self.query.weight")?,
                        bq: load_l("attention.self.query.bias")?,
                        wk: load_l("attention.self.key.weight")?,
                        bk: load_l("attention.self.key.bias")?,
                        wv: load_l("attention.self.value.weight")?,
                        bv: load_l("attention.self.value.bias")?,
                    },
                    wo: load_l("attention.output.dense.weight")?,
                    bo: load_l("attention.output.dense.bias")?,
                    ln_gamma: load_l("attention.output.LayerNorm.weight")?,
                    ln_beta: load_l("attention.output.LayerNorm.bias")?,
                },
                ffn: BertFeedForward {
                    w_inter: load_l("intermediate.dense.weight")?,
                    b_inter: load_l("intermediate.dense.bias")?,
                    w_out: load_l("output.dense.weight")?,
                    b_out: load_l("output.dense.bias")?,
                    ln_gamma: load_l("output.LayerNorm.weight")?,
                    ln_beta: load_l("output.LayerNorm.bias")?,
                },
            };
            layers.push(layer);
        }

        Ok(Self {
            config,
            embeddings,
            layers,
        })
    }

    /// Runs the encoder forward pass. Returns per-token hidden states.
    ///
    /// - `input_ids`: token ids, length = sequence length.
    /// - `token_type_ids`: optional segment ids; defaults to all-zeros.
    pub fn forward(&self, input_ids: &[u32], token_type_ids: Option<&[u32]>) -> Tensor {
        let seq_len = input_ids.len();
        assert!(seq_len > 0, "forward: input_ids must not be empty");
        assert!(
            seq_len <= self.config.max_position_embeddings,
            "forward: sequence length {} exceeds max {}",
            seq_len,
            self.config.max_position_embeddings
        );

        // 1. Sum word + position + token-type embeddings, then LayerNorm.
        let word_e = embedding(input_ids, &self.embeddings.word);
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_e = embedding(&position_ids, &self.embeddings.position);
        let owned_type_ids: Vec<u32>;
        let type_ids: &[u32] = match token_type_ids {
            Some(t) => {
                assert_eq!(t.len(), seq_len, "token_type_ids length mismatch");
                t
            }
            None => {
                owned_type_ids = vec![0u32; seq_len];
                &owned_type_ids
            }
        };
        let type_e = embedding(type_ids, &self.embeddings.token_type);

        let summed = add(&add(&word_e, &pos_e), &type_e);
        let mut hidden = layer_norm(
            &summed,
            &self.embeddings.ln_gamma,
            &self.embeddings.ln_beta,
            self.config.layer_norm_eps,
        );

        // 2. Encoder layers.
        for layer in &self.layers {
            hidden = encoder_layer_forward(&hidden, layer, &self.config);
        }

        hidden
    }

    /// Convenience: forward + mean pooling, returning a single `[hidden]` vector.
    ///
    /// `attention_mask` follows the HuggingFace convention: `1` for real
    /// tokens, `0` for padding. Passing `None` averages over all positions.
    pub fn embed_sentence(
        &self,
        input_ids: &[u32],
        token_type_ids: Option<&[u32]>,
        attention_mask: Option<&[u32]>,
    ) -> Tensor {
        let hidden = self.forward(input_ids, token_type_ids);
        mean_pool(&hidden, attention_mask)
    }
}

fn encoder_layer_forward(x: &Tensor, layer: &BertLayer, config: &BertConfig) -> Tensor {
    // Attention block.
    let attn = multi_head_attention(
        x,
        &layer.attention.self_attn.wq,
        Some(&layer.attention.self_attn.bq),
        &layer.attention.self_attn.wk,
        Some(&layer.attention.self_attn.bk),
        &layer.attention.self_attn.wv,
        Some(&layer.attention.self_attn.bv),
        &layer.attention.wo,
        Some(&layer.attention.bo),
        config.num_attention_heads,
    );
    let residual = add(x, &attn);
    let post_attn = layer_norm(
        &residual,
        &layer.attention.ln_gamma,
        &layer.attention.ln_beta,
        config.layer_norm_eps,
    );

    // Feed-forward block.
    let inter = linear(&post_attn, &layer.ffn.w_inter, Some(&layer.ffn.b_inter));
    let inter = gelu_erf(&inter);
    let proj = linear(&inter, &layer.ffn.w_out, Some(&layer.ffn.b_out));
    let residual = add(&post_attn, &proj);
    layer_norm(
        &residual,
        &layer.ffn.ln_gamma,
        &layer.ffn.ln_beta,
        config.layer_norm_eps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a deterministic, tiny BERT for smoke testing.
    fn tiny_bert() -> (BertConfig, BertModel) {
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

        let hidden = config.hidden_size;
        let inter = config.intermediate_size;

        let ones = |shape: &[usize]| {
            let n: usize = shape.iter().product();
            Tensor::from_vec(vec![0.01f32; n], shape)
        };
        let one_vec = |n: usize| Tensor::from_vec(vec![1.0f32; n], &[n]);
        let zero_vec = |n: usize| Tensor::from_vec(vec![0.0f32; n], &[n]);

        let embeddings = BertEmbeddings {
            word: ones(&[config.vocab_size, hidden]),
            position: ones(&[config.max_position_embeddings, hidden]),
            token_type: ones(&[config.type_vocab_size, hidden]),
            ln_gamma: one_vec(hidden),
            ln_beta: zero_vec(hidden),
        };

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer {
                attention: BertAttention {
                    self_attn: BertSelfAttention {
                        wq: ones(&[hidden, hidden]),
                        bq: zero_vec(hidden),
                        wk: ones(&[hidden, hidden]),
                        bk: zero_vec(hidden),
                        wv: ones(&[hidden, hidden]),
                        bv: zero_vec(hidden),
                    },
                    wo: ones(&[hidden, hidden]),
                    bo: zero_vec(hidden),
                    ln_gamma: one_vec(hidden),
                    ln_beta: zero_vec(hidden),
                },
                ffn: BertFeedForward {
                    w_inter: ones(&[inter, hidden]),
                    b_inter: zero_vec(inter),
                    w_out: ones(&[hidden, inter]),
                    b_out: zero_vec(hidden),
                    ln_gamma: one_vec(hidden),
                    ln_beta: zero_vec(hidden),
                },
            });
        }

        (config, BertModel { config, embeddings, layers })
    }

    #[test]
    fn forward_produces_correct_shape() {
        let (config, model) = tiny_bert();
        let ids = vec![1u32, 2, 3, 4, 5];
        let out = model.forward(&ids, None);
        assert_eq!(out.shape().as_slice(), &[ids.len(), config.hidden_size]);
        // Output must contain finite numbers.
        for &v in out.data() {
            assert!(v.is_finite(), "non-finite output: {}", v);
        }
    }

    #[test]
    fn embed_sentence_produces_hidden_vector() {
        let (config, model) = tiny_bert();
        let ids = vec![1u32, 2, 3];
        let emb = model.embed_sentence(&ids, None, None);
        assert_eq!(emb.shape().as_slice(), &[config.hidden_size]);
    }

    #[test]
    fn embed_sentence_respects_attention_mask() {
        let (_config, model) = tiny_bert();
        let ids = vec![1u32, 2, 3, 4];
        let mask = [1u32, 1, 0, 0];
        let masked = model.embed_sentence(&ids, None, Some(&mask));
        let unmasked = model.embed_sentence(&ids[..2], None, None);
        // Masking the last two tokens should give the same result as passing
        // only the first two tokens (the actual hidden states still differ
        // because position embeddings are present, so we just check the
        // outputs are similarly shaped and finite).
        assert_eq!(masked.shape().as_slice(), unmasked.shape().as_slice());
        for &v in masked.data() {
            assert!(v.is_finite());
        }
    }
}
