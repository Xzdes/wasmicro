//! T5 encoder-decoder transformer.
//!
//! Supports `t5-small`, `t5-base`, `t5-large`, `google/flan-t5-*`, and other
//! models that follow the HuggingFace `T5ForConditionalGeneration` weight layout.
//!
//! Key differences from BERT / GPT-2:
//! - **RMSNorm** instead of LayerNorm (no bias, no mean subtraction).
//! - **Relative position biases** instead of absolute position embeddings.
//! - **No bias** on any linear projection.
//! - **Encoder + decoder** (seq2seq). The decoder uses causal self-attention
//!   plus cross-attention to encoder output.
//! - **Gated FFN** (T5v1.1 / Flan-T5): `gelu(W0 x) ⊙ W1 x` before the
//!   output projection. Set `is_gated_act = true` for these models.

use crate::error::{Error, Result};
use crate::loader::ModelFile;
use crate::ops::activations::{gelu_tanh, relu};
use crate::ops::attention::{cross_attention_from_qkv, multi_head_attention_with_bias};
use crate::ops::elementwise::{add, mul};
use crate::ops::embedding::embedding;
use crate::ops::linear::linear;
use crate::ops::matmul::matmul_t_b;
use crate::ops::rms_norm::rms_norm;
use crate::tensor::Tensor;

/// Architectural hyperparameters for a T5 model.
#[derive(Debug, Clone, Copy)]
pub struct T5Config {
    /// Model embedding dimension (`d_model`). 512 for T5-small.
    pub d_model: usize,
    /// Feed-forward inner dimension (`d_ff`). 2048 for T5-small.
    pub d_ff: usize,
    /// Head dimension (`d_kv`). 64 for T5-small.
    pub d_kv: usize,
    /// Number of attention heads. 8 for T5-small.
    pub num_heads: usize,
    /// Number of encoder layers. 6 for T5-small.
    pub num_encoder_layers: usize,
    /// Number of decoder layers. 6 for T5-small.
    pub num_decoder_layers: usize,
    /// Vocabulary size. 32128 for T5.
    pub vocab_size: usize,
    /// Number of relative attention buckets (default 32).
    pub relative_attention_num_buckets: usize,
    /// Maximum relative distance for bucketing (default 128).
    pub relative_attention_max_distance: usize,
    /// RMSNorm epsilon. T5 uses `1e-6`.
    pub layer_norm_eps: f32,
    /// Use gated GELU FFN (true for T5v1.1 and Flan-T5, false for original T5).
    pub is_gated_act: bool,
}

impl T5Config {
    /// Config for `t5-small`.
    pub fn t5_small() -> Self {
        Self {
            d_model: 512,
            d_ff: 2048,
            d_kv: 64,
            num_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            vocab_size: 32128,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            is_gated_act: false,
        }
    }

    /// Config for `t5-base`.
    pub fn t5_base() -> Self {
        Self {
            d_model: 768,
            d_ff: 3072,
            d_kv: 64,
            num_heads: 12,
            num_encoder_layers: 12,
            num_decoder_layers: 12,
            vocab_size: 32128,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            is_gated_act: false,
        }
    }

    /// Config for `google/flan-t5-small`.
    pub fn flan_t5_small() -> Self {
        Self {
            d_model: 512,
            d_ff: 1024,
            d_kv: 64,
            num_heads: 6,
            num_encoder_layers: 8,
            num_decoder_layers: 8,
            vocab_size: 32128,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            is_gated_act: true,
        }
    }

    /// Parses a HuggingFace `config.json` string for T5.
    pub fn from_config_json(json: &str) -> Result<Self> {
        let extract_usize = |key: &str| -> Option<usize> {
            let pattern = format!("\"{key}\":");
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            let end = rest.find(|c: char| !c.is_ascii_digit())?;
            if end == 0 {
                return None;
            }
            rest[..end].parse().ok()
        };
        let extract_f32 = |key: &str| -> Option<f32> {
            let pattern = format!("\"{key}\":");
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            let end = rest
                .find(|c: char| !matches!(c, '-' | '+' | '.' | 'e' | 'E') && !c.is_ascii_digit())?;
            if end == 0 {
                return None;
            }
            rest[..end].parse().ok()
        };
        let extract_bool = |key: &str| -> Option<bool> {
            let pattern = format!("\"{key}\":");
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            if rest.starts_with("true") {
                Some(true)
            } else if rest.starts_with("false") {
                Some(false)
            } else {
                None
            }
        };

        let d_model =
            extract_usize("d_model").ok_or(Error::InvalidInput("config.json: missing d_model"))?;
        let num_layers = extract_usize("num_layers").unwrap_or(6);
        Ok(Self {
            d_model,
            d_ff: extract_usize("d_ff").unwrap_or(4 * d_model),
            d_kv: extract_usize("d_kv").unwrap_or(64),
            num_heads: extract_usize("num_heads").unwrap_or(d_model / 64),
            num_encoder_layers: extract_usize("num_encoder_layers").unwrap_or(num_layers),
            num_decoder_layers: extract_usize("num_decoder_layers").unwrap_or(num_layers),
            vocab_size: extract_usize("vocab_size")
                .ok_or(Error::InvalidInput("config.json: missing vocab_size"))?,
            relative_attention_num_buckets: extract_usize("relative_attention_num_buckets")
                .unwrap_or(32),
            relative_attention_max_distance: extract_usize("relative_attention_max_distance")
                .unwrap_or(128),
            layer_norm_eps: extract_f32("layer_norm_epsilon")
                .or_else(|| extract_f32("layer_norm_eps"))
                .unwrap_or(1e-6),
            is_gated_act: extract_bool("is_gated_act").unwrap_or(false),
        })
    }
}

// ─── weight structs ──────────────────────────────────────────────────────────

struct T5SelfAttention {
    q_w: Tensor, // [inner_dim, d_model]
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor, // [d_model, inner_dim]
    // Relative position bias — only present on the first layer.
    rel_bias: Option<Tensor>, // [num_heads, num_buckets]
}

struct T5CrossAttention {
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
}

struct T5Ffn {
    wi_w: Tensor,              // [d_ff, d_model] — or wi_0 for gated
    wi_gate_w: Option<Tensor>, // [d_ff, d_model] wi_1 for gated
    wo_w: Tensor,              // [d_model, d_ff]
}

struct T5EncoderLayer {
    self_attn: T5SelfAttention,
    self_attn_ln: Tensor,
    ffn: T5Ffn,
    ffn_ln: Tensor,
}

struct T5DecoderLayer {
    self_attn: T5SelfAttention,
    self_attn_ln: Tensor,
    cross_attn: T5CrossAttention,
    cross_attn_ln: Tensor,
    ffn: T5Ffn,
    ffn_ln: Tensor,
}

/// Full T5 encoder-decoder model.
///
/// Use [`T5Model::encode`] to get encoder representations, and
/// [`T5Model::generate_greedy`] for sequence-to-sequence generation.
pub struct T5Model {
    /// Architectural configuration.
    pub config: T5Config,
    shared_emb: Tensor, // [vocab, d_model]
    encoder_layers: Vec<T5EncoderLayer>,
    encoder_final_ln: Tensor,
    decoder_layers: Vec<T5DecoderLayer>,
    decoder_final_ln: Tensor,
    lm_head: Tensor, // [vocab, d_model]
    /// EOS / pad token id. T5 uses id 1 (`</s>`).
    pub eos_token_id: u32,
    /// Decoder start token id. T5 uses id 0 (`<pad>`).
    pub decoder_start_token_id: u32,
}

impl T5Model {
    /// Loads a T5 model from a parsed safetensors file.
    ///
    /// Tensor names follow `T5ForConditionalGeneration` layout.
    pub fn from_safetensors(file: &ModelFile, config: T5Config) -> Result<Self> {
        let load = |name: &str| -> Result<Tensor> { file.get(name)?.to_tensor() };
        let _inner_dim = config.d_kv * config.num_heads;

        let load_self_attn = |prefix: &str, layer_idx: usize| -> Result<T5SelfAttention> {
            let p = format!("{prefix}.block.{layer_idx}.layer.0.SelfAttention");
            let rel_bias = if layer_idx == 0 {
                load(&format!("{p}.relative_attention_bias.weight")).ok()
            } else {
                None
            };
            Ok(T5SelfAttention {
                q_w: load(&format!("{p}.q.weight"))?,
                k_w: load(&format!("{p}.k.weight"))?,
                v_w: load(&format!("{p}.v.weight"))?,
                o_w: load(&format!("{p}.o.weight"))?,
                rel_bias,
            })
        };

        let load_ffn = |prefix: &str, layer_idx: usize, layer_offset: usize| -> Result<T5Ffn> {
            let p = format!("{prefix}.block.{layer_idx}.layer.{layer_offset}");
            let name = if config.is_gated_act {
                "DenseActDense"
            } else {
                "DenseReluDense"
            };
            let ffn_p = format!("{p}.{name}");
            let (wi_w, wi_gate_w) = if config.is_gated_act {
                (
                    load(&format!("{ffn_p}.wi_0.weight"))?,
                    Some(load(&format!("{ffn_p}.wi_1.weight"))?),
                )
            } else {
                (load(&format!("{ffn_p}.wi.weight"))?, None)
            };
            Ok(T5Ffn {
                wi_w,
                wi_gate_w,
                wo_w: load(&format!("{ffn_p}.wo.weight"))?,
            })
        };

        let mut encoder_layers = Vec::with_capacity(config.num_encoder_layers);
        for i in 0..config.num_encoder_layers {
            encoder_layers.push(T5EncoderLayer {
                self_attn: load_self_attn("encoder", i)?,
                self_attn_ln: load(&format!("encoder.block.{i}.layer.0.layer_norm.weight"))?,
                ffn: load_ffn("encoder", i, 1)?,
                ffn_ln: load(&format!("encoder.block.{i}.layer.1.layer_norm.weight"))?,
            });
        }

        let mut decoder_layers = Vec::with_capacity(config.num_decoder_layers);
        for i in 0..config.num_decoder_layers {
            let dec_rel_bias = if i == 0 {
                load(&format!(
                    "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
                ))
                .ok()
            } else {
                None
            };
            decoder_layers.push(T5DecoderLayer {
                self_attn: T5SelfAttention {
                    q_w: load(&format!("decoder.block.{i}.layer.0.SelfAttention.q.weight"))?,
                    k_w: load(&format!("decoder.block.{i}.layer.0.SelfAttention.k.weight"))?,
                    v_w: load(&format!("decoder.block.{i}.layer.0.SelfAttention.v.weight"))?,
                    o_w: load(&format!("decoder.block.{i}.layer.0.SelfAttention.o.weight"))?,
                    rel_bias: dec_rel_bias,
                },
                self_attn_ln: load(&format!("decoder.block.{i}.layer.0.layer_norm.weight"))?,
                cross_attn: T5CrossAttention {
                    q_w: load(&format!(
                        "decoder.block.{i}.layer.1.EncDecAttention.q.weight"
                    ))?,
                    k_w: load(&format!(
                        "decoder.block.{i}.layer.1.EncDecAttention.k.weight"
                    ))?,
                    v_w: load(&format!(
                        "decoder.block.{i}.layer.1.EncDecAttention.v.weight"
                    ))?,
                    o_w: load(&format!(
                        "decoder.block.{i}.layer.1.EncDecAttention.o.weight"
                    ))?,
                },
                cross_attn_ln: load(&format!("decoder.block.{i}.layer.1.layer_norm.weight"))?,
                ffn: load_ffn("decoder", i, 2)?,
                ffn_ln: load(&format!("decoder.block.{i}.layer.2.layer_norm.weight"))?,
            });
        }

        let shared_emb = load("shared.weight")?;
        let lm_head = file
            .get("lm_head.weight")
            .ok()
            .and_then(|v| v.to_tensor().ok())
            .unwrap_or_else(|| shared_emb.clone());

        Ok(Self {
            config,
            shared_emb,
            encoder_layers,
            encoder_final_ln: load("encoder.final_layer_norm.weight")?,
            decoder_layers,
            decoder_final_ln: load("decoder.final_layer_norm.weight")?,
            lm_head,
            eos_token_id: 1,
            decoder_start_token_id: 0,
        })
    }

    /// Runs the encoder forward pass.
    ///
    /// Returns hidden states `[input_len, d_model]`.
    pub fn encode(&self, input_ids: &[u32]) -> Tensor {
        let mut h = embedding(input_ids, &self.shared_emb);
        let enc_bias = self
            .encoder_layers
            .first()
            .and_then(|l| l.self_attn.rel_bias.as_ref())
            .map(|b| {
                compute_relative_bias(
                    b,
                    input_ids.len(),
                    input_ids.len(),
                    true,
                    self.config.relative_attention_num_buckets,
                    self.config.relative_attention_max_distance,
                )
            });

        for (i, layer) in self.encoder_layers.iter().enumerate() {
            let bias = if i == 0 { enc_bias.as_ref() } else { None };
            h = encoder_layer_forward(&h, layer, &self.config, bias);
        }
        rms_norm(&h, &self.encoder_final_ln, self.config.layer_norm_eps)
    }

    /// Runs the decoder forward pass for one step, producing logits.
    ///
    /// `decoder_ids`: decoder input ids (including `decoder_start_token_id`).
    /// `encoder_output`: from [`T5Model::encode`].
    /// Returns logits `[decoder_len, vocab_size]`.
    pub fn decode_step(&self, decoder_ids: &[u32], encoder_output: &Tensor) -> Tensor {
        let mut h = embedding(decoder_ids, &self.shared_emb);
        let dec_bias = self
            .decoder_layers
            .first()
            .and_then(|l| l.self_attn.rel_bias.as_ref())
            .map(|b| {
                compute_relative_bias(
                    b,
                    decoder_ids.len(),
                    decoder_ids.len(),
                    false,
                    self.config.relative_attention_num_buckets,
                    self.config.relative_attention_max_distance,
                )
            });

        for (i, layer) in self.decoder_layers.iter().enumerate() {
            let bias = if i == 0 { dec_bias.as_ref() } else { None };
            h = decoder_layer_forward(&h, layer, encoder_output, &self.config, bias);
        }
        let h = rms_norm(&h, &self.decoder_final_ln, self.config.layer_norm_eps);
        matmul_t_b(&h, &self.lm_head)
    }

    /// Greedy sequence-to-sequence generation.
    ///
    /// Encodes `encoder_ids`, then decodes greedily up to `max_new_tokens`.
    /// Returns the generated token ids (not including the start token).
    pub fn generate_greedy(&self, encoder_ids: &[u32], max_new_tokens: usize) -> Vec<u32> {
        let encoder_out = self.encode(encoder_ids);
        let mut decoder_ids = vec![self.decoder_start_token_id];
        let mut output = Vec::new();

        for _ in 0..max_new_tokens {
            let logits = self.decode_step(&decoder_ids, &encoder_out);
            let vocab = self.config.vocab_size;
            let last_off = (decoder_ids.len() - 1) * vocab;
            let next = argmax(&logits.data()[last_off..last_off + vocab]);
            let next_id = next as u32;
            output.push(next_id);
            decoder_ids.push(next_id);
            if next_id == self.eos_token_id {
                break;
            }
        }
        output
    }
}

// ─── forward helpers ─────────────────────────────────────────────────────────

fn encoder_layer_forward(
    x: &Tensor,
    layer: &T5EncoderLayer,
    config: &T5Config,
    position_bias: Option<&Tensor>,
) -> Tensor {
    let _inner_dim = config.d_kv * config.num_heads;

    // Self-attention sublayer
    let normed = rms_norm(x, &layer.self_attn_ln, config.layer_norm_eps);
    let q = linear(&normed, &layer.self_attn.q_w, None);
    let k = linear(&normed, &layer.self_attn.k_w, None);
    let v = linear(&normed, &layer.self_attn.v_w, None);
    // Reshape to [seq, inner_dim] if needed (they already are)
    let attn = multi_head_attention_with_bias(&q, &k, &v, config.num_heads, position_bias, false);
    let attn_proj = linear(&attn, &layer.self_attn.o_w, None);
    let x = add(x, &attn_proj);

    // FFN sublayer
    let normed = rms_norm(&x, &layer.ffn_ln, config.layer_norm_eps);
    let ffn_out = ffn_forward(&normed, &layer.ffn, config);
    add(&x, &ffn_out)
}

fn decoder_layer_forward(
    x: &Tensor,
    layer: &T5DecoderLayer,
    encoder_out: &Tensor,
    config: &T5Config,
    position_bias: Option<&Tensor>,
) -> Tensor {
    // Causal self-attention
    let normed = rms_norm(x, &layer.self_attn_ln, config.layer_norm_eps);
    let q = linear(&normed, &layer.self_attn.q_w, None);
    let k = linear(&normed, &layer.self_attn.k_w, None);
    let v = linear(&normed, &layer.self_attn.v_w, None);
    let self_attn =
        multi_head_attention_with_bias(&q, &k, &v, config.num_heads, position_bias, true);
    let self_proj = linear(&self_attn, &layer.self_attn.o_w, None);
    let x = add(x, &self_proj);

    // Cross-attention (Q from decoder, K/V from encoder)
    let normed = rms_norm(&x, &layer.cross_attn_ln, config.layer_norm_eps);
    let q = linear(&normed, &layer.cross_attn.q_w, None);
    let k = linear(encoder_out, &layer.cross_attn.k_w, None);
    let v = linear(encoder_out, &layer.cross_attn.v_w, None);
    let cross = cross_attention_from_qkv(&q, &k, &v, config.num_heads);
    let cross_proj = linear(&cross, &layer.cross_attn.o_w, None);
    let x = add(&x, &cross_proj);

    // FFN
    let normed = rms_norm(&x, &layer.ffn_ln, config.layer_norm_eps);
    let ffn_out = ffn_forward(&normed, &layer.ffn, config);
    add(&x, &ffn_out)
}

fn ffn_forward(x: &Tensor, ffn: &T5Ffn, config: &T5Config) -> Tensor {
    if config.is_gated_act {
        // Gated GELU: gelu(W0 x) ⊙ W1 x
        let h0 = linear(x, &ffn.wi_w, None);
        let h0 = gelu_tanh(&h0);
        let h1 = linear(x, ffn.wi_gate_w.as_ref().unwrap(), None);
        let gated = mul(&h0, &h1);
        linear(&gated, &ffn.wo_w, None)
    } else {
        // ReLU FFN
        let h = linear(x, &ffn.wi_w, None);
        let h = relu(&h);
        linear(&h, &ffn.wo_w, None)
    }
}

// ─── T5 relative position bias ───────────────────────────────────────────────

/// Computes T5's relative position bias tensor `[num_heads, q_len, k_len]`.
///
/// `bidirectional = true` for encoder self-attention; `false` for causal
/// (decoder self-attention).
pub fn compute_relative_bias(
    bias_weight: &Tensor, // HuggingFace layout: [num_buckets, num_heads]
    q_len: usize,
    k_len: usize,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> Tensor {
    // HF stores relative_attention_bias as an Embedding [num_buckets, num_heads]
    let sh = bias_weight.shape().as_slice();
    let num_heads = sh[sh.len() - 1]; // last dim is num_heads
    let bw = bias_weight.data();
    let mut out = vec![0.0f32; num_heads * q_len * k_len];
    for qi in 0..q_len {
        for ki in 0..k_len {
            let rel = ki as i32 - qi as i32;
            let bucket = relative_position_bucket(rel, bidirectional, num_buckets, max_distance);
            for h in 0..num_heads {
                // weight layout: [bucket, head]
                out[h * q_len * k_len + qi * k_len + ki] = bw[bucket * num_heads + h];
            }
        }
    }
    Tensor::from_vec(out, &[num_heads, q_len, k_len])
}

fn relative_position_bucket(
    relative_position: i32,
    bidirectional: bool,
    mut num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut relative_buckets = 0usize;
    let abs_pos = if bidirectional {
        num_buckets /= 2;
        if relative_position > 0 {
            relative_buckets += num_buckets;
        }
        relative_position.unsigned_abs() as usize
    } else {
        // Causal: only attend to past, clamp positives to 0
        (-relative_position).max(0) as usize
    };

    let max_exact = num_buckets / 2;
    let bucket = if abs_pos < max_exact {
        abs_pos
    } else if max_exact == 0 || max_distance <= max_exact {
        max_exact
    } else {
        let log_val = (abs_pos as f32 / max_exact as f32).ln()
            / (max_distance as f32 / max_exact as f32).ln()
            * (num_buckets - max_exact) as f32;
        (max_exact + log_val as usize).min(num_buckets - 1)
    };

    relative_buckets + bucket
}

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_position_bucket_bidirectional() {
        // Identical positions: bucket 0 or num_buckets/2 offset
        let b0 = relative_position_bucket(0, true, 32, 128);
        let bpos = relative_position_bucket(1, true, 32, 128);
        let bneg = relative_position_bucket(-1, true, 32, 128);
        assert!(
            bpos > b0,
            "positive should map to higher bucket with bidirectional"
        );
        let _ = bneg;
    }

    #[test]
    fn relative_position_bucket_causal() {
        // Past positions (negative relative) get non-zero buckets
        let b0 = relative_position_bucket(0, false, 32, 128);
        let b_past = relative_position_bucket(-5, false, 32, 128);
        assert_eq!(b0, 0);
        assert!(b_past > 0);
    }

    #[test]
    fn compute_bias_shape() {
        // HuggingFace layout: [num_buckets=32, num_heads=8]
        let bw = Tensor::from_vec(vec![0.1f32; 32 * 8], &[32, 8]);
        let bias = compute_relative_bias(&bw, 4, 4, true, 32, 128);
        // Output should be [num_heads=8, q_len=4, k_len=4]
        assert_eq!(bias.shape().as_slice(), &[8, 4, 4]);
    }

    #[test]
    fn from_config_json_parses_t5_fields() {
        let json = r#"{"model_type":"t5","d_model":512,"d_ff":2048,"d_kv":64,"num_heads":8,"num_layers":6,"vocab_size":32128}"#;
        let cfg = T5Config::from_config_json(json).unwrap();
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.d_ff, 2048);
        assert_eq!(cfg.num_heads, 8);
    }

    #[test]
    fn from_config_json_flan_t5_gated() {
        let json = r#"{"model_type":"t5","d_model":512,"d_ff":1024,"d_kv":64,"num_heads":6,"num_layers":8,"vocab_size":32128,"is_gated_act":true}"#;
        let cfg = T5Config::from_config_json(json).unwrap();
        assert!(cfg.is_gated_act);
    }
}
