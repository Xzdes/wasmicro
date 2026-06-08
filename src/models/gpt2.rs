//! GPT-2 decoder-only transformer.
//!
//! Supports loading from HuggingFace `GPT2LMHeadModel` safetensors saves.
//! GPT-2 uses `Conv1D` internally, which stores weights as `[in, out]`;
//! this loader transposes them to `[out, in]` at load time so the same
//! `matmul_t_b` path used by BERT can be reused throughout.
//!
//! # Generation
//!
//! [`Gpt2Model::generate_greedy`] runs greedy (argmax) decoding without a
//! KV cache — it re-runs the full forward pass for each new token.
//! This is simple and correct; add a KV cache if generation speed matters.

use crate::error::{Error, Result};
use crate::loader::ModelFile;
use crate::ops::activations::gelu_tanh;
use crate::ops::attention::causal_multi_head_attention_from_qkv;
use crate::ops::elementwise::add;
use crate::ops::embedding::embedding;
use crate::ops::layernorm::layer_norm;
use crate::ops::linear::linear;
use crate::ops::matmul::matmul_t_b;
use crate::tensor::Tensor;

/// Architectural hyperparameters for a GPT-2 model.
#[derive(Debug, Clone, Copy)]
pub struct Gpt2Config {
    /// Embedding / hidden dimension (`n_embd`). 768 for GPT-2 small.
    pub hidden_size: usize,
    /// Number of transformer blocks (`n_layer`). 12 for GPT-2 small.
    pub num_hidden_layers: usize,
    /// Number of attention heads (`n_head`). 12 for GPT-2 small.
    pub num_attention_heads: usize,
    /// FFN inner dimension. Typically `4 * hidden_size`.
    pub intermediate_size: usize,
    /// Vocabulary size. 50257 for GPT-2.
    pub vocab_size: usize,
    /// Maximum supported sequence length. 1024 for GPT-2.
    pub max_position_embeddings: usize,
    /// LayerNorm epsilon. GPT-2 uses `1e-5`.
    pub layer_norm_eps: f32,
}

impl Gpt2Config {
    /// Config for the original `gpt2` (117 M parameters).
    pub fn gpt2_small() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            layer_norm_eps: 1e-5,
        }
    }

    /// Config for `gpt2-medium` (345 M parameters).
    pub fn gpt2_medium() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            layer_norm_eps: 1e-5,
        }
    }

    /// Parses a HuggingFace `config.json` string.
    pub fn from_config_json(json: &str) -> Result<Self> {
        let extract_usize = |key: &str| -> Option<usize> {
            let pattern = format!("\"{key}\":");
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            let end = rest.find(|c: char| !c.is_ascii_digit())?;
            if end == 0 { return None; }
            rest[..end].parse().ok()
        };
        let extract_f32 = |key: &str| -> Option<f32> {
            let pattern = format!("\"{key}\":");
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            let end = rest.find(|c: char| {
                !matches!(c, '-' | '+' | '.' | 'e' | 'E') && !c.is_ascii_digit()
            })?;
            if end == 0 { return None; }
            rest[..end].parse().ok()
        };

        // n_embd / n_head / n_layer are HF's field names for GPT-2.
        let hidden_size = extract_usize("n_embd")
            .or_else(|| extract_usize("hidden_size"))
            .ok_or(Error::InvalidInput("config.json: missing n_embd/hidden_size"))?;
        let num_hidden_layers = extract_usize("n_layer")
            .or_else(|| extract_usize("num_hidden_layers"))
            .ok_or(Error::InvalidInput("config.json: missing n_layer/num_hidden_layers"))?;
        let num_attention_heads = extract_usize("n_head")
            .or_else(|| extract_usize("num_attention_heads"))
            .ok_or(Error::InvalidInput("config.json: missing n_head/num_attention_heads"))?;

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size: extract_usize("n_inner")
                .unwrap_or(4 * hidden_size),
            vocab_size: extract_usize("vocab_size")
                .ok_or(Error::InvalidInput("config.json: missing vocab_size"))?,
            max_position_embeddings: extract_usize("n_positions")
                .or_else(|| extract_usize("max_position_embeddings"))
                .unwrap_or(1024),
            layer_norm_eps: extract_f32("layer_norm_epsilon")
                .or_else(|| extract_f32("layer_norm_eps"))
                .unwrap_or(1e-5),
        })
    }
}

struct Gpt2Attention {
    c_attn_w: Tensor, // [3*hidden, hidden]  (transposed from Conv1D [hidden, 3*hidden])
    c_attn_b: Tensor, // [3*hidden]
    c_proj_w: Tensor, // [hidden, hidden]     (transposed)
    c_proj_b: Tensor, // [hidden]
}

struct Gpt2Mlp {
    fc_w: Tensor,   // [4*hidden, hidden]  (transposed)
    fc_b: Tensor,   // [4*hidden]
    proj_w: Tensor, // [hidden, 4*hidden]  (transposed)
    proj_b: Tensor, // [hidden]
}

struct Gpt2Layer {
    ln_1_w: Tensor,
    ln_1_b: Tensor,
    attn: Gpt2Attention,
    ln_2_w: Tensor,
    ln_2_b: Tensor,
    mlp: Gpt2Mlp,
}

/// Full GPT-2 decoder.
///
/// Construct via [`Gpt2Model::from_safetensors`]. The model owns its weights.
pub struct Gpt2Model {
    /// Architectural config.
    pub config: Gpt2Config,
    wte: Tensor,      // token embeddings [vocab, hidden]
    wpe: Tensor,      // position embeddings [max_pos, hidden]
    layers: Vec<Gpt2Layer>,
    ln_f_w: Tensor,   // final layer norm weight
    ln_f_b: Tensor,   // final layer norm bias
    lm_head: Tensor,  // [vocab, hidden] (may be tied with wte)
    /// EOS token id. GPT-2 uses `<|endoftext|>` (id 50256).
    pub eos_token_id: Option<u32>,
}

impl Gpt2Model {
    /// Loads a GPT-2 model from a parsed safetensors file.
    ///
    /// Supports two HuggingFace naming conventions:
    /// - **With prefix**: `transformer.wte.weight`, `transformer.h.{i}.attn.c_attn.weight`, …
    ///   (used by some checkpoints such as `gpt2-medium`)
    /// - **Without prefix**: `wte.weight`, `h.{i}.attn.c_attn.weight`, …
    ///   (used by `openai-community/gpt2` and most current exports)
    ///
    /// The prefix is detected automatically by probing `transformer.wte.weight`.
    pub fn from_safetensors(file: &ModelFile, config: Gpt2Config) -> Result<Self> {
        let load = |name: &str| -> Result<Tensor> { file.get(name)?.to_tensor() };

        // Conv1D weights are stored [in, out] — transpose to [out, in] for matmul_t_b.
        let load_conv = |name: &str| -> Result<Tensor> {
            let t = load(name)?;
            Ok(transpose_2d(&t))
        };

        // Auto-detect whether weights include the "transformer." prefix.
        let pfx: &str = if file.get("transformer.wte.weight").is_ok() {
            "transformer."
        } else {
            ""
        };

        let wte = load(&format!("{pfx}wte.weight"))?;
        let wpe = load(&format!("{pfx}wpe.weight"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("{pfx}h.{i}");
            layers.push(Gpt2Layer {
                ln_1_w: load(&format!("{p}.ln_1.weight"))?,
                ln_1_b: load(&format!("{p}.ln_1.bias"))?,
                attn: Gpt2Attention {
                    c_attn_w: load_conv(&format!("{p}.attn.c_attn.weight"))?,
                    c_attn_b: load(&format!("{p}.attn.c_attn.bias"))?,
                    c_proj_w: load_conv(&format!("{p}.attn.c_proj.weight"))?,
                    c_proj_b: load(&format!("{p}.attn.c_proj.bias"))?,
                },
                ln_2_w: load(&format!("{p}.ln_2.weight"))?,
                ln_2_b: load(&format!("{p}.ln_2.bias"))?,
                mlp: Gpt2Mlp {
                    fc_w: load_conv(&format!("{p}.mlp.c_fc.weight"))?,
                    fc_b: load(&format!("{p}.mlp.c_fc.bias"))?,
                    proj_w: load_conv(&format!("{p}.mlp.c_proj.weight"))?,
                    proj_b: load(&format!("{p}.mlp.c_proj.bias"))?,
                },
            });
        }

        let ln_f_w = load(&format!("{pfx}ln_f.weight"))?;
        let ln_f_b = load(&format!("{pfx}ln_f.bias"))?;

        // lm_head may or may not be saved — tied to wte if absent.
        let lm_head = file
            .get("lm_head.weight")
            .ok()
            .and_then(|v| v.to_tensor().ok())
            .unwrap_or_else(|| wte.clone());

        Ok(Self {
            config,
            wte,
            wpe,
            layers,
            ln_f_w,
            ln_f_b,
            lm_head,
            eos_token_id: Some(50256),
        })
    }

    /// Runs the transformer and returns hidden states `[seq_len, hidden_size]`.
    pub fn forward(&self, input_ids: &[u32]) -> Tensor {
        let seq_len = input_ids.len();
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();

        let mut h = add(
            &embedding(input_ids, &self.wte),
            &embedding(&pos_ids, &self.wpe),
        );

        for layer in &self.layers {
            // Pre-norm → attention → residual
            let ln1 = layer_norm(&h, &layer.ln_1_w, &layer.ln_1_b, self.config.layer_norm_eps);
            let qkv = linear(&ln1, &layer.attn.c_attn_w, Some(&layer.attn.c_attn_b));
            let (q, k, v) = split_qkv(&qkv, self.config.hidden_size);
            let attn_out = causal_multi_head_attention_from_qkv(&q, &k, &v, self.config.num_attention_heads);
            let attn_proj = linear(&attn_out, &layer.attn.c_proj_w, Some(&layer.attn.c_proj_b));
            h = add(&h, &attn_proj);

            // Pre-norm → FFN → residual
            let ln2 = layer_norm(&h, &layer.ln_2_w, &layer.ln_2_b, self.config.layer_norm_eps);
            let fc = linear(&ln2, &layer.mlp.fc_w, Some(&layer.mlp.fc_b));
            let act = gelu_tanh(&fc);
            let proj = linear(&act, &layer.mlp.proj_w, Some(&layer.mlp.proj_b));
            h = add(&h, &proj);
        }

        layer_norm(&h, &self.ln_f_w, &self.ln_f_b, self.config.layer_norm_eps)
    }

    /// Returns next-token logits for every position: shape `[seq_len, vocab_size]`.
    pub fn logits(&self, input_ids: &[u32]) -> Tensor {
        let hidden = self.forward(input_ids);
        matmul_t_b(&hidden, &self.lm_head)
    }

    /// Returns the next-token logits for the last input position: shape `[vocab_size]`.
    pub fn next_token_logits(&self, input_ids: &[u32]) -> Tensor {
        let all = self.logits(input_ids);
        let vocab = self.config.vocab_size;
        let last_off = (input_ids.len() - 1) * vocab;
        Tensor::from_vec(
            all.data()[last_off..last_off + vocab].to_vec(),
            &[vocab],
        )
    }

    /// Greedy autoregressive generation.
    ///
    /// Appends tokens to `prompt_ids` one at a time until `max_new_tokens`
    /// are generated or the EOS token is produced. Returns the full sequence
    /// (prompt + generated tokens).
    ///
    /// Uses full re-computation each step (no KV cache). For short outputs
    /// and small models this is practical in WASM.
    pub fn generate_greedy(&self, prompt_ids: &[u32], max_new_tokens: usize) -> Vec<u32> {
        let mut ids = prompt_ids.to_vec();
        for _ in 0..max_new_tokens {
            if ids.len() >= self.config.max_position_embeddings {
                break;
            }
            let logits = self.next_token_logits(&ids);
            let next = argmax(logits.data());
            ids.push(next as u32);
            if self.eos_token_id == Some(next as u32) {
                break;
            }
        }
        ids
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

/// Splits a `[seq, 3*hidden]` tensor into three `[seq, hidden]` tensors (Q, K, V).
fn split_qkv(qkv: &Tensor, hidden: usize) -> (Tensor, Tensor, Tensor) {
    let seq = qkv.shape().as_slice()[0];
    let data = qkv.data();
    let stride = 3 * hidden;
    let mut q = Vec::with_capacity(seq * hidden);
    let mut k = Vec::with_capacity(seq * hidden);
    let mut v = Vec::with_capacity(seq * hidden);
    for t in 0..seq {
        let off = t * stride;
        q.extend_from_slice(&data[off..off + hidden]);
        k.extend_from_slice(&data[off + hidden..off + 2 * hidden]);
        v.extend_from_slice(&data[off + 2 * hidden..off + 3 * hidden]);
    }
    (
        Tensor::from_vec(q, &[seq, hidden]),
        Tensor::from_vec(k, &[seq, hidden]),
        Tensor::from_vec(v, &[seq, hidden]),
    )
}

/// Transposes a 2-D tensor: `[m, n]` → `[n, m]`.
fn transpose_2d(t: &Tensor) -> Tensor {
    let s = t.shape().as_slice();
    assert_eq!(s.len(), 2, "transpose_2d: expected 2D tensor");
    let (m, n) = (s[0], s[1]);
    let src = t.data();
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = src[i * n + j];
        }
    }
    Tensor::from_vec(out, &[n, m])
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

    fn tiny_config() -> Gpt2Config {
        Gpt2Config {
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 16,
            layer_norm_eps: 1e-5,
        }
    }

    fn ones(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec(vec![0.01f32; n], shape)
    }

    fn build_tiny_gpt2() -> Gpt2Model {
        let config = tiny_config();
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let v = config.vocab_size;
        let max_pos = config.max_position_embeddings;

        let layers = (0..config.num_hidden_layers)
            .map(|_| Gpt2Layer {
                ln_1_w: Tensor::from_vec(vec![1.0f32; h], &[h]),
                ln_1_b: Tensor::from_vec(vec![0.0f32; h], &[h]),
                attn: Gpt2Attention {
                    c_attn_w: ones(&[3 * h, h]),
                    c_attn_b: Tensor::from_vec(vec![0.0f32; 3 * h], &[3 * h]),
                    c_proj_w: ones(&[h, h]),
                    c_proj_b: Tensor::from_vec(vec![0.0f32; h], &[h]),
                },
                ln_2_w: Tensor::from_vec(vec![1.0f32; h], &[h]),
                ln_2_b: Tensor::from_vec(vec![0.0f32; h], &[h]),
                mlp: Gpt2Mlp {
                    fc_w: ones(&[inter, h]),
                    fc_b: Tensor::from_vec(vec![0.0f32; inter], &[inter]),
                    proj_w: ones(&[h, inter]),
                    proj_b: Tensor::from_vec(vec![0.0f32; h], &[h]),
                },
            })
            .collect();

        Gpt2Model {
            config,
            wte: ones(&[v, h]),
            wpe: ones(&[max_pos, h]),
            layers,
            ln_f_w: Tensor::from_vec(vec![1.0f32; h], &[h]),
            ln_f_b: Tensor::from_vec(vec![0.0f32; h], &[h]),
            lm_head: ones(&[v, h]),
            eos_token_id: Some(50256),
        }
    }

    #[test]
    fn forward_shape() {
        let model = build_tiny_gpt2();
        let ids = vec![1u32, 2, 3];
        let out = model.forward(&ids);
        assert_eq!(out.shape().as_slice(), &[3, model.config.hidden_size]);
        for &v in out.data() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn logits_shape() {
        let model = build_tiny_gpt2();
        let ids = vec![1u32, 2];
        let out = model.logits(&ids);
        assert_eq!(
            out.shape().as_slice(),
            &[2, model.config.vocab_size]
        );
    }

    #[test]
    fn generate_greedy_extends_sequence() {
        let model = build_tiny_gpt2();
        let prompt = vec![1u32, 2];
        let out = model.generate_greedy(&prompt, 3);
        assert!(out.len() >= prompt.len());
        assert!(out.len() <= prompt.len() + 3);
    }

    #[test]
    fn from_config_json_parses_gpt2_fields() {
        let json = r#"{"model_type":"gpt2","n_embd":768,"n_layer":12,"n_head":12,"vocab_size":50257,"n_positions":1024}"#;
        let cfg = Gpt2Config::from_config_json(json).unwrap();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.vocab_size, 50257);
    }

    #[test]
    fn split_qkv_correct() {
        // qkv = [[0,1,2,3, 4,5,6,7, 8,9,10,11]]  hidden=4
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let qkv = Tensor::from_vec(data, &[1, 12]);
        let (q, k, v) = split_qkv(&qkv, 4);
        assert_eq!(q.data(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(k.data(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(v.data(), &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn transpose_2d_correct() {
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t2 = transpose_2d(&t);
        assert_eq!(t2.shape().as_slice(), &[3, 2]);
        assert_eq!(t2.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
