# wasmicro

**Tiny transformer inference for the web. One file. No build step.**

`wasmicro` runs transformer models (embeddings, classifiers, small LLMs) in
any JavaScript environment — browser, Node.js, Cloudflare Workers, Electron —
with a single small `.wasm` file. The same crate also runs natively, so the
same code powers your tests, your benchmarks, and your production website.

## Status

Pre-alpha. Working today:

- **Tensor core.** Owned `Tensor` with inline shape. No `Rc<RefCell>`, no
  autograd, no training state.
- **Forward ops.** `matmul`, `matmul_t_b`, `linear`, `embedding`, `softmax`,
  `layer_norm`, `relu`, `silu`, `gelu_tanh`, `gelu_erf`, elementwise math,
  multi-head self-attention, mean pooling, and weight-only quantized linear
  paths for `i8`, affine `u8`, and packed `q4` weights.
- **BERT encoder.** Full forward pass against the HuggingFace BERT weight
  layout (`bert-base-uncased`, `sentence-transformers/*`, etc.). Linear
  weights may be `F32`, `I8`, or affine `U8/q8` with companion scale tensors.
- **WordPiece tokenizer.** `WordPieceTokenizer::from_vocab_bytes(&[u8])`
  loads external `vocab.txt` bytes and produces `input_ids`,
  `token_type_ids`, and `attention_mask`.
- **Model loader.** `ModelFile::parse(&[u8])` reads safetensors with a
  hand-rolled JSON parser. No `serde`, no `serde_json` in the library.
- **Converter CLI.** `wasmicro-convert <hf-model-id> <out-dir>` downloads
  a model from the HuggingFace Hub, validates it, and can write an `i8` or
  `u8/q8` weight-only quantized BERT file.
- **WASM build + demo.** GitHub Actions builds the WASM bundle and deploys
  a live demo page on every push to `main`.

## Quick start (using wasmicro in another project)

The most convenient way is a **path dependency** while iterating locally:

```toml
[dependencies]
wasmicro = { path = "../wasmicro" }
```

A **git dependency** is just as easy:

```toml
[dependencies]
wasmicro = { git = "https://github.com/Xzdes/wasmicro" }
```

Once it is published, **crates.io** will be the recommended path:

```toml
[dependencies]
wasmicro = "0.2.0"
```

Use it:

```rust
use std::fs;
use wasmicro::{
    models::bert::{BertConfig, BertModel},
    ModelFile, WordPieceTokenizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = fs::read("model.safetensors")?;
    let vocab_bytes = fs::read("vocab.txt")?;

    let file = ModelFile::parse(&model_bytes)?;
    let tokenizer = WordPieceTokenizer::from_vocab_bytes(&vocab_bytes)?;
    let config = BertConfig::mini_lm_l6_v2();
    let model = BertModel::from_safetensors(&file, config, "")?;

    let embedding = model.embed_text(&tokenizer, "hello world", 128)?;

    println!("embedding dim: {:?}", embedding.shape().as_slice());
    Ok(())
}
```

## Get a model

```bash
# Build the converter (one-time)
cargo build --release -p wasmicro-convert

# Download all-MiniLM-L6-v2 from the HuggingFace Hub
./target/release/wasmicro-convert \
    sentence-transformers/all-MiniLM-L6-v2 \
    ./models/mini-lm

# Optional: also write model.i8.safetensors with quantized BERT linear weights
./target/release/wasmicro-convert \
    sentence-transformers/all-MiniLM-L6-v2 \
    ./models/mini-lm \
    --quantize i8
```

Output:

```
models/mini-lm/
├── model.safetensors    (~ 87 MB, ready to pass to ModelFile::parse)
├── model.i8.safetensors (optional, when --quantize i8 is used)
├── config.json
├── vocab.txt
└── tokenizer.json
```

## Building

```bash
# Native — tests, benchmarks, examples.
cargo test --workspace
cargo run --example load_safetensors

# WASM bundle (browser, ES modules).
wasm-pack build --release --target web --no-opt \
    --out-dir demo/pkg --out-name wasmicro --features wasm
wasm-opt --enable-bulk-memory --enable-nontrapping-float-to-int -Oz \
    demo/pkg/wasmicro_bg.wasm -o demo/pkg/wasmicro_bg.wasm

# Repeatable size report for the WASM bundle and npm dry-run package.
powershell -ExecutionPolicy Bypass -File tools/measure-size.ps1

# Serve the demo locally
cd demo && python -m http.server 8080
```

## Demo

A live demo is built and deployed automatically by GitHub Actions on every
push to `main`. The workflow is at `.github/workflows/pages.yml`.

To enable Pages on your fork:
1. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. Push to `main`. The `pages` workflow builds the WASM bundle, runs
   `wasm-opt -Oz`, and publishes `demo/` to Pages.

## Project layout

```
wasmicro/
├── src/                       # the library
│   ├── lib.rs
│   ├── tensor.rs              # owned f32 tensor + inline shape
│   ├── tokenizer.rs           # minimal WordPiece tokenizer
│   ├── quant.rs               # weight-only quantized storage types
│   ├── loader.rs              # safetensors parser (no serde)
│   ├── error.rs
│   ├── ops/                   # forward ops: matmul, attention, layernorm, ...
│   ├── models/
│   │   └── bert.rs            # BertModel + forward + from_safetensors
│   └── wasm.rs                # wasm-bindgen surface (feature = "wasm")
├── tools/
│   ├── wasmicro-convert/      # CLI to download & validate HF models
│   └── measure-size.ps1       # WASM/npm size report
├── tests/                     # integration tests via the public API
├── examples/                  # runnable demos
├── demo/                      # static site for GitHub Pages
└── .github/workflows/         # CI + Pages deploy
```

## Design rules

These are non-negotiable. Code that breaks them gets reverted.

1. **Tiny WASM bundle.** Target: < 250 KB after `wasm-opt -Oz`.
2. **Forward only.** No autograd, no optimizers, no training.
3. **Owned tensors.** `Vec<f32>`, no `Rc`, no `RefCell`.
4. **No heavy dependencies.** The library's default build pulls in only
   `bytemuck`. No `ndarray`, `candle`, `rayon`, `serde_json`, `chrono`.
   (The `wasmicro-convert` CLI is a separate crate — it can have any deps
   it likes.)
5. **The host owns bytes.** `ModelFile::parse(&[u8])` works for files,
   fetches, `mmap`, `ArrayBuffer` — all the same to us.
6. **Ops are free functions.** Layers are functions, not objects.

## Roadmap

- [x] Project skeleton
- [x] Plain tensor + shape
- [x] Forward ops: matmul, linear, embedding, softmax, layernorm, GELU/SiLU/ReLU
- [x] safetensors loader with no `serde`
- [x] Multi-head attention + mean pooling
- [x] BERT encoder forward + `from_safetensors`
- [x] HuggingFace → wasmicro converter CLI
- [x] CI + GitHub Pages deploy workflow
- [x] WASM demo page
- [x] WordPiece tokenizer from external `vocab.txt`
- [x] End-to-end semantic-search demo: text -> WordPiece -> BERT embeddings -> cosine ranking
- [x] Weight-only quantized linear ops: `i8`, affine `u8`, packed `q4`
- [x] Quantized BERT linear loading for `i8` and affine `u8/q8`
- [x] Repeatable WASM/npm size measurement script
- [ ] Real `all-MiniLM-L6-v2` semantic-search demo
- [x] Converter quantization pipeline for BERT linear weights
- [ ] WASM SIMD128 paths
- [ ] GPT-2 + KV-cache
- [ ] WebGPU backend

## License

MIT OR Apache-2.0
