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
  multi-head self-attention, mean pooling.
- **BERT encoder.** Full forward pass against the HuggingFace BERT weight
  layout (`bert-base-uncased`, `sentence-transformers/*`, etc.).
- **Model loader.** `ModelFile::parse(&[u8])` reads safetensors with a
  hand-rolled JSON parser. No `serde`, no `serde_json` in the library.
- **Converter CLI.** `wasmicro-convert <hf-model-id> <out-dir>` downloads
  a model from the HuggingFace Hub and validates it.
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
wasmicro = "0.0.1"
```

Use it:

```rust
use std::fs;
use wasmicro::{models::bert::{BertConfig, BertModel}, ModelFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read("model.safetensors")?;
    let file = ModelFile::parse(&bytes)?;

    let config = BertConfig::mini_lm_l6_v2();
    let model = BertModel::from_safetensors(&file, config, "")?;

    let input_ids = vec![101u32, 7592, 2088, 102]; // [CLS] hello world [SEP]
    let embedding = model.embed_sentence(&input_ids, None, None);

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
```

Output:

```
models/mini-lm/
├── model.safetensors    (~ 87 MB, ready to pass to ModelFile::parse)
├── config.json
└── tokenizer.json
```

## Building

```bash
# Native — tests, benchmarks, examples.
cargo test --workspace
cargo run --example load_safetensors

# WASM bundle (browser, ES modules).
wasm-pack build --release --target web --features wasm \
    --out-dir demo/pkg --out-name wasmicro
wasm-opt -Oz demo/pkg/wasmicro_bg.wasm -o demo/pkg/wasmicro_bg.wasm

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
│   ├── loader.rs              # safetensors parser (no serde)
│   ├── error.rs
│   ├── ops/                   # forward ops: matmul, attention, layernorm, …
│   ├── models/
│   │   └── bert.rs            # BertModel + forward + from_safetensors
│   └── wasm.rs                # wasm-bindgen surface (feature = "wasm")
├── tools/
│   └── wasmicro-convert/      # CLI to download & validate HF models
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
- [ ] WordPiece tokenizer (so the demo can run end-to-end without pre-tokenized input)
- [ ] Real `all-MiniLM-L6-v2` semantic-search demo
- [ ] WASM SIMD128 paths
- [ ] int8 quantization
- [ ] GPT-2 + KV-cache
- [ ] WebGPU backend

## License

MIT OR Apache-2.0
