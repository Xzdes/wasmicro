# wasmicro

**Tiny multilingual transformer inference for the web.**

[![crates.io](https://img.shields.io/crates/v/wasmicro.svg)](https://crates.io/crates/wasmicro)
[![npm](https://img.shields.io/npm/v/wasmicro.svg)](https://www.npmjs.com/package/wasmicro)
[![docs.rs](https://docs.rs/wasmicro/badge.svg)](https://docs.rs/wasmicro)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

A 93 KB WebAssembly bundle that runs WordPiece + BERT inference in any
JavaScript environment — browser, Node, Cloudflare Workers, Electron — or
natively from Rust. WordPiece tokenization and BERT forward outputs match
HuggingFace `transformers` to within `f32` round-off (`1e-6`) on every input
we have tested, including Russian, Chinese, and Spanish.

## What works today — and what we verified against

| Component | Verified against | Result |
|---|---|---|
| BERT encoder forward | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace `transformers` | max abs error **1e-6**, cosine **1.000000** |
| WordPiece tokenizer | `bert-base-multilingual-cased` on 8 RU / ZH / ES / EN / mixed cases | **8 / 8 exact id match** |
| End-to-end semantic search | 3 queries × 6 documents | **3 / 3 queries** rank expected document at top-1 |
| WASM bundle | `wasm-opt -Oz` on release build | **93 KB** |

Reproducible from the [`wasmicro-verify`](#verification) sub-project — every
claim in this README is backed by a binary that downloads the real model and
compares numbers.

## Install

### Rust

```toml
[dependencies]
wasmicro = "0.2.1"
```

### JavaScript

```bash
npm install wasmicro
```

```js
import init, { WasmBertModel, WasmWordPieceTokenizer } from "wasmicro";

await init();
// Fetch model.safetensors + vocab.txt from your CDN of choice.
const modelBytes = new Uint8Array(await (await fetch("/model.safetensors")).arrayBuffer());
const vocabBytes = new Uint8Array(await (await fetch("/vocab.txt")).arrayBuffer());

const tokenizer = new WasmWordPieceTokenizer(vocabBytes, /*lowercase=*/ true);
const model = new WasmBertModel(
  modelBytes,
  /* hidden_size */ 384, /* num_layers */ 6, /* num_heads */ 12,
  /* intermediate */ 1536, /* vocab */ 30522, /* max_pos */ 512, /* type_vocab */ 2,
  /* prefix */ "",
);
const embedding = model.embed_text(tokenizer, "hello world", 128);
console.log(`dim=${embedding.length}`); // 384
```

The shipped `.wasm` is 93 KB. Compared to common alternatives the engine is
**18×–250× smaller**; the model file is unchanged.

| Runtime | WASM/JS payload |
|---|---|
| **wasmicro** | **93 KB** |
| Candle WASM | 1.5–5 MB |
| transformers.js | ~10 MB |
| ONNX Runtime Web | 8–20 MB |

## Quick start (Rust)

```rust
use std::fs;
use wasmicro::{
    models::bert::{BertConfig, BertModel},
    ModelFile, WordPieceTokenizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_bytes = fs::read("models/mini-lm/model.safetensors")?;
    let vocab_bytes = fs::read("models/mini-lm/vocab.txt")?;

    let file = ModelFile::parse(&model_bytes)?;
    let tokenizer = WordPieceTokenizer::from_vocab_bytes(&vocab_bytes)?;
    let model = BertModel::from_safetensors(&file, BertConfig::mini_lm_l6_v2(), "")?;

    let embedding = model.embed_text(&tokenizer, "hello world", 128)?;
    println!("embedding dim: {:?}", embedding.shape().as_slice());
    Ok(())
}
```

## Multilingual

The WordPiece tokenizer is Unicode-aware:

- Splits each CJK ideograph into its own token (matches HuggingFace).
- Lowercases via Unicode (`char::to_lowercase`) — handles Cyrillic, Greek,
  Latin Extended, etc.
- Recognises Unicode whitespace (NBSP, ideographic space, …).
- Treats Unicode punctuation as its own token (CJK comma, Spanish `¿`/`¡`,
  French guillemets, …).

To work with non-English text, use a multilingual vocabulary. Example:

```rust
use wasmicro::tokenizer::{WordPieceOptions, WordPieceTokenizer};

let vocab = std::fs::read("models/multilingual/vocab.txt")?;
let tokenizer = WordPieceTokenizer::from_vocab_bytes_with_options(
    &vocab,
    // bert-base-multilingual-cased: keep case, no accent stripping.
    WordPieceOptions { lowercase: false, max_input_chars_per_word: 100 },
)?;
let encoded = tokenizer.encode("Привет, мир!", 32)?;
// -> [CLS] При ##вет , мир ! [SEP]
```

Accent stripping (NFD + combining-mark removal) is **not** implemented; pick
a `*-cased` multilingual vocabulary if your inputs contain accents.

## Get a model

```bash
# Build the converter once.
cargo build --release -p wasmicro-convert

# Download all-MiniLM-L6-v2 from the HuggingFace Hub.
./target/release/wasmicro-convert \
    sentence-transformers/all-MiniLM-L6-v2 \
    ./models/mini-lm

# Optional: also write model.i8.safetensors with weight-only int8 quantization.
./target/release/wasmicro-convert \
    sentence-transformers/all-MiniLM-L6-v2 \
    ./models/mini-lm \
    --quantize i8
```

Resulting directory:

```
models/mini-lm/
├── model.safetensors      (~87 MB, ready for ModelFile::parse)
├── model.i8.safetensors   (optional, --quantize i8)
├── config.json
├── vocab.txt              (pass to WordPieceTokenizer::from_vocab_bytes)
└── tokenizer.json
```

## Building from source

```bash
# Native: tests, benchmarks, examples.
cargo test --workspace
cargo run --example load_safetensors

# WASM bundle (SIMD128 is enabled automatically by .cargo/config.toml).
wasm-pack build --release --target web --no-opt \
    --out-dir demo/pkg --out-name wasmicro \
    . -- --features wasm

wasm-opt --enable-bulk-memory --enable-nontrapping-float-to-int --enable-simd \
    -Oz demo/pkg/wasmicro_bg.wasm -o demo/pkg/wasmicro_bg.wasm

# Optional: repeatable size report.
powershell -ExecutionPolicy Bypass -File tools/measure-size.ps1

# Serve the demo locally.
cd demo && python -m http.server 8080
```

`.cargo/config.toml` sets `target-feature=+simd128` for `wasm32-unknown-unknown`,
so every `wasm-pack build` ships SIMD128 kernels. To target very old browsers
(<2022), pass `RUSTFLAGS="-C target-feature=-simd128"`.

## Verification

The [`wasmicro-verify`](../wasmicro-verify) sibling project is the source of
truth for every numeric claim above.

```bash
cd ../wasmicro-verify

# 1. Generate HuggingFace reference outputs (Python + transformers).
python python/reference.py
python python/multilingual_tokens.py

# 2. Run wasmicro on the same inputs and compare.
cargo run --release --bin wasmicro-verify     # BERT forward vs HF
cargo run --release --bin e2e_search          # text -> tokenize -> embed -> rank
cargo run --release --bin multilingual_tokens # WordPiece vs HF on RU/ZH/ES
```

Expected outcome — all three exit `0` with detailed per-case reports. CI will
gate releases on these in a future revision.

## Project layout

```
wasmicro/
├── src/                          # the library (default deps: bytemuck only)
│   ├── lib.rs
│   ├── tensor.rs                 # owned f32 tensor + inline shape
│   ├── tokenizer.rs              # Unicode WordPiece tokenizer
│   ├── quant.rs                  # i8, u8 affine, q4 packed quantized tensors
│   ├── loader.rs                 # safetensors parser (no serde)
│   ├── error.rs
│   ├── ops/                      # matmul (+SIMD128), attention, layernorm, …
│   ├── models/
│   │   └── bert.rs               # BertModel + forward + from_safetensors
│   └── wasm.rs                   # wasm-bindgen surface (feature = "wasm")
├── tools/
│   ├── wasmicro-convert/         # CLI to download, validate, quantize HF models
│   └── measure-size.ps1          # WASM/npm size report
├── tests/                        # integration tests via the public API
├── examples/                     # runnable demos
├── demo/                         # static site deployed to GitHub Pages
├── .cargo/config.toml            # enables SIMD128 by default for wasm32
└── .github/workflows/            # CI + Pages deploy
```

## Design rules

These are non-negotiable. Code that breaks them gets reverted.

1. **Tiny WASM bundle.** Current: 93 KB. Cap: 250 KB after `wasm-opt -Oz`.
2. **Forward only.** No autograd, no optimizers, no training state.
3. **Owned tensors.** `Vec<f32>`. No `Rc`, no `RefCell`.
4. **Minimal dependencies.** The library's default build pulls in only
   `bytemuck`. No `ndarray`, no `candle`, no `rayon`, no `serde_json`, no
   `chrono`. The `wasmicro-convert` CLI is a separate crate with its own
   deps (`hf-hub`, etc.) and never ships in the WASM.
5. **The host owns bytes.** `ModelFile::parse(&[u8])` — same code path for
   files, fetches, `mmap`, or `ArrayBuffer`.
6. **Ops are free functions.** Layers are functions, not objects.

## Honest limitations

- Only the BERT encoder architecture is supported. No GPT, T5, Whisper, ViT,
  CLIP, or any decoder/encoder-decoder model yet.
- No accent stripping (NFD + mark removal). Use `*-cased` multilingual
  vocabularies if your inputs include accents.
- No batching. Encoding multiple sentences runs them sequentially.
- CPU only. No WebGPU backend; matmul uses naive `ikj` with WASM SIMD128
  inner kernels. Production-scale throughput is not the target.
- No zero-config import. You must download the model, copy `vocab.txt`, and
  pass the config fields explicitly. Higher-level pipelines (à la
  `pipeline('feature-extraction', '...')`) are not provided.

If any of these matter for your use case, prefer
[transformers.js](https://github.com/xenova/transformers.js) or
[Candle](https://github.com/huggingface/candle) — they are far more
feature-complete.

## Roadmap

- [x] Project skeleton
- [x] Plain tensor + inline shape
- [x] Forward ops: matmul, linear, embedding, softmax, layernorm, GELU/SiLU/ReLU
- [x] safetensors loader with no `serde`
- [x] Multi-head attention + mean pooling
- [x] BERT encoder forward + `from_safetensors`
- [x] Numerical parity with HuggingFace on `all-MiniLM-L6-v2` (1e-6)
- [x] HuggingFace → wasmicro converter CLI
- [x] WordPiece tokenizer with Unicode awareness (CJK split, Unicode case)
- [x] Multilingual parity test against `bert-base-multilingual-cased` (8/8)
- [x] Weight-only quantized linear ops: `i8`, affine `u8`, packed `q4`
- [x] Quantized BERT linear loading (`i8`, `u8/q8`)
- [x] Converter quantization pipeline (`--quantize i8`)
- [x] WASM SIMD128 kernels for `matmul` and `matmul_t_b`
- [x] End-to-end semantic-search verifier (text → embedding → ranking)
- [x] CI + GitHub Pages deploy workflow
- [x] WASM demo page
- [x] Published to crates.io and npm
- [ ] Live demo with a downloadable model bundle on GitHub Pages
- [ ] NFD accent-stripping path for uncased multilingual vocabularies
- [ ] Zero-config import: `wasmicro::embed("text")` with auto-fetch of HF assets
- [ ] Browser benchmark numbers (tokens/s on M-series, mid-tier x86, Android)
- [ ] GPT-2 + KV-cache
- [ ] WebGPU backend

## License

MIT OR Apache-2.0
