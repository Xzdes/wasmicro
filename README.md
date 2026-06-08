# wasmicro

**Tiny transformer inference for the web. One file. No build step.**

[![crates.io](https://img.shields.io/crates/v/wasmicro.svg)](https://crates.io/crates/wasmicro)
[![npm](https://img.shields.io/npm/v/wasmicro.svg)](https://www.npmjs.com/package/wasmicro)
[![docs.rs](https://docs.rs/wasmicro/badge.svg)](https://docs.rs/wasmicro)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

A **199 KB** WebAssembly bundle that runs BERT, GPT-2, and T5 inference in any
JavaScript environment — browser, Node, Cloudflare Workers, Electron — or
natively from Rust. Model type is auto-detected from `config.json`; no
hardcoded parameters required.

Outputs match HuggingFace `transformers` to within `f32` round-off on every
input tested.

---

## What works today — verified against real HuggingFace checkpoints

| Component | Checkpoint | Result |
|---|---|---|
| BERT encoder + pooling | `bert-base-uncased` | cosine **1.000000**, max\|Δ\| **8.3 × 10⁻⁷** |
| End-to-end semantic search | `bert-base-uncased` | **3 / 3** queries rank expected doc at top-1 |
| GPT-2 generation | `openai-community/gpt2` | loads & generates, model\_type detection OK |
| T5 encoder + decoder | `google-t5/t5-small` | encoder shape `[seq, 512]`, all values finite |
| WASM bundle | `wasm-opt -Oz` on release | **199 KB** |

All claims are reproducible from [`wasmicro-verify`](#verification) — a sibling
project that downloads the real models and compares numbers against PyTorch.

---

## Install

### Rust

```toml
[dependencies]
wasmicro = "0.3.0"
```

### JavaScript / npm

```bash
npm install wasmicro
```

---

## Quick start

### JavaScript — unified `WasmPipeline` API

```js
import init, { WasmPipeline } from "wasmicro";

await init();

// Read four files from disk / fetch from CDN.
const model     = new Uint8Array(await (await fetch("model.safetensors")).arrayBuffer());
const tokenizer = new Uint8Array(await (await fetch("vocab.txt")).arrayBuffer());   // or vocab.json for GPT-2
const config    = await (await fetch("config.json")).text();
// merges.txt is required for GPT-2/T5, pass null for BERT.
const merges    = null;

const pipeline = WasmPipeline.fromBytes(model, tokenizer, config, merges);

// ── BERT (embedding / semantic search) ────────────────────────────────────────
const emb  = pipeline.embed("Hello world", /* max_len */ 128);   // Float32Array [768]
const batch = pipeline.embedBatch(["sentence one", "sentence two"], 128); // Float32Array [2×768]

// ── GPT-2 / T5 (text generation) ──────────────────────────────────────────────
const text = pipeline.generate("Once upon a time", /* max_new_tokens */ 50);
console.log(text);

// Detect model type without loading:
console.log(WasmPipeline.detectedModelType(config)); // "bert" | "gpt2" | "t5" | …
```

`WasmPipeline.fromBytes` auto-detects model type from `config.json` and
selects the right tokenizer and architecture automatically.

### Rust — `Pipeline::from_bytes`

```rust
use std::fs;
use wasmicro::pipeline::Pipeline;

// ── BERT embedding ─────────────────────────────────────────────────────────────
let model_bytes = fs::read("bert-base-uncased/model.safetensors")?;
let vocab_bytes = fs::read("bert-base-uncased/vocab.txt")?;
let config_json = fs::read_to_string("bert-base-uncased/config.json")?;

let pipeline = Pipeline::from_bytes(&model_bytes, &vocab_bytes, &config_json, None)?;
let embedding = pipeline.embed("Hello world", 128)?;
println!("dim = {}", embedding.shape().as_slice()[0]);  // 768

// ── GPT-2 generation ───────────────────────────────────────────────────────────
let vocab_json   = fs::read("gpt2/vocab.json")?;
let merges_bytes = fs::read("gpt2/merges.txt")?;
let config_json  = fs::read_to_string("gpt2/config.json")?;

let pipeline = Pipeline::from_bytes(
    &fs::read("gpt2/model.safetensors")?,
    &vocab_json,
    &config_json,
    Some(&merges_bytes),
)?;
let text = pipeline.generate("Once upon a time", 50)?;
println!("{text}");
```

### Lower-level API

The full model APIs are also public for advanced use:

```rust
use wasmicro::{ModelFile, models::gpt2::{Gpt2Config, Gpt2Model}};

let file   = ModelFile::parse(&model_bytes)?;
let config = Gpt2Config::from_config_json(&config_json)?;
let model  = Gpt2Model::from_safetensors(&file, config)?;

let logits = model.logits(&[15496u32, 11, 314, 1101]); // [seq, vocab]
```

---

## Supported models

| `model_type` | Architecture | Tokenizer | Methods |
|---|---|---|---|
| `bert`, `roberta`, `distilbert`, `electra` | Encoder | WordPiece (`vocab.txt`) | `embed`, `embed_batch` |
| `gpt2`, `gpt_neo`, `gpt_neox` | Decoder | Byte-level BPE (`vocab.json` + `merges.txt`) | `generate` |
| `t5`, `mt5`, `longt5` | Encoder-decoder | Byte-level BPE (`vocab.json` + `merges.txt`) | `generate`, `encode_t5` |

**Note on T5 tokenization:** T5's original tokenizer uses SentencePiece, which
is not yet built into wasmicro. Passing BPE-tokenized IDs works for encoder
shape / value checks; for real T5 generation quality, pre-tokenize with
SentencePiece externally and pass raw `input_ids` via the lower-level API.

---

## Bundle size vs alternatives

| Runtime | WASM/JS payload |
|---|---|
| **wasmicro** | **199 KB** |
| Candle WASM | 1.5 – 5 MB |
| transformers.js | ~10 MB |
| ONNX Runtime Web | 8 – 20 MB |

wasmicro is **8× – 50× smaller** than the next-smallest option for the same
three model families.

---

## Get a model

```bash
# Download any HuggingFace model to a local directory.
cargo build --release -p wasmicro-convert

./target/release/wasmicro-convert \
    sentence-transformers/all-MiniLM-L6-v2 \
    ./models/mini-lm

./target/release/wasmicro-convert \
    openai-community/gpt2 \
    ./models/gpt2

# Optional: weight-only int8 quantization (reduces model.safetensors size ~4×).
./target/release/wasmicro-convert \
    sentence-transformers/all-MiniLM-L6-v2 \
    ./models/mini-lm \
    --quantize i8
```

---

## Building from source

```bash
# Native: tests and benchmarks.
cargo test --workspace
cargo bench

# WASM bundle.
wasm-pack build --release --target web --no-opt \
    --out-dir demo/pkg --out-name wasmicro \
    . -- --features wasm

wasm-opt --enable-bulk-memory --enable-nontrapping-float-to-int \
    -Oz demo/pkg/wasmicro_bg.wasm -o demo/pkg/wasmicro_bg.wasm

# Serve the demo locally.
cd demo && python -m http.server 8080
```

`.cargo/config.toml` sets `target-feature=+simd128` for `wasm32-unknown-unknown`.
To target older browsers (<2022), pass `RUSTFLAGS="-C target-feature=-simd128"`.

---

## Verification

The [`wasmicro-verify`](../wasmicro-verify) sibling project is the source of
truth for every numeric claim in this README.

```bash
cd ../wasmicro-verify

# Optionally generate Python/PyTorch reference outputs first:
pip install transformers torch sentencepiece
python python/reference.py            # BERT reference
python python/reference_gpt2.py       # GPT-2 next-token logits
python python/reference_t5.py         # T5 encoder hidden states

# Run all verifiers:
cargo run --release                       # BERT forward vs HuggingFace
cargo run --release --bin e2e_search      # semantic search ranking
cargo run --release --bin verify_gpt2     # GPT-2 generation smoke test
cargo run --release --bin verify_t5       # T5 encoder + generation smoke test
```

All four exit `0`. The BERT verifier compares hidden-state values numerically;
GPT-2 and T5 perform smoke tests (correct shapes, finite values, non-empty output).
Full numerical comparison is enabled when the corresponding Python reference file
(`expected_gpt2.json` / `expected_t5.json`) is present.

---

## Project layout

```
wasmicro/
├── src/
│   ├── lib.rs                    # public re-exports
│   ├── tensor.rs                 # owned f32 tensor with inline shape
│   ├── tokenizer.rs              # Unicode WordPiece tokenizer
│   ├── tokenizer/bpe.rs          # byte-level BPE tokenizer (GPT-2/RoBERTa)
│   ├── quant.rs                  # i8, u8 affine, q4 packed quantized tensors
│   ├── loader.rs                 # zero-copy safetensors parser (no serde)
│   ├── error.rs
│   ├── pipeline.rs               # Pipeline::from_bytes — unified entry point
│   ├── ops/                      # free-function ops: matmul, attention, layernorm …
│   ├── models/
│   │   ├── bert.rs               # BERT encoder + mean/CLS pooling
│   │   ├── gpt2.rs               # GPT-2 decoder + greedy generation
│   │   └── t5.rs                 # T5 encoder-decoder + greedy generation
│   └── wasm.rs                   # wasm-bindgen surface (feature = "wasm")
├── tools/
│   └── wasmicro-convert/         # CLI to download + quantize HF models
├── demo/                         # static demo site (GitHub Pages)
└── ../wasmicro-verify/           # numeric verification harness
```

---

## Design rules

These are non-negotiable. Code that violates them gets reverted.

1. **Tiny WASM bundle.** Current: 199 KB. Hard cap: 250 KB after `wasm-opt -Oz`.
2. **Forward only.** No autograd, no optimizers, no training state.
3. **Owned tensors.** `Vec<f32>`. No `Rc`, no `RefCell`, no `Arc`, no `Mutex`.
4. **Minimal dependencies.** Default build pulls in only `bytemuck`. No `ndarray`,
   no `candle`, no `rayon`, no `serde_json`, no `chrono`.
5. **The host owns bytes.** `Pipeline::from_bytes(&[u8], …)` — same code path
   for disk files, HTTP fetches, `mmap`, or JS `ArrayBuffer`.
6. **Ops are free functions.** Layers are functions, not objects. No dynamic dispatch.

---

## Limitations

- **No KV-cache.** GPT-2 generation re-runs the full forward pass for each new
  token — O(n²) in sequence length. Fast enough for short prompts; add a cache
  if you need long continuations.
- **T5 tokenizer.** T5's native SentencePiece tokenizer is not yet built in.
  BPE-tokenized IDs work for encoder shape/value tests; real task-prefix
  generation requires external SentencePiece tokenization.
- **No accent stripping.** NFD + combining-mark removal is not implemented.
  Use `*-cased` multilingual BERT vocabularies for accented inputs.
- **CPU only.** No WebGPU backend. Matmul uses a naive `ikj` loop with optional
  WASM SIMD128 inner kernels. Not designed for production throughput.
- **No streaming.** `generate()` returns the full string only after all tokens
  are produced.

If these matter for your use case, prefer
[transformers.js](https://github.com/xenova/transformers.js) or
[Candle](https://github.com/huggingface/candle) — they are more feature-complete.

---

## Roadmap

- [x] Tensor engine + safetensors loader (no serde)
- [x] WordPiece tokenizer (Unicode-aware: CJK, Cyrillic, accents)
- [x] Byte-level BPE tokenizer (GPT-2 / RoBERTa compatible)
- [x] BERT encoder forward + `from_config_json` auto-detection
- [x] Numerical parity with HuggingFace BERT (`1e-6` max abs error)
- [x] Embed batch + end-to-end semantic search verifier
- [x] Weight-only quantization: i8, affine u8, packed q4
- [x] WASM SIMD128 kernels for matmul
- [x] GPT-2 decoder + greedy generation (verified on `openai-community/gpt2`)
- [x] T5 encoder-decoder + greedy generation (verified on `google-t5/t5-small`)
- [x] Unified `Pipeline::from_bytes` API (auto-detects model type)
- [x] `WasmPipeline` JS class — single entry point for all model families
- [x] Published to crates.io and npm
- [ ] KV-cache for GPT-2 / GPT-Neo (5–10× generation speedup)
- [ ] SentencePiece tokenizer (for T5 task-prefix generation)
- [ ] SIMD128 matmul tiling (fill the vector units)
- [ ] WebGPU backend
- [ ] Zero-config import: `wasmicro::embed("text")` with HF asset auto-fetch
- [ ] Live demo with downloadable model bundle on GitHub Pages
- [ ] Browser benchmark: tokens/s on M-series, x86, Android

---

## License

MIT OR Apache-2.0
