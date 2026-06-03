# AGENTS.md

Onboarding notes for LLM coding assistants (Claude Code, Cursor, Aider, etc.)
working on `wasmicro`. Read this before making changes.

## What this crate is

A minimal forward-only inference library for transformer models. Current
WASM bundle is **93 KB** after `wasm-opt -Oz`. Hard ceiling is 250 KB.
The same crate runs natively for tests, benchmarks, and the verification
suite that proves numerical parity with HuggingFace.

### Verified claims (as of v0.2.1)

- BERT forward on `sentence-transformers/all-MiniLM-L6-v2` matches the
  `transformers` reference to within `1e-6`, cosine `1.000000`.
- WordPiece tokenizer matches `bert-base-multilingual-cased` exactly on
  8 / 8 test cases covering English, Russian, Chinese, Spanish, and
  mixed-script inputs.
- WASM SIMD128 is on by default for `wasm32-unknown-unknown` via
  `.cargo/config.toml`; the scalar fallback covers every other target.

When you add or change code, **do not regress these claims**. Run the
verification commands at the end of this file before declaring work done.

## What this crate is NOT

- Not a training framework. No autograd, no backward pass, no optimizers.
- Not a general tensor library. It exists to run transformer-shaped
  computations. Reject feature requests that pull the crate away from that.
- Not a wrapper around `candle`, `tch`, `ort`, or `ndarray`. Adding any of
  these as a dependency defeats the purpose.

## Hard rules (library crate `wasmicro`)

These exist so the WASM bundle stays small and predictable.

1. **No `Rc`, no `RefCell`, no `Arc`, no `Mutex` in the tensor or op layer.**
   Tensors own their data. If you need an out parameter, take `&mut Tensor`.
2. **No autograd state on `Tensor`.** No `requires_grad`, no `grad`,
   no `ctx`. Forward ops produce new owned tensors or write into out params.
3. **No `ndarray`, no `candle*`, no `rayon`, no `serde_json`, no `chrono`,
   no `getrandom`** in the default build. The default dependency set is
   exactly `bytemuck`. Adding anything else needs an explicit justification
   in the PR description and a corresponding WASM size measurement.
4. **No `std::fs` inside the library.** The host provides bytes via
   `ModelFile::parse(&[u8])`. Examples and tests may read files; the library
   itself may not.
5. **All identifiers, comments, doc-comments, and error messages in English.**
   No exceptions.
6. **Ops are free functions.** Do not introduce a `trait Layer` or a
   `Module` zoo. Models are user-written structs of weights with their own
   `forward` method.
7. **Public errors go through `wasmicro::Error`.** Do not return
   `Box<dyn Error>` or panic on user input (panics are fine for internal
   shape contracts).
8. **Every new op needs unit tests with known-good values.** Use
   approximate-equality helpers for floats; do not rely on bitwise equality.

## CLI crate `wasmicro-convert`

The converter at `tools/wasmicro-convert/` is a separate workspace member
and is NOT subject to the same dependency restrictions as the library. It
can use `hf-hub`, `serde_json`, `reqwest`, `clap` — anything reasonable for
a desktop CLI. It never ships in the WASM bundle.

## File map

```
src/
  lib.rs                Crate root. Module declarations + re-exports.
  error.rs              `Error` enum and `Result<T>` alias.
  tensor.rs             `Tensor` and `Shape`. Owned data, inline shape.
  loader.rs             Safetensors parser + hand-rolled JSON.
  wasm.rs               `#[wasm_bindgen]` bindings (only with `wasm` feature).
  ops/
    mod.rs              Module aggregator.
    matmul.rs           `matmul` (A @ B) and `matmul_t_b` (A @ B^T).
    linear.rs           `linear(x, W, b)` — built on `matmul_t_b` + `add_bias`.
    embedding.rs        `embedding(ids, W)` — row lookup.
    elementwise.rs      `add`, `sub`, `mul`, `scale`, `add_bias`.
    softmax.rs          `softmax_last_dim`, numerically stable.
    layernorm.rs        `layer_norm`, Welford single-pass.
    activations.rs      `relu`, `silu`, `gelu_tanh`, `gelu_erf`.
    attention.rs        `multi_head_attention`, `mean_pool`.
  models/
    mod.rs              Module aggregator.
    bert.rs             `BertConfig`, `BertModel`, `forward`, `from_safetensors`.
tests/                  Integration tests that consume the public API only.
examples/               Runnable demos.
tools/wasmicro-convert/ Standalone CLI: download HF model + validate.
demo/                   Static site deployed to GitHub Pages.
.github/workflows/      CI (test + clippy + wasm check) and Pages deploy.
```

## Conventions

### Tensor convention

- 32-bit float (`f32`) only. f16/bf16/int8 support, when added, will come
  as separate dtypes and conversion paths — never silent.
- Row-major layout. `[m, k] @ [k, n] -> [m, n]`.
- Linear weights match PyTorch: `[out_features, in_features]`. Use
  `matmul_t_b` when applying them.
- Multi-head attention treats batch=1 implicitly. Inputs are 2D
  `[seq_len, hidden]`. Add batched paths only when needed.

### Ops API shape

```rust
/// One-line doc explaining the math.
///
/// - `x`: shape and meaning.
/// - returned tensor shape: ...
pub fn my_op(x: &Tensor /* , ... */) -> Tensor {
    // validate shapes with `assert_eq!` / `assert!` and clear messages
    // produce output as `Vec<f32>` of the right size
    // return `Tensor::from_vec(out, &[...])`
}
```

### Errors

`error::Error` is a flat enum with no heap-allocated payloads in the common
path. The static `&'static str` in `InvalidHeader(...)` is the only context
payload — use it for parser-level specificity. Do not add `String`
payloads casually.

### Testing

- Unit tests live next to their code in `#[cfg(test)] mod tests`.
- Integration tests under `tests/` exercise only the public API the way a
  downstream user would.
- Run before every commit: `cargo test --workspace && cargo check --target
  wasm32-unknown-unknown --features wasm`.

## Commands

```bash
# Native check + test (whole workspace)
cargo check --workspace
cargo test  --workspace

# WASM target check (catches std accidentally leaking into the library)
cargo check --target wasm32-unknown-unknown --features wasm

# Build the WASM bundle for a browser. SIMD128 is enabled automatically by
# .cargo/config.toml; --enable-simd on wasm-opt is mandatory because the
# emitted .wasm contains v128 instructions.
wasm-pack build --release --target web --no-opt \
    --out-dir demo/pkg --out-name wasmicro \
    . -- --features wasm
wasm-opt --enable-bulk-memory --enable-nontrapping-float-to-int --enable-simd \
    -Oz demo/pkg/wasmicro_bg.wasm -o demo/pkg/wasmicro_bg.wasm

# Build the converter
cargo build --release -p wasmicro-convert

# Publish dry-run for the library
cargo publish --dry-run --allow-dirty -p wasmicro

# Run the verification suite (downloads HF reference models).
# All three must pass before tagging a release.
cd ../wasmicro-verify
python python/reference.py
python python/multilingual_tokens.py
cargo run --release --bin wasmicro-verify
cargo run --release --bin e2e_search
cargo run --release --bin multilingual_tokens
```

## When in doubt

- Smaller surface beats more features.
- Fewer dependencies beat faster code that needs a new dependency.
- A copy of 20 lines beats pulling in a 5,000-line crate.
- If you cannot justify a change as "this makes the WASM bundle smaller or
  the cold start faster, or it adds a transformer architecture we want to
  support" — do not make the change.
