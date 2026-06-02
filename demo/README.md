# wasmicro demo

The static page deployed to GitHub Pages. The CI workflow at
`.github/workflows/pages.yml` builds the WASM bundle into `demo/pkg/` and
uploads the whole `demo/` directory as the Pages artifact — there is no
build step you need to run for production.

## Local preview

```bash
# Build the WASM bundle (from the repo root)
wasm-pack build --release --target web --features wasm --out-dir demo/pkg --out-name wasmicro

# Optimize
wasm-opt -Oz demo/pkg/wasmicro_bg.wasm -o demo/pkg/wasmicro_bg.wasm

# Serve
cd demo
python -m http.server 8080
# open http://localhost:8080/
```

## What the demo shows

1. **Bundle size, cold load time, library version** — a quick sanity card so
   you see the deployed bundle is what you expect.
2. **Matmul benchmark** — runs `n × n` matmul inside WASM, reports GFLOPS.
3. **BERT inference** — pick a `model.safetensors` file (e.g. the one
   produced by `wasmicro-convert sentence-transformers/all-MiniLM-L6-v2`).
   The page parses it and runs a tiny dummy inference. Real tokenization is
   not yet included — token ids are hardcoded in this demo, by design,
   while the WordPiece tokenizer is being built.
