# wasmicro demo

The static page deployed to GitHub Pages. The CI workflow at
`.github/workflows/pages.yml` builds the WASM bundle into `demo/pkg/` and
uploads the whole `demo/` directory as the Pages artifact — there is no
build step you need to run for production.

## Local preview

```bash
# Build the WASM bundle (from the repo root)
wasm-pack build --release --target web --no-opt --out-dir demo/pkg --out-name wasmicro --features wasm

# Optimize
wasm-opt --enable-bulk-memory --enable-nontrapping-float-to-int --enable-simd -Oz demo/pkg/wasmicro_bg.wasm -o demo/pkg/wasmicro_bg.wasm

# Serve
cd demo
python -m http.server 8080
# open http://localhost:8080/
```

## What the demo shows

1. **Bundle size, cold load time, library version** — a quick sanity card so
   you see the deployed bundle is what you expect.
2. **Matmul benchmark** — runs `n × n` matmul inside WASM, reports GFLOPS.
3. **Semantic search** — pick a `model.safetensors` file (e.g. the one
   produced by `wasmicro-convert sentence-transformers/all-MiniLM-L6-v2`)
   and its `vocab.txt`. The page tokenizes text with WordPiece, runs the
   encoder, normalizes embeddings, and ranks documents by cosine similarity.
