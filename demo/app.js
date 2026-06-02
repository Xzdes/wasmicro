// Entry point for the wasmicro demo page.
//
// Imports the wasm-pack output from ./pkg/wasmicro.js. The CI workflow at
// .github/workflows/pages.yml builds this directory and uploads it to
// GitHub Pages — there is no local build step required to view changes.

import init, {
  version,
  matmul_bench,
  init_panic_hook,
  WasmBertModel,
} from "./pkg/wasmicro.js";

const $ = (id) => document.getElementById(id);

async function fetchBundleSize() {
  try {
    const res = await fetch("./pkg/wasmicro_bg.wasm", { method: "HEAD" });
    const len = res.headers.get("Content-Length");
    if (!len) return "unknown";
    const kb = (Number(len) / 1024).toFixed(1);
    return `${kb} KB`;
  } catch {
    return "unknown";
  }
}

async function main() {
  // 1. Load WASM and measure cold start.
  const t0 = performance.now();
  await init();
  const t1 = performance.now();

  init_panic_hook();

  $("status").textContent = "ready";
  $("version").textContent = `v${version()}`;
  $("load-time").textContent = `${(t1 - t0).toFixed(1)} ms`;
  $("bundle-size").textContent = await fetchBundleSize();

  // 2. Matmul benchmark.
  $("run-bench").addEventListener("click", () => {
    const n = Number($("n-input").value) || 128;
    $("bench-result").textContent = "running…";
    // Run on next frame so the UI can update first.
    requestAnimationFrame(() => {
      // Warm-up.
      matmul_bench(32);
      const start = performance.now();
      const head = matmul_bench(n);
      const elapsed = (performance.now() - start) / 1000;
      const flops = 2 * n * n * n;
      const gflops = flops / elapsed / 1e9;
      $("bench-result").textContent = [
        `n = ${n}`,
        `time = ${(elapsed * 1000).toFixed(2)} ms`,
        `GFLOPS = ${gflops.toFixed(2)}`,
        `first-cell sanity = ${head}`,
      ].join("\n");
    });
  });

  // 3. BERT model loading.
  let modelBytes = null;
  $("model-file").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    modelBytes = new Uint8Array(await file.arrayBuffer());
    $("model-status").textContent =
      `selected: ${file.name} (${(modelBytes.byteLength / 1024 / 1024).toFixed(2)} MB)\n` +
      "click Load model to parse.";
    $("load-model").disabled = false;
  });

  $("load-model").addEventListener("click", () => {
    if (!modelBytes) return;
    try {
      // Hardcoded to all-MiniLM-L6-v2 config for the demo.
      const t0 = performance.now();
      const model = new WasmBertModel(
        modelBytes,
        /* hidden_size            */ 384,
        /* num_hidden_layers      */ 6,
        /* num_attention_heads    */ 12,
        /* intermediate_size      */ 1536,
        /* vocab_size             */ 30522,
        /* max_position_embeddings*/ 512,
        /* type_vocab_size        */ 2,
        /* prefix                 */ "",
      );
      const t1 = performance.now();

      // Run a tiny inference on dummy input ids (just to prove the pipeline works).
      const ids = new Uint32Array([101, 7592, 2088, 102]); // [CLS] hello world [SEP]
      const t2 = performance.now();
      const emb = model.embed(ids);
      const t3 = performance.now();

      $("model-status").textContent = [
        `loaded in ${(t1 - t0).toFixed(1)} ms`,
        `inference on ${ids.length} tokens: ${(t3 - t2).toFixed(1)} ms`,
        `embedding dim: ${emb.length}`,
        `first 8 values: ${[...emb.slice(0, 8)].map((v) => v.toFixed(4)).join(", ")}`,
      ].join("\n");
    } catch (err) {
      $("model-status").textContent = `error: ${err.message ?? err}`;
    }
  });
}

main();
