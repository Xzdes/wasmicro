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
  WasmWordPieceTokenizer,
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
  let vocabBytes = null;
  let model = null;
  let tokenizer = null;

  const refreshLoadButton = () => {
    $("load-model").disabled = !(modelBytes && vocabBytes);
  };

  const resetLoadedModel = () => {
    model = null;
    tokenizer = null;
    $("run-search").disabled = true;
  };

  $("model-file").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    resetLoadedModel();
    modelBytes = new Uint8Array(await file.arrayBuffer());
    $("model-status").textContent =
      `selected: ${file.name} (${(modelBytes.byteLength / 1024 / 1024).toFixed(2)} MB)\n` +
      "select vocab.txt, then click Load.";
    refreshLoadButton();
  });

  $("vocab-file").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    resetLoadedModel();
    vocabBytes = new Uint8Array(await file.arrayBuffer());
    $("model-status").textContent =
      `selected: ${file.name} (${(vocabBytes.byteLength / 1024).toFixed(1)} KB)\n` +
      "select model.safetensors, then click Load.";
    refreshLoadButton();
  });

  $("load-model").addEventListener("click", () => {
    if (!modelBytes || !vocabBytes) return;
    try {
      // Hardcoded to all-MiniLM-L6-v2 config for the demo.
      const t0 = performance.now();
      tokenizer = new WasmWordPieceTokenizer(vocabBytes, true);
      model = new WasmBertModel(
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

      $("run-search").disabled = false;
      $("model-status").textContent = [
        `loaded in ${(t1 - t0).toFixed(1)} ms`,
        "ready for semantic search",
      ].join("\n");
    } catch (err) {
      resetLoadedModel();
      $("model-status").textContent = `error: ${err.message ?? err}`;
    }
  });

  $("run-search").addEventListener("click", () => {
    if (!model || !tokenizer) return;
    try {
      const text = $("query-text").value || "hello world";
      const maxLen = Number($("max-len").value) || 32;
      const t2 = performance.now();
      const queryEmbedding = normalize(model.embed_text(tokenizer, text, maxLen));
      const documents = $("documents")
        .value
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);

      const results = documents.map((document, index) => {
        const embedding = normalize(model.embed_text(tokenizer, document, maxLen));
        return {
          index,
          document,
          score: dot(queryEmbedding, embedding),
        };
      });
      results.sort((a, b) => b.score - a.score);
      const t3 = performance.now();

      $("model-status").textContent = [
        `query: "${text}"`,
        `documents: ${documents.length}`,
        `search time: ${(t3 - t2).toFixed(1)} ms`,
        "",
        ...results.map(
          (result, rank) =>
            `${rank + 1}. score=${result.score.toFixed(4)} doc#${result.index + 1}: ${result.document}`,
        ),
      ].join("\n");
    } catch (err) {
      $("model-status").textContent = `error: ${err.message ?? err}`;
    }
  });
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function normalize(vector) {
  let norm = 0;
  for (const value of vector) {
    norm += value * value;
  }
  norm = Math.sqrt(norm);
  if (norm === 0) return vector;
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    out[i] = vector[i] / norm;
  }
  return out;
}

main();
