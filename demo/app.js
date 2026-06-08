/**
 * wasmicro demo — unified Pipeline API.
 *
 * Supports BERT (embedding / semantic search), GPT-2 and T5 (text generation).
 * Model type is auto-detected from config.json — no hardcoded parameters needed.
 */

import init, {
  version,
  matmul_bench,
  init_panic_hook,
  WasmPipeline,
} from "./pkg/wasmicro.js";

const $ = (id) => document.getElementById(id);

// ─── bootstrap ────────────────────────────────────────────────────────────────

async function main() {
  const t0 = performance.now();
  await init();
  const t1 = performance.now();
  init_panic_hook();

  $("status").textContent    = "ready";
  $("version").textContent   = `v${version()}`;
  $("load-time").textContent = `${(t1 - t0).toFixed(1)} ms`;

  try {
    const res = await fetch("./pkg/wasmicro_bg.wasm", { method: "HEAD" });
    const kb  = Number(res.headers.get("Content-Length") ?? 0) / 1024;
    $("bundle-size").textContent = kb ? `${kb.toFixed(1)} KB` : "unknown";
  } catch { $("bundle-size").textContent = "unknown"; }

  setupBench();
  setupPipeline();
}

// ─── matmul benchmark ────────────────────────────────────────────────────────

function setupBench() {
  $("run-bench").addEventListener("click", () => {
    const n = Number($("n-input").value) || 128;
    $("bench-result").textContent = "running…";
    requestAnimationFrame(() => {
      try {
        matmul_bench(32); // warm-up
        const t0  = performance.now();
        const val = matmul_bench(n);
        const ms  = performance.now() - t0;
        const gf  = 2 * n ** 3 / (ms / 1000) / 1e9;
        $("bench-result").textContent =
          `n=${n}  time=${ms.toFixed(2)} ms  GFLOPS=${gf.toFixed(2)}  cell[0]=${val}`;
      } catch (e) {
        $("bench-result").textContent = `error: ${e.message ?? e}`;
      }
    });
  });
}

// ─── Pipeline loader ─────────────────────────────────────────────────────────

function setupPipeline() {
  // File state
  let files = { model: null, tokenizer: null, config: null, merges: null };
  let pipeline = null;
  let modelType = "";

  // Update the "Load" button state
  const refreshBtn = () => {
    $("load-pipeline").disabled = !(files.model && files.tokenizer && files.config);
  };

  // Collect uploaded files
  ["model", "tokenizer", "config", "merges"].forEach((key) => {
    const el = $(`file-${key}`);
    if (!el) return;
    el.addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      files[key] = new Uint8Array(await f.arrayBuffer());
      $("pipeline-status").textContent = `${key}: ${f.name} (${(files[key].byteLength / 1024).toFixed(1)} KB)`;
      refreshBtn();
    });
  });

  // Load the pipeline
  $("load-pipeline").addEventListener("click", async () => {
    $("pipeline-status").textContent = "loading…";
    pipeline = null;
    try {
      const configJson = new TextDecoder().decode(files.config);
      modelType = WasmPipeline.detectedModelType(configJson);

      const t0 = performance.now();
      pipeline = WasmPipeline.fromBytes(
        files.model,
        files.tokenizer,
        configJson,
        files.merges ?? null,
      );
      const ms = (performance.now() - t0).toFixed(1);

      $("pipeline-status").textContent =
        `loaded in ${ms} ms — model_type: "${modelType}"`;

      // Show the right panel
      $("bert-section").hidden    = !["bert","roberta","distilbert","electra"].includes(modelType);
      $("gen-section").hidden     = !["gpt2","gpt_neo","gpt_neox","t5","mt5","longt5"].includes(modelType);
      $("gen-title").textContent  = modelType.startsWith("t5") || modelType.startsWith("mt5")
        ? "T5 generation (include task prefix, e.g. "translate English to French: Hello")"
        : "GPT-2 generation";

    } catch (e) {
      $("pipeline-status").textContent = `error: ${e.message ?? e}`;
    }
  });

  // ── BERT: semantic search ──────────────────────────────────────────────────

  $("run-search").addEventListener("click", () => {
    if (!pipeline) return;
    try {
      const query   = $("query-text").value.trim() || "hello world";
      const maxLen  = Number($("max-len").value) || 64;
      const docs    = $("documents").value.split("\n").map(s => s.trim()).filter(Boolean);
      if (!docs.length) { $("search-output").textContent = "(no documents)"; return; }

      const t0    = performance.now();
      const qEmb  = normalize(pipeline.embed(query, maxLen));
      const all   = pipeline.embedBatch(docs, maxLen);
      const dim   = all.length / docs.length;

      const scored = docs.map((doc, i) => {
        const dEmb = normalize(all.slice(i * dim, (i + 1) * dim));
        return { doc, score: dot(qEmb, dEmb) };
      }).sort((a, b) => b.score - a.score);

      const ms = (performance.now() - t0).toFixed(1);
      $("search-output").textContent = [
        `query: "${query}"  (${ms} ms, dim=${dim})`,
        "",
        ...scored.map((r, i) => `${i + 1}. [${r.score.toFixed(4)}] ${r.doc}`),
      ].join("\n");
    } catch (e) {
      $("search-output").textContent = `error: ${e.message ?? e}`;
    }
  });

  // ── GPT-2 / T5: text generation ───────────────────────────────────────────

  $("run-generate").addEventListener("click", () => {
    if (!pipeline) return;
    $("gen-output").textContent = "generating…";
    const prompt  = $("gen-prompt").value || "Once upon a time";
    const maxToks = Number($("gen-tokens").value) || 50;

    requestAnimationFrame(() => {
      try {
        const t0   = performance.now();
        const text = pipeline.generate(prompt, maxToks);
        const ms   = (performance.now() - t0).toFixed(0);
        $("gen-output").textContent = `[${ms} ms]\n${text}`;
      } catch (e) {
        $("gen-output").textContent = `error: ${e.message ?? e}`;
      }
    });
  });
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function normalize(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n);
  if (n === 0) return Float32Array.from(v);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

main();
