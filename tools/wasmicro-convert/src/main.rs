//! `wasmicro-convert`: download a HuggingFace model and prepare it for wasmicro.
//!
//! In the initial version, HuggingFace safetensors files are already in the
//! format wasmicro expects, so this tool's job is:
//!
//!   1. Download `model.safetensors` (and `config.json`, `tokenizer.json` if
//!      present) from the Hub.
//!   2. Validate that the safetensors header parses and contains the tensor
//!      names a BERT encoder needs.
//!   3. Copy the files into a local output directory ready to be served
//!      next to a wasmicro WASM bundle.
//!
//! Future versions will add int8 quantization and weight-name remapping.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use hf_hub::api::sync::ApiBuilder;
use wasmicro::ModelFile;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprint_usage();
        return ExitCode::from(2);
    }

    let model_id = &args[1];
    let output_dir = PathBuf::from(&args[2]);

    if let Err(e) = run(model_id, &output_dir) {
        eprintln!("error: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn eprint_usage() {
    eprintln!("usage: wasmicro-convert <hf-model-id> <output-dir>");
    eprintln!();
    eprintln!("examples:");
    eprintln!("  wasmicro-convert sentence-transformers/all-MiniLM-L6-v2 ./models/mini-lm");
    eprintln!("  wasmicro-convert bert-base-uncased ./models/bert-base");
}

fn run(model_id: &str, output_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    println!("[1/3] Downloading {model_id} ...");

    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(model_id.to_string());

    // Required: the model itself.
    let model_path = repo.get("model.safetensors")?;
    let dst_model = output_dir.join("model.safetensors");
    fs::copy(&model_path, &dst_model)?;

    // Best-effort: config.
    let dst_config = output_dir.join("config.json");
    match repo.get("config.json") {
        Ok(src) => {
            fs::copy(&src, &dst_config)?;
            println!("    + config.json");
        }
        Err(e) => println!("    - config.json: {e}"),
    }

    // Best-effort: tokenizer.
    let dst_tokenizer = output_dir.join("tokenizer.json");
    match repo.get("tokenizer.json") {
        Ok(src) => {
            fs::copy(&src, &dst_tokenizer)?;
            println!("    + tokenizer.json");
        }
        Err(e) => println!("    - tokenizer.json: {e}"),
    }

    // 2. Validate.
    println!("[2/3] Validating safetensors ...");
    let bytes = fs::read(&dst_model)?;
    let file = ModelFile::parse(&bytes)?;
    let names: Vec<&str> = file.names().collect();
    println!("    {} tensors in file", names.len());

    let looks_like_bert = names
        .iter()
        .any(|n| n.contains("embeddings.word_embeddings"));
    if looks_like_bert {
        println!("    detected: BERT-style encoder");
    } else {
        println!("    warning: no BERT word embeddings found — model may not be a BERT encoder");
    }

    // 3. Report.
    println!("[3/3] Done.");
    println!();
    println!("Output directory: {}", output_dir.display());
    println!("  model.safetensors    {} bytes", bytes.len());
    if dst_config.exists() {
        println!("  config.json          {} bytes", fs::metadata(&dst_config)?.len());
    }
    if dst_tokenizer.exists() {
        println!("  tokenizer.json       {} bytes", fs::metadata(&dst_tokenizer)?.len());
    }
    println!();
    println!("Next: load these files into wasmicro::ModelFile::parse(...)");
    Ok(())
}
