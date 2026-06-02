//! `wasmicro-convert`: download a HuggingFace model and prepare it for wasmicro.
//!
//! HuggingFace safetensors files are already close to the format wasmicro
//! expects, so this tool's job is:
//!
//!   1. Download `model.safetensors` (and `config.json`, `vocab.txt`,
//!      `tokenizer.json` if present) from the Hub.
//!   2. Validate that the safetensors header parses and contains the tensor
//!      names a BERT encoder needs.
//!   3. Optionally write a weight-only quantized BERT safetensors file for
//!      linear layers.
//!   4. Copy the files into a local output directory ready to be served
//!      next to a wasmicro WASM bundle.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use hf_hub::api::sync::ApiBuilder;
use wasmicro::{Dtype, ModelFile, QuantizedTensorI8, QuantizedTensorU8};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuantMode {
    I8,
    U8,
}

impl QuantMode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "i8" => Some(Self::I8),
            "u8" | "q8" => Some(Self::U8),
            _ => None,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::U8 => "u8",
        }
    }
}

struct CliArgs {
    model_id: String,
    output_dir: PathBuf,
    quantize: Option<QuantMode>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let args = match parse_args(&args) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("error: {e}");
            eprint_usage();
            return ExitCode::from(2);
        }
    };

    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    if args.len() < 3 {
        return Err("missing required arguments".to_string());
    }

    let mut quantize = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--quantize" => {
                let Some(value) = args.get(i + 1) else {
                    return Err("--quantize requires i8, u8, or q8".to_string());
                };
                quantize = Some(
                    QuantMode::parse(value)
                        .ok_or_else(|| "--quantize must be one of: i8, u8, q8".to_string())?,
                );
                i += 2;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(CliArgs {
        model_id: args[1].clone(),
        output_dir: PathBuf::from(&args[2]),
        quantize,
    })
}

fn eprint_usage() {
    eprintln!("usage: wasmicro-convert <hf-model-id> <output-dir> [--quantize i8|u8|q8]");
    eprintln!();
    eprintln!("examples:");
    eprintln!("  wasmicro-convert sentence-transformers/all-MiniLM-L6-v2 ./models/mini-lm");
    eprintln!("  wasmicro-convert bert-base-uncased ./models/bert-base --quantize i8");
}

fn run(args: &CliArgs) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(&args.output_dir)?;
    println!("[1/4] Downloading {} ...", args.model_id);

    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(args.model_id.to_string());

    // Required: the model itself.
    let model_path = repo.get("model.safetensors")?;
    let dst_model = args.output_dir.join("model.safetensors");
    fs::copy(&model_path, &dst_model)?;

    // Best-effort: config.
    let dst_config = args.output_dir.join("config.json");
    match repo.get("config.json") {
        Ok(src) => {
            fs::copy(&src, &dst_config)?;
            println!("    + config.json");
        }
        Err(e) => println!("    - config.json: {e}"),
    }

    // Best-effort: WordPiece vocabulary.
    let dst_vocab = args.output_dir.join("vocab.txt");
    match repo.get("vocab.txt") {
        Ok(src) => {
            fs::copy(&src, &dst_vocab)?;
            println!("    + vocab.txt");
        }
        Err(e) => println!("    - vocab.txt: {e}"),
    }

    // Best-effort: tokenizer metadata.
    let dst_tokenizer = args.output_dir.join("tokenizer.json");
    match repo.get("tokenizer.json") {
        Ok(src) => {
            fs::copy(&src, &dst_tokenizer)?;
            println!("    + tokenizer.json");
        }
        Err(e) => println!("    - tokenizer.json: {e}"),
    }

    // 2. Validate.
    println!("[2/4] Validating safetensors ...");
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

    let mut quantized_path = None;
    if let Some(mode) = args.quantize {
        println!(
            "[3/4] Quantizing BERT linear weights ({}) ...",
            mode.suffix()
        );
        let quantized = quantize_safetensors(&bytes, mode)?;
        let path = args
            .output_dir
            .join(format!("model.{}.safetensors", mode.suffix()));
        fs::write(&path, quantized)?;
        println!("    + {}", path.file_name().unwrap().to_string_lossy());
        quantized_path = Some(path);
    } else {
        println!("[3/4] Quantization skipped.");
    }

    // 4. Report.
    println!("[4/4] Done.");
    println!();
    println!("Output directory: {}", args.output_dir.display());
    println!("  model.safetensors    {} bytes", bytes.len());
    if let Some(path) = &quantized_path {
        println!(
            "  {}    {} bytes",
            path.file_name().unwrap().to_string_lossy(),
            fs::metadata(path)?.len()
        );
    }
    if dst_config.exists() {
        println!(
            "  config.json          {} bytes",
            fs::metadata(&dst_config)?.len()
        );
    }
    if dst_vocab.exists() {
        println!(
            "  vocab.txt            {} bytes",
            fs::metadata(&dst_vocab)?.len()
        );
    }
    if dst_tokenizer.exists() {
        println!(
            "  tokenizer.json       {} bytes",
            fs::metadata(&dst_tokenizer)?.len()
        );
    }
    println!();
    println!("Next: load model.safetensors or model.<mode>.safetensors into ModelFile::parse(...) and vocab.txt into WordPieceTokenizer::from_vocab_bytes(...)");
    Ok(())
}

struct OutputTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn quantize_safetensors(
    bytes: &[u8],
    mode: QuantMode,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = ModelFile::parse(bytes)?;
    let mut out = Vec::new();

    for name in file.names() {
        let view = file.get(name)?;
        if should_quantize_bert_linear(name, view.dtype, view.shape) {
            let weight = view.to_tensor()?;
            match mode {
                QuantMode::I8 => {
                    let q = QuantizedTensorI8::quantize_per_row(&weight);
                    out.push(OutputTensor {
                        name: name.to_string(),
                        dtype: "I8",
                        shape: view.shape.to_vec(),
                        data: q.data().iter().map(|v| *v as u8).collect(),
                    });
                    out.push(OutputTensor {
                        name: format!("{name}.scale"),
                        dtype: "F32",
                        shape: vec![q.scales().len()],
                        data: f32_slice_to_le_bytes(q.scales()),
                    });
                }
                QuantMode::U8 => {
                    let q = QuantizedTensorU8::quantize_per_row(&weight);
                    out.push(OutputTensor {
                        name: name.to_string(),
                        dtype: "U8",
                        shape: view.shape.to_vec(),
                        data: q.data().to_vec(),
                    });
                    out.push(OutputTensor {
                        name: format!("{name}.scale"),
                        dtype: "F32",
                        shape: vec![q.scales().len()],
                        data: f32_slice_to_le_bytes(q.scales()),
                    });
                    out.push(OutputTensor {
                        name: format!("{name}.zero_point"),
                        dtype: "U8",
                        shape: vec![q.zero_points().len()],
                        data: q.zero_points().to_vec(),
                    });
                }
            }
        } else {
            out.push(OutputTensor {
                name: name.to_string(),
                dtype: dtype_name(view.dtype),
                shape: view.shape.to_vec(),
                data: view.raw.to_vec(),
            });
        }
    }

    Ok(write_safetensors(&out))
}

fn should_quantize_bert_linear(name: &str, dtype: Dtype, shape: &[usize]) -> bool {
    dtype == Dtype::F32
        && shape.len() == 2
        && [
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
            "attention.output.dense.weight",
            "intermediate.dense.weight",
            "output.dense.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::I8 => "I8",
        Dtype::U8 => "U8",
        Dtype::I32 => "I32",
        Dtype::I64 => "I64",
        Dtype::Bool => "BOOL",
    }
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn write_safetensors(tensors: &[OutputTensor]) -> Vec<u8> {
    let mut offset = 0usize;
    let mut header = String::from("{");
    for (i, tensor) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let end = offset + tensor.data.len();
        header.push('"');
        header.push_str(&json_escape(&tensor.name));
        header.push_str("\":{");
        header.push_str("\"dtype\":\"");
        header.push_str(tensor.dtype);
        header.push_str("\",\"shape\":[");
        for (dim_i, dim) in tensor.shape.iter().enumerate() {
            if dim_i > 0 {
                header.push(',');
            }
            header.push_str(&dim.to_string());
        }
        header.push_str("],\"data_offsets\":[");
        header.push_str(&offset.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");
        offset = end;
    }
    header.push('}');

    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for tensor in tensors {
        out.extend_from_slice(&tensor.data);
    }
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_tensor(name: &str, shape: &[usize], values: &[f32]) -> OutputTensor {
        OutputTensor {
            name: name.to_string(),
            dtype: "F32",
            shape: shape.to_vec(),
            data: f32_slice_to_le_bytes(values),
        }
    }

    fn sample_model() -> Vec<u8> {
        write_safetensors(&[
            f32_tensor(
                "embeddings.word_embeddings.weight",
                &[2, 2],
                &[0.0, 1.0, 2.0, 3.0],
            ),
            f32_tensor(
                "encoder.layer.0.intermediate.dense.weight",
                &[2, 2],
                &[0.0, 1.0, -2.0, 2.0],
            ),
            f32_tensor("encoder.layer.0.intermediate.dense.bias", &[2], &[0.0, 0.0]),
        ])
    }

    #[test]
    fn quantize_i8_only_changes_bert_linear_weights() {
        let output = quantize_safetensors(&sample_model(), QuantMode::I8).unwrap();
        let file = ModelFile::parse(&output).unwrap();

        assert_eq!(
            file.get("embeddings.word_embeddings.weight").unwrap().dtype,
            Dtype::F32
        );
        assert_eq!(
            file.get("encoder.layer.0.intermediate.dense.weight")
                .unwrap()
                .dtype,
            Dtype::I8
        );
        assert_eq!(
            file.get("encoder.layer.0.intermediate.dense.weight.scale")
                .unwrap()
                .dtype,
            Dtype::F32
        );
    }

    #[test]
    fn quantize_u8_adds_zero_points() {
        let output = quantize_safetensors(&sample_model(), QuantMode::U8).unwrap();
        let file = ModelFile::parse(&output).unwrap();

        assert_eq!(
            file.get("encoder.layer.0.intermediate.dense.weight")
                .unwrap()
                .dtype,
            Dtype::U8
        );
        assert_eq!(
            file.get("encoder.layer.0.intermediate.dense.weight.zero_point")
                .unwrap()
                .dtype,
            Dtype::U8
        );
    }
}
