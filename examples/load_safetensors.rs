//! Example: load a safetensors file from disk and print a summary.
//!
//! Usage:
//!
//! ```bash
//! cargo run --example load_safetensors -- path/to/model.safetensors
//! ```
//!
//! For a quick smoke test without downloading a real model, run without
//! arguments — the example builds a synthetic safetensors blob in memory
//! and parses it back.

use std::env;
use std::fs;
use std::process::ExitCode;

use wasmicro::{ops, Dtype, ModelFile, Tensor};

fn main() -> ExitCode {
    let arg = env::args().nth(1);
    let bytes = match arg {
        Some(path) => match fs::read(&path) {
            Ok(b) => {
                println!("Loaded {} bytes from {}", b.len(), path);
                b
            }
            Err(e) => {
                eprintln!("failed to read {}: {}", path, e);
                return ExitCode::from(1);
            }
        },
        None => {
            println!("No file given; building a synthetic safetensors blob.");
            synthetic_safetensors()
        }
    };

    let model = match ModelFile::parse(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse failed: {}", e);
            return ExitCode::from(1);
        }
    };

    println!("\nTensors:");
    for name in model.names() {
        let view = model.get(name).expect("name from iterator must exist");
        println!(
            "  {:<32} dtype={:?} shape={:?}",
            name, view.dtype, view.shape
        );
    }

    // Demonstrate running a Linear layer if a weight + bias pair is present.
    if let (Ok(w_view), Ok(b_view)) = (model.get("weight"), model.get("bias")) {
        if w_view.dtype == Dtype::F32 && b_view.dtype == Dtype::F32 {
            let weight = w_view.to_tensor().expect("weight to_tensor");
            let bias = b_view.to_tensor().expect("bias to_tensor");
            let in_features = weight.shape().as_slice()[1];

            let input = Tensor::from_vec(vec![1.0; in_features], &[1, in_features]);
            let output = ops::linear::linear(&input, &weight, Some(&bias));
            println!(
                "\nLinear(input=ones[1, {}]) -> output shape {:?}",
                in_features,
                output.shape().as_slice()
            );
            println!(
                "first few outputs: {:?}",
                &output.data()[..output.numel().min(8)]
            );
        }
    }

    ExitCode::SUCCESS
}

/// Builds a minimal in-memory safetensors blob containing a 4x3 weight matrix
/// and a 4-element bias.
fn synthetic_safetensors() -> Vec<u8> {
    let weight: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let bias: Vec<f32> = vec![0.01, 0.02, 0.03, 0.04];

    let mut data_bytes = Vec::new();
    for &v in &weight {
        data_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let weight_end = data_bytes.len();
    for &v in &bias {
        data_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let bias_end = data_bytes.len();

    let header = format!(
        r#"{{"weight":{{"dtype":"F32","shape":[4,3],"data_offsets":[0,{weight_end}]}},"bias":{{"dtype":"F32","shape":[4],"data_offsets":[{weight_end},{bias_end}]}},"__metadata__":{{"format":"pt"}}}}"#,
    );
    let header = header.into_bytes();

    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data_bytes);
    out
}
