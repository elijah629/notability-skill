//! Fixed, unseen generation requests for visual review. Reads weights only.
use anyhow::{Context, Result};
use ink_corpus::{file_hash, save};
use ink_neural::Runtime;
use serde_json::json;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let decoder = PathBuf::from(args.next().context("decoder JSON required")?);
    let out = PathBuf::from(args.next().context("output directory required")?);
    let runtime = Runtime::load(&decoder)?;
    let checksum = file_hash(&decoder)?;
    fs::create_dir_all(&out)?;
    let mut reports = vec![];
    let requested: Vec<String> = args.collect();
    let requests: Vec<(String, String)> = if requested.is_empty() {
        [
            ("hello", "Hello, World!"),
            ("lowercase", "abcdefghijklmnopqrstuvwxyz"),
            ("uppercase", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            ("digits", "0123456789"),
            (
                "sentence",
                r"\text{The quick brown fox jumps over the lazy dog.}",
            ),
            ("boxed", r"\boxed{Hello, World!}"),
            ("matrix", r"\begin{bmatrix}1&2\\3&4\end{bmatrix}"),
            ("fraction", r"\left[\frac{T+t}{2}\right]"),
            ("T-Gamma", r"T\quad\Gamma\quad t"),
            ("root", r"\sqrt{x^2+1}"),
            ("repeated", r"\text{balloon minimum}"),
            ("equation", "x+y=12"),
        ]
        .into_iter()
        .map(|(a, b)| (a.into(), b.into()))
        .collect()
    } else {
        requested
            .into_iter()
            .enumerate()
            .map(|(i, input)| (format!("custom-{i}"), input))
            .collect()
    };
    for (name, input) in requests {
        match runtime.draw(&input, 419, &checksum, 0.) {
            Ok(drawing) => {
                fs::write(out.join(format!("{name}.svg")), ink_layout::svg(&drawing)?)?;
                save(&out.join(format!("{name}.ink.json")), &drawing)?;
                reports.push(json!({"name":name,"input":input,"generated":true,"strokes":drawing.glyphs.iter().map(|g|g.strokes.len()).sum::<usize>(),"procedural_structures":drawing.procedures.len()}));
            }
            Err(error) => reports.push(
                json!({"name":name,"input":input,"generated":false,"error":error.to_string()}),
            ),
        }
    }
    let result = json!({"model":runtime.model_id(),"decoder_sha256":checksum,"seed":419,"variation":0,"content_recognition":"requires visual review; generation success is not recognition","results":reports});
    save(&out.join("gallery.json"), &result)?;
    println!("{result}");
    Ok(())
}
