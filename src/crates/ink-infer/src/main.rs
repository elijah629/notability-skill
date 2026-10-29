use anyhow::{Context, Result};
use clap::Parser;
use std::{fs, path::PathBuf};
#[derive(Parser)]
#[command(about = "Generate neural vector handwriting without Python or a training runtime")]
struct Cli {
    #[arg(long)]
    input: String,
    #[arg(long, default_value = "output/handwriting")]
    out: PathBuf,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 0.0)]
    variation: f32,
    #[arg(long)]
    decoder: Option<PathBuf>,
}
fn run() -> Result<()> {
    let c = Cli::parse();
    let path = c.decoder.unwrap_or(
        std::env::current_exe()?
            .parent()
            .context("binary directory")?
            .parent()
            .context("skill directory")?
            .join("decoder.json"),
    );
    let decoder = ink_neural::Runtime::load(&path)?;
    let seed = c.seed.map(Ok).unwrap_or_else(ink_neural::random_seed)?;
    let drawing = decoder.draw(&c.input, seed, &ink_corpus::file_hash(&path)?, c.variation)?;
    if let Some(p) = c.out.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(c.out.with_extension("svg"), ink_layout::svg(&drawing)?)?;
    ink_corpus::save(&c.out.with_extension("ink.json"), &drawing)?;
    println!(
        "Wrote {} ({} neural glyphs, model {})",
        c.out.with_extension("svg").display(),
        ink_layout::drawing_labels(&drawing).len(),
        decoder.model_id()
    );
    println!("Seed: {seed}; variation: {}", c.variation);
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
