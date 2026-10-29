//! Prepare a small train-only content gate from unchanged complete source rows.
//! Usage: cargo run -p ink-model --example curriculum -- CORPUS OUT LABEL...
use anyhow::{Context, Result, ensure};
use ink_corpus::{Sample, audit, file_hash, load, save};
use serde_json::json;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let corpus = PathBuf::from(args.next().context("corpus JSONL required")?);
    let out = PathBuf::from(args.next().context("fresh output directory required")?);
    let labels: Vec<_> = args.collect();
    ensure!(
        labels.len() >= 2,
        "at least two distinct supplied labels required"
    );
    ensure!(!out.exists(), "output already exists");
    let all = load(&corpus)?;
    audit(&all)?;
    let mut rows: Vec<Sample> = Vec::new();
    let mut excluded = vec![];
    for label in &labels {
        let mut written = vec![];
        ink_layout::labels(&ink_layout::parse(label)?, &mut written);
        for row in all.iter().filter(|s| {
            s.split == "train"
                && &s.label == label
                && written.len() > 1
                && s.strokes.iter().all(|p| p.points.len() == 1)
        }) {
            excluded.push(json!({"id":row.sample_id,"label":row.label,"reason":"multi-glyph label has no native curve; unusable content gate, source retained unchanged"}));
        }
        let row = all
            .iter()
            .filter(|s| s.split == "train" && &s.label == label)
            .filter(|s| written.len() <= 1 || s.strokes.iter().any(|p| p.points.len() > 1))
            .min_by_key(|s| s.strokes.iter().map(|p| p.points.len()).sum::<usize>())
            .with_context(|| format!("no complete train row for {label:?}"))?;
        ensure!(
            !rows.iter().any(|r| r.sample_id == row.sample_id),
            "duplicate label"
        );
        rows.push(row.clone());
    }
    // Retained for split/audit contracts only; the trainer never optimizes these.
    for split in ["validation", "test"] {
        rows.push(
            all.iter()
                .find(|s| s.split == split)
                .context("missing holdout")?
                .clone(),
        );
    }
    audit(&rows)?;
    fs::create_dir_all(&out)?;
    let encoded = rows
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?
        .join("\n")
        + "\n";
    fs::write(out.join("corpus.jsonl"), encoded)?;
    let receipt = json!({"role":"diagnostic curriculum, never a general model", "source_sha256":file_hash(&corpus)?,
        "subset_sha256":file_hash(&out.join("corpus.jsonl"))?, "excluded_from_diagnostic_only":excluded,"rows":rows.iter().map(|s| json!({"id":s.sample_id,"label":s.label,"split":s.split,"strokes":s.strokes.len()})).collect::<Vec<_>>()});
    save(&out.join("receipt.json"), &receipt)?;
    println!("{receipt}");
    Ok(())
}
