//! Audit train-only whole-row pen traversal without changing source arrays.
use anyhow::Result;
fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("corpus JSONL required");
    let mut reports = vec![];
    for sample in ink_corpus::load(std::path::Path::new(&path))?
        .iter()
        .filter(|s| s.split == "train")
    {
        let tokens = ink_layout::tokenize(&sample.label)?;
        let labels: Vec<_> = tokens.iter().filter(|s| s.as_str() != "<space>").collect();
        if labels.len() < 4
            || tokens.iter().filter(|s| s.as_str() == "<space>").count() < labels.len() - 1
        {
            continue;
        }
        let b = ink_corpus::bounds(sample);
        let cell = (b[2] - b[0]) / labels.len() as f32;
        let x: Vec<_> = sample
            .strokes
            .iter()
            .map(|s| s.points.iter().map(|p| p[0]).sum::<f32>() / s.points.len() as f32)
            .collect();
        let jumps = x.windows(2).filter(|p| p[0] - p[1] > cell * 1.5).count();
        if jumps > 0 {
            reports.push(serde_json::json!({"label":sample.label,"strokes":sample.strokes.len(),"large_backward_cell_jumps":jumps}));
        }
    }
    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
