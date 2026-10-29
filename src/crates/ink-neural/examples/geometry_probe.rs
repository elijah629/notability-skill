//! Fixed-condition geometry diagnostic; never reads the source corpus.
use anyhow::{Context, Result, ensure};
use std::{fs, path::PathBuf};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("decoder JSON required")?);
    let out = PathBuf::from(args.next().context("output directory required")?);
    let d = ink_neural::Runtime::load(&model)?;
    let hash = ink_corpus::file_hash(&model)?;
    fs::create_dir_all(&out)?;
    let input = r"\begin{gathered}abcdefghijklmnopqrstuvwxyz\\ABCDEFGHIJKLMNOPQRSTUVWXYZ\\0123456789+-=\\\text{Hello, World! Quick brown fox jumps.}\end{gathered}";
    let drawing = d.draw(input, 419, &hash, 0.)?;
    fs::write(out.join("alphabet.svg"), ink_layout::svg(&drawing)?)?;
    ink_corpus::save(&out.join("alphabet.ink.json"), &drawing)?;
    let minus = d.draw("-", 419, &hash, 0.)?;
    let s = &minus.glyphs[0].strokes[0];
    let a = s.points[0];
    let b = *s.points.last().unwrap();
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let norm = dx.hypot(dy).max(1e-6);
    let arc: f32 = s
        .points
        .windows(2)
        .map(|p| (p[1][0] - p[0][0]).hypot(p[1][1] - p[0][1]))
        .sum();
    let rms = (s
        .points
        .iter()
        .map(|p| {
            let distance = ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / norm;
            distance.powi(2)
        })
        .sum::<f32>()
        / s.points.len() as f32)
        .sqrt()
        / norm;
    let straight = rms < 0.01
        && norm > 0.05
        && norm / arc.max(1e-6) > 0.99
        && minus.glyphs[0].strokes.len() == 1;
    let mut loops = serde_json::Map::new();
    for label in ["o", "O", "0"] {
        let glyph = d.draw(label, 419, &hash, 0.)?;
        let mut gap = f32::INFINITY;
        for s in &glyph.glyphs[0].strokes {
            let mut b = [
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ];
            for p in &s.points {
                b[0] = b[0].min(p[0]);
                b[1] = b[1].min(p[1]);
                b[2] = b[2].max(p[0]);
                b[3] = b[3].max(p[1]);
            }
            let diagonal = (b[2] - b[0]).hypot(b[3] - b[1]);
            let a = s.points[0];
            let b = s.points.last().unwrap();
            if diagonal > 0.1 {
                gap = gap.min((a[0] - b[0]).hypot(a[1] - b[1]) / diagonal);
            }
        }
        loops.insert(
            label.into(),
            serde_json::json!({"relative_endpoint_gap":gap,"closed":gap<0.05}),
        );
    }
    let report = serde_json::json!({"seed":419,"variation":0,"decoder_sha256":hash,"minus_chord_relative_rmse":rms,"minus_chord_to_path_ratio":norm/arc.max(1e-6),"minus_contract":straight,"closed_loop_gaps":loops,"recognition":"not independently scored; inspect alphabet.svg"});
    ink_corpus::save(&out.join("geometry.json"), &report)?;
    println!("{report}");
    ensure!(
        straight,
        "minus must be one noncollapsed straight stroke, with less than 1 percent chord deviation"
    );
    Ok(())
}
