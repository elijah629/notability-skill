//! Autonomous generation versus complete train sources. Never feeds source ink.
use anyhow::{Context, Result, ensure};
use ink_corpus::sequence::FEATURES;
use ink_corpus::{Sample, Stroke, bounds, file_hash, load, save};
use ink_layout::Glyph;
use ink_neural::sequence::SequenceDecoder;
use ink_neural::sequence::{HIDDEN, MIXTURES, STYLE};
use serde_json::json;
use std::{fs, path::PathBuf};

fn cloud(strokes: &[Stroke]) -> Vec<[f32; 2]> {
    scaled_cloud(strokes, false)
}
fn height_cloud(strokes: &[Stroke]) -> Vec<[f32; 2]> {
    scaled_cloud(strokes, true)
}
fn scaled_cloud(strokes: &[Stroke], height_units: bool) -> Vec<[f32; 2]> {
    let mut points = vec![];
    for stroke in strokes {
        points.push(stroke.points[0]);
        for p in stroke.points.windows(4).step_by(3) {
            for i in 1..=8 {
                let t = i as f32 / 8.;
                let u = 1. - t;
                points.push(std::array::from_fn(|a| {
                    u.powi(3) * p[0][a]
                        + 3. * u * u * t * p[1][a]
                        + 3. * u * t * t * p[2][a]
                        + t.powi(3) * p[3][a]
                }));
            }
        }
    }
    let b = bounds(&Sample {
        sample_id: String::new(),
        label: String::new(),
        source_sha256: String::new(),
        split: String::new(),
        strokes: strokes.to_vec(),
        metadata: serde_json::Value::Null,
    });
    let scale = if height_units {
        (b[3] - b[1]).max(strokes.iter().map(|s| s.width).fold(0_f32, f32::max))
    } else {
        (b[2] - b[0]).max(b[3] - b[1])
    }
    .max(1e-6);
    points
        .iter()
        .map(|p| [(p[0] - b[0]) / scale, (p[1] - b[1]) / scale])
        .collect()
}
fn chamfer(a: &[[f32; 2]], b: &[[f32; 2]]) -> f32 {
    let directed = |a: &[[f32; 2]], b: &[[f32; 2]]| {
        a.iter()
            .map(|p| {
                b.iter()
                    .map(|q| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2))
                    .fold(f32::INFINITY, f32::min)
            })
            .sum::<f32>()
            / a.len() as f32
    };
    ((directed(a, b) + directed(b, a)) * 0.5).sqrt()
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("decoder JSON required")?);
    let corpus = PathBuf::from(args.next().context("diagnostic corpus required")?);
    let out = PathBuf::from(args.next().context("output directory required")?);
    fs::create_dir_all(&out)?;
    let decoder = SequenceDecoder::load(&model)?;
    let hash = file_hash(&model)?;
    let rows = load(&corpus)?;
    let mut reports = vec![];
    for (i, row) in rows.iter().filter(|s| s.split == "train").enumerate() {
        let generated = decoder.generate_tokens(&ink_layout::tokenize(&row.label)?, 419, 0., 419);
        let mut labels = vec![];
        ink_layout::labels(&ink_layout::parse(&row.label)?, &mut labels);
        let source = if decoder.em_normalization {
            ink_corpus::sequence::encode_em(row, ink_layout::tokenize(&row.label)?.len())?
        } else {
            ink_corpus::sequence::encode(row)?
        };
        let original = ink_corpus::sequence::decode(&source.events, [0., 0.], 1.)?;
        let (states, mean) = decoder.condition(&row.label)?;
        let expected_events = (decoder.length.apply(&mean)?[0]
            + if decoder.length_per_token {
                (states.len() as f32).ln()
            } else {
                0.
            })
        .exp();
        let mut hidden = vec![0.; HIDDEN];
        let mut previous = [0.; FEATURES];
        let mut teacher = vec![];
        for (t, target) in source.events.iter().enumerate() {
            let (next, y) = decoder.step(
                &previous,
                &hidden,
                &states,
                &mean,
                &[0.; STYLE],
                t as f32 / source.events.len() as f32,
            )?;
            hidden = next;
            let mix = y[3..3 + MIXTURES]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let mut event = *target;
            event[..7].copy_from_slice(&y[3 + MIXTURES + mix * 14..3 + MIXTURES + mix * 14 + 7]);
            event = ink_corpus::sequence::canonical_feedback(event);
            teacher.push(event);
            previous = *target;
        }
        let teacher_strokes = ink_corpus::sequence::decode(&teacher, [0., 0.], 1.)?;
        let teacher_error = chamfer(&cloud(&original), &cloud(&teacher_strokes));
        let write = |strokes: Vec<Stroke>, name: &str| -> Result<()> {
            let glyph = Glyph {
                run_labels: labels.clone(),
                condition_label: None,
                label: row.label.clone(),
                mode: name.into(),
                model_id: decoder.model_id.clone(),
                decoder_sha256: hash.clone(),
                latent_seed: 419,
                strokes,
                transform: [1., 1., 0., 0.],
            };
            let drawing = ink_layout::generate_sequence(&row.label, 419, glyph)?;
            fs::write(
                out.join(format!("{i}-{name}.svg")),
                ink_layout::svg(&drawing)?,
            )?;
            Ok(())
        };
        write(original.clone(), "source")?;
        write(teacher_strokes, "teacher-source-guided")?;
        match generated {
            Ok(strokes) => {
                let error = chamfer(&cloud(&original), &cloud(&strokes));
                let height_error = chamfer(&height_cloud(&original), &height_cloud(&strokes));
                let degenerate = labels.len() > 1 && original.iter().all(|s| s.points.len() == 1);
                let legacy_pass = !degenerate && strokes.len() == original.len() && error < 0.06;
                let pass = legacy_pass && height_error < 0.06;
                reports.push(json!({"label":row.label,"source":row.sample_id,"source_degenerate_for_label":degenerate,"source_strokes":original.len(),"generated_strokes":strokes.len(),"source_events":source.events.len(),"expected_events":expected_events,"generated_events":strokes.iter().map(|s| 1+(s.points.len()-1)/3).sum::<usize>()+1,"normalized_chamfer":error,"height_chamfer":height_error,"legacy_extent_gate_pass":legacy_pass,"teacher_source_guided_chamfer":teacher_error,"pass":pass}));
                write(strokes,"generated")?;
            }
            Err(error) => reports.push(json!({"label":row.label,"source":row.sample_id,"error":error.to_string(),"pass":false})),
        }
    }
    let passed = reports.iter().filter(|r| r["pass"] == true).count();
    let report = json!({"decoder_sha256":hash,"samples":reports.len(),"passed":passed,"geometry_gate":"v2: exact stroke count, nondegenerate source, extent Chamfer < 0.06 and height Chamfer < 0.06; both normalize isotropically without changing aspect ratio; pen width floors zero-height scale","limits":"train-only geometric curriculum gate; not recognition, style or generalization evidence","results":reports});
    save(&out.join("report.json"), &report)?;
    println!("{report}");
    ensure!(
        passed == report["samples"].as_u64().unwrap() as usize,
        "autonomous content gate failed"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wide_text_cannot_hide_a_badly_misplaced_middle_stroke() {
        let line = |x: f32, i: usize| Stroke {
            points: vec![[x, 0.], [x, 0.3], [x, 0.7], [x, 1.]],
            width: 0.02,
            source_index: i,
            uuid: format!("fixture-{i}"),
        };
        let source = vec![line(0., 0), line(5., 1), line(10., 2)];
        let generated = vec![line(0., 0), line(5.3, 1), line(10., 2)];
        assert!(
            chamfer(&cloud(&source), &cloud(&generated)) < 0.06,
            "fixture must reproduce the old false-positive extent gate"
        );
        assert!(
            chamfer(&height_cloud(&source), &height_cloud(&generated)) > 0.06,
            "a 0.3-height relocation must fail regardless of sentence width"
        );
    }
}
