//! Fixed-condition review of learned glyph modes. No source trajectories are read.
use anyhow::{Context, Result, ensure};
use ink_corpus::{file_hash, save};
use ink_neural::{Runtime, sequence::SequenceDecoder};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub fn glyphs(
    model: &Path,
    out: &Path,
    selection: Option<&Path>,
    reference: Option<&Path>,
) -> Result<Value> {
    let mut manifest = super::show(model)?;
    let decoder_path = model.join("decoder.json");
    let hash = file_hash(&decoder_path)?;
    let mut decoder = SequenceDecoder::load(&decoder_path)?;
    let (reference_head, reference_id, reference_hash) = if let Some(path) = reference {
        let reference_manifest = super::show(path)?;
        ensure!(
            reference_manifest["dataset_sha256"] == manifest["dataset_sha256"],
            "reference corpus mismatch"
        );
        let frozen = SequenceDecoder::load(&path.join("decoder.json"))?
            .glyph_head
            .context("reference head missing")?;
        ensure!(
            frozen.anchor_head.is_none(),
            "reference must be a single trained expert"
        );
        (
            Some(frozen),
            reference_manifest["id"].clone(),
            reference_manifest["decoder_sha256"].clone(),
        )
    } else {
        (
            decoder
                .glyph_head
                .as_ref()
                .and_then(|h| h.anchor_head.as_deref().cloned()),
            manifest["glyph_mode_review"]["reference_model"].clone(),
            manifest["glyph_mode_review"]["reference_decoder_sha256"].clone(),
        )
    };
    let head = decoder.glyph_head.as_ref().context("glyph head required")?;
    let mut sources = vec![vec![BTreeSet::<String>::new(); head.modes]; head.vocabulary.len()];
    let support_path = model.join("glyph-topology-evidence.json");
    let known_support = support_path.exists();
    if known_support {
        ensure!(
            manifest["glyph_training"]["topology_evidence_sha256"] == file_hash(&support_path)?,
            "topology support hash mismatch"
        );
        let evidence: BTreeMap<String, (String, Vec<(usize, usize)>)> =
            serde_json::from_slice(&fs::read(support_path)?)?;
        for (source, choices) in evidence.values() {
            for &(token, mode) in choices {
                ensure!(
                    token < sources.len() && mode < head.modes,
                    "invalid mode evidence"
                );
                sources[token][mode].insert(source.clone());
            }
        }
    }
    if let Some(path) = selection {
        ensure!(
            manifest["approved_for_default"] != true,
            "approved models are immutable"
        );
        ensure!(
            known_support,
            "mode calibration requires private training support state"
        );
        let review: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            review["model"] == manifest["id"] && review["decoder_sha256"] == hash,
            "stale mode selection"
        );
        ensure!(
            review["reference_model"] == reference_id
                && review["reference_decoder_sha256"] == reference_hash,
            "reference changed since mode review"
        );
        ensure!(
            review["seed"].as_u64() == Some(419) && review["variation"].as_f64() == Some(0.),
            "mode review must use seed 419 and variation zero"
        );
        let choices = review["modes"]
            .as_object()
            .context("mode choices missing")?;
        let mut preferred = if head.preferred_modes.is_empty() {
            vec![0; head.vocabulary.len()]
        } else {
            head.preferred_modes.clone()
        };
        let mut anchors: BTreeSet<_> = head.anchor_conditions.iter().cloned().collect();
        let mut weak_conditions = BTreeSet::new();
        for (label, mode) in choices {
            let condition = ink_layout::neural_condition(label).to_string();
            let token = head
                .vocabulary
                .binary_search(&condition)
                .map_err(|_| anyhow::anyhow!("unknown selected condition {label}"))?;
            if mode.as_str() == Some("anchor") {
                ensure!(
                    reference_head
                        .as_ref()
                        .is_some_and(|h| h.supported.contains(&condition)),
                    "unsupported reference condition"
                );
                anchors.insert(condition.clone());
                continue;
            }
            let mode = mode.as_u64().context("mode must be an integer")? as usize;
            ensure!(mode < head.modes, "selected mode out of range");
            if sources[token][mode].is_empty()
                && !(mode == 0 && sources[token].iter().all(BTreeSet::is_empty))
            {
                let path = model.join("glyph-lineage.json");
                ensure!(
                    manifest["glyph_lineage_sha256"] == file_hash(&path)?,
                    "weak-mode lineage integrity"
                );
                let lineage: Vec<Value> = serde_json::from_slice(&fs::read(path)?)?;
                ensure!(
                    lineage.iter().any(|row| row["split"] == "train"
                        && row["conditions"]
                            .as_array()
                            .is_some_and(|v| v.iter().any(|s| s.as_str() == Some(&condition)))),
                    "selected condition has no whole-row train supervision"
                );
                weak_conditions.insert(condition.clone());
            }
            preferred[token] = mode;
            anchors.remove(&condition);
        }
        let selected = decoder.glyph_head.as_mut().unwrap();
        selected.preferred_modes = preferred;
        selected.anchor_conditions = anchors.into_iter().collect();
        selected.anchor_head = if selected.anchor_conditions.is_empty() {
            None
        } else {
            reference_head.clone().map(Box::new)
        };
        decoder.validate()?;
        fs::create_dir_all(out)?;
        let backup = out.join(format!("before-{hash}"));
        ensure!(!backup.exists(), "calibration backup already exists");
        fs::create_dir_all(&backup)?;
        for name in ["decoder.json", "glyph-decoder.json", "manifest.json"] {
            fs::copy(model.join(name), backup.join(name))?;
        }
        let update = (|| -> Result<()> {
            save(&decoder_path, &decoder)?;
            save(
                &model.join("glyph-decoder.json"),
                decoder.glyph_head.as_ref().unwrap(),
            )?;
            manifest["decoder_sha256"] = json!(file_hash(&decoder_path)?);
            manifest["glyph_decoder_sha256"] = json!(file_hash(&model.join("glyph-decoder.json"))?);
            manifest["glyph_mode_review"] = json!({"source_decoder_sha256":hash,"selection_sha256":file_hash(path)?,"conditions":choices.len(),"seed":419,"variation":0,"reference_model":reference_id,"reference_decoder_sha256":reference_hash,"weak_context_modes":weak_conditions,"basis":"visual selection among learned supported modes and frozen trained reference weights; weak-context choices are explicit, not accepted character annotations"});
            save(&model.join("manifest.json"), &manifest)?;
            Ok(())
        })();
        if let Err(error) = update {
            for name in ["decoder.json", "glyph-decoder.json", "manifest.json"] {
                fs::copy(backup.join(name), model.join(name))
                    .context("restore calibration backup")?;
            }
            return Err(error);
        }
        return Ok(
            json!({"model":manifest["id"],"calibrated_conditions":choices.len(),"decoder_sha256":manifest["decoder_sha256"],"approved_for_default":false}),
        );
    }
    fs::create_dir_all(out)?;
    let mut labels: Vec<_> = head.supported.clone();
    labels.sort_by_key(|s| {
        let c = s.chars().next().unwrap_or(' ');
        (
            if s.len() == 1 && c.is_ascii_lowercase() {
                0
            } else if s.len() == 1 && c.is_ascii_uppercase() {
                1
            } else if s.len() == 1 && c.is_ascii_digit() {
                2
            } else {
                3
            },
            s.clone(),
        )
    });
    let modes = head.modes;
    let columns = modes + usize::from(reference_head.is_some());
    let vocabulary = head.vocabulary.clone();
    let original = head.preferred_modes.clone();
    let mut pages = vec![];
    let mut failures = vec![];
    let mut selection_modes = serde_json::Map::new();
    for (page, chunk) in labels.chunks(6).enumerate() {
        let width = 1.4 + columns as f32 * 2.2;
        let height = 0.7 + chunk.len() as f32 * 1.9;
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" width=\"1800\" height=\"{}\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
            height / width * 1800.
        );
        for mode in 0..columns {
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"0.35\" font-size=\"0.24\" font-family=\"sans-serif\">{}</text>",
                1.4 + mode as f32 * 2.2,
                if mode < modes {
                    format!("mode {mode}")
                } else {
                    "anchor".into()
                }
            ));
        }
        for (row, label) in chunk.iter().enumerate() {
            let condition = ink_layout::neural_condition(label).to_string();
            let token = vocabulary.binary_search(&condition).unwrap();
            let y = 0.6 + row as f32 * 1.9;
            let preferred = original.get(token).copied().unwrap_or(0);
            selection_modes.insert(
                condition.clone(),
                if head.anchor_conditions.contains(&condition) {
                    json!("anchor")
                } else {
                    json!(preferred)
                },
            );
            svg.push_str(&format!(
                "<text x=\"0.05\" y=\"{}\" font-size=\"0.28\" font-family=\"sans-serif\">{}</text>",
                y + 0.8,
                xml(label)
            ));
            let mut support: Vec<_> = sources[token].iter().map(|s| Some(s.len())).collect();
            if reference_head.is_some() {
                support.push(None);
            }
            for (mode, support) in support.iter().enumerate() {
                let x = 1.4 + mode as f32 * 2.2;
                let mut variant = decoder.clone();
                let head = variant.glyph_head.as_mut().unwrap();
                if head.preferred_modes.is_empty() {
                    head.preferred_modes = vec![0; vocabulary.len()];
                }
                if mode < modes {
                    head.preferred_modes[token] = mode;
                    head.anchor_conditions.clear();
                } else {
                    head.anchor_head = reference_head.clone().map(Box::new);
                    head.anchor_conditions = vec![condition.clone()];
                }
                let input = if label.starts_with('\\') {
                    label.clone()
                } else if label == "{" || label == "}" {
                    format!("\\{label}")
                } else {
                    format!("\\text{{{label}}}")
                };
                match Runtime::Sequence(Box::new(variant)).draw(&input, 419, &hash, 0.) {
                    Ok(drawing) => {
                        let rendered = ink_layout::svg(&drawing)?;
                        let (_, body) = rendered.split_once('\n').context("SVG header")?;
                        let body = body.rsplit_once("</svg>").context("SVG closing tag")?.0;
                        let scale = (1.6 / drawing.width).min(1.2 / drawing.height);
                        svg.push_str(&format!(
                            "<g transform=\"translate({} {}) scale({scale})\">{body}</g>",
                            x + (1.8 - drawing.width * scale) * 0.5,
                            y + 0.1
                        ));
                    }
                    Err(error) => {
                        failures.push(json!({"label":label,"mode":mode,"error":error.to_string()}));
                    }
                }
                svg.push_str(&format!("<text x=\"{x}\" y=\"{}\" font-size=\"0.17\" fill=\"#555\" font-family=\"sans-serif\">{}{} </text>",y+1.7,if mode==modes {"trained reference".into()} else if known_support {format!("sources: {}",support.unwrap_or(0))} else {"sources: ?".into()},if mode==preferred {"; selected"} else {""}));
            }
        }
        svg.push_str("</svg>\n");
        let file = format!("page-{:02}.svg", page + 1);
        fs::write(out.join(&file), svg)?;
        pages.push(json!({"file":file,"labels":chunk}));
    }
    let template = json!({"model":manifest["id"],"decoder_sha256":hash,"reference_model":reference_id,"reference_decoder_sha256":reference_hash,"seed":419,"variation":0,"modes":selection_modes});
    save(&out.join("selection.json"), &template)?;
    let report = json!({"model":manifest["id"],"decoder_sha256":hash,"reference_model":reference_id,"reference_decoder_sha256":reference_hash,"seed":419,"variation":0,"pages":pages,"failures":failures,"selection":out.join("selection.json"),"source_support_known":known_support});
    save(&out.join("review.json"), &report)?;
    Ok(report)
}
