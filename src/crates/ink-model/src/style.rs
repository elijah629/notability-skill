//! Read-only evidence for handwriting revision. Diagnostics never create glyph labels.
use anyhow::{Result, ensure};
use ink_corpus::Sample;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// Mixed Greek practice may use a serifed Tau-like form for supplied Latin T.
/// Keep its original complete-row label, but do not certify it as a Latin T anchor.
pub fn latin_anchor_allowed(conditions: &[String], label: &str) -> bool {
    label != "T"
        || !conditions.iter().any(|s| {
            [
                "\\Phi", "\\Pi", "\\Theta", "\\Sigma", "\\Omega", "\\Delta", "\\Lambda", "\\tau",
            ]
            .contains(&s.as_str())
        })
}

/// Public motion pretraining and personal fine-tuning are separate corpus stages.
/// Missing metadata means historical personal data, never an implicit public license.
pub fn domain(samples: &[Sample]) -> Result<&'static str> {
    let mut domains = BTreeSet::new();
    let mut writers = BTreeMap::<String, String>::new();
    for sample in samples {
        let domain = sample.metadata["data_domain"]
            .as_str()
            .unwrap_or("personal");
        ensure!(
            matches!(domain, "personal" | "public"),
            "unknown data domain for {}",
            sample.sample_id
        );
        domains.insert(domain);
        if domain == "public" {
            for key in ["dataset", "license", "source_url", "writer_id"] {
                ensure!(
                    sample.metadata["public"][key]
                        .as_str()
                        .is_some_and(|s| !s.trim().is_empty()),
                    "public sample {} missing {key}",
                    sample.sample_id
                );
            }
            ensure!(
                sample.metadata["public"]["kind"] == "human",
                "public pretraining requires human trajectories: {}",
                sample.sample_id
            );
            let writer = format!(
                "{}:{}",
                sample.metadata["public"]["dataset"], sample.metadata["public"]["writer_id"]
            );
            if let Some(split) = writers.insert(writer, sample.split.clone()) {
                ensure!(
                    split == sample.split,
                    "public writer crosses dataset splits"
                );
            }
        }
    }
    ensure!(
        domains.len() <= 1,
        "public and personal corpora must be trained in separate stages"
    );
    Ok(if domains.contains("public") {
        "public"
    } else {
        "personal"
    })
}

pub fn audit(samples: &[Sample]) -> Result<Value> {
    let targets = [
        "T", "t", "\\Gamma", "Γ", "(", ")", "[", "]", "{", "}", "|", "\\cdot", ";",
    ];
    let mut support = BTreeMap::new();
    let mut complete_runs = vec![];
    let mut new_rows = vec![];
    for target in targets {
        let mut isolated = vec![];
        let mut practice = vec![];
        let mut contexts = vec![];
        let mut sources = BTreeSet::new();
        for sample in samples.iter().filter(|s| s.split == "train") {
            let tokens = ink_layout::tokenize(&sample.label)?;
            let labels: Vec<_> = tokens.iter().filter(|t| t.as_str() != "<space>").collect();
            if !labels.iter().any(|t| t.as_str() == target) {
                continue;
            }
            sources.insert(sample.source_sha256.clone());
            let record = json!({"sample_id":sample.sample_id,"source_sha256":sample.source_sha256,"label":sample.label,"strokes":sample.strokes.len(),"latin_anchor_eligible":latin_anchor_allowed(&tokens,target)});
            if labels.len() == 1 {
                isolated.push(record);
            } else if tokens
                .iter()
                .all(|t| t == "<space>" || t.chars().count() == 1 || ink_corpus::glyph_label(t))
                && tokens
                    .windows(2)
                    .all(|p| p[0] == "<space>" || p[1] == "<space>")
            {
                practice.push(record);
            } else {
                contexts.push(record);
            }
        }
        support.insert(target, json!({"isolated":isolated,"separated_practice":practice,"whole_contexts":contexts,"distinct_train_sources":sources.len(),"isolated_authorship":"supplied labels; no new independent certification"}));
    }
    for sample in samples.iter().filter(|s| s.split == "train") {
        let tokens = ink_layout::tokenize(&sample.label)?;
        let flat = tokens
            .iter()
            .all(|t| t == "<space>" || !(t.len() > 1 && t.starts_with('<') && t.ends_with('>')));
        let letters = tokens.iter().filter(|t| t.as_str() != "<space>").count();
        if flat && letters > 1 {
            let b = ink_corpus::bounds(sample);
            let pitch = (b[2] - b[0]) / letters as f32;
            let wide: Vec<_> = sample
                .strokes
                .iter()
                .filter(|s| {
                    let lo = s.points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
                    let hi = s
                        .points
                        .iter()
                        .map(|p| p[0])
                        .fold(f32::NEG_INFINITY, f32::max);
                    hi - lo > pitch * 1.8
                })
                .map(|s| s.source_index)
                .collect();
            complete_runs.push(json!({"sample_id":sample.sample_id,"source_sha256":sample.source_sha256,"label":sample.label,"characters":letters,"strokes":sample.strokes.len(),"wide_stroke_candidates":wide,"few_strokes_candidate":sample.strokes.len()<letters,"use":"complete original trajectory; no glyph assignments"}));
        }
        if sample.sample_id.starts_with("patch-") {
            new_rows.push(json!({"sample_id":sample.sample_id,"label":sample.label,"source_sha256":sample.source_sha256,"strokes":sample.strokes.len(),"split":sample.split}));
        }
    }
    Ok(
        json!({"character_support":support,"complete_train_runs":complete_runs,"new_supplied_rows":new_rows,
        "candidate_interpretation":"wide/few strokes identify rows to inspect, not certified joins or character boundaries",
        "naturalness":"judge ordered pen motion, observed joining frequency, baseline drift and spacing separately from character identity",
        "T_review":"compare complete T/t contexts and isolated/practice evidence; never relabel a T as Gamma from shape or OCR alone",
        "test_policy":"held-out rows excluded from this diagnosis; no updates or label changes"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_evidence_is_never_reported_as_isolated_and_holdout_is_excluded() {
        let row = |id: &str, label: &str, split: &str| Sample {
            sample_id: id.into(),
            label: label.into(),
            split: split.into(),
            source_sha256: "a".repeat(64),
            metadata: Value::Null,
            strokes: vec![ink_corpus::Stroke {
                points: vec![[0., 0.], [0.2, 0.1], [0.8, 0.2], [1., 0.3]],
                width: 0.02,
                source_index: 7,
                uuid: "native".into(),
            }],
        };
        let report = audit(&[
            row("word", r"\text{There}", "train"),
            row("isolated", "T", "test"),
        ])
        .unwrap();
        assert_eq!(
            report["character_support"]["T"]["isolated"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            report["character_support"]["T"]["whole_contexts"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            report["complete_train_runs"][0]["wide_stroke_candidates"],
            json!([7])
        );
    }
    #[test]
    fn public_pretraining_cannot_mix_domains_or_leak_writers() {
        let public = Sample {
            sample_id: "public".into(),
            label: "T".into(),
            split: "train".into(),
            source_sha256: "a".repeat(64),
            strokes: vec![],
            metadata: json!({"data_domain":"public","public":{"dataset":"fixture","license":"test-only","source_url":"https://example.invalid/fixture","writer_id":"writer-1","kind":"human"}}),
        };
        assert_eq!(domain(std::slice::from_ref(&public)).unwrap(), "public");
        let mut private = public.clone();
        private.metadata = Value::Null;
        assert!(domain(&[public.clone(), private]).is_err());
        let mut holdout = public.clone();
        holdout.split = "test".into();
        assert!(domain(&[public.clone(), holdout]).is_err());
        let mut unlicensed = public;
        unlicensed.metadata["public"]["license"] = Value::Null;
        assert!(domain(&[unlicensed]).is_err());
    }
    #[test]
    fn mixed_greek_t_forms_do_not_become_latin_anchors() {
        assert!(!latin_anchor_allowed(&["\\Phi".into(), "T".into()], "T"));
        assert!(latin_anchor_allowed(&["S".into(), "T".into()], "T"));
        assert!(latin_anchor_allowed(&["\\Phi".into(), "T".into()], "\\Phi"));
    }
}
