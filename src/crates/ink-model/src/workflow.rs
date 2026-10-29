//! Read-only preflight shared by dry-run and real training. No GPU or model allocation.
use super::TrainOptions;
use anyhow::{Context, Result, ensure};
use ink_corpus::{audit, file_hash, load};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs};

pub fn training_plan(options: &TrainOptions<'_>) -> Result<Value> {
    ensure!(
        (1..=28800).contains(&options.seconds),
        "Training budget must be 1s–8h; use --for 30m or --seconds 1800"
    );
    ensure!(
        !options.resume || options.model.is_some(),
        "Resume needs a model: nix run .#training -- resume latest --for 30m"
    );
    let samples = load(options.corpus).with_context(|| {
        format!(
            "Cannot read corpus {}. Run nix run . -- data audit; use --corpus for another dataset",
            options.corpus.display()
        )
    })?;
    let audit = audit(&samples)?;
    let domain = super::style::domain(&samples)?;
    let train: Vec<_> = samples.iter().filter(|s| s.split == "train").collect();
    ensure!(!train.is_empty(), "Corpus has no train samples");
    let mut tokens = BTreeSet::from(["\0".to_owned()]);
    for row in &train {
        tokens.extend(
            ink_layout::tokenize(&row.label)
                .with_context(|| format!("Invalid training label for {}", row.sample_id))?,
        );
    }
    let corpus_hash = file_hash(options.corpus)?;
    let recipe: Value =
        serde_json::from_str(include_str!("../../../recipes/sequence-cvae-v14.json"))?;
    let mut execution = super::execution::ExecutionPlan::new(options.cpu);
    let mut expected_batches = None;
    let mut parent = options
        .model
        .map(|p| super::resolve(Some(p), options.models))
        .transpose()?;
    if parent.is_none() && !options.resume && options.models.exists() {
        let rows = super::list(options.models)?;
        for row in rows.as_array().context("model list")? {
            let m = &row["manifest"];
            let path = std::path::PathBuf::from(row["path"].as_str().context("model path")?);
            if m["kind"] == "sequence-cvae-v3"
                && m["recipe"] == recipe
                && m["dataset_sha256"] == corpus_hash
                && path.join("checkpoint.safetensors").exists()
            {
                let vocab: Vec<String> =
                    serde_json::from_slice(&fs::read(path.join("vocabulary.json"))?)?;
                if vocab == tokens.iter().cloned().collect::<Vec<_>>() {
                    parent = Some(path);
                }
            }
        }
    }
    if let Some(path) = &parent {
        let m = super::show(path)?;
        ensure!(
            m["checkpoint_sha256"] == file_hash(&path.join("checkpoint.safetensors"))?,
            "Parent checkpoint checksum mismatch"
        );
        if options.resume {
            execution = super::execution::ExecutionPlan::resume(&m)?;
            expected_batches = m["batches_per_epoch"].as_u64();
            ensure!(
                m["dataset_sha256"] == corpus_hash,
                "Dataset changed since this checkpoint. Create a new version with nix run .#training -- train --from {} --for 30m",
                m["id"].as_str().unwrap_or("latest")
            );
            ensure!(
                m["kind"] == "conditional-vae-v2" || m["recipe"] == recipe,
                "Recipe changed since this checkpoint. Create a new version with nix run .#training -- train --from {} --for 30m",
                m["id"].as_str().unwrap_or("latest")
            );
            ensure!(
                m["optimizer_sha256"] == file_hash(&path.join("optimizer.safetensors"))?,
                "Optimizer checksum mismatch"
            );
            if m["kind"] == "sequence-cvae-v3" {
                let vocab: Vec<String> =
                    serde_json::from_slice(&fs::read(path.join("vocabulary.json"))?)?;
                ensure!(
                    vocab == tokens.iter().cloned().collect::<Vec<_>>(),
                    "Resume vocabulary mismatch; start a new version with --from"
                );
                if path.join("glyph-training.json").exists() {
                    let head: Value =
                        serde_json::from_slice(&fs::read(path.join("glyph-training.json"))?)?;
                    let expected: Value = serde_json::from_str(include_str!(
                        "../../../recipes/compositional-vector-v20.json"
                    ))?;
                    ensure!(
                        head["recipe"] == expected,
                        "Auxiliary recipe changed; start a new version with --from"
                    );
                }
            }
            ensure!(
                m["training_seconds"].as_f64().unwrap_or(28800.) < 28800.,
                "Model reached its 8h cumulative budget; start a new model"
            );
        } else {
            ensure!(
                m["kind"] == "sequence-cvae-v3",
                "Transfer needs a sequence model; omit --from to start fresh"
            );
            ensure!(
                m["train_lineage_sha256"] == file_hash(&path.join("train-lineage.json"))?,
                "Parent lineage checksum mismatch"
            );
            let rows: Vec<Value> =
                serde_json::from_slice(&fs::read(path.join("train-lineage.json"))?)?;
            let holdout: BTreeSet<_> = samples
                .iter()
                .filter(|s| s.split != "train")
                .map(|s| s.source_sha256.as_str())
                .collect();
            ensure!(
                !rows.iter().any(|s| s["source_sha256"]
                    .as_str()
                    .is_some_and(|s| holdout.contains(s))),
                "Parent train sources overlap current holdout"
            );
        }
    }
    let lengths = train
        .iter()
        .map(|r| ink_corpus::sequence::encode(r).map(|s| s.events.len()))
        .collect::<Result<Vec<_>>>()?;
    let batches_per_epoch = execution.groups(&lengths).len();
    ensure!(
        expected_batches.is_none_or(|n| n == batches_per_epoch as u64),
        "Saved batch boundaries changed; start a new version with --from"
    );
    Ok(
        json!({"training_started":false,"action":"training-plan","mode":if options.resume{"resume"}else if parent.is_some(){"transfer into new model"}else{"new model"},"parent":parent,"corpus":options.corpus,"corpus_sha256":corpus_hash,"training_samples":train.len(),"validation_samples":audit["splits"]["validation"],"test_samples":audit["splits"]["test"],"training_domain":domain,"seconds":options.seconds,"device":if options.cpu{"CPU"}else{"GPU"},"recipe":if options.resume{"checkpoint recipe"}else{"sequence-cvae-v14"},"execution_plan":execution,"batches_per_epoch":batches_per_epoch,"output":"existing model for resume; new mNNN directory otherwise","training_compiled":cfg!(feature="rocm"),"status":"ready; no weights updated or model directory created"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dry_run_validates_complete_splits_without_creating_a_model() {
        let root = std::env::temp_dir().join(format!("ink-plan-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let corpus = root.join("fixture.jsonl");
        let models = root.join("models");
        let mut data = String::new();
        for (i, split) in ["train", "validation", "test"].into_iter().enumerate() {
            let sample = ink_corpus::Sample {
                sample_id: format!("fixture-{i}"),
                label: "th".into(),
                source_sha256: format!("{i:064x}"),
                split: split.into(),
                metadata: Value::Null,
                strokes: vec![ink_corpus::Stroke {
                    points: vec![[0., 0.], [0.1, 0.2], [0.5, 0.8], [1., 1.]],
                    width: 0.02,
                    source_index: 0,
                    uuid: "test-only".into(),
                }],
            };
            data.push_str(&serde_json::to_string(&sample).unwrap());
            data.push('\n');
        }
        fs::write(&corpus, data).unwrap();
        let options = TrainOptions {
            model: None,
            corpus: &corpus,
            models: &models,
            seconds: 1800,
            resume: false,
            repo: &root,
            cpu: false,
            gpu_index: None,
        };
        let plan = training_plan(&options).unwrap();
        assert_eq!(plan["training_started"], false);
        assert_eq!(plan["seconds"], 1800);
        assert_eq!(plan["training_samples"], 1);
        assert_eq!(plan["execution_plan"]["max_batch_samples"], 64);
        assert_eq!(plan["batches_per_epoch"], 1);
        assert!(!models.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
