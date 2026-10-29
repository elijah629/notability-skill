use anyhow::{Context, Result, ensure};
use ink_corpus::{file_hash, save};
use ort::{Allocator, Checkpoint, Session, Tensor, Trainer};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command, time::Instant};

pub fn backend_gate(out: &Path, repo: &Path) -> Result<Value> {
    fs::create_dir_all(out)?;
    let start = Instant::now();
    let runtime = std::env::var("ORT_DYLIB_PATH")
        .context("ORT_DYLIB_PATH unset; enter pinned Nix Rust environment")?;
    ensure!(
        Path::new(&runtime).is_file(),
        "ONNX Runtime shared library missing: {runtime}"
    );
    ort::init_from(&runtime)
        .with_name("ink-cpu-training-gate")
        .commit()?;
    let mut weight_updates = 0;
    let result = (|| -> Result<Value> {
        ort::training_api()
            .context("backend gate: installed ONNX Runtime exposes no training API")?;
        let graph = out.join("artifacts");
        if !graph.exists() {
            let status = Command::new("nix")
                .arg("develop")
                .arg(format!("{}#graph", repo.display()))
                .args(["--command", "python"])
                .arg(repo.join("src/graphs/gate.py"))
                .arg(&graph)
                .stdout(std::process::Stdio::from(std::io::stderr()))
                .status()?;
            ensure!(
                status.success(),
                "offline ONNX training graph construction failed"
            );
        }
        for name in [
            "checkpoint",
            "training_model.onnx",
            "eval_model.onnx",
            "optimizer_model.onnx",
        ] {
            ensure!(
                graph.join(name).is_file(),
                "incomplete graph artifacts: {name}; use a fresh gate directory"
            );
        }
        let trainer = Trainer::new_from_artifacts(
            Session::builder()?.with_intra_threads(1)?,
            Allocator::default(),
            &graph,
            None,
        )?;
        trainer.optimizer().set_lr(0.1)?;
        trainer.export(out.join("before.onnx"), ["prediction"])?;
        let initial_prediction = predict(&out.join("before.onnx"))?;
        let mut losses = Vec::new();
        for _ in 0..10 {
            trainer.optimizer().reset_grad()?;
            let x = Tensor::from_array(([1usize, 1], vec![1f32]))?;
            let y = Tensor::from_array(([1usize, 1], vec![2f32]))?;
            let output = trainer.step(ort::inputs![x]?, ort::inputs![y]?)?;
            let (_, values) = output[0].try_extract_raw_tensor::<f32>()?;
            let loss = values[0];
            ensure!(loss.is_finite(), "nonfinite gate loss");
            losses.push(loss);
            drop(output);
            trainer.optimizer().step()?;
            weight_updates += 1;
        }
        ensure!(losses.last() < losses.first(), "gate loss did not fall");
        trainer.checkpoint().save(out.join("checkpoint"), true)?;
        trainer.export(out.join("updated.onnx"), ["prediction"])?;
        ensure!(
            file_hash(&out.join("before.onnx"))? != file_hash(&out.join("updated.onnx"))?,
            "exported decoder weights unchanged"
        );
        drop(trainer);
        let restored = Trainer::new_from_artifacts(
            Session::builder()?.with_intra_threads(1)?,
            Allocator::default(),
            &graph,
            Some(Checkpoint::load(out.join("checkpoint"))?),
        )?;
        restored.export(out.join("restored.onnx"), ["prediction"])?;
        let updated = predict(&out.join("updated.onnx"))?;
        let reloaded = predict(&out.join("restored.onnx"))?;
        ensure!(
            (2.0 - updated).abs() < (2.0 - initial_prediction).abs(),
            "exported decoder did not improve prediction"
        );
        ensure!(
            (updated - reloaded).abs() < 1e-6 && updated.abs() > 1e-6,
            "checkpoint reload mismatch"
        );
        restored.optimizer().set_lr(0.1)?;
        restored.optimizer().reset_grad()?;
        let output = restored.step(
            ort::inputs![Tensor::from_array(([1usize, 1], vec![1f32]))?]?,
            ort::inputs![Tensor::from_array(([1usize, 1], vec![2f32]))?]?,
        )?;
        drop(output);
        restored.optimizer().step()?;
        weight_updates += 1;
        restored.export(out.join("resumed.onnx"), ["prediction"])?;
        let resumed = predict(&out.join("resumed.onnx"))?;
        ensure!(
            (2.0 - resumed).abs() < (2.0 - reloaded).abs(),
            "resumed optimizer did not update decoder weights"
        );
        Ok(
            json!({"status":"passed","weight_updates":weight_updates,"initial_prediction":initial_prediction,"resumed_prediction":resumed,"initial_loss":losses[0],"last_loss":losses[9],"losses":losses,"updated_prediction":updated,"reloaded_prediction":reloaded,"checkpoint_reload":true,"resume_update":true,"updated_decoder_export":true}),
        )
    })();
    let report = match &result {
        Ok(v) => v.clone(),
        Err(e) => {
            json!({"status":"blocked","weight_updates":weight_updates,"neural_complete":false,"error":format!("{e:#}"),"fallback":"none; handwriting VAE uses the native Rust backend"})
        }
    };
    let report = json!({"ort_api_version":ort_sys::ORT_API_VERSION,"runtime_path":runtime,"runtime_sha256":file_hash(Path::new(&runtime))?,"elapsed_seconds":start.elapsed().as_secs_f64(),"gate":report});
    save(&out.join("gate-report.json"), &report)?;
    result.map(|_| report)
}

fn predict(path: &Path) -> Result<f32> {
    let session = Session::builder()?
        .with_intra_threads(1)?
        .commit_from_file(path)?;
    let input = Tensor::from_array(([1usize, 1], vec![1f32]))?;
    let output = session.run(ort::inputs![input]?)?;
    let value = output[0].try_extract_raw_tensor::<f32>()?.1[0];
    ensure!(value.is_finite(), "nonfinite decoder prediction");
    Ok(value)
}
