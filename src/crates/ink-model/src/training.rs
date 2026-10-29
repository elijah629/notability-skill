use anyhow::{Context, Result, ensure};
use ink_corpus::{Sample, audit, file_hash, glyph_label, load, save, tensor};
use ink_neural::{Decoder, LATENT, Linear, OUTPUT, XY};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, process::Command, time::Instant};
use tch::{Device, Kind, Tensor, nn};
const LR: f64 = 0.001;
struct Network {
    store: nn::VarStore,
    e1: nn::Linear,
    e2: nn::Linear,
    mu: nn::Linear,
    logvar: nn::Linear,
    d1: nn::Linear,
    d2: nn::Linear,
    out: nn::Linear,
}
impl Network {
    fn new(device: Device, classes: usize) -> Self {
        let store = nn::VarStore::new(device);
        let r = &store.root();
        let linear = |name: &str, a, b| nn::linear(r / name, a, b, Default::default());
        Self {
            e1: linear("e1", (OUTPUT + classes) as i64, 128),
            e2: linear("e2", 128, 64),
            mu: linear("mu", 64, LATENT as i64),
            logvar: linear("logvar", 64, LATENT as i64),
            d1: linear("d1", (LATENT + classes) as i64, 128),
            d2: linear("d2", 128, 128),
            out: linear("out", 128, OUTPUT as i64),
            store,
        }
    }
    fn encode(&self, x: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        let h = Tensor::cat(&[x, c], 1)
            .apply(&self.e1)
            .relu()
            .apply(&self.e2)
            .relu();
        (h.apply(&self.mu), h.apply(&self.logvar).clamp(-8., 4.))
    }
    fn decode(&self, z: &Tensor, c: &Tensor) -> Tensor {
        Tensor::cat(&[z, c], 1)
            .apply(&self.d1)
            .relu()
            .apply(&self.d2)
            .relu()
            .apply(&self.out)
            .sigmoid()
    }
    fn loss(&self, x: &Tensor, c: &Tensor, noise: bool) -> Tensor {
        let (mu, lv) = self.encode(x, c);
        let z = if noise {
            &mu + (&lv * 0.5).exp() * Tensor::randn_like(&mu)
        } else {
            mu.shallow_clone()
        };
        let y = self.decode(&z, c);
        let mask = x.narrow(1, XY as i64, 8);
        let expanded = mask
            .unsqueeze(-1)
            .expand([-1, 8, 48], true)
            .reshape([-1, XY as i64]);
        let xy = x.narrow(1, 0, XY as i64);
        let yp = y.narrow(1, 0, XY as i64);
        let mse = ((&yp - &xy).square() * &expanded).sum(Kind::Float) / expanded.sum(Kind::Float);
        let predicted = y.narrow(1, XY as i64, 8).clamp(1e-6, 1. - 1e-6);
        let one = Tensor::ones_like(&mask);
        let bce = -(&mask * predicted.log() + (&one - &mask) * (&one - predicted).log())
            .mean(Kind::Float);
        let a = yp.reshape([-1, 8, 24, 2]);
        let b = xy.reshape([-1, 8, 24, 2]);
        let velocity = ((a.narrow(2, 1, 23) - a.narrow(2, 0, 23))
            - (b.narrow(2, 1, 23) - b.narrow(2, 0, 23)))
        .square()
            * mask.unsqueeze(-1).unsqueeze(-1);
        let velocity = velocity.sum(Kind::Float) / (mask.sum(Kind::Float) * 46.);
        let kl = (mu.square() + lv.exp() - &lv - 1.).mean(Kind::Float) * 0.5;
        mse + bce * 0.05 + velocity * 0.1 + kl * 0.0005
    }
    fn export(&self, id: &str, vocab: &[String]) -> Result<Decoder> {
        let layer = |l: &nn::Linear| -> Result<Linear> {
            let size = l.ws.size();
            Ok(Linear {
                weight: Vec::<f32>::try_from(l.ws.to_device(Device::Cpu).view([-1]))?,
                bias: Vec::<f32>::try_from(
                    l.bs.as_ref().context("linear bias")?.to_device(Device::Cpu),
                )?,
                input: size[1] as usize,
            })
        };
        let d = Decoder {
            schema_version: 2,
            kind: "conditional-vae-v2".into(),
            model_id: id.into(),
            vocabulary: vocab.to_vec(),
            layers: vec![layer(&self.d1)?, layer(&self.d2)?, layer(&self.out)?],
            latent_scale: 0.35,
        };
        d.validate()?;
        Ok(d)
    }
}
struct Adam {
    states: BTreeMap<String, (Tensor, Tensor)>,
    step: u64,
}
impl Adam {
    fn new() -> Self {
        Self {
            states: BTreeMap::new(),
            step: 0,
        }
    }
    fn update(&mut self, net: &Network) -> Result<()> {
        self.step += 1;
        tch::no_grad(|| -> Result<()> {
            for (name, mut weight) in net.store.variables() {
                let grad = weight.grad();
                ensure!(
                    grad.defined() && grad.device() == weight.device(),
                    "gradient missing or wrong device"
                );
                let (m, v) = self
                    .states
                    .entry(name)
                    .or_insert_with(|| (Tensor::zeros_like(&weight), Tensor::zeros_like(&weight)));
                *m = &*m * 0.9 + &grad * 0.1;
                *v = &*v * 0.999 + grad.square() * 0.001;
                let delta = (&*m / (1. - 0.9_f64.powf(self.step as f64)))
                    / ((&*v / (1. - 0.999_f64.powf(self.step as f64))).sqrt() + 1e-8)
                    * LR;
                let _ = weight.f_sub_(&delta)?;
                weight.zero_grad();
            }
            Ok(())
        })
    }
    fn save(&self, path: &Path) -> Result<()> {
        let mut tensors = vec![];
        for (name, (m, v)) in &self.states {
            tensors.push((format!("m.{name}"), m.to_device(Device::Cpu)));
            tensors.push((format!("v.{name}"), v.to_device(Device::Cpu)));
        }
        Tensor::write_safetensors(&tensors, path)?;
        Ok(())
    }
    fn restore(path: &Path, step: u64, net: &Network) -> Result<Self> {
        let mut tensors: BTreeMap<_, _> = Tensor::read_safetensors(path)?.into_iter().collect();
        let mut states = BTreeMap::new();
        for (name, w) in net.store.variables() {
            let m = tensors
                .remove(&format!("m.{name}"))
                .context("Adam first moment missing")?;
            let v = tensors
                .remove(&format!("v.{name}"))
                .context("Adam second moment missing")?;
            ensure!(
                m.size() == w.size() && v.size() == w.size(),
                "Adam state shape mismatch"
            );
            states.insert(name, (m.to_device(w.device()), v.to_device(w.device())));
        }
        ensure!(tensors.is_empty(), "unexpected optimizer tensors");
        Ok(Self { states, step })
    }
}
fn batches(
    samples: &[Sample],
    vocab: &[String],
    split: &str,
    device: Device,
) -> Result<(Tensor, Tensor, Vec<String>)> {
    let mut x = vec![];
    let mut c = vec![];
    let mut ids = vec![];
    for s in samples
        .iter()
        .filter(|s| s.split == split && glyph_label(&s.label) && s.strokes.len() <= 8)
    {
        let Some(index) = vocab.iter().position(|l| l == &s.label) else {
            continue;
        };
        let (xy, mask) = tensor(s)?;
        x.extend(xy);
        x.extend(mask);
        let mut one = vec![0f32; vocab.len()];
        one[index] = 1.;
        c.extend(one);
        ids.push(s.sample_id.clone());
    }
    ensure!(
        !ids.is_empty(),
        "no eligible atomic {split} samples with trained labels"
    );
    Ok((
        Tensor::from_slice(&x)
            .view([ids.len() as i64, OUTPUT as i64])
            .to_device(device),
        Tensor::from_slice(&c)
            .view([ids.len() as i64, vocab.len() as i64])
            .to_device(device),
        ids,
    ))
}
fn parity(net: &Network, d: &Decoder, device: Device) -> Result<f32> {
    let mut max: f32 = 0.;
    for (i, label) in d.vocabulary.iter().enumerate() {
        let z = ink_neural::normal(419 + i as u64, LATENT);
        let mut c = vec![0f32; d.vocabulary.len()];
        c[i] = 1.;
        let native = net.decode(
            &Tensor::from_slice(&z)
                .view([1, LATENT as i64])
                .to_device(device),
            &Tensor::from_slice(&c)
                .view([1, c.len() as i64])
                .to_device(device),
        );
        let native = Vec::<f32>::try_from(native.to_device(Device::Cpu).view([-1]))?;
        let rust = d.forward(label, &z)?;
        for (a, b) in native.iter().zip(rust) {
            max = max.max((a - b).abs());
        }
    }
    ensure!(max < 2e-5, "native/Rust decoder parity failed: {max}");
    Ok(max)
}
pub fn train(options: super::TrainOptions<'_>) -> Result<Value> {
    let super::TrainOptions {
        model,
        corpus,
        models,
        seconds,
        resume,
        repo,
        cpu,
        gpu_index: index,
    } = options;
    ensure!(
        (1..=28800).contains(&seconds),
        "time budget must be 1..28800 seconds"
    );
    ensure!(!resume || model.is_some(), "resume requires --model mNNN");
    ensure!(
        resume || model.is_none(),
        "new training allocates a version; use --resume to update an existing model"
    );
    let samples = load(corpus)?;
    audit(&samples)?;
    let data_hash = file_hash(corpus)?;
    let vocab: Vec<_> = samples
        .iter()
        .filter(|s| s.split == "train" && glyph_label(&s.label) && s.strokes.len() <= 8)
        .map(|s| s.label.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    ensure!(!vocab.is_empty(), "no eligible train glyphs");
    let (device, hardware) = if cpu {
        ensure!(index.is_none(), "--gpu-index cannot be used with --cpu");
        (Device::Cpu, json!({"device":"CPU"}))
    } else {
        super::rocm::device(index)?
    };
    tch::set_num_threads(2);
    tch::manual_seed(419);
    let (x, c, ids) = batches(&samples, &vocab, "train", device)?;
    let mut net = Network::new(device, vocab.len());
    let mut adam = Adam::new();
    let mut previous_seconds = 0.;
    let target = if resume {
        let p = super::resolve(model, models)?;
        let m = super::show(&p)?;
        ensure!(
            m["dataset_sha256"] == data_hash,
            "resume corpus changed; start a new version"
        );
        ensure!(
            m["recipe"]["id"] == "glyph-cvae-v2",
            "resume recipe mismatch"
        );
        let expected: Vec<String> = serde_json::from_slice(&fs::read(p.join("vocabulary.json"))?)?;
        ensure!(expected == vocab, "resume vocabulary changed");
        ensure!(
            m["checkpoint_sha256"] == file_hash(&p.join("checkpoint.safetensors"))?
                && m["optimizer_sha256"] == file_hash(&p.join("optimizer.safetensors"))?,
            "checkpoint integrity mismatch"
        );
        net.store.load(p.join("checkpoint.safetensors"))?;
        adam = Adam::restore(
            &p.join("optimizer.safetensors"),
            m["updates"].as_u64().context("update count")?,
            &net,
        )?;
        previous_seconds = m["training_seconds"].as_f64().unwrap_or(0.);
        p
    } else {
        fs::create_dir_all(models)?;
        let mut p = None;
        for i in 0..100000 {
            let target = models.join(format!("m{i:03}"));
            if target.join("retired.json").exists() {
                continue;
            }
            match fs::create_dir(&target) {
                Ok(()) => {
                    p = Some(target);
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }
        }
        p.context("model IDs exhausted")?
    };
    ensure!(
        previous_seconds < 28800.,
        "model has exhausted eight-hour training budget"
    );
    let seconds = (seconds as f64).min(28800. - previous_seconds);
    let id = target
        .file_name()
        .context("model ID")?
        .to_string_lossy()
        .to_string();
    super::progress::log(format!(
        "Training {id}: {} glyphs, {} samples, {device:?}; {:.0}s budget",
        vocab.len(),
        ids.len(),
        seconds
    ));
    let initial = tch::no_grad(|| net.loss(&x, &c, false).double_value(&[]));
    let started = Instant::now();
    let mut progress = super::progress::EpochProgress::new(&format!("{id} · legacy"), 1, adam.step);
    let old_steps = adam.step;
    let mut trace = vec![];
    while started.elapsed().as_secs_f64() < seconds && adam.step - old_steps < 20000 {
        progress.start();
        tch::manual_seed(419 + adam.step as i64);
        let loss = net.loss(&x, &c, true);
        ensure!(loss.device() == device, "loss moved off training device");
        let value = loss.double_value(&[]);
        ensure!(
            value.is_finite(),
            "nonfinite loss; checkpoint not published"
        );
        loss.backward();
        adam.update(&net)?;
        // The historical network uses the entire corpus in each update.
        if let Some(summary) = progress.update(value, ids.len()) {
            trace.push(json!({"epoch":summary.epoch,"update":adam.step,"mean_loss":summary.mean_loss,"seconds":started.elapsed().as_secs_f64()}));
        }
    }
    ensure!(adam.step > old_steps, "no updates completed");
    progress.message("Saving and validating checkpoint…");
    let final_loss = tch::no_grad(|| net.loss(&x, &c, false).double_value(&[]));
    ensure!(final_loss.is_finite(), "nonfinite final loss");
    net.store.save(target.join("checkpoint.safetensors"))?;
    adam.save(&target.join("optimizer.safetensors"))?;
    let decoder = net.export(&id, &vocab)?;
    let error = tch::no_grad(|| parity(&net, &decoder, device))?;
    // Restore encoder/decoder and full Adam moments, then exercise an actual update.
    let mut restored = Network::new(device, vocab.len());
    restored.store.load(target.join("checkpoint.safetensors"))?;
    let restored_loss = tch::no_grad(|| restored.loss(&x, &c, false).double_value(&[]));
    ensure!(
        (restored_loss - final_loss).abs() < 1e-7,
        "checkpoint reload differs"
    );
    let mut restored_adam =
        Adam::restore(&target.join("optimizer.safetensors"), adam.step, &restored)?;
    restored.loss(&x, &c, false).backward();
    restored_adam.update(&restored)?;
    let after_resume = tch::no_grad(|| restored.loss(&x, &c, false).double_value(&[]));
    ensure!(
        (after_resume - restored_loss).abs() > 1e-12 && after_resume.is_finite(),
        "restored optimizer did not update weights"
    );
    net.loss(&x, &c, false).backward();
    adam.update(&net)?;
    let continued = net.export(&id, &vocab)?;
    let updated = restored.export(&id, &vocab)?;
    let continued_bytes = serde_json::to_vec(&continued)?;
    ensure!(
        continued_bytes == serde_json::to_vec(&updated)?,
        "restored Adam update differs from uninterrupted update"
    );
    // The verification update is not part of the published training state.
    adam.step -= 1;
    let resume_error = tch::no_grad(|| parity(&restored, &updated, device))?;
    ensure!(
        serde_json::to_vec(&updated)? != serde_json::to_vec(&decoder)?,
        "decoder weights unchanged after restored update"
    );
    save(&target.join("decoder.json"), &decoder)?;
    save(&target.join("vocabulary.json"), &vocab)?;
    save(
        &target.join("train-lineage.json"),
        &json!({"sample_ids":ids,"split":"train","source_hashes":samples.iter().filter(|s|ids.contains(&s.sample_id)).map(|s|&s.source_sha256).collect::<std::collections::BTreeSet<_>>()}),
    )?;
    let python_runtime_loaded = fs::read_to_string("/proc/self/maps")?.contains("libpython");
    ensure!(
        !python_runtime_loaded,
        "Python runtime loaded in native trainer"
    );
    let recipe: Value = serde_json::from_str(include_str!("../../../recipes/glyph-cvae-v2.json"))?;
    let git = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()?;
    ensure!(git.status.success(), "source commit unavailable");
    let dirty = Command::new("git")
        .args(["diff", "--binary", "HEAD"])
        .current_dir(repo)
        .output()?;
    let manifest = json!({"schema_version":2,"id":id,"kind":"conditional-vae-v2","seed":419,"parent_model":null,"recipe":recipe,"dataset_sha256":data_hash,"source_commit":String::from_utf8(git.stdout)?.trim(),"source_diff_sha256":ink_corpus::hash(&dirty.stdout),"nix_lock_sha256":file_hash(&repo.join("flake.lock"))?,"decoder_sha256":file_hash(&target.join("decoder.json"))?,"vocabulary_sha256":file_hash(&target.join("vocabulary.json"))?,"checkpoint_sha256":file_hash(&target.join("checkpoint.safetensors"))?,"optimizer_sha256":file_hash(&target.join("optimizer.safetensors"))?,"train_lineage_sha256":file_hash(&target.join("train-lineage.json"))?,"updates":adam.step,"training_seconds":previous_seconds+started.elapsed().as_secs_f64(),"hardware":hardware,"python_runtime_loaded":python_runtime_loaded,"glyph_labels":vocab.len(),"training_samples":ids.len(),"initial_loss":initial,"final_loss":final_loss,"rust_decoder_max_error":error,"checkpoint_reload_identical":true,"adam_resume_matches_uninterrupted":true,"restored_update_loss":after_resume,"restored_decoder_max_error":resume_error,"limitations":["84 atomic train samples and 20 labels in supplied corpus","no atomic test samples","style not independently scored","supplied labels lack independent second review","fixed pen width; learned points are interpolated, not original cubic controls"]});
    save(&target.join("manifest.json"), &manifest)?;
    let run = repo
        .join("runs")
        .join(&id)
        .join(format!("update-{}", adam.step));
    fs::create_dir_all(&run)?;
    save(&run.join("loss.json"), &trace)?;
    save(&run.join("training.json"), &manifest)?;
    let evaluation = evaluate(
        &target,
        corpus,
        "validation",
        &target.join("evaluation.json"),
    )?;
    progress.finish();
    super::progress::log(format!(
        "Saved {id}: {} updates; decoder parity {:.2e}",
        adam.step - old_steps,
        error
    ));
    Ok(
        json!({"model":target,"updates":adam.step,"initial_loss":initial,"final_loss":final_loss,"glyph_labels":vocab.len(),"evaluation":evaluation,"logs":run}),
    )
}
pub fn evaluate(model: &Path, corpus: &Path, split: &str, out: &Path) -> Result<Value> {
    ensure!(
        ["validation", "test"].contains(&split),
        "evaluation requires validation or test"
    );
    let m = super::show(model)?;
    ensure!(
        m["dataset_sha256"] == file_hash(corpus)?,
        "evaluation corpus mismatch"
    );
    let samples = load(corpus)?;
    audit(&samples)?;
    let d = Decoder::load(&model.join("decoder.json"))?;
    let eligible = samples
        .iter()
        .filter(|s| {
            s.split == split
                && glyph_label(&s.label)
                && s.strokes.len() <= 8
                && d.vocabulary.contains(&s.label)
        })
        .count();
    if eligible == 0 {
        let v = json!({"split":split,"eligible_atomic_samples":0,"status":"unavailable; no covered atomic held-out ink","style":"not scored"});
        save(out, &v)?;
        return Ok(v);
    }
    let (x, c, ids) = batches(&samples, &d.vocabulary, split, Device::Cpu)?;
    let mut net = Network::new(Device::Cpu, d.vocabulary.len());
    net.store.load(model.join("checkpoint.safetensors"))?;
    let report = tch::no_grad(|| -> Result<Value> {
        let (mu, _) = net.encode(&x, &c);
        let y = net.decode(&mu, &c);
        let mask = x.narrow(1, XY as i64, 8);
        let expanded = mask
            .unsqueeze(-1)
            .expand([-1, 8, 48], true)
            .reshape([-1, XY as i64]);
        let rmse = ((y.narrow(1, 0, XY as i64) - x.narrow(1, 0, XY as i64)).square() * &expanded)
            .sum(Kind::Float)
            / expanded.sum(Kind::Float);
        let accuracy = y
            .narrow(1, XY as i64, 8)
            .ge(0.5)
            .eq_tensor(&mask.ge(0.5))
            .to_kind(Kind::Float)
            .mean(Kind::Float)
            .double_value(&[]);
        Ok(
            json!({"split":split,"eligible_atomic_samples":ids.len(),"sample_ids":ids,"geometry":{"posterior_xy_rmse":rmse.sqrt().double_value(&[]),"stroke_mask_accuracy":accuracy},"content":{"status":"not independently scored; label conditioning is not recognition accuracy"},"style":{"status":"not independently scored; loss is not style quality"},"excluded_context_or_oversized_or_unknown":samples.iter().filter(|s|s.split==split).count()-ids.len()}),
        )
    })?;
    save(out, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_vae_updates_checkpoint_adam_resume_and_rust_export() -> Result<()> {
        tch::set_num_threads(1);
        tch::manual_seed(419);
        let net = Network::new(Device::Cpu, 2);
        let c = Tensor::eye(2, (Kind::Float, Device::Cpu));
        let x = Tensor::cat(
            &[
                Tensor::ones([2, 384], (Kind::Float, Device::Cpu)) * 0.2,
                Tensor::ones([2, 8], (Kind::Float, Device::Cpu)),
            ],
            1,
        );
        let mut adam = Adam::new();
        let initial = net.loss(&x, &c, false).double_value(&[]);
        for _ in 0..64 {
            net.loss(&x, &c, false).backward();
            adam.update(&net)?;
        }
        let final_loss = net.loss(&x, &c, false).double_value(&[]);
        ensure!(final_loss < initial * 0.5, "VAE failed to learn");
        let path = std::env::temp_dir().join(format!("ink-vae-contract-{}", std::process::id()));
        fs::create_dir_all(&path)?;
        net.store.save(path.join("checkpoint.safetensors"))?;
        adam.save(&path.join("optimizer.safetensors"))?;
        let mut restored = Network::new(Device::Cpu, 2);
        restored.store.load(path.join("checkpoint.safetensors"))?;
        let mut restored_adam =
            Adam::restore(&path.join("optimizer.safetensors"), adam.step, &restored)?;
        for n in [&net, &restored] {
            n.loss(&x, &c, false).backward();
        }
        adam.update(&net)?;
        restored_adam.update(&restored)?;
        let vocab = vec!["x".into(), "y".into()];
        let a = net.export("fixture", &vocab)?;
        let b = restored.export("fixture", &vocab)?;
        ensure!(
            serde_json::to_vec(&a)? == serde_json::to_vec(&b)?,
            "resumed Adam differs"
        );
        parity(&restored, &b, Device::Cpu)?;
        let drawing = b.draw("x^2", 419, "test");
        ensure!(drawing.is_err(), "missing glyph must fail");
        let drawing = b.draw("x_y", 419, "test")?;
        ensure!(drawing.glyphs.len() == 2, "glyph omitted");
        fs::remove_dir_all(path)?;
        Ok(())
    }
}
