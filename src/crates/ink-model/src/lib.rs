use anyhow::{Context, Result, ensure};
pub mod progress;
pub mod review;
pub mod style;
pub mod workflow;
use ink_corpus::{file_hash, load, save};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
mod execution;
mod gate;
pub use gate::backend_gate;
#[cfg(feature = "rocm")]
mod compositional;
#[cfg(feature = "rocm")]
mod rocm;
#[cfg(feature = "rocm")]
mod sequence;
#[cfg(feature = "rocm")]
mod training;
pub fn rocm_gate(out: &Path, repo: &Path, index: Option<usize>) -> Result<Value> {
    #[cfg(feature = "rocm")]
    {
        rocm::gate(out, repo, index)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (out, repo, index);
        anyhow::bail!("GPU backend requires nix develop .#rocm and --features rocm")
    }
}
pub struct TrainOptions<'a> {
    pub model: Option<&'a Path>,
    pub corpus: &'a Path,
    pub models: &'a Path,
    pub seconds: u64,
    pub resume: bool,
    pub repo: &'a Path,
    pub cpu: bool,
    pub gpu_index: Option<usize>,
}
pub fn train(options: TrainOptions<'_>) -> Result<Value> {
    let preflight = progress::Progress::spinner("Validate data and parent");
    workflow::training_plan(&options)?;
    preflight.finish("ready");
    #[cfg(feature = "rocm")]
    {
        if options.resume && options.model.is_some() {
            let path = resolve(options.model, options.models)?;
            if show(&path)?["kind"] == "conditional-vae-v2" {
                return training::train(options);
            }
        }
        sequence::train(options)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = options;
        anyhow::bail!(
            "Training needs native LibTorch. Run nix run .#training -- train (pinned Rust/ROCm environment)"
        )
    }
}
pub fn resolve(model: Option<&Path>, models: &Path) -> Result<PathBuf> {
    if let Some(p) = model {
        if p == Path::new("latest") {
            let mut paths = if models.exists() {
                fs::read_dir(models)?
                    .map(|entry| entry.map(|e| e.path()))
                    .collect::<std::io::Result<Vec<_>>>()?
            } else {
                vec![]
            };
            paths.sort();
            for path in paths.into_iter().rev() {
                if path.join("manifest.json").exists() && show(&path).is_ok() {
                    return Ok(path);
                }
            }
            anyhow::bail!(
                "No valid trained models. Start with nix run .#training -- train --dry-run"
            );
        }
        let resolved = if p.exists() {
            p.to_owned()
        } else {
            models.join(p)
        };
        ensure!(
            resolved.join("manifest.json").is_file(),
            "Model {} not found. Run nix run . -- models or use --model latest",
            p.display()
        );
        return Ok(resolved);
    }
    let mut paths = vec![];
    if models.exists() {
        for entry in fs::read_dir(models)? {
            let mut p = entry?.path();
            if p.join("approved").join("manifest.json").exists() {
                p = p.join("approved");
            }
            if p.join("decoder.json").exists()
                && fs::read(p.join("manifest.json"))
                    .ok()
                    .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                    .is_some_and(|m| {
                        m["kind"] == "conditional-vae-v2" || m["approved_for_default"] == true
                    })
                && show(&p).is_ok_and(|m| {
                    m["kind"] == "conditional-vae-v2" || m["approved_for_default"] == true
                })
            {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths.pop().context(
        "No default model. Preview a candidate with --model latest, or inspect nix run . -- models",
    )
}
pub fn show(model: &Path) -> Result<Value> {
    let m = manifest(model)?;
    ink_neural::Runtime::load(&model.join("decoder.json"))?;
    Ok(m)
}
/// Inventory checks file integrity; execution additionally validates neural shapes.
fn manifest(model: &Path) -> Result<Value> {
    let m: Value = serde_json::from_slice(&fs::read(model.join("manifest.json"))?)?;
    ensure!(
        m["kind"] == "conditional-vae-v2" || m["kind"] == "sequence-cvae-v3",
        "retired exemplar model; run ink train"
    );
    ensure!(
        m["decoder_sha256"] == file_hash(&model.join("decoder.json"))?,
        "decoder checksum mismatch"
    );
    ensure!(
        m["vocabulary_sha256"] == file_hash(&model.join("vocabulary.json"))?,
        "vocabulary checksum mismatch"
    );
    Ok(m)
}
pub fn list(models: &Path) -> Result<Value> {
    let mut result = vec![];
    if models.exists() {
        for entry in fs::read_dir(models)? {
            let p = entry?.path();
            if p.join("manifest.json").exists()
                && let Ok(m) = manifest(&p)
            {
                result.push(json!({"path":p,"manifest":m}));
            }
        }
    }
    result.sort_by_key(|v| v["path"].as_str().unwrap_or("").to_owned());
    Ok(json!(result))
}
pub fn render(model: &Path, input: &str, seed: u64, out: &Path) -> Result<Value> {
    render_with_variation(model, input, seed, out, 1.0)
}
pub fn render_with_variation(
    model: &Path,
    input: &str,
    seed: u64,
    out: &Path,
    variation: f32,
) -> Result<Value> {
    let m = show(model)?;
    let d = ink_neural::Runtime::load(&model.join("decoder.json"))?;
    let drawing = d.draw(
        input,
        seed,
        m["decoder_sha256"].as_str().context("decoder checksum")?,
        variation,
    )?;
    if let Some(p) = out.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(out.with_extension("svg"), ink_layout::svg(&drawing)?)?;
    save(&out.with_extension("ink.json"), &drawing)?;
    Ok(
        json!({"svg":out.with_extension("svg"),"ink":out.with_extension("ink.json"),"glyphs":ink_layout::drawing_labels(&drawing).len(),"requested_symbols":ink_layout::drawing_labels(&drawing).len(),"generated_strokes":drawing.glyphs.iter().map(|g|g.strokes.len()).sum::<usize>(),"content_recognition":"not scored","neural_runs":drawing.glyphs.iter().filter(|g| !g.run_labels.is_empty()).count(),"procedural_symbols":0,"procedural_structures":drawing.procedures.len(),"pen_width":drawing.pen_width,"seed":seed,"variation":variation,"model":m["id"],"training_domain":m["training_domain"],"mode":if drawing.glyphs.iter().any(|g| !g.run_labels.is_empty()){"joint-neural-sequence"}else{"neural-glyph-layout"}}),
    )
}
pub fn evaluate(model: &Path, corpus: &Path, split: &str, out: &Path) -> Result<Value> {
    #[cfg(feature = "rocm")]
    {
        if show(model)?["kind"] == "sequence-cvae-v3" {
            sequence::evaluate(model, corpus, split, out)
        } else {
            training::evaluate(model, corpus, split, out)
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (model, corpus, split, out);
        anyhow::bail!("Posterior evaluation needs native training build: nix develop .#rocm")
    }
}
pub fn collection(model: &Path, inputs: &[String], out: &Path) -> Result<Value> {
    let d = ink_neural::Runtime::load(&model.join("decoder.json"))?;
    let mut missing = vec![];
    for input in inputs {
        let mut labels = vec![];
        ink_layout::labels(&ink_layout::parse(input)?, &mut labels);
        for label in labels {
            let label = ink_layout::neural_condition(&label).to_string();
            if !ink_layout::procedural_symbol(&label)
                && !d.vocabulary().contains(&label)
                && !missing.contains(&label)
            {
                missing.push(label);
            }
        }
    }
    let result = json!({"missing":missing,"available":d.vocabulary(),"target_per_glyph":10});
    save(out, &result)?;
    Ok(result)
}
pub fn export(model: &Path, out: &Path, repo: &Path) -> Result<Value> {
    let mut manifest = show(model)?;
    ensure!(!out.exists(), "export requires fresh directory");
    ensure!(
        !out.with_extension("zip").exists(),
        "Export archive already exists. Choose a fresh --out destination"
    );
    fs::create_dir_all(out)?;
    let progress = progress::Progress::spinner("Export skill");
    progress.message("Build standalone inference binary");
    let status = progress.suspend(|| {
        Command::new("cargo")
            .env_remove("CARGO_TARGET_DIR")
            .args([
                "build",
                "--locked",
                "--offline",
                "--release",
                "--manifest-path",
            ])
            .arg(repo.join("src/Cargo.toml"))
            .args(["-p", "ink-infer"])
            .status()
    })?;
    ensure!(status.success(), "build inference entrypoint failed");
    progress.message("Bundle decoder, provenance and licenses");
    fs::create_dir(out.join("bin"))?;
    fs::copy(
        repo.join("src/target/release/ink-infer"),
        out.join("bin/ink-infer"),
    )?;
    fs::copy(model.join("vocabulary.json"), out.join("vocabulary.json"))?;
    let decoder = fs::read(model.join("decoder.json"))?;
    fs::write(out.join("decoder.json"), compact_json(&decoder)?)?;
    manifest["export_source_decoder_sha256"] = json!(file_hash(&model.join("decoder.json"))?);
    manifest["decoder_sha256"] = json!(file_hash(&out.join("decoder.json"))?);
    manifest["export_decoder_format"] =
        json!("lossless JSON whitespace compaction; original numeric tokens preserved");
    save(&out.join("manifest.json"), &manifest)?;
    fs::copy(repo.join("flake.lock"), out.join("flake.lock"))?;
    fs::write(
        out.join("flake.nix"),
        include_str!("../../../export-flake.nix"),
    )?;
    fs::write(
        out.join("SKILL.md"),
        "---\nname: personal-vector-handwriting\ndescription: Generate personal-style SVG handwriting from text or LaTeX with an exported Rust neural decoder.\n---\n\nRun from this skill directory:\n\n```sh\nnix run path:. -- --input '\\boxed{Hello, World!}' --out output/hello --seed 419 --variation 0\n```\n\nUse --seed 419 --variation 0 for stable, controlled output. Nonzero variation is experimental and may produce jagged or jittery letters; inspect every result. --variation accepts 0-2. Omit --seed only for an exploratory run and record the returned seed. The bundled x86_64 Linux binary requires its linked system libraries; use nix run to provision the pinned runtime on another Nix host. Direct use: bin/ink-infer --decoder decoder.json --input 'Hello' --out output/hello --seed 419 --variation 0. Generation writes .svg and .ink.json, and reports errors before writing unsupported requests. Inspect SVG readability before delivery; successful generation proves neither recognition accuracy nor personal authorship.\n\nDecoder supports trained conditions and a bounded LaTeX grammar, including boxes and matrices. Structural paths carry procedural provenance. Neural glyphs carry model, decoder hash and seed provenance. Composed symbols record their learned donor in condition_label and carry neural-composed-symbol provenance. Multiplication dots carry procedural-symbol provenance and use the same fixed document pen as letters and structural paths. Weights contain no source trajectory lookup; Python and training libraries are unnecessary. Supplied labels lack independent second review. Read bundled manifest limitations before claiming coverage or quality.\n",
    )?;
    fs::write(
        out.join("LICENSES.md"),
        "Private personal handwriting derivative; intended for the owner's use. No public corpus included. Do not redistribute without the owner's permission. The inference binary incorporates Rust crates under MIT/Apache-2.0 and other permissive licenses; see bundled dependency metadata.\n",
    )?;
    let meta = Command::new("cargo")
        .args([
            "metadata",
            "--locked",
            "--offline",
            "--filter-platform",
            "x86_64-unknown-linux-gnu",
            "--format-version",
            "1",
            "--manifest-path",
        ])
        .arg(repo.join("src/Cargo.toml"))
        .output()?;
    ensure!(
        meta.status.success(),
        "dependency metadata failed: {}",
        String::from_utf8_lossy(&meta.stderr)
    );
    let meta: Value = serde_json::from_slice(&meta.stdout)?;
    let packages = meta["packages"].as_array().context("cargo packages")?;
    let root = packages
        .iter()
        .find(|p| p["name"] == "ink-infer")
        .context("inference package missing")?["id"]
        .as_str()
        .context("package ID")?;
    let nodes = meta["resolve"]["nodes"]
        .as_array()
        .context("cargo dependency graph")?;
    let mut selected = std::collections::BTreeSet::new();
    let mut pending = vec![root.to_owned()];
    while let Some(id) = pending.pop() {
        if selected.insert(id.clone())
            && let Some(node) = nodes.iter().find(|n| n["id"] == id)
        {
            for dep in node["dependencies"].as_array().context("dependencies")? {
                pending.push(dep.as_str().context("dependency ID")?.to_owned());
            }
        }
    }
    let mut licenses = Vec::new();
    let mut hashes = BTreeMap::new();
    for package in packages
        .iter()
        .filter(|p| p["id"].as_str().is_some_and(|id| selected.contains(id)))
    {
        licenses.push(json!({"name":package["name"],"version":package["version"],"license":package["license"]}));
        let directory = Path::new(
            package["manifest_path"]
                .as_str()
                .context("dependency manifest")?,
        )
        .parent()
        .context("dependency directory")?;
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .context("license file name")?
                .to_string_lossy();
            if path.is_file() && (name.starts_with("LICENSE") || name.starts_with("COPYING")) {
                let relative = format!(
                    "licenses/{}-{}/{}",
                    package["name"].as_str().context("package name")?,
                    package["version"].as_str().context("package version")?,
                    name
                );
                let target = out.join(&relative);
                fs::create_dir_all(target.parent().context("license directory")?)?;
                fs::copy(path, &target)?;
                hashes.insert(relative, file_hash(&target)?);
            }
        }
    }
    save(&out.join("dependency-licenses.json"), &licenses)?;
    for name in [
        "bin/ink-infer",
        "decoder.json",
        "vocabulary.json",
        "manifest.json",
        "flake.nix",
        "flake.lock",
        "SKILL.md",
        "LICENSES.md",
        "dependency-licenses.json",
    ] {
        hashes.insert(name.to_owned(), file_hash(&out.join(name))?);
    }
    save(&out.join("checksums.json"), &hashes)?;
    let mut uncompressed_bytes = fs::metadata(out.join("checksums.json"))?.len();
    for name in hashes.keys() {
        uncompressed_bytes += fs::metadata(out.join(name))?.len();
    }
    ensure!(
        uncompressed_bytes < 25_000_000,
        "skill export must be under 25 MB uncompressed: {uncompressed_bytes} bytes"
    );
    progress.message("Write checksums and ZIP archive");
    let archive = out.with_extension("zip");
    ensure!(!archive.exists(), "export archive already exists");
    let mut zip = zip::ZipWriter::new(fs::File::create(&archive)?);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, _) in hashes {
        zip.start_file(
            &name,
            if name.starts_with("bin/") {
                options.unix_permissions(0o755)
            } else {
                options.unix_permissions(0o644)
            },
        )?;
        std::io::copy(&mut fs::File::open(out.join(&name))?, &mut zip)?;
    }
    zip.start_file("checksums.json", options)?;
    std::io::copy(&mut fs::File::open(out.join("checksums.json"))?, &mut zip)?;
    zip.finish()?;
    progress.finish(format!(
        "saved · {:.2} MB uncompressed",
        uncompressed_bytes as f64 / 1_000_000.
    ));
    Ok(
        json!({"directory":out,"archive":archive,"model":manifest["id"],"kind":manifest["kind"],"uncompressed_bytes":uncompressed_bytes,"decoder_sha256":manifest["decoder_sha256"],"source_decoder_sha256":manifest["export_source_decoder_sha256"],"excluded":["raw data","Python","trainer","note editing"]}),
    )
}

fn compact_json(input: &[u8]) -> Result<Vec<u8>> {
    let _: Value = serde_json::from_slice(input)?;
    let mut output = Vec::with_capacity(input.len());
    let mut quoted = false;
    let mut escaped = false;
    for &byte in input {
        if quoted {
            output.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
            output.push(byte);
        } else if !matches!(byte, b' ' | b'\n' | b'\r' | b'\t') {
            output.push(byte);
        }
    }
    Ok(output)
}

fn valid_neural_ink(g: &ink_layout::Glyph, hash: &str) -> bool {
    matches!(
        g.mode.as_str(),
        "neural" | "neural-run" | "neural-run-public"
    ) && g.decoder_sha256 == hash
        && !g.strokes.is_empty()
        && g.strokes.iter().all(|s| !s.uuid.starts_with("procedural"))
}
pub fn verify(model: &Path, corpus: &Path, out: &Path) -> Result<Value> {
    let m = show(model)?;
    if m["kind"] == "sequence-cvae-v3" {
        return verify_sequence(model, corpus, out);
    }
    ensure!(m["dataset_sha256"] == file_hash(corpus)?, "corpus mismatch");
    let d = ink_neural::Runtime::load(&model.join("decoder.json"))?;
    let samples = load(corpus)?;
    let seen: std::collections::BTreeSet<_> = samples.iter().map(|s| s.label.as_str()).collect();
    fs::create_dir_all(out)?;
    let mut examples = vec![];
    for a in 1..=9 {
        for b in 1..=9 {
            if examples.len() == 20 {
                break;
            }
            let input = format!("\\frac{{x^{{{a}{b}}}}}{{\\sqrt{{y_{{{b}{a}}}}}}}");
            if seen.contains(input.as_str()) {
                continue;
            }
            let drawing = d.draw(
                &input,
                419,
                m["decoder_sha256"].as_str().context("checksum")?,
                0.0,
            )?;
            let again = d.draw(
                &input,
                419,
                m["decoder_sha256"].as_str().context("checksum")?,
                0.0,
            )?;
            ensure!(
                serde_json::to_vec(&drawing)? == serde_json::to_vec(&again)?,
                "seed not reproducible"
            );
            let mut labels = vec![];
            ink_layout::labels(&ink_layout::parse(&input)?, &mut labels);
            ensure!(
                drawing
                    .glyphs
                    .iter()
                    .map(|g| g.label.clone())
                    .collect::<Vec<_>>()
                    == labels,
                "glyph omitted"
            );
            for g in &drawing.glyphs {
                ensure!(
                    valid_neural_ink(g, m["decoder_sha256"].as_str().unwrap_or("")),
                    "missing neural provenance"
                );
                for p in g.strokes.iter().flat_map(|s| &s.points) {
                    let x = p[0] * g.transform[0] + g.transform[2];
                    let y = p[1] * g.transform[1] + g.transform[3];
                    ensure!(
                        x.is_finite()
                            && y.is_finite()
                            && x >= 0.
                            && y >= 0.
                            && x <= drawing.width + 1e-5
                            && y <= drawing.height + 1e-5,
                        "invalid geometry"
                    );
                }
            }
            let name = format!("formula-{:02}", examples.len() + 1);
            fs::write(out.join(format!("{name}.svg")), ink_layout::svg(&drawing)?)?;
            save(&out.join(format!("{name}.ink.json")), &drawing)?;
            examples.push(
                json!({"input":input,"svg":format!("{name}.svg"),"glyphs":drawing.glyphs.len()}),
            );
        }
    }
    let result = json!({"model":m["id"],"examples":examples,"seed_reproducible":true,"finite_geometry":true,"all_glyphs_neural":true,"style":"not independently scored","reference_attachments":"not supplied"});
    save(&out.join("verification.json"), &result)?;
    Ok(result)
}

fn variation_report(decoder: &ink_neural::Runtime, checksum: &str) -> Result<Value> {
    let input = r"\text{Hello lllll 0123456789}";
    let reference = decoder.draw(input, 419, checksum, 0.)?;
    let distance = |a: &ink_layout::Glyph, b: &ink_layout::Glyph| {
        let mut sum = 0_f64;
        let mut count = 0;
        for (x, y) in a
            .strokes
            .iter()
            .flat_map(|s| &s.points)
            .zip(b.strokes.iter().flat_map(|s| &s.points))
        {
            for k in 0..2 {
                sum += (x[k] as f64 - y[k] as f64).powi(2);
                count += 1;
            }
        }
        (sum / count.max(1) as f64).sqrt()
    };
    let letters: Vec<_> = reference.glyphs.iter().filter(|g| g.label == "l").collect();
    let repeated: Vec<_> = letters.windows(2).map(|p| distance(p[0], p[1])).collect();
    let low_a = decoder.draw(input, 419, checksum, 0.)?;
    let low_b = decoder.draw(input, 419, checksum, 0.)?;
    let zero_equal = ink_layout::svg(&low_a)? == ink_layout::svg(&low_b)?;
    Ok(
        json!({"input":input,"seed":419,"variation":0,"within_document_repeated_l_rmse":if letters.len()>1{json!(repeated)}else{Value::Null},"letter_geometry_measurement":if letters.len()>1{"isolated outputs"}else{"not measured: jointly generated strokes have no certified letter segmentation"},"joint_runs":reference.glyphs.iter().filter(|g| !g.run_labels.is_empty()).count(),"procedural_paths":reference.procedures.len(),"fixed_seed_svg_identical":zero_equal,"interpretation":"content, personal style and natural pen motion require separate review; coordinate diversity alone is not naturalness"}),
    )
}

fn verify_sequence(model: &Path, corpus: &Path, out: &Path) -> Result<Value> {
    let manifest = show(model)?;
    ensure!(
        manifest["dataset_sha256"] == file_hash(corpus)?,
        "verification corpus mismatch"
    );
    let decoder = ink_neural::Runtime::load(&model.join("decoder.json"))?;
    let checksum = manifest["decoder_sha256"].as_str().context("checksum")?;
    let samples = load(corpus)?;
    ink_corpus::audit(&samples)?;
    fs::create_dir_all(out)?;
    let mut records = vec![];
    let mut failures = vec![];
    let fresh = [
        r"\boxed{Hello, World!}",
        r"\begin{bmatrix}1&2\\3&4\end{bmatrix}",
        r"\begin{cases}x+y=3\\x-y=1\end{cases}",
        r"\sqrt[3]{x_1^2+2}",
        r"\text{Hello hello lllll WWW iii gypq}",
        r"\boxed{\frac{a+b}{\sqrt{x}}}",
        r"\text{A new sentence, never copied.}",
        r"\left(\frac{1}{2}\right)",
    ];
    let inputs: Vec<_> = samples
        .iter()
        .map(|s| (s.sample_id.clone(), s.label.clone()))
        .chain(
            fresh
                .iter()
                .enumerate()
                .map(|(i, s)| (format!("unseen-{i:02}"), s.to_string())),
        )
        .collect();
    let progress = progress::Progress::count("Verify generation", inputs.len());
    for (i, (id, input)) in inputs.into_iter().enumerate() {
        let result = (|| -> Result<Value> {
            let seed = 419;
            let drawing = decoder.draw(&input, seed, checksum, 0.)?;
            let again = decoder.draw(&input, seed, checksum, 0.)?;
            ensure!(
                serde_json::to_vec(&drawing)? == serde_json::to_vec(&again)?,
                "seed not reproducible"
            );
            let mut expected = vec![];
            ink_layout::labels(&ink_layout::parse(&input)?, &mut expected);
            ensure!(
                ink_layout::drawing_labels(&drawing) == expected,
                "requested glyph omitted"
            );
            for g in &drawing.glyphs {
                ensure!(valid_neural_ink(g, checksum), "missing neural provenance");
                for p in g.strokes.iter().flat_map(|s| &s.points) {
                    let x = p[0] * g.transform[0] + g.transform[2];
                    let y = p[1] * g.transform[1] + g.transform[3];
                    ensure!(
                        x.is_finite()
                            && y.is_finite()
                            && x >= 0.
                            && y >= 0.
                            && x <= drawing.width
                            && y <= drawing.height,
                        "glyph outside SVG bounds"
                    );
                }
            }
            for p in drawing.procedures.iter().flat_map(|p| &p.points) {
                ensure!(
                    p.iter().all(|v| v.is_finite())
                        && p[0] >= 0.
                        && p[0] <= drawing.width
                        && p[1] >= 0.
                        && p[1] <= drawing.height,
                    "structure outside SVG bounds"
                );
            }
            let file = format!("sample-{i:04}.svg");
            fs::write(out.join(&file), ink_layout::svg(&drawing)?)?;
            if id.starts_with("unseen-") {
                save(&out.join(format!("{id}.ink.json")), &drawing)?;
            }
            Ok(
                json!({"sample_id":id,"input":input,"svg":file,"requested_glyphs":expected.len(),"seed":419,"variation":0,"terminated":true,"finite_bounds":true,"seed_reproducible":true}),
            )
        })();
        match result {
            Ok(record) => records.push(record),
            Err(e) => failures.push(json!({"sample_id":id,"input":input,"error":e.to_string()})),
        }
        progress.set(i + 1);
        progress.message(format!("{} failures", failures.len()));
        if (i + 1) % 100 == 0 {
            progress.note(format!(
                "Verification: {} requests · {} failures",
                i + 1,
                failures.len()
            ));
        }
    }
    progress.finish(format!(
        "{} passed · {} failed",
        records.len(),
        failures.len()
    ));
    let result = json!({"model":manifest["id"],"decoder_sha256":checksum,"dataset_sha256":manifest["dataset_sha256"],"dataset_requests":samples.len(),"unseen_requests":fresh.len(),"passed":records.len(),"failed":failures.len(),"records":records,"failures":failures,"variation":variation_report(&decoder,checksum)?, "complete_content_requests":"AST glyph coverage only; recognition accuracy not independently scored","style":"requires visual review","approved_for_default":false});
    save(&out.join("verification.json"), &result)?;
    Ok(result)
}

pub fn preview_source(corpus: &Path, id: &str, out: &Path) -> Result<Value> {
    let samples = load(corpus)?;
    let s = samples
        .iter()
        .find(|s| s.sample_id == id)
        .context("source sample not found")?;
    let b = ink_corpus::bounds(s);
    let scale = 1. / (b[3] - b[1]).max(0.01);
    let drawing = ink_layout::Drawing {
        schema_version: 1,
        input: s.label.clone(),
        seed: 0,
        width: (b[2] - b[0]) * scale + 0.5,
        height: 1.5,
        pen_width: None,
        glyphs: vec![ink_layout::Glyph {
            run_labels: vec![],
            condition_label: None,
            label: s.label.clone(),
            mode: "supplied-source-preview".into(),
            model_id: String::new(),
            decoder_sha256: s.source_sha256.clone(),
            latent_seed: 0,
            strokes: s.strokes.clone(),
            transform: [scale, scale, 0.25 - b[0] * scale, 0.25 - b[1] * scale],
        }],
        procedures: vec![],
    };
    if let Some(p) = out.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(out, ink_layout::svg(&drawing)?)?;
    Ok(
        json!({"preview":out,"sample_id":id,"label":s.label,"strokes":s.strokes.len(),"mode":"unchanged supplied vectors; review not certified"}),
    )
}

/// Approval refers to an immutable decoder hash; resume retains the approved inference snapshot.
pub fn approve(model: &Path, review_path: &Path) -> Result<Value> {
    let mut manifest = show(model)?;
    ensure!(
        manifest["kind"] == "sequence-cvae-v3",
        "approval is for sequence models"
    );
    let review: Value = serde_json::from_slice(&fs::read(review_path)?)?;
    ensure!(
        review["model"] == manifest["id"] && review["decoder_sha256"] == manifest["decoder_sha256"],
        "review model/decoder mismatch"
    );
    ensure!(
        review["visual_review"]["status"] == "passed",
        "visual review has not passed"
    );
    if matches!(
        manifest["recipe"]["id"].as_str(),
        Some(
            "sequence-cvae-v4"
                | "sequence-cvae-v5"
                | "sequence-cvae-v6"
                | "sequence-cvae-v7"
                | "sequence-cvae-v8"
                | "sequence-cvae-v9"
                | "sequence-cvae-v10"
                | "sequence-cvae-v11"
                | "sequence-cvae-v12"
                | "sequence-cvae-v13"
                | "sequence-cvae-v14"
        )
    ) {
        for axis in [
            "content",
            "personal_style",
            "natural_motion",
            "contextual_joining",
            "T_vs_Gamma",
            "learned_symbols",
        ] {
            ensure!(
                review["visual_review"][axis]["status"] == "passed",
                "whole-run review missing passed {axis} evidence"
            );
            ensure!(
                review["visual_review"][axis]["evidence"]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty()),
                "whole-run review missing {axis} explanation"
            );
        }
        ensure!(
            manifest["training_domain"] != "public",
            "public-style pretraining cannot be promoted as personal handwriting"
        );
    }
    let inspected = review["visual_review"]["artifacts"]
        .as_array()
        .context("visual artifacts missing")?;
    ensure!(
        inspected.len() >= 4,
        "review needs alphabet, text, structure and variation artifacts"
    );
    for artifact in inspected {
        let path = artifact["path"].as_str().context("review artifact path")?;
        ensure!(
            artifact["sha256"] == file_hash(Path::new(path))?,
            "review artifact changed"
        );
    }
    let verification_path = review["verification"]
        .as_str()
        .context("verification path missing")?;
    let verification: Value = serde_json::from_slice(&fs::read(verification_path)?)?;
    ensure!(
        verification["model"] == manifest["id"]
            && verification["decoder_sha256"] == manifest["decoder_sha256"]
            && verification["dataset_sha256"] == manifest["dataset_sha256"],
        "verification decoder/dataset mismatch"
    );
    if model.join("corpus-audit.json").exists() {
        let audit: Value = serde_json::from_slice(&fs::read(model.join("corpus-audit.json"))?)?;
        ensure!(
            verification["dataset_requests"] == audit["samples"],
            "verification omitted corpus samples"
        );
    }
    ensure!(
        verification["failed"] == 0
            && verification["dataset_requests"].as_u64().unwrap_or(0) > 0
            && verification["passed"].as_u64()
                == Some(
                    verification["dataset_requests"].as_u64().unwrap_or(0)
                        + verification["unseen_requests"].as_u64().unwrap_or(0)
                ),
        "complete generation verification has not passed"
    );
    manifest["approved_for_default"] = json!(true);
    manifest["quality_review_sha256"] = json!(file_hash(review_path)?);
    manifest["verification_sha256"] = json!(file_hash(Path::new(verification_path))?);
    let staging = model.join(".approved-staging");
    ensure!(
        !staging.exists(),
        "approval staging exists; inspect previous interrupted approval"
    );
    fs::create_dir(&staging)?;
    for name in ["decoder.json", "vocabulary.json"] {
        fs::copy(model.join(name), staging.join(name))?;
    }
    fs::copy(review_path, staging.join("quality-review.json"))?;
    save(&staging.join("manifest.json"), &manifest)?;
    show(&staging)?;
    let approved = model.join("approved");
    let history = model.join("approved-history");
    fs::create_dir_all(&history)?;
    let hash = manifest["decoder_sha256"]
        .as_str()
        .context("decoder hash")?;
    let immutable = history.join(hash);
    if immutable.exists() {
        ensure!(
            show(&immutable)?["decoder_sha256"] == manifest["decoder_sha256"],
            "immutable snapshot mismatch"
        );
        fs::remove_dir_all(&staging)?;
    } else {
        fs::rename(&staging, &immutable)?;
    }
    let link = model.join(".approved-link");
    ensure!(!link.exists(), "approval link staging exists");
    std::os::unix::fs::symlink(Path::new("approved-history").join(hash), &link)?;
    fs::rename(&link, &approved)?;
    fs::copy(review_path, model.join("quality-review.json"))?;
    save(&model.join("manifest.json"), &manifest)?;
    Ok(
        json!({"model":manifest["id"],"approved_for_default":true,"decoder_sha256":manifest["decoder_sha256"],"inference_snapshot":approved}),
    )
}

pub fn snapshot(model: &Path, out: &Path) -> Result<Value> {
    if !model.join("glyph-training.json").exists() {
        ensure!(!out.exists(), "snapshot requires fresh output directory");
        let manifest_bytes = fs::read(model.join("manifest.json"))?;
        let mut manifest: Value = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            manifest["kind"] == "sequence-cvae-v3",
            "snapshot requires a sequence model"
        );
        let decoder_bytes = fs::read(model.join("decoder.json"))?;
        let vocabulary_bytes = fs::read(model.join("vocabulary.json"))?;
        ensure!(
            manifest["decoder_sha256"] == ink_corpus::hash(&decoder_bytes)
                && manifest["vocabulary_sha256"] == ink_corpus::hash(&vocabulary_bytes),
            "publication is changing; retry snapshot after save completes"
        );
        let decoder: ink_neural::sequence::SequenceDecoder =
            serde_json::from_slice(&decoder_bytes)?;
        decoder.validate()?;
        ensure!(
            fs::read(model.join("manifest.json"))? == manifest_bytes,
            "publication changed while snapshotting; retry"
        );
        manifest["approved_for_default"] = json!(false);
        manifest["snapshot_source"] = json!(model);
        manifest["snapshot_source_manifest_sha256"] = json!(ink_corpus::hash(&manifest_bytes));
        manifest["snapshot_checkpoint_sha256"] = manifest["checkpoint_sha256"].clone();
        manifest["snapshot_kind"] =
            json!("published sequence inference; no optimizer or source corpus");
        manifest
            .as_object_mut()
            .unwrap()
            .remove("quality_review_sha256");
        manifest
            .as_object_mut()
            .unwrap()
            .remove("verification_sha256");
        fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
        fs::create_dir(out)?;
        fs::write(out.join("decoder.json"), decoder_bytes)?;
        fs::write(out.join("vocabulary.json"), vocabulary_bytes)?;
        save(&out.join("manifest.json"), &manifest)?;
        return Ok(
            json!({"snapshot":out,"model":manifest["id"],"decoder_sha256":manifest["decoder_sha256"],"updates":manifest["updates"],"approved_for_default":false}),
        );
    }
    #[cfg(feature = "rocm")]
    {
        compositional::snapshot(model, out)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (model, out);
        anyhow::bail!("checkpoint snapshot needs native build; use nix run . --")
    }
}

#[cfg(test)]
mod export_contracts {
    #[test]
    fn compaction_preserves_strings_escapes_and_numeric_tokens() {
        let source = br#" { "name": "a b \\\" c", "weights": [ 1.234567890123456789, -0.0, 1e-09 ], "slash": "\\" } "#;
        let compact = super::compact_json(source).unwrap();
        assert_eq!(
            std::str::from_utf8(&compact).unwrap(),
            r#"{"name":"a b \\\" c","weights":[1.234567890123456789,-0.0,1e-09],"slash":"\\"}"#
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(source).unwrap(),
            serde_json::from_slice::<serde_json::Value>(&compact).unwrap()
        );
        assert!(super::compact_json(b"{bad json}").is_err());
    }
}
