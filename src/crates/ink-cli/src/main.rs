use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Instant,
};
#[derive(Parser)]
#[command(
    name = "ink",
    version,
    about = "Train personal neural handwriting; generate SVG from text and LaTeX",
    after_help = "Quick start:\n  ink guide\n  ink train --from latest --for 30m --dry-run\n  ink train --from latest --for 30m\n  ink resume latest --for 30m\n  ink generate 'Hello, World!' --model latest\n  ink verify latest\n  ink export latest\n\nGPU training is default; --cpu selects CPU. --model latest explicitly previews an unapproved candidate.\nUse nix run . -- for inference or nix run .#training -- for native training."
)]
struct Cli {
    /// Print machine-readable JSON instead of concise human output.
    #[arg(long, global = true)]
    json: bool,
    /// Suppress progress and timing; keep results and errors.
    #[arg(short, long, global = true)]
    quiet: bool,
    /// Terminal progress policy. Always also animates redirected stderr.
    #[arg(long, global = true, value_enum, default_value = "auto")]
    progress: ProgressMode,
    /// Disable animation (plain text stage updates remain).
    #[arg(long, global = true, conflicts_with = "progress")]
    no_progress: bool,
    #[command(subcommand)]
    command: Commands,
}
#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    OrtCpu,
    Rocm,
}
#[derive(Clone, Copy, ValueEnum)]
enum ProgressMode {
    Auto,
    Always,
    Never,
}
#[derive(Args)]
struct TrainingArgs {
    #[arg(long, default_value = "data/personal/corpus.jsonl")]
    corpus: PathBuf,
    #[arg(long, default_value = "models")]
    models: PathBuf,
    /// Time budget: 30m, 2h, 1h30m, or 120s. Default: 2m; maximum: 8h.
    #[arg(long="for", value_parser=parse_duration, conflicts_with="seconds")]
    duration: Option<u64>,
    /// Time budget in seconds (compatible with existing scripts).
    #[arg(long, value_parser=clap::value_parser!(u64).range(1..=28800))]
    seconds: Option<u64>,
    /// Validate data, recipe and parent without starting training or creating a model.
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    cpu: bool,
    #[arg(long, conflicts_with = "cpu")]
    gpu_index: Option<usize>,
}
fn parse_duration(input: &str) -> std::result::Result<u64, String> {
    let mut total = 0_u64;
    let mut number = String::new();
    for c in input.trim().chars() {
        if c.is_ascii_digit() {
            number.push(c);
            continue;
        }
        let multiplier = match c {
            's' => 1,
            'm' => 60,
            'h' => 3600,
            _ => return Err("Use 30m, 2h, 1h30m or 120s".into()),
        };
        let n: u64 = number
            .parse()
            .map_err(|_| "A duration needs a number before each unit")?;
        total = n
            .checked_mul(multiplier)
            .and_then(|n| total.checked_add(n))
            .ok_or("Duration is too large")?;
        number.clear();
    }
    if !number.is_empty() {
        total = total
            .checked_add(number.parse().map_err(|_| "Duration is too large")?)
            .ok_or("Duration is too large")?;
    }
    if !(1..=28800).contains(&total) {
        return Err("Time budget must be between 1s and 8h".into());
    }
    Ok(total)
}
#[derive(Subcommand)]
enum Commands {
    /// Show copyable commands for the complete handwriting workflow.
    Guide,
    /// Show runtime availability and latest trained model.
    Doctor,
    /// Train a NEW model; optionally initialize it from an existing model.
    Train {
        #[command(flatten)]
        options: TrainingArgs,
        /// Parent model ID, directory, or latest. Creates a new model, preserving the parent.
        #[arg(long = "from", visible_alias = "model")]
        model: Option<PathBuf>,
        #[arg(long, requires = "model", hide = true)]
        resume: bool,
    },
    /// Continue the SAME model and optimizer; requires matching recipe and dataset.
    Resume {
        /// Model ID, directory, or latest.
        model: PathBuf,
        #[command(flatten)]
        options: TrainingArgs,
    },
    /// Generate neural handwriting from text or whitelisted LaTeX.
    Generate {
        /// Text or LaTeX expression. Quote shell-sensitive characters.
        #[arg(required_unless_present = "input")]
        expression: Option<String>,
        /// Alternative to positional expression, useful for scripts.
        #[arg(long, conflicts_with = "expression")]
        input: Option<String>,
        /// Model ID or directory. Default: latest quality-approved neural model.
        #[arg(short = 'm', long)]
        model: Option<PathBuf>,
        /// Output stem; writes .svg and .ink.json.
        #[arg(short = 'o', long, default_value = "output/handwriting")]
        out: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
        /// Style latent amplitude 0..2; explicit seed is reproducible. Quality needs review.
        #[arg(long, default_value_t = 0.0)]
        variation: f32,
    },
    /// List successfully trained models and their coverage.
    Models,
    /// Verify all corpus requests and new combinations for a model.
    Verify {
        model: PathBuf,
        #[arg(long, default_value = "data/personal/corpus.jsonl")]
        corpus: PathBuf,
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Export a standalone inference skill to a new directory and ZIP.
    Export {
        model: PathBuf,
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Measure posterior reconstruction on untouched held-out sources.
    Evaluate {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "data/personal/corpus.jsonl")]
        corpus: PathBuf,
        #[arg(long, default_value = "validation")]
        split: String,
        #[arg(long, default_value = "reports/evaluation.json")]
        out: PathBuf,
    },
    /// Inspect a trained model or prove backend weight updates.
    Model {
        #[command(subcommand)]
        command: Model,
    },
    /// Import, audit or inspect labeled vector data; phrases are never guessed apart.
    Data {
        #[command(subcommand)]
        command: Data,
    },
    /// Package a standalone Rust neural inference skill.
    Skill {
        #[command(subcommand)]
        command: Skill,
    },
    /// Run formatting, workspace tests and Clippy; optionally verify every dataset label.
    Check {
        /// Also run native fixture weight-update tests (never trains the personal corpus).
        #[arg(long)]
        training_tests: bool,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "data/personal/corpus.jsonl")]
        corpus: PathBuf,
        #[arg(long, default_value = "reports/neural-verification")]
        out: PathBuf,
    },
}
#[derive(Subcommand)]
enum Model {
    List,
    /// Review learned glyph modes at seed 419/variation 0; apply a supported selection.
    Glyphs {
        model: PathBuf,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long, default_value = "reports/glyph-modes")]
        out: PathBuf,
        #[arg(long)]
        selection: Option<PathBuf>,
    },
    /// Freeze a saved training checkpoint for inference and visual review.
    Snapshot {
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Promote an exact decoder after complete generation checks and visual review.
    Approve {
        model: PathBuf,
        #[arg(long)]
        review: PathBuf,
    },
    /// Generate every dataset label and fresh requests; report all failures.
    Verify {
        model: PathBuf,
        #[arg(long, default_value = "data/personal/corpus.jsonl")]
        corpus: PathBuf,
        #[arg(long, default_value = "reports/sequence-verification")]
        out: PathBuf,
    },
    Show {
        model: PathBuf,
    },
    Gate {
        #[arg(long, default_value = "runs/backend-gate")]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "ort-cpu")]
        backend: Backend,
        #[arg(long)]
        gpu_index: Option<usize>,
    },
}
#[derive(Subcommand)]
enum Data {
    /// Draw unchanged source vectors for one supplied sample.
    Preview {
        sample: String,
        #[arg(long, default_value = "data/personal/corpus.jsonl")]
        corpus: PathBuf,
        #[arg(long, default_value = "reports/source-preview.svg")]
        out: PathBuf,
    },
    Import {
        input: PathBuf,
        #[arg(long, default_value = "data/personal")]
        out: PathBuf,
    },
    Merge {
        input: PathBuf,
        #[arg(long, default_value = "data/personal")]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Prepare {
        input: PathBuf,
        #[arg(long, default_value = "data/personal")]
        out: PathBuf,
    },
    Audit {
        #[arg(default_value = "data/personal/corpus.jsonl")]
        input: PathBuf,
    },
    Coverage {
        #[arg(default_value = "data/personal/corpus.jsonl")]
        corpus: PathBuf,
    },
    Review {
        #[arg(default_value = "data/personal/corpus.jsonl")]
        input: PathBuf,
    },
    Collection {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long,num_args=1..)]
        input: Vec<String>,
        #[arg(long, default_value = "reports/missing-glyphs.json")]
        out: PathBuf,
    },
}
#[derive(Subcommand)]
enum Skill {
    Export {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "exports/personal-handwriting")]
        out: PathBuf,
    },
}
fn corpus_path(p: &Path) -> PathBuf {
    if p.is_dir() {
        p.join("corpus.jsonl")
    } else {
        p.into()
    }
}
fn run_training(
    model: Option<PathBuf>,
    options: TrainingArgs,
    resume: bool,
    repo: &Path,
) -> Result<Value> {
    let corpus = corpus_path(&options.corpus);
    let train = ink_model::TrainOptions {
        model: model.as_deref(),
        corpus: &corpus,
        models: &options.models,
        seconds: options.duration.or(options.seconds).unwrap_or(120),
        resume,
        repo,
        cpu: options.cpu,
        gpu_index: options.gpu_index,
    };
    if options.dry_run {
        let progress = ink_model::progress::Progress::spinner("Validate training plan");
        let plan = ink_model::workflow::training_plan(&train)?;
        progress.finish("ready · no training started");
        Ok(plan)
    } else {
        ink_model::train(train)
    }
}
fn execute(command: Commands) -> Result<Value> {
    let repo = std::env::var_os("INK_REPO")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    if let Some(workspace) = std::env::var_os("INK_WORKSPACE") {
        std::env::set_current_dir(workspace)?;
    } else if repo.join(".private/models").is_dir() {
        std::env::set_current_dir(repo.join(".private"))?;
    }
    let models = Path::new("models");
    let default_model = std::env::var_os("INK_DEFAULT_MODEL").map(PathBuf::from);
    let resolve = |p: Option<&Path>| ink_model::resolve(p.or(default_model.as_deref()), models);
    Ok(match command {
        Commands::Guide => json!({"guide":[
            ["Inspect", "nix run . -- doctor"],
            ["Data", "nix run . -- data review"],
            ["Plan only", "nix run .#training -- train --from latest --for 30m --dry-run"],
            ["Train NEW version", "nix run .#training -- train --from latest --for 30m"],
            ["Resume SAME version", "nix run .#training -- resume latest --for 30m"],
            ["Generate", "nix run . -- generate 'Hello, World!' --model latest"],
            ["Verify", "nix run . -- verify latest"],
            ["Evaluate", "nix run .#training -- evaluate --model latest"],
            ["Export", "nix run . -- export latest"],
            ["Code checks", "nix run . -- check"]
        ],"notes":"latest explicitly selects the newest valid candidate, not an approved default. Use the ID printed after training to pin a specific model. Only evaluate --split test after final selection. Public-data import is not implemented; public pretraining remains a plan."}),
        Commands::Doctor => {
            let corpus = Path::new("data/personal/corpus.jsonl");
            json!({"doctor":true,"runtime":"Rust","recipe":"sequence-cvae-v12 / compositional-vector-v20","training_compiled":cfg!(feature="rocm"),"corpus":corpus,"corpus_exists":corpus.is_file(),"default_model":resolve(None).ok(),"latest_candidate":resolve(Some(Path::new("latest"))).ok(),"training_command":"nix run .#training -- train --from latest --for 30m --dry-run","python":"not needed"})
        }
        Commands::Train {
            model,
            options,
            resume,
        } => run_training(model, options, resume, &repo)?,
        Commands::Resume { model, options } => run_training(Some(model), options, true, &repo)?,
        Commands::Generate {
            model,
            expression,
            input,
            out,
            seed,
            variation,
        } => ink_model::render_with_variation(
            &resolve(model.as_deref())?,
            &expression.or(input).context("expression required")?,
            seed.map(Ok).unwrap_or_else(ink_neural::random_seed)?,
            &out,
            variation,
        )?,
        Commands::Models => ink_model::list(models)?,
        Commands::Verify { model, corpus, out } => {
            let model = resolve(Some(&model))?;
            let id = ink_model::show(&model)?["id"]
                .as_str()
                .context("model id")?
                .to_owned();
            let out = out.unwrap_or_else(|| PathBuf::from(format!("reports/{id}-verification")));
            let result = ink_model::verify(&model, &corpus_path(&corpus), &out)?;
            ensure!(
                result["failed"].as_u64().unwrap_or(0) == 0,
                "Generation verification failed; inspect {}",
                out.display()
            );
            result
        }
        Commands::Export { model, out } => {
            let model = resolve(Some(&model))?;
            let id = ink_model::show(&model)?["id"]
                .as_str()
                .context("model id")?
                .to_owned();
            let out = out.unwrap_or_else(|| PathBuf::from(format!("exports/{id}-skill")));
            ensure!(
                !out.exists(),
                "Export {} already exists. Choose a fresh directory with -o exports/{id}-skill-new",
                out.display()
            );
            ink_model::export(&model, &out, &repo)?
        }
        Commands::Evaluate {
            model,
            corpus,
            split,
            out,
        } => ink_model::evaluate(&resolve(model.as_deref())?, &corpus, &split, &out)?,
        Commands::Model { command } => match command {
            Model::List => ink_model::list(models)?,
            Model::Glyphs {
                model,
                out,
                selection,
                reference,
            } => ink_model::review::glyphs(
                &resolve(Some(&model))?,
                &out,
                selection.as_deref(),
                reference
                    .as_deref()
                    .map(|p| resolve(Some(p)))
                    .transpose()?
                    .as_deref(),
            )?,
            Model::Snapshot { model, out } => ink_model::snapshot(&resolve(Some(&model))?, &out)?,
            Model::Approve { model, review } => {
                ink_model::approve(&resolve(Some(&model))?, &review)?
            }
            Model::Verify { model, corpus, out } => {
                let result =
                    ink_model::verify(&resolve(Some(&model))?, &corpus_path(&corpus), &out)?;
                ensure!(
                    result["failed"].as_u64().unwrap_or(0) == 0,
                    "Generation verification failed; inspect {}",
                    out.display()
                );
                result
            }
            Model::Show { model } => ink_model::show(&resolve(Some(&model))?)?,
            Model::Gate {
                out,
                backend,
                gpu_index,
            } => match backend {
                Backend::OrtCpu => ink_model::backend_gate(&out, &repo)?,
                Backend::Rocm => ink_model::rocm_gate(&out, &repo, gpu_index)?,
            },
        },
        Commands::Data { command } => match command {
            Data::Preview {
                sample,
                corpus,
                out,
            } => ink_model::preview_source(&corpus, &sample, &out)?,
            Data::Import { input, out } | Data::Prepare { input, out } => {
                ink_corpus::import(&input, &out)?
            }
            Data::Merge { input, corpus, out } => {
                ink_corpus::additive::merge(&input, &corpus, &out)?
            }
            Data::Audit { input } => ink_corpus::audit(&ink_corpus::load(&corpus_path(&input))?)?,
            Data::Coverage { corpus } => {
                let samples = ink_corpus::load(&corpus)?;
                let progress =
                    ink_model::progress::Progress::count("Inspect corpus", samples.len());
                let mut counts = BTreeMap::<String, BTreeMap<String, usize>>::new();
                let mut splits = BTreeMap::<String, usize>::new();
                let mut conditions = BTreeMap::<String, BTreeMap<String, usize>>::new();
                let mut syntax_errors = vec![];
                let mut max_events = 0;
                let mut usage = vec![];
                for (i, s) in samples.iter().enumerate() {
                    *splits.entry(s.split.clone()).or_default() += 1;
                    let sequence = ink_corpus::sequence::encode(s)?;
                    max_events = max_events.max(sequence.events.len());
                    let syntax = ink_layout::parse(&s.label);
                    if let Ok(ast) = &syntax {
                        let mut labels = vec![];
                        ink_layout::labels(ast, &mut labels);
                        for label in labels {
                            *conditions
                                .entry(label)
                                .or_default()
                                .entry(s.split.clone())
                                .or_default() += 1;
                        }
                    }
                    if let Err(e) = &syntax {
                        syntax_errors.push(
                            json!({"sample_id":s.sample_id,"input":s.label,"error":e.to_string()}),
                        );
                    }
                    usage.push(json!({"sample_id":s.sample_id,"split":s.split,"use":if s.split=="train"{"whole-sample training"}else{"held-out evaluation"},"events":sequence.events.len(),"syntax_supported":syntax.is_ok()}));
                    progress.set(i + 1);
                }
                progress.finish("coverage inspected");
                for s in samples {
                    if ink_corpus::glyph_label(&s.label) {
                        *counts
                            .entry(s.label)
                            .or_default()
                            .entry(s.split)
                            .or_default() += 1;
                    }
                }
                json!({"semantic_glyph_conditions":conditions,"condition_counts":"label occurrences in whole samples; not segmented glyph samples","atomic_glyphs":counts,"whole_sample_splits":splits,"max_sequence_events":max_events,"syntax_errors":syntax_errors,"sample_usage":usage,"phrase_segmentation":"never inferred"})
            }
            Data::Review { input } => {
                let samples = ink_corpus::load(&corpus_path(&input))?;
                ink_corpus::audit(&samples)?;
                json!({"samples":samples.len(),"corpus_sha256":ink_corpus::file_hash(&corpus_path(&input))?,"independent_second_pass":false,"page_transforms":"unresolved","status":"user-supplied labels; not independently certified","label_geometry_warnings":ink_corpus::label_geometry_warnings(&samples),"handwriting":ink_model::style::audit(&samples)?})
            }
            Data::Collection { model, input, out } => {
                ink_model::collection(&resolve(model.as_deref())?, &input, &out)?
            }
        },
        Commands::Skill {
            command: Skill::Export { model, out },
        } => ink_model::export(&resolve(model.as_deref())?, &out, &repo)?,
        Commands::Check {
            model,
            corpus,
            out,
            training_tests,
        } => {
            ensure!(
                !training_tests || cfg!(feature = "rocm"),
                "Native fixture tests require nix run . -- check --training-tests"
            );
            let progress = ink_model::progress::Progress::count("Code checks", 3);
            for (i, operation) in ["fmt", "test", "clippy"].into_iter().enumerate() {
                progress.message(operation);
                let mut cmd = Command::new("cargo");
                cmd.arg(operation);
                if operation == "fmt" {
                    cmd.args(["--all", "--check"]);
                } else {
                    cmd.args(["--locked", "--offline", "--workspace"]);
                    if cfg!(feature = "rocm") && (operation != "test" || training_tests) {
                        cmd.args(["--features", "rocm"]);
                    }
                    if operation == "clippy" {
                        cmd.arg("--all-targets");
                    }
                }
                cmd.arg("--manifest-path").arg(repo.join("src/Cargo.toml"));
                if operation == "clippy" {
                    cmd.args(["--", "-D", "warnings"]);
                }
                let output = progress
                    .suspend(|| cmd.output())
                    .context("run checks inside Nix development shell")?;
                if !output.status.success() {
                    eprint!(
                        "{}{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                    anyhow::bail!("{operation} failed");
                }
                progress.set(i + 1);
                progress.note(format!("{operation}: passed"));
            }
            progress.finish("passed");
            if let Some(model) = model {
                let result = ink_model::verify(&resolve(Some(&model))?, &corpus, &out)?;
                ensure!(
                    result["failed"].as_u64().unwrap_or(0) == 0,
                    "generation verification failed; see {}",
                    out.join("verification.json").display()
                );
                result
            } else {
                json!({"checks":"passed"})
            }
        }
    })
}
fn human(v: &Value) {
    if let Some(steps) = v["guide"].as_array() {
        println!("Handwriting workflow\n");
        for row in steps {
            println!(
                "{}\n  {}\n",
                row[0].as_str().unwrap_or(""),
                row[1].as_str().unwrap_or("")
            );
        }
        println!("{}", v["notes"].as_str().unwrap_or(""));
        return;
    }
    if v["doctor"] == true {
        println!("Handwriting workspace");
        println!(
            "  Training runtime: {}",
            if v["training_compiled"] == true {
                "available"
            } else {
                "use nix run . -- to load the pinned runtime"
            }
        );
        println!(
            "  Corpus: {} ({})",
            v["corpus"].as_str().unwrap_or("?"),
            if v["corpus_exists"] == true {
                "found"
            } else {
                "missing"
            }
        );
        println!("  Recipe: {}", v["recipe"].as_str().unwrap_or("?"));
        println!(
            "  Default model: {}",
            v["default_model"].as_str().unwrap_or("none")
        );
        println!(
            "  Latest candidate: {} (quality not implied)",
            v["latest_candidate"].as_str().unwrap_or("none")
        );
        println!("\nNext: nix run . -- guide");
        return;
    }
    if v["action"] == "training-plan" {
        println!("Training plan · {}", v["mode"].as_str().unwrap_or("?"));
        println!(
            "  Parent: {}",
            v["parent"].as_str().unwrap_or("none; fresh initialization")
        );
        println!(
            "  Data: {} train / {} validation / {} test · {}",
            v["training_samples"],
            v["validation_samples"],
            v["test_samples"],
            v["training_domain"].as_str().unwrap_or("?")
        );
        println!(
            "  Budget: {} seconds · {}",
            v["seconds"],
            v["device"].as_str().unwrap_or("?")
        );
        println!("  Recipe: {}", v["recipe"].as_str().unwrap_or("?"));
        println!(
            "  Batches: up to {} rows · {} batches/epoch",
            v["execution_plan"]["max_batch_samples"], v["batches_per_epoch"]
        );
        println!(
            "\nReady. No training started, no model created. Remove --dry-run to run this plan."
        );
        return;
    }
    if v.get("handwriting").is_some() {
        println!("Data review · {} samples", v["samples"]);
        println!(
            "  New worksheet rows: {}",
            v["handwriting"]["new_supplied_rows"]
                .as_array()
                .map_or(0, Vec::len)
        );
        let t = &v["handwriting"]["character_support"]["T"];
        println!(
            "  T evidence: {} isolated / {} practice rows / {} contexts",
            t["isolated"].as_array().map_or(0, Vec::len),
            t["separated_practice"].as_array().map_or(0, Vec::len),
            t["whole_contexts"].as_array().map_or(0, Vec::len)
        );
        println!(
            "  Label geometry warnings: {}",
            v["label_geometry_warnings"].as_array().map_or(0, Vec::len)
        );
        println!(
            "  Supplied labels; independent second review absent. No labels or weights changed."
        );
        println!("\nDetails: nix run . -- --json data review > reports/data-review.json");
        return;
    }
    if v.get("source_groups").is_some() && v.get("strokes").is_some() {
        println!(
            "Corpus audit · {} samples · {} strokes · {} source groups",
            v["samples"], v["strokes"], v["source_groups"]
        );
        println!(
            "  Split: {} train / {} validation / {} test",
            v["splits"]["train"], v["splits"]["validation"], v["splits"]["test"]
        );
        println!(
            "  Source split leakage: {} · duplicate strokes: {}",
            v["source_split_leakage"], v["duplicate_stroke_count"]
        );
        return;
    }
    if let Some(path) = v["snapshot"].as_str() {
        println!(
            "Saved inference snapshot: {path} ({} glyph conditions; quality pending)",
            v["glyph_conditions"]
        );
        return;
    }
    if v.get("dataset_requests").is_some() {
        println!(
            "Generation requests: {} passed, {} failed; report records content requests, not recognition accuracy.",
            v["passed"], v["failed"]
        );
        return;
    }
    if v.get("whole_samples").is_some() {
        println!(
            "{}: {} of {} whole samples evaluated; {} failures",
            v["split"].as_str().unwrap_or("evaluation"),
            v["evaluated_samples"],
            v["whole_samples"],
            v["failures"].as_array().map_or(0, Vec::len)
        );
        println!("Geometry: {}", v["geometry"]);
        println!("Content/style require free-generation review.");
        return;
    }
    if let Some(examples) = v["examples"].as_array() {
        println!(
            "Verified {} unseen formulas: all glyphs neural, finite bounds, fixed-seed reproducibility.",
            examples.len()
        );
        return;
    }
    if v.get("eligible_atomic_samples").is_some() {
        println!(
            "{}: {} eligible held-out glyphs",
            v["split"].as_str().unwrap_or("evaluation"),
            v["eligible_atomic_samples"]
        );
        if let Some(rmse) = v["geometry"]["posterior_xy_rmse"].as_f64() {
            println!(
                "Posterior XY RMSE: {rmse:.6}; stroke mask accuracy: {:.2}%",
                v["geometry"]["stroke_mask_accuracy"].as_f64().unwrap_or(0.) * 100.
            );
        }
        if let Some(status) = v["status"].as_str() {
            println!("{status}");
        }
        println!("Content/style quality: not independently scored.");
        return;
    }
    if let Some(archive) = v["archive"].as_str() {
        println!("Exported neural inference skill: {archive}");
        println!("Directory: {}", v["directory"].as_str().unwrap_or("?"));
        return;
    }
    if v["kind"] == "sequence-cvae-v3" && v.get("id").is_some() {
        println!(
            "Model {} · {} whole train samples · {} glyph conditions · {} updates",
            v["id"].as_str().unwrap_or("?"),
            v["training_samples"],
            v["glyph_conditions"],
            v["updates"]
        );
        println!(
            "Default approved: {}; Rust decoder error: {}",
            v["approved_for_default"], v["rust_decoder_max_error"]
        );
        return;
    }
    if v["kind"] == "conditional-vae-v2" && v.get("id").is_some() {
        println!(
            "Model {} · {} glyphs · {} updates",
            v["id"].as_str().unwrap_or("?"),
            v["glyph_labels"],
            v["updates"]
        );
        println!(
            "Training samples: {}; loss: {}; time: {}s",
            v["training_samples"], v["final_loss"], v["training_seconds"]
        );
        println!(
            "Checkpoint reload: {}; Rust decoder maximum error: {}",
            v["checkpoint_reload_identical"], v["rust_decoder_max_error"]
        );
        println!(
            "Dataset SHA256: {}",
            v["dataset_sha256"].as_str().unwrap_or("?")
        );
        return;
    }

    if let Some(a) = v.as_array() {
        if a.is_empty() {
            println!("No trained neural models. Run ink train.");
        }
        println!("MODEL   SAMPLES  UPDATES  APPROVED  RECIPE");
        for row in a {
            let m = &row["manifest"];
            if m["kind"] == "sequence-cvae-v3" {
                println!(
                    "{:<7} {:>7} {:>8} {:>9}  {}",
                    m["id"].as_str().unwrap_or("?"),
                    m["training_samples"],
                    m["updates"],
                    m["approved_for_default"],
                    m["recipe"]["id"].as_str().unwrap_or("legacy")
                );
                continue;
            }
            println!(
                "{}  {} glyphs  {} updates  loss {}",
                m["id"].as_str().unwrap_or("?"),
                m["glyph_labels"],
                m["updates"],
                m["final_loss"]
            );
        }
        if !a.is_empty() {
            println!(
                "\nPreview candidate: nix run . -- generate 'Hello' --model latest\nlatest is explicit candidate selection; it does not approve a model."
            );
        }
        return;
    }
    if let Some(svg) = v["svg"].as_str() {
        println!(
            "Wrote {svg} (requested symbols: {}; pen strokes: {}; model {})",
            v["glyphs"],
            v["generated_strokes"],
            v["model"].as_str().unwrap_or("?")
        );
        println!("Ink JSON: {}", v["ink"].as_str().unwrap_or(""));
        println!("Seed: {}; variation: {}", v["seed"], v["variation"]);
        return;
    }
    if v.get("updates").is_some() && v.get("model").is_some() {
        if v.get("training_samples").is_some() {
            println!(
                "Model: {} · {} whole samples · {} glyph conditions · {} sequence updates",
                v["model"].as_str().unwrap_or("?"),
                v["training_samples"],
                v["glyph_conditions"],
                v["updates"]
            );
            println!("Default approved: {}", v["approved_for_default"]);
            if let Some(id) = v["model"]
                .as_str()
                .and_then(|s| Path::new(s).file_name())
                .and_then(|s| s.to_str())
            {
                println!(
                    "\nPreview: nix run . -- generate 'Hello, World!' --model {id}\nContinue: nix run .#training -- resume {id} --for 30m\nVerify: nix run . -- verify {id}"
                );
            }
            return;
        }
        println!(
            "Model: {} · {} updates · {} glyphs",
            v["model"].as_str().unwrap_or("?"),
            v["updates"],
            v["glyph_labels"]
        );
        return;
    }
    if let Some(a) = v["atomic_glyphs"].as_object() {
        if let Some(conditions) = v["semantic_glyph_conditions"].as_object() {
            let count = conditions
                .values()
                .filter(|c| c["train"].as_u64().unwrap_or(0) > 0)
                .count();
            println!(
                "Whole-sample train glyph conditions: {count}; historical atomic labels: {}",
                a.values()
                    .filter(|c| c["train"].as_u64().unwrap_or(0) > 0)
                    .count()
            );
            println!("Conditions are label occurrences, not inferred glyph annotations.");
        }
        if let Some(splits) = v["whole_sample_splits"].as_object() {
            println!(
                "Whole samples: train {}, validation {}, test {}; max {} events; syntax failures {}",
                splits["train"],
                splits["validation"],
                splits["test"],
                v["max_sequence_events"],
                v["syntax_errors"].as_array().map_or(0, Vec::len)
            );
        }
        println!("Glyph       Train  Validation  Test");
        for (l, c) in a {
            println!(
                "{l:<10} {:>5} {:>11} {:>5}",
                c["train"].as_u64().unwrap_or(0),
                c["validation"].as_u64().unwrap_or(0),
                c["test"].as_u64().unwrap_or(0)
            );
        }
        return;
    }
    if let Some(a) = v.as_object() {
        for (k, val) in a {
            match val {
                Value::String(s) => println!("{}: {s}", k.replace('_', " ")),
                _ => println!("{}: {val}", k.replace('_', " ")),
            }
        }
    }
}
fn main() -> ExitCode {
    let c = Cli::parse();
    let mode = if c.json || c.no_progress {
        ink_model::progress::Mode::Never
    } else {
        match c.progress {
            ProgressMode::Auto => ink_model::progress::Mode::Auto,
            ProgressMode::Always => ink_model::progress::Mode::Always,
            ProgressMode::Never => ink_model::progress::Mode::Never,
        }
    };
    ink_model::progress::configure(mode, c.quiet || c.json);
    let started = Instant::now();
    let label = match &c.command {
        Commands::Doctor
        | Commands::Models
        | Commands::Model {
            command: Model::List | Model::Show { .. },
        } => Some("Inspect models"),
        Commands::Generate { .. } => Some("Generate handwriting"),
        Commands::Data {
            command: Data::Coverage { .. },
        } => None,
        Commands::Data { .. } => Some("Inspect data"),
        _ => None,
    };
    let activity = label.map(ink_model::progress::Progress::spinner);
    match execute(c.command) {
        Ok(v) => {
            if let Some(activity) = activity {
                activity.finish("done");
            }
            if c.json {
                println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            } else {
                human(&v);
            }
            if !c.json && !c.quiet {
                println!("Completed in {:.2}s", started.elapsed().as_secs_f64());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            drop(activity);
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod cli_contracts {
    use super::*;
    #[test]
    fn durations_aliases_and_resume_have_unambiguous_modes() {
        let cli = Cli::try_parse_from([
            "ink",
            "train",
            "--from",
            "latest",
            "--for",
            "1h30m",
            "--dry-run",
        ])
        .unwrap();
        match cli.command {
            Commands::Train {
                model,
                options,
                resume,
            } => {
                assert_eq!(model.unwrap(), PathBuf::from("latest"));
                assert_eq!(options.duration, Some(5400));
                assert!(options.dry_run);
                assert!(!resume);
            }
            _ => panic!("wrong training mode"),
        }
        let old = Cli::try_parse_from([
            "ink",
            "train",
            "--model",
            "m027",
            "--seconds",
            "120",
            "--resume",
        ])
        .unwrap();
        assert!(matches!(old.command, Commands::Train { resume: true, .. }));
        assert!(matches!(
            Cli::try_parse_from(["ink", "resume", "latest", "--for", "30m"])
                .unwrap()
                .command,
            Commands::Resume { .. }
        ));
        for args in [
            vec!["ink", "train", "--for", "0s"],
            vec!["ink", "train", "--for", "9h"],
            vec!["ink", "train", "--for", "20m", "--seconds", "5"],
            vec!["ink", "resume"],
            vec!["ink", "train", "--cpu", "--gpu-index", "0"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
        }
        assert!(parse_duration("18446744073709551615h").is_err());
    }
    #[test]
    fn safe_generation_defaults_and_plain_progress_are_parseable() {
        let cli = Cli::try_parse_from([
            "ink",
            "--no-progress",
            "generate",
            "Hello",
            "-m",
            "latest",
            "-o",
            "output/hello",
        ])
        .unwrap();
        assert!(cli.no_progress);
        assert!(matches!(
            cli.command,
            Commands::Generate { variation: 0.0, .. }
        ));
        assert!(Cli::try_parse_from(["ink", "--json", "data", "review"]).is_ok());
        assert!(
            Cli::try_parse_from(["ink", "--progress", "always", "--no-progress", "models"])
                .is_err()
        );
    }
}
