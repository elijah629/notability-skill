# Use your own handwriting

The public repository ships the framework and a separate owner-approved m027 inference download. To train your own style, supply your own writing and labels. No original source samples are distributed.

## Set up a local checkout

Clone the repository URL you were given and enter the checkout. Keep your original notes read-only. Create `.private/models/` and put your corpus under `.private/data/personal/`. Git ignores the whole `.private/` directory. The `ink` CLI detects that workspace and keeps relative model, run and output paths there.

```sh
mkdir -p .private/models .private/data/personal
```

The pinned development and training environment targets x86_64 Linux. Training uses native Rust `tch` with LibTorch. The verified GPU backend targets the RX 9070 XT, and native CPU training is available with `--cpu`. The native runtime is large. See [the backend setup](rocm-training.md) before building it. The inference-only build in [INSTALL.md](../INSTALL.md) needs none of these training libraries.

## Prepare labeled vector ink

The trainer consumes JSONL, one `Sample` object per line. The schema is defined in [ink-corpus](../src/crates/ink-corpus/src/lib.rs). A sample has these fields:

| Field | Meaning |
| --- | --- |
| `sample_id` | A unique sample identifier |
| `label` | The complete text or supported LaTeX for the sample |
| `source_sha256` | The SHA256 of the original source, shared by related samples |
| `split` | `train`, `validation` or `test` |
| `strokes` | Ordered strokes with `points`, `source_index`, `uuid` and positive `width` |
| `metadata` | Provenance, annotation and review information |

Each stroke is a single dot or cubic control points with length `3*n+1`. Preserve original stroke order, pen lifts and cubic controls. Do not convert a raster image into invented vector ground truth. Supply labels for complete examples without guessing per-character boundaries.

All examples from the same source stay in the same split. A source stroke may appear only once. All three partitions must be nonempty. Build your own importer for your note format, or adapt an existing exporter to this schema. The framework's ZIP importer expects its specific manifest and vector-array format; it does not accept arbitrary Notability backups, PDFs or scanned worksheets.

Write your corpus to `.private/data/personal/corpus.jsonl`. The wrapper runs from `.private/`, so its default and the relative `--corpus data/personal/corpus.jsonl` argument both select that file. Absolute `--corpus` paths also work. The private dataset used in the research is not required and is not available from the public repository.

Review labels and authorship before accepting samples. The original research records review limitations rather than treating inferred labels as certified data. Keep validation and test sources out of annotation or model-selection shortcuts.

## Audit and plan

From the repository root:

```sh
nix run . -- data audit data/personal/corpus.jsonl
nix run . -- data coverage data/personal/corpus.jsonl
nix run .#training -- train --corpus data/personal/corpus.jsonl --cpu --for 2m --dry-run
```

The dry run checks corpus, recipe and parent compatibility. It does not allocate a model or start weight updates. Review its result before starting training. A model trained on a different person's corpus is not a default parent for your data.

## Train and inspect

Run training only when you intend to spend the compute budget:

```sh
nix run .#training -- train --corpus data/personal/corpus.jsonl --cpu --for 2m
# On the supported GPU setup, omit --cpu.
```

Use the model ID printed by the command. Replace `m000` below with that ID:

```sh
nix run . -- generate 'x^2+1' --model m000 --seed 419 --variation 0 --out output/first
nix run . -- verify m000 --corpus data/personal/corpus.jsonl
nix run .#training -- evaluate --model m000 --corpus data/personal/corpus.jsonl --split validation
```

Render and read the actual output. Check the alphabet, symbols, repeated letters, formulas and unseen combinations you need. A short run is a pipeline check, not a promise of readable handwriting. The current sequence research has not solved reliable connected sentence generation. Additional training alone has not established that result.

Resume with the same corpus and recipe:

```sh
nix run .#training -- resume m000 --corpus data/personal/corpus.jsonl --cpu --for 2m
```

A recipe change needs a new model version. Evaluate the test split only after final model selection, rather than using it to tune the model.

## Install your decoder

Build `ink-infer` using [INSTALL.md](../INSTALL.md), then install the skill with your model:

```sh
node scripts/install-skill.mjs \
  --to /path/to/your/agent/skills/notability-form-latex \
  --decoder .private/models/m000/decoder.json \
  --binary src/target/release/ink-infer
```

Your installed skill contains private weights. Keep it out of public commits and releases. `nix run . -- export m000` creates a separate inference-only package, but it still contains those private weights. The public-source export includes only the owner-approved m027 release, not your private model export.
