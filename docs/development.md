# Personal neural handwriting

Train from personal vector ink; generate SVG and ink JSON from text or LaTeX. The `nix run . --` wrapper loads the pinned Rust/ROCm environment. Invoke it from this repository, or use its absolute path from another directory.

Current sequence recipe is **v14**: learned monotonic text/ink alignment, reversible em coordinates, separate pen-lift/curve geometry objectives and complete-prompt conditioning. Full-data m034–m036 still fail readability. The owner requires complete connected sentence generation; m027 remains a glyph baseline and does not satisfy that requirement. New whole-word training is under review; no new default is approved. See [repair evidence](model-repair-2026-10-18.md).

For readable controlled glyph composition now:

```sh
nix run . -- generate '\boxed{Hello, World!}' --model m027 --seed 419 --variation 0
```

This preserved expert composes trained glyphs; it does not establish successful whole-text joining. `latest` selects experimental weights and can still produce unreadable output.

## Start here

```sh
nix run . -- guide
nix run . -- doctor
nix run . -- data review
nix run . -- models
```

## Train a new version

First inspect a read-only plan. The second command actually starts training. New training follows the owner's renewed authorization and requires actual output review; decreasing loss is insufficient.

```sh
nix run .#training -- train --from latest --for 30m --dry-run
nix run .#training -- train --from latest --for 30m
```

`--from latest` explicitly selects the newest valid candidate as the parent. Training allocates a new `mNNN` directory, retains the parent, and starts fresh optimizers. Use an exact model ID such as `--from m027` to pin the parent. Omitting `--from` lets the trainer choose a matching parent or initialize a fresh model.

The wrapper builds optimized release code. New GPU runs automatically use up to **64 rows per batch**, reducing the batch size when padding would exceed **16,384 events**; full original rows are retained. The current corpus has **35 batches per epoch**, versus 253 at the old batch size of eight. Immutable padded inputs are cached on the GPU when their total is at most 256 MiB, and the text encoder runs once per loss. CPU runs retain a maximum of eight rows. The effective `execution_plan` overrides the architecture recipe's reference batch size and is saved with every checkpoint. No precision reduction or coordinate approximation is used.

Time budgets accept `120s`, `30m`, `2h`, or `1h30m`; default is two minutes. `--seconds 1800` remains supported. GPU is default; add `--cpu` for CPU, or `--gpu-index N` for a specific GPU. The cumulative per-model limit is eight hours. Sequence progress shows one bar per integer epoch, with batches and epoch ETA. A completed epoch retains one line with sample-weighted mean loss. A time-budget stop saves the partial epoch without reporting partial loss; resume continues its batch position. If an epoch began in an earlier run, its completion omits mean loss because the earlier batches are unavailable. Saving/checking can take additional time. Loss and progress do not establish handwriting quality.

## Continue the same version

After a new version has been trained:

```sh
nix run .#training -- resume latest --for 30m --dry-run
nix run .#training -- resume latest --for 30m
```

Resume retains the model ID and optimizer state. Dataset, recipe and vocabulary must match. On a mismatch, the CLI prints the command for creating a new version instead. Checkpoints are periodic; an interrupted process can lose work since its last completed save. The old `train --model ID --resume --seconds N` form still works.

Resume also retains its saved batch boundaries. Earlier recipes require transfer into a fresh v14 model; m033/m034 cannot resume under changed recipes. Larger batches change the optimizer trajectory and examples seen during warm-up; assess generated output. A historical full-corpus GPU forward/backward benchmark measured **13.96s → 2.85s per epoch (4.9×)**; optimizer updates, checkpointing and the glyph auxiliary were excluded. See [the throughput evidence](training-throughput-2026-10-16.md).

## Generate and inspect

```sh
nix run . -- generate 'Hello, World!' --model latest
nix run . -- generate '\left[\frac{T+t}{2}\right]' -m latest -o output/formula --seed 419
```

The default output is `output/handwriting.svg` plus `.ink.json`. `-m` and `-o` shorten `--model` and `--out`. `latest` is explicit candidate selection; it does **not** approve that model. Omit `--model` only when you intend to use the approved/default policy, which currently retains a legacy v2 fallback.

Variation defaults to **0**, matching the controlled m027 demos. Use `--seed 419` for repeatability. Nonzero `--variation` remains experimental, especially for older glyph models. The revised whole-sequence decoder must learn joining, pen lifts, baseline motion and personal identity from real trajectories; changing variation alone cannot establish natural handwriting.

## Verify, evaluate and export

```sh
nix run . -- verify latest
nix run .#training -- evaluate --model latest
nix run . -- export latest
nix run . -- check
```

Verification writes `reports/MODEL-verification/` and exits unsuccessfully if requests fail. Evaluation defaults to validation and writes `reports/evaluation.json`; only run `--split test` after final model selection. The corpus must match the model's training hash, so m027 cannot be verified/evaluated against the newly extended active corpus. Generation requests and reconstruction metrics are separate from recognition, personal style and natural joining.

Export creates `exports/MODEL-skill/` and its ZIP without overwriting an existing delivery. Use `-o exports/my-new-skill` for another fresh destination. The standalone inference package contains weights and Rust inference, without private source trajectories, Python or training libraries. Exports are diagnostic candidates until visual quality is approved. Expert approval still uses `model approve MODEL --review FILE` with hash-pinned verification and visual evidence.

`check` runs formatting, default workspace tests and native Clippy without weight-update tests. Explicit `check --training-tests` also runs native fixture optimizer tests; these are separate from personal-corpus training. Existing `model verify` and `skill export` commands remain available.

## Terminal and scripting

Terminal progress includes stage names, elapsed time, counts and ETA. Training retains one line per completed epoch; batch updates redraw that line without printing loss. The current recipe assigns the complete budget to the sequence model and skips the glyph auxiliary. Preparation and export use spinners when a total is unknown. Data review prints a compact summary; full details are opt-in:

```sh
nix run . -- --json data review > reports/data-review.json
nix run . -- --no-progress train --from latest --for 30m --dry-run
nix run . -- --quiet generate 'Hello' -m latest
```

Animation goes to stderr and is hidden automatically outside a terminal or with `TERM=dumb`. `--no-progress` keeps plain stage messages. `--progress always` forces animation, including redirected stderr. `--json` disables animation and keeps stdout machine-readable. `--quiet` suppresses progress/timing while retaining results and errors. Every command has `--help`.

The personal corpus retains 2,022 train / 251 validation / 251 test samples and original source-grouped splits. No new accepted glyph segmentation is inferred. Public pretraining is an optional design with provenance and split guards; a public-data importer is still required. Models, private data, reports, outputs, exports and retired implementations are ignored by Git.

See [CLI QA](cli-2026-10-16.md), [handwriting revision](handwriting-revision-2026-10-15.md), [architecture](vector-architecture.md), [data contract](data.md) and [GPU setup](rocm-training.md).
