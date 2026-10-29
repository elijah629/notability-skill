# Full-dataset handwriting handoff — 2026-10-15

Latest continuation: [October 18 model repair](model-repair-2026-10-18.md). The owner has newly authorized training to obtain a working model. Source recipe is now v14 with learned alignment, em units and separate MOVE/CURVE geometry supervision; full-data m034–m036 remain unreadable. A larger whole-word curriculum is under review. The owner explicitly requires whole connected sentences; glyph composition does not count as completion. The old ten-minute authorization below remains historical and is not the authority for this new work.

Current continuation: [October 18 squiggle diagnosis](squiggle-diagnosis-2026-10-18.md). Source recipe v8; latest full-corpus candidate m033 remains unreadable and unapproved. The owner authorized at most ten minutes of GPU work while preserving old models; conservative accounting is 581.05 seconds and all workers are stopped. Do not start further GPU training under this authorization. m027–m030 and the active corpus are unchanged. Older authorizations below do not authorize another run. The prior [loss-normalization correction](loss-normalization-2026-10-16.md), [throughput revision](training-throughput-2026-10-16.md) and [natural motion revision](handwriting-revision-2026-10-15.md) remain historical evidence.

Latest continuation: [m027 training, final output and standalone export](handwriting-m027-2026-10-15.md). Tests fix seed 419 and variation 0. The m020 checkpoint below remains historical evidence. Earlier m021–m023 experiments are recorded in [the pen/legibility continuation](handwriting-pen-legibility-2026-10-15.md).

This is the continuation record for the implementation requested by the owner. The latest candidate is **m020**, unapproved. Its 300-second run completed and saved both optimizers, checkpoints and decoder. There is no new quality-approved default. The user requested this progress checkpoint before further work.

## Scope and data authority

Use only the existing ZIP. Preserve supplied source groups and split: 2,001 train / 251 validation / 251 test, 2,503 total, 25,960 strokes. Do not re-review original notes, import public data, change supplied labels, invent character annotations or move holdout rows into train. These explicit session instructions authorize using the supplied labels with their limitations; they do not certify the independent reviews otherwise required for newly accepted data.

Corpus SHA256: `5c33fdf811918ea5466c22a5801bb3c8aef7c63a3ff2ce590b9334fbbe20d862`. ZIP and original cubic data remain unchanged. The old 84-row/20-condition limitation was the v2 atomic-only/eight-stroke training filter, not the corpus size. Full sequence preparation now retains all whole rows and every cubic event. Maximum encoded row length is 1,299 events.

Supplied singleton `0` visibly draws an open V/U. It remains in full supervision and weak auxiliary training; strong ordered anchoring is disabled and the warning is explicit. Singleton `9` also has an unusual supplied shape; it was inspected without relabeling. The `k`/`l` practice row belongs to holdout, so those letters must learn from complete ordinary train words. Internal attention/transport assignments are weak model variables, not reviewed character labels.

## Implemented code

- `ink-corpus/src/sequence.rs`: reversible variable-length MOVE/CURVE/EOS encoding; cubic controls, pen order and widths retained. Coverage accounts for each original row.
- `ink-model/src/sequence.rs`: Rust/tch ROCm text-conditioned CVAE, GRU text encoder, masked attention recurrent decoder, 16-dimensional style latent, five Gaussian coordinate/width mixtures, learned duration, KL warm-up, prior exposure, teacher-input dropout and conservative geometry augmentation. All 2,001 train rows cycle through length-bucketed batches. Fixed 20,000-update stopping is removed.
- `ink-neural/src/sequence.rs`: pure Rust inference with matching GRU/attention math, schema v3 and legacy v2 loading. Raw whole-sequence prior generation remains unreliable; practical generation uses the learned compositional head through the AST.
- `ink-model/src/compositional.rs` and `ink-neural/src/compositional.rs`: whole-row weak vector supervision on 1,617 flat-layout train rows / 137 direct glyph conditions. Learned advances, shared document and local noise, canonical em coordinates, prefix stroke counts, smooth cubic interpolation and per-token learned residual logits. Auxiliary output has 12 stroke slots and 24 sampled points per stroke; this bound does not truncate the full sequence corpus. No runtime source/exemplar trajectory lookup exists.
- Auxiliary recipe v11 added ordered arc-length sampling across cubic segments, reverse-traversal weak costs and balanced entropic whole-stroke transport. Earlier segment-index sampling over-weighted dense cubic regions. Recipe v12 supervises continuous occupancy counts before prefix-mask saturation, including singleton rows. Versioned recipes are retained.
- `ink-layout/src/lib.rs`: AST grammar/layout for boxes, matrix families, cases, aligned, gathered, array, fractions, indexed roots, scripts, stretchy delimiters, text, accents, spacing and supplied symbol commands. Baseline, x-height, descenders and punctuation replace equal-height glyph normalization. Final bounds include controls and pen thickness. Structural strokes have procedural provenance and seeded mild variation.
- CLI preserves positional input and existing options. `--variation` defaults to 1, accepts 0–2; omitted seed is random and reported. Grammar/model errors precede files. `data preview`, coverage, evaluate, model verify/snapshot/approve, check and export are implemented. Reports distinguish grammar, training evidence, reconstruction geometry, diversity and visual quality.
- Checkpoints store sequence/head weights, Adam moments, update/refinement cursors, lineage and hashes. RNG/augmentation and sampling derive from saved progress. Resume rejects recipe/data/vocabulary mismatches. New runs get a new model ID and may warm-start compatible graph weights with recorded parents; this is distinct from resume.
- Approval requires complete matching-decoder generation verification and hashed visual artifacts. It creates immutable approved inference history and an atomic `approved` symlink. Later training does not overwrite the approved inference snapshot. No new candidate has passed this gate.

Repository paths above are relative to `src/crates/`. Read [architecture](vector-architecture.md), [data contract](data.md) and [sequence status](sequence-status.md) for permanent contracts. Research references: [Graves text-conditioned sequence synthesis](https://arxiv.org/abs/1308.0850) and [Ha/Eck vector sketch generation](https://arxiv.org/abs/1704.03477); the implementation adapts these ideas and does not claim to reproduce either paper's demonstrated quality.

## Current model state

| Model | Recipe | Saved state and outcome |
| --- | --- | --- |
| m007–m008 | early sequence/head | Development failures; duration/condition coverage/readability problems. Retained, unapproved. |
| m009–m012 | whole-row head extensions | Expanded flat-row evidence and rare conditions; geometry checks passed but several glyphs malformed. |
| m013–m016 | prefix counts, spacing, balanced anchors, residual refinement | Improved punctuation/interpolation; raw sequence probes still unreadable and isolated glyphs failed review. |
| m017 | head v9 | Point-set loss without adequate curvature regularization produced zigzags. Training interrupted after saved checkpoint; unapproved. |
| m018 | head v10 | Completed 1,802.54 seconds, 5,883 sequence updates; 416,180 head Adam updates plus 258,325 token refinement updates. Full generation contracts passed; visual alphabet failed. |
| m019 | head v11 | Whole-sequence phase completed, 5,643 updates. Head interrupted deliberately to fix saturated stroke-count gradients. Saved head state: 24,997 Adam updates / 120.96 seconds. Root decoder is the earlier sequence publication; use a frozen head snapshot for inspection. |
| **m020** | **head v12** | Completed 302.78 seconds, all 2,001 rows visited, 966 sequence updates; head 39,154 Adam updates and 20,632 refinement updates / 269.51 seconds. Saved final decoder includes head. Still unapproved. |

m020 final decoder SHA256: `c53059f3dca9927b7178d1b762f35be6ab99f30a43ec6c5e051b360a3605b93d`.

m020 native/Rust maximum errors: sequence `8.58306884765625e-6`, glyph head `2.384185791015625e-7`. Actual GPU uninterrupted/restored Adam update difference: `0`. Hardware is the verified RX 9070 XT, PCI `0000:03:00.0`, gfx1201, 128 SIMD; generic HIP name alone is not the identity check.

m020 final `Hello, World!` boxed output is recognizable and unclipped. Its final lowercase/digit sheet still fails overall quality: `b`, `k`, `q`, `9` and the minus shape need work. Do not describe the model as universally readable, or promote it on the strength of the Hello example. Direct whole-sequence prior probes are a separate unresolved failure.

## Verification already completed

- Current code: 17 workspace contract tests, formatting and Clippy with warnings denied. Tests include actual sequence/head updates, original cubic round trips, full Adam restore, resumed-update equality, Rust/native parity, source split/stroke identity, grammar, baseline/width geometry and arc-length density invariance.
- m018 final decoder: **2,511/2,511** fresh generation requests passed (all 2,503 labels plus eight unseen combinations). This checks requested AST glyph sequence, finite coordinates, termination, bounds, provenance and fixed-seed equality. It does not recognize rendered text.
- m018 validation: **251/251**, no evaluation failures. Teacher-forced posterior delta RMSE `0.0213854658`; prior `0.0213553722`; event accuracy `0.9264426917`; relative duration error `0.1848037573`. Latent posterior variances and KL are in the report. Content/style remain unscored. Test split has not been used for final selection.
- Actual inference CLI checks passed: byte-identical fixed seed, changed geometry with another seed, variation-zero SVG equality across seeds, reported random seeds, unsupported command/ragged matrix/invalid variation failing before SVG or ink JSON creation.
- m018 diagnostic ZIP export unpacked under `/tmp/ink-export-m018`: all checksums passed and generated SVG matched CLI bytes. ELF links only libc/libm/libgcc, no Torch/Python. Bundled Nix flake provisions matching runtime on another Nix host; the binary alone is not a universally portable static executable.
- Pinned Nix production builds passed for both v11 and current v12. The final handoff check also repeated all 17 workspace tests successfully. Documentation links and `git diff --check` passed.

The added verification diversity report measures raw glyph-coordinate RMS across seeds and repeated `l` occurrences inside one document, plus variation-zero equality. m018's earlier full verification predates this addition; the next full model verification must populate these measurements. Visible variation and readability still require actual review.

## Saved local evidence

Models, data, generated outputs, reports and exports are gitignored and remain on this workspace. A Git commit saves code/docs/recipes, not private weights. Preserve these directories when moving hosts:

- `models/m020/`: final manifest, decoder/vocabulary, sequence and glyph checkpoint/optimizer safetensors, train and glyph lineage, corpus audit, head training state and parent record.
- `models/m018/`, `models/m019/` and earlier model directories: experiment/checkpoint history. Original m002/m003 remain intact.
- `reports/full-coverage-final.json`: all-row usage/syntax/event inventory.
- `reports/m018-verification/verification.json` and all generated dataset-label SVGs.
- `reports/m018-validation.json`, `reports/m018-quality-review.json` and `reports/m020-quality-review.json` (both failed), `reports/cli-contracts-m018/report.json`.
- `reports/m019-snapshot-01/`, `reports/m020-snapshot-01/`: immutable diagnostic decoder copies. These are intermediate weights, not the final m020 decoder.
- `reports/sequence-probes-m016/` and `reports/sequence-probes-m018/`: direct sequence output failures; source previews `reports/source-{H,ef,zero,nine,ab}*` retain their provenance.
- `output/sequence/hello-m020-final.{svg,png,ink.json}` and `alphabet-m020-final.{svg,png,ink.json}`: latest inspected final artifacts, seed 419 / variation 1.
- `exports/m018-candidate/` and `.zip`: unapproved diagnostic export.
- `exports/m020-training-checkpoint-20261015.tar.gz`: complete m020 training-directory backup, including weights, both optimizers, lineage and manifest. SHA256 `5fce94ff298e90e7c74536104806cdff672479ca5eb3c2ca960a9cf1a57efb2d`. The original private corpus/ZIP remain separately on disk.

## Continuation order and completion criteria

1. Start with `model show m020`, inspect saved hashes and final alphabet/Hello SVGs. No training worker should remain active after this checkpoint. Use the pinned `nix run . --` wrapper; `src/target/debug/ink` is stale historical code. Current native debug executable is under `src/target/rocm/debug/`; release `ink-infer` is inference-only.
2. Improve failed glyphs and raw sequence synthesis using only allowed full train rows. Review actual generated output after short experiments. The v12 count correction helped counts and `t`/`+`, but did not solve content quality. Repeated time-only training or lower loss is not sufficient evidence. Account for source-label ambiguity honestly.
3. Resume m020 only with exactly matching full sequence v3/head v12 recipes. Changed training recipes require a new ID. m019 has an interrupted head and old recipe; changing current source to v12 then resuming it is correctly rejected. A new compatible warm-start run is allowed and recorded.
4. After final weight changes, run full generation verification, validation, and only then final independent test evaluation. Review uppercase/lowercase/digits/all supplied symbols, boxed Hello, all structural families, a long sentence, repeated characters and unseen combinations. Record failed cases separately. Verify actual learned seed diversity, zero variation, optimizer recovery and standalone export again for the selected decoder.
5. Approve only an exact final decoder that passes both full request contracts and visual quality. Re-test the two requested commands without `--model`. Export that approved snapshot and compare unpacked Rust output. Until then the default remains the existing legacy fallback; explicit `--model m020` is necessary for these new candidates.

```sh
nix run . -- generate '\boxed{Hello, World!}' --model m020 --out output/hello --seed 419
nix run . -- generate '\begin{bmatrix}1&2\\3&4\end{bmatrix}' --model m020 --out output/matrix --seed 419
nix run . -- model show m020
nix run . -- model verify m020 --out reports/m020-verification
nix run .#training -- evaluate --model m020 --split validation --out reports/m020-validation.json
# Only after selection/tuning is final:
nix run .#training -- evaluate --model m020 --split test --out reports/m020-test.json
nix run .#training -- train --model m020 --resume --seconds 300
nix run . -- check --model m020 --out reports/m020-verification-final
nix run . -- skill export --model m020 --out exports/m020-candidate
```

GPU authorization was eight hours total for this implementation, including tuning/validation. Saved manifests and interrupted head checkpoints account for approximately **8,048 seconds (2.24 hours)** of training. This is a lower bound: interrupted work between checkpoints and validation/gate time are not all in that sum. Reserve a conservative **2.5 hours already spent**, leaving at most **5.5 hours**, and record future actual wall time. Code currently enforces an eight-hour cumulative cap per model; it does not implement an aggregate session-wide GPU meter. The operator must enforce the total authorization across new IDs.

Known engineering gaps: sequential checkpoint files are hash-verified but lack an atomic whole-checkpoint transaction/recovery journal and a training lock. Approval snapshots are immutable/atomic separately. No final m020 all-label verification, test evaluation, accepted visual review, final standalone export or default promotion has been performed.

## Source commits

- `63e871d`: whole-sample sequence training, compositional generation, extended AST/CLI/lifecycle and documentation.
- `a2fdcea`: balanced whole-stroke transport and arc-length geometry.
- `d028b99`: learned diversity report and acceptance limits.
- `e5c38f9`: continuous stroke-count supervision before prefix saturation (head v12).

The source commits and recipes preserve the implementation. A future claim of completion must resolve the explicit visual failures above, not merely cite this handoff or the 2,511 geometry checks.
