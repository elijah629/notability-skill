# Handwriting revision: natural motion and personal identity

This is a **source/model-recipe revision, not a trained model delivery**. No training command, backward pass, optimizer update, calibration, model promotion, public-data download, or corpus relabeling was performed. Existing m027 weights and exports are retained. The revised decoder must be trained and visually evaluated before claiming improved handwriting.

## Findings from original vectors

The active corpus now contains 2,524 samples: 2,022 train, 251 validation, 251 test; 26,618 original strokes. SHA256: `fcb45645b2e2b89d7041c16afc6d4cf6fc49c39737bb9a0e9e5a247157521557`. The latest worksheet contributes 21 complete train rows and 658 strokes. These additions postdate m027, whose training used the earlier 2,503-row corpus.

The original `There are more of them and they are stronger` and `The reeds by the Pharaoh's daughter` rows visibly contain contextual joins, uneven spacing, changing letter sizes, and baseline movement. Per-character bounding-box normalization and fixed baseline placement erase these relationships. Increasing latent amplitude in the old isolated-glyph head instead introduces jagged geometry; this is a separate failure from handwriting identity.

The supplied train labels contain no `\Gamma` or literal Γ condition. That alone does not prove all T labels are correct. There are no isolated T samples, two separated T practice rows, and 25 complete T contexts from 17 train sources. Both practice rows come from the same source, so they are not independent writer/style evidence:

- `s_91ec7cbf31758adcc8ace2fa`: six S followed by six T; visible flat top bars crossing centered stems.
- `s_5eac586628c5f50527c9c097`: six Phi followed by six supplied T labels; visible right hooks and lower serifs, resembling the current generated T more closely than the worksheet T.

The second row is a style/semantic ambiguity, not a certified Gamma relabeling. The new auxiliary training guard prevents T conditions in mixed Greek-capital/T practice from contributing strong Latin T gradients or preferred-topology evidence. Other supported conditions in the row remain eligible. The original complete row remains in sequence supervision with its supplied label. No stroke subset was accepted as a new glyph annotation, and no independent second review was claimed.

Read-only Rust diagnostics are available through `ink --json data review data/personal/corpus.jsonl`. They distinguish isolated samples, separated practice and whole-context evidence; identify rows worth checking for cross-letter strokes; and exclude held-out sources from this diagnosis. Wide/few-stroke flags are inspection hints, not inferred letter boundaries or certified joins. Private receipts and unchanged-vector previews are under `reports/natural-v4/`.

## Implemented model and runtime changes

`sequence-cvae-v4` keeps the variable-length cubic representation and original pen order. Generation decodes the **complete expression jointly**, including words, brackets, punctuation, fractions, radicals, accents and enclosures. There is one isotropic placement transform for the complete drawing. No character splitting, artificial connecting stroke, per-letter baseline snapping, or post-generation jitter is used. `run_labels` records requested content without claiming stroke-to-letter alignment. AST coverage remains a request-accounting contract, not recognition accuracy.

The exported `whole_run` capability is enabled by the new recipe. Historical decoder files default to false; m027 is not silently presented as a trained whole-run model. Historical glyph layout now decodes stretch/matrix delimiters and punctuation under their own learned conditions. Semicolon/accent/arrow/epsilon/alpha donor substitutions and procedural character construction are retired. The old glyph layout still uses procedural *layout rules* for fractions, roots and enclosures; the new complete-sequence path creates none. Old m027 naturalness remains unresolved.

Joining must be contextual and occasional. The same character pair may continue within one native stroke or use a pen lift, including when stroke endpoints touch. MOVE/CURVE supervision retains this distinction; character boundaries do not force a lift, and endpoint proximity does not manufacture a connector. An end-to-end contract checks both versions of the same pair through cubic encoding, decoding, whole-run layout and SVG pen paths. This proves representation/runtime support, not that an untrained decoder has learned the owner's joining habits. Future approval requires a separate explained `contextual_joining` review with both naturally joined and separated examples.

New training dedicates 80% of the personal run budget to the sequence model instead of 10%. Every fourth batch uses actual recurrent self-fed predictions mixed with teacher inputs; exposure warms to 35%, and learned duration is used after warmup. An accumulated absolute-path loss supplements relative cubic likelihood, preserving source layout instead of allowing local delta errors to hide global drift. Event loss mixes uniform weights with equal mass per original pen stroke, so short crossbars and dots are not overwhelmed by dense long strokes. This uses genuine pen lifts, never character segmentation. Geometry augmentation and teacher-input zeroing are removed. Whole-run inference uses the most probable mixture's mean; learned document/run style latents supply diversity without independent coordinate noise.

`compositional-vector-v20` is an auxiliary for supplied singletons and explicitly spaced practice. Ordinary connected words never enter its disjoint glyph loss. Canonical training retains source traversal direction and source-relative curvature; the straight-line forcing penalty is removed. Ambiguous singleton supervision retains its existing restrictions. A corpus with no isolated/practice evidence can train complete sequences without requiring an empty auxiliary head.

The superseded implementation is preserved under ignored `legacy/model-experiments/pre-natural-v4/`. Recipes v3/v19 and historical model/export artifacts remain available.

## Public pretraining and transfer

The optional plan is `src/recipes/public-vector-pretraining-v1.json`; it is not a downloaded or prepared corpus. [IAM-OnDB](https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database) supplies human online text-line trajectories, under non-commercial research terms with registration requested. [MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting) supplies online math trajectories under CC-BY-NC-SA-4.0 data terms; its synthetic data is excluded from this plan. Math pretraining alone does not teach connected English handwriting. A dedicated Rust importer is still required before either dataset can be used.

Rust now enforces separate public and personal corpus stages. Public rows require dataset, license, source URL, writer ID and human-trajectory metadata; writer overlap across public splits and mixed public/personal corpora are rejected. Public pretraining uses the complete-sequence objective; output carries `neural-run-public` provenance and cannot be promoted as personal handwriting.

Explicit warm-start transfer may cross compatible recipes/corpora into a fresh model ID. It verifies checkpoint and lineage hashes, rejects overlap between parent train sources and the current held-out sources, maps embeddings by exact semantic token, and starts fresh optimizers. This permits a future public-motion checkpoint to seed personal-only fine-tuning, and permits m027 to seed a new version using the additive worksheet. Automatic parent selection remains restricted to matching recipe, corpus and vocabulary. Resume remains exact and rejects mismatches.

## Verification and remaining quality gates

Regression checks cover neural bracket/punctuation routing, complete ordered cross-letter cubics and delayed crossbars, one whole-run transform, no procedural paths in complete math decoding, fixed-seed behavior, absence of independent coordinate noise, historical capability defaults, public-domain boundaries and T anchor ambiguity. Native scheduled decoding is checked with forward-only tensors, including teacher-forward parity at zero self-feed probability. No update-producing native training tests were run.

Validation passed: 22 default workspace tests, one native forward-only regression, formatting, all-target native Clippy with warnings denied, and `git diff --check`. The joining follow-up adds the same-pair joined/lifted contract to the default suite. m027 decoder/checkpoint/optimizer and active corpus hashes match their existing manifests. Backward/optimizer behavior and trained free-generation quality for v4 remain untested, consistent with the owner's instruction not to train now.

Actual m027 inference also generated `\left[\frac{T+t}{2}\right]` through learned bracket conditions, with neural decoder provenance for both brackets. That diagnostic verifies routing, not personal bracket accuracy or the new untrained sequence model's quality.

Future v4 promotion requires separate passed, explained reviews for content, personal style, natural motion, contextual joining, T-versus-Gamma discrimination, and learned symbols, plus hash-pinned artifacts and complete generation verification. Review the worksheet-like phrases, Latin T practice, serifed mixed-Greek practice, repeated letters, joined/unjoined writing, brackets and full math expressions. Coordinate diversity, decreasing loss and AST labels do not establish these qualities. Personal held-outs and their source groups remain unchanged.
