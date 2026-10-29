# Squiggle diagnosis — October 18, 2026

**Outcome: complete-text generation is still unreadable. No model was approved or promoted.** Source recipe v8 corrects demonstrated conditioning and feedback problems, but the bounded experiments do not establish usable handwriting, natural joining, learned punctuation quality or personal fidelity.

The owner explicitly authorized at most ten minutes of GPU training and required preserving old models. All GPU workers stopped. Conservative wall accounting is **581.05 / 600 seconds**: m031 123.44 s, interrupted m032 charged 120 s, m033 302.21 s, separate one-source diagnostic 35.40 s. Saved checkpoint times omit interrupted unsaved work, so they are not the budget accounting. No further GPU training is authorized by this trial.

## Reproduction and diagnosis

Frozen m030 generated a single U-shaped stroke for `Hello, World!` at seed 419, variation zero. Source-guided teacher reconstruction also failed: source event types, original previous events and posterior style did not recover readable letters. This rules out treating the problem as ordinary stylistic variation or solely exposure to self-fed coordinates.

The decoder GRU received one constant sentence mean at every event. Local text attention entered only after recurrence. v8 instead supplies the current soft monotonic encoded-text window to the recurrent state, consistently in native teacher forcing, scheduled rollout and exported Rust inference. A deterministic regression test failed before the change with “current character never reached recurrent state” and passes afterward. The historical constant-mean convention remains available through serde defaults.

Other observed risks were dominant CURVE predictions suppressing rare pen lifts/EOS, dense cubic controls weakening geometry supervision, and mismatched unused controls/width feedback. Recipe history records the bounded experiments:

- v6: balanced MOVE/CURVE/EOS event loss; canonical feedback; direct greedy-mixture geometry and cubic second-difference supervision; zero/prior/posterior style exposure; per-token duration; complete-sequence budget.
- v7: narrower soft attention and geometry gain based on original event count per supplied token, bounded 1–64. No source resampling or inferred character segmentation.
- v8: current text window enters the GRU before recurrence. Tensor shapes remain compatible for explicit transfer; new recipes require fresh optimizers. Old duration heads are reset when their parameterization differs.

Frozen inference counterfactuals tested raw versus canonical feedback, sampled event kinds, sampled mixture components, and mixture density peaks instead of highest mixture mass. None restored readable complete text. These policies remain explicitly ignored diagnostic tests, not new production sampling defaults. Diagnostic rendering appends EOS on an exhausted rollout and records `terminated: false`; production retains its termination error.

## Actual updates and visual results

| Candidate | Recipe | Saved updates | Saved training seconds | Result |
| --- | --- | ---: | ---: | --- |
| m031, transferred from m030 | v6 | 967 | 120.179089 | Unreadable |
| m032, transferred from m031 | v7 | 516 | 60.017774 | Interrupted after a saved checkpoint; unreadable |
| m033, transferred from m032 | v8 | 2,239 | 300.566408 | Final `x` and `Hello, World!` unreadable |
| Separate diagnostic m000, transferred from m033 | v8 | 2,885 | 35.067611 | Learned one two-stroke `x`; not a general handwriting model |

Each full-corpus run uses all **2,022 train rows**, with **35 batches per epoch** under the bounded GPU execution plan. The isolated diagnostic uses the exact original train row `s_75cf18379676be431fd1cba3`; its separate corpus retains one unchanged validation and one unchanged test row without optimizing them. This probe is outside the main model store and must never be selected as a general candidate. It demonstrates that the decoder can learn a native cubic `x`; it does not prove cross-sample textual alignment or unseen text quality.

Final m033 teacher/native-export parity checks covered every event of the inspected train sources: 13-event `x`, maximum parameter error **2.6226043701171875e-6**; 636-event sentence, **5.7220458984375e-6**. Matching inference math does not imply correct learned content. Even source-guided m033 reconstruction remained visibly wrong.

Final decoder SHA256 values:

```text
m031 434d2b8bef2928173888e5387d6b0da938e4c4881631858b945fac744b50bccf
m032 7e47e7648763e267d48e9b74c9b6af981bb0be14f2e9450e6b1b0062ab95aaf8
m033 fcc50c1e6828515be8e05496d70c21d07d16053b2dffa22c9882bf206b1669a2
```

Local ignored evidence lives in `reports/squiggle-diagnosis-20261018/`: `outcome.json`, `preserved-hashes.json`, `pilot1.log`–`pilot3.log`, `overfit.log`, `final-teacher/teacher-report.json`, `final-teacher/contact.png`, `density-rollouts/rollouts.json`, `density-rollouts/contact.png`, `final-hello.png`, `final-x.png` and `overfit-x.png`. Source-guided reconstructions are marked diagnostic and must not be presented as autonomous generation.

## Data, compatibility and verification

The active corpus is unchanged: 2,524 rows, split 2,022 / 251 / 251; SHA256 `fcb45645b2e2b89d7041c16afc6d4cf6fc49c39737bb9a0e9e5a247157521557`. Original controls, widths, pen lifts, source grouping and labels remain intact. No public data was downloaded, no samples relabeled or newly accepted, and no holdout rows were used to select these fixes. All files under m027–m030 match the pre-trial integrity record.

Historical decoders default to their original sentence-mean conditioning, attention strength 4, old duration convention and old feedback. New exports explicitly record v8 semantics. Geometry/normalization do not introduce procedural brackets or connector strokes. Generation CLI reports requested symbols and actual pen strokes separately, with content recognition explicitly unscored.

Contract coverage includes recurrent local-text influence, historical defaults, source event balancing, canonical feedback, reversible source geometry and native teacher/scheduled-forward equivalence at zero sampling probability. Final `nix run . -- --json check` passed formatting, default tests and native Clippy without optimizer tests. The explicit native forward-only fixture also passed without updates. `train --from m033 --for 30m --dry-run` reported v8, 2,022 train rows and 35 batches per epoch without creating a model or starting training. Final integrity verification matched all 49 recorded old-model/corpus files. Full model approval remains blocked by visual failure regardless of these engineering checks.

## Next quality gate

The contrast between a learned single-source `x` and failed full-corpus output supports investigating cross-sample content/alignment and training interference. It does not establish their sole cause. Before another full run, require a small multi-label train-only curriculum to reconstruct distinct supplied labels and generate their characters autonomously, with explicit content and pen-lift review. If an alignment or recognition objective is introduced, preserve original complete trajectories and treat inferred assignments as model variables, never accepted source labels. Do not use more variation, lower scalar loss or blind longer training as the acceptance criterion.

Only after that gate passes should complete text, T versus Gamma, learned brackets/math marks, occasional observed joins, personal style and untouched validation sources be reviewed. No final test selection, independent approval, default promotion or release export was performed here.
