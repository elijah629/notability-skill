# Model repair — October 18, 2026

The owner requested a working model and explicitly allowed further training. They subsequently clarified that complete connected sentence generation is required; per-glyph composition is not an acceptable completion. This is new authorization; the earlier ten-minute trial remains closed. Existing models and original data are preserved.

## Preserved glyph baseline — not the requested completion

`nix run . -- generate '\boxed{Hello, World!}' --model m027 --seed 419 --variation 0` produces readable, composed glyph handwriting. This is the preserved m027 expert, not a newly successful whole-text network. Nonzero variation and natural joining remain unapproved.

The fresh standalone package is `exports/m027-working-20261018.zip` (17,783,660 uncompressed bytes). It was unpacked to `/tmp/ink-working-20261018`; all 106 checksums passed, and its Rust binary reproduced the application Hello SVG byte for byte without the corpus or training runtime. The source decoder remains `cac8182d095b4e4bd76d57b8c32f59e5700c64f484f19f6345d5846c283701cc`; compact delivery decoder is `6bccb239ebefcd7202a041101b3255bc1fdbdd556e134574078b98db7634cb5e`. This remains a diagnostic, explicitly selected package.

## Reproductions and fixes

Frozen m033 produces one stroke for `Hello, World!`. An autonomous, train-only five-label gate failed 0/5 before training. Source-free generation is compared against unchanged complete source rows using stroke counts and a normalized sampled-path Chamfer threshold of 0.06. This is a geometric curriculum gate, not independent recognition or generalization evidence.

- A fixed attention window in normalized sentence progress grows wider in characters as the sentence grows. A deterministic test demonstrated that a 20-character prompt blurred the current character away. v9 measures window distances in token units, identically in native teacher forcing, scheduled recurrence and exported inference. Historical exports retain their original units.
- v8/v9 introduced fully self-fed geometry after only 1,500 updates, together with prior/posterior style perturbations. v10 first trains canonical zero-style source feedback for 10,000 updates, then ramps alternate scheduled batches over 20,000 further updates. Learned-duration feedback starts after the full ramp; KL starts after canonical warmup. A loss-path regression test catches premature feedback/style perturbation.
- Local text alone cannot specify complete-run geometry at the first event. v11 concatenates local context, complete-text mean, style, progress and explicit log token count into the GRU. The first-event regression test failed before this change. Old input columns are copied to their exact new positions; new context/count columns start at zero. An actual native transfer test proves unchanged predictions before training the new columns.
- `model snapshot` assumed every sequence model had a glyph-training checkpoint. Whole-sequence-only models now freeze their hash-verified published decoder and vocabulary directly, with approval claims cleared. The contract rejects corrupt publications and existing destinations.
- Full-data v11 exposed a remaining loss-unit problem: complete-path SSE used whole-sample width units, so a tenfold wider sentence weakened identical physical path error by 100 times. The regression measured 0.025 versus 0.00025. v12 uses source-height-scaled smooth L1 (gain 1–32, beta 0.1), independently of the native event-density gain used by the separate local-control loss. The physical-unit invariance test now passes. Original coordinates/events are unchanged.

All original cubic controls, labels, event order, pen lifts, widths and source-grouped splits are unchanged. No new character segmentation, accepted annotation, public data or exemplar lookup was added.

## Actual curriculum training

Separate ignored diagnostic corpora retain unchanged complete train rows and one unchanged validation/test row for split-contract checks; only train rows are optimized. These models live outside the primary model store and must never be selected as general handwriting models.

| Experiment | Train rows | Saved updates | Budget | Final autonomous gate |
| --- | ---: | ---: | ---: | --- |
| v8 singletons | 5 | 6,232 | 120 s | 5/5 |
| v8 mixed singletons/strings | 9 | 5,543 | 180 s | 4/9 |
| v9 mixed strings | 9 | 5,475 | 180 s | 6/9 |
| v10 canonical warmup | 9 | 9,944 | 120 s | 8/9 at the inspected checkpoint; A failed |
| v11 complete-prompt context | 9 | 11,793 | 180 s | **9/9**, exact stroke and event counts |
| v12 source-height path loss | 9 | 4,962 | 60 s | **9/9**, exact stroke and event counts |

The final v11 gate contains `H`, `x`, `y`, `2`, `A`, `xy`, `2x`, `AA`, `123`. Autonomous normalized path errors are 0.00381–0.01538. Final decoder SHA256 is `ecfb703d818fa88051f050643c03d6e5a60505ddc63a4566353191fc2e26a501`; its immutable inference snapshot is `reports/model-repair-20261018/gate-v11-passed/`. Source-guided teacher reconstructions are separately marked and never counted as autonomous output.

## Full-data continuation

The passing small curriculum permits a full-data trial, not a claim of general completion. m034 transferred m033 into v11 with fresh Adam state and the same 2,022 train / 251 validation / 251 test rows. It was stopped after the physical loss-unit failure was demonstrated. Its last saved state contains 12,269 updates / 350 epochs / 783.872684359 training seconds; unsaved in-flight work is additional. Frozen galleries and all nine free short-label probes still failed. This is not a successful general model.

m035 transferred m034 into v12 with fresh Adam and a 30-minute requested budget. Repeated frozen whole-sentence galleries remained unreadable. The worker was stopped after its saved 14,801 updates / 422 epochs / 1,144.958897366 training seconds; additional unsaved work is bounded by the checkpoint interval. Its frozen 13,499-update gallery still shows curves rather than requested letters. No new default is approved, and no test-set selection has occurred. The v12 small-gate decoder is `94fc8bc3a54f73abe7378b6829d7340e0f91f7e6c2b2f3b9e3d92892062f0adc`, with autonomous errors 0.00169–0.01050. These small-model results do not certify m035.

The active corpus SHA256 remains `fcb45645b2e2b89d7041c16afc6d4cf6fc49c39737bb9a0e9e5a247157521557`. A receipt verifies all 344 pre-existing model files unchanged. Hardware is the RX 9070 XT, PCI `0000:03:00.0`, gfx1201, 128 SIMD. Saved publications verify native/export parity and an identical restored next Adam update.

The full native `check --training-tests` suite exposed a stale pre-v20 auxiliary test that expected ordinary words to supply isolated glyph supervision. The test now verifies the existing separated-practice-only policy and preserves ordinary words for whole-sequence training. No production auxiliary behavior changed.

Default and native workspace tests, formatting, Clippy with warnings denied, actual sequence/auxiliary optimizer updates, export parity, exact restored Adam updates and the new loss-unit regression all pass under `nix run . -- --json check --training-tests`.

Local receipts and galleries are under `reports/model-repair-20261018/`. The Rust examples `curriculum`, `content_gate` and `gallery` make preparation, autonomous checks and visual requests repeatable. Preserve both successful small-model evidence and failed full-data galleries; lower scalar loss is not acceptance.

## v13: learned alignment and em units

The fixed linear clock could memorize nine seen labels but failed unseen `H2Axy`. This invalidates that small gate as generalization evidence. v13 adds a separate recurrent monotonic Gaussian window, conditioned on pen history and complete text. It can hold or advance a letter independently of a guessed total duration. Full-source end coverage regularizes the window without labeling character boundaries. This adapts the monotonic-window idea from [Graves (2013/2014)](https://arxiv.org/abs/1308.0850); it is not a reproduction or a claim of that paper's quality.

Whole-width coordinates also made the same letter's motions shrink in longer text. v13 uses reversible em units based on source height, width/token count and pen diameter. The regression explicitly reproduces the legacy shrinkage, then verifies identical motions and inverse original controls/widths under em units. No source geometry is resampled or changed. Density amplification is removed for the local em loss; whole-path source-height loss remains.

Long native/export testing uncovered naive float32 window accumulation drift missed by short probes. Fused scans now accumulate in double precision; pure Rust retains compensated sums. The expanded 64/400-cubic, unequal-padding teacher/scheduled/export test passes. Historical fixed-clock loading is unchanged. These tests establish mathematical transport, not readable free generation.

A separate 39-row source-preserving curriculum contains 30 complete words and the nine earlier labels. The minimum-control `\text{have}` row turned out to be a single tap, not a connected word. Its zero Chamfer and correct stroke count exposed a false-positive curriculum gate. The gate now rejects multi-glyph labels with no original curve, and diagnostic selection records/excludes such rows while leaving the active corpus unchanged. The other existing `have` row has three strokes / 240 controls and remains eligible. All rows are unchanged train sources; retained validation/test rows are audit-only. The run transfers the prior small model with fresh Adam and newly initialized alignment weights. It does not replace the full corpus or count as a general trained delivery. Frozen probes include connected training words and unseen word combinations. The final quality result remains pending at this checkpoint.

The first 39-row v13 trial completed 5,805 actual updates in 300 seconds; its initial/final batch losses were 174.4543 / -2.6543, not comparable acceptance metrics. At 4,644 updates it passed 12/39 old geometric checks, including the invalid tap, so only 11 were nondegenerate. The rendered `feel` begins to resemble its source, but free word content and unseen sentences still fail. This is progress evidence, not a working delivery. The first unseen spaced probes correctly failed for an untrained space token; the next diagnostic set includes original multiword rows.

## v14: sparse pen-lift geometry

The v13 full trial m036 and 53-row diagnostic m001 were stopped when a reproducible geometry objective failure was demonstrated. Their last saved states are 1,079 updates / 496.568705398 seconds and 1,023 updates / 485.99475311 seconds respectively; unsaved work is additional, bounded by each checkpoint interval. m036 used all 2,022 unchanged train rows and copied all 175 parent token embeddings. Its earlier frozen free galleries still fail. Both workers are stopped; published weights remain preserved.

A frozen, zero-style, source-guided text ablation on 30 nondegenerate words establishes that v13 does use the requested text: correct versus reversed text event CE is 0.005576 / 3.783877, and source-height coordinate MSE is 0.001774 / 0.019457. However, MOVE location RMSE is 0.24490 source heights, versus curve-control RMSE 0.01760. Strong teacher classification can therefore coexist with badly misplaced autonomous pen strokes. This is not independent recognition evidence. The diagnostic performs zero optimizer updates, verifies its checkpoint hash and never changes a supplied training label.

The actual v13 local geometry objective weakened identical pen-lift error from 0.000574713 to 0.000059312 as following native curves increased tenfold. The regression fails before the fix. v14 assigns equal objective mass to present MOVE and CURVE classes. MOVE locations/width use source-height units, smooth L1 beta 0.02 and equal weight for each pen lift; they do not inherit following-stroke density weights. Curves and source-relative curvature retain density-scaled precision demands without resampling or modifying original controls. The regression also covers a short second native stroke; inverse physical-unit conversion keeps the actual pen-lift error fixed.

This is a training-objective change with the same exported v13 recurrent graph. New training must transfer into fresh optimizers; old recipes cannot resume under v14. A passing mathematical test is not a successful handwriting result. Full connected sentence generation remains the completion gate.

The v14 53-row trial completed 4,165 updates / 300.156569373 seconds, native/export max error 5.6266785e-5 and exact restored next Adam update difference 0. Its decoder is `31bbe843debd363a3bebc105aefaca194678ceea03ea33828e778f24550b2742`. Free source-relative geometry/count gates rose from 21/53 to 34/53. On the same 45 eligible source-guided words/phrases, median MOVE RMSE fell from 0.25534 to 0.09060 source heights; p90 fell from 0.53052 to 0.28429. Curve-control RMSE fell from 0.02185 to 0.01167. These are teacher-guided diagnostics; free unseen sentences still fail visually. The correct original `have` word is visibly recognizable, while novel combinations remain malformed. No delivery/default is approved.

m037 now transfers that actual complete-phrase curriculum checkpoint into all 2,022 unchanged personal train rows / 175 tokens, with fresh Adam and a one-hour budget. The current small curriculum supplies 35 token embeddings; unseen symbols initialize separately. This is still a candidate under training, not a working model claim. No test-set selection has occurred. All m000–m036 publications and the active corpus remain preserved.

The diagnostic gate was tightened after visual failures showed that extent-normalized Chamfer can hide letter displacement in wide text. A regression reproduces a 0.3-letter-height middle-stroke error that passes the old 0.06 whole-extent threshold. Gate v2 preserves aspect ratio and additionally requires Chamfer < 0.06 in height/pen-width units. On the exact same immutable v14 decoder, this passes only **13/53**, versus 34/53 under the old extent metric. Both measurements remain recorded; the stronger result supersedes the weaker gate for future acceptance. Neither metric is independent recognition, and unseen sentence output still fails. The dedicated example regression and complete native checks pass.
