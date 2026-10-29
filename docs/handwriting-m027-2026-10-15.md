# m027: trained modes and fixed-condition legibility review

Latest delivery: [under-25-MB skill export, PDF demo and natural-variation findings](handwriting-export-2026-10-15.md). Nonzero variation remains jagged/jittery and has not passed the owner's naturalness review. Use variation 0 for controlled output.

The new inference candidate is **m027**, recipe `compositional-vector-v19`, with decoder SHA256 `cac8182d095b4e4bd76d57b8c32f59e5700c64f484f19f6345d5846c283701cc`. The owner's expression at `output/handwriting.svg` now uses this decoder, seed **419**, variation **0**. The previous m020 output is preserved at `reports/legibility-v19/m020-before-final.*`. No default promotion or universal recognition/perfection claim is made.

## Actual training and diagnosis

| Candidate | Recipe | Training seconds | Result |
| --- | --- | ---: | --- |
| m024 | v16 | 600.360 | Eight topology modes; global stroke-order alignment failed on real cross-cell pen movement. |
| m025 | v17 | 600.352 | Spatial whole-row assignment accommodates delayed strokes; unrestricted context mixing damaged trusted anchors. |
| m026 | v18 | 900.357 | Protect existing practice anchors, reuse short contexts for missing conditions, supervise near-closed source paths. |
| m027 | v19 | 600.358 | Match source curvature rather than flattening corners; visually select learned modes and preserve seven useful frozen reference conditions. |

m027 warm-started m026 and performed **1,986 sequence updates**, **123,192 head Adam updates**, and **75,067 token refinement updates**. Its uncalibrated trained decoder hash was `b559edece182212e043787854bc0db10f596ae1421a3db8cb182e36b444d5623`. Native/Rust all-mode head maximum difference is `4.172325134277344e-7`; sequence maximum difference is `0.0000133514404296875`. Restored eight-mode optimizer next-update difference was zero. These are real native Rust/tch weight updates, not replacement font/exemplar paths.

This continuation spent 2,701.427 seconds of training, about 45 minutes. Adding the prior conservative 2.75-hour accounting gives about **3.50 hours spent / 4.50 hours remaining** from the owner's eight-hour ceiling. No training worker remains running.

## Reuse and selection

All 2,001 complete train rows remain in sequence training; 1,617 complete flat rows support 137 auxiliary conditions. Corpus SHA256 remains `5c33fdf811918ea5466c22a5801bb3c8aef7c63a3ff2ce590b9334fbbe20d862`. Validation and test retain 251 complete rows each. No original trajectory, supplied label, split, or accepted character annotation was changed. Transient spatial correspondences and source-support evidence remain internal training state.

All 23 pages of eight-mode/reference grids were visually inspected. The hash-pinned selection is tracked in `src/recipes/reviews/m027.json`; its SHA256 is `d288f87ca4e1bbda555cd6ca833a1dd66414aa44dd1f982e82e88d85aeccdd88`. Calibration validates candidate/reference identities before mutation and backs up the original inference files. Stale selections were rejected without modifying the decoder in an isolated contract check.

The final decoder embeds frozen **trained m020 neural weights** for `&`, `/`, `\Phi`, `\Pi`, `\infty`, `\pm`, and `v`. The reference decoder hash is `c53059f3dca9927b7178d1b762f35be6ab99f30a43ec6c5e051b360a3605b93d`. This preserves sharper `v` and clear sparse symbols without loading another model or looking up source ink at runtime. The selected `R` mode has weak whole-row train evidence rather than a canonical hard-cell assignment; calibration records this explicitly. Selection is visual inference calibration, not new accepted source annotation. Resuming an optimizer checkpoint does not preserve these inference selections automatically.

Structural composition now reuses learned comma for semicolon; learned `a/e/o/u/E/n` for `á/é/ó/ú/É/ñ`; learned `?` rotated for `¿`; and learned right arrow rotated and resized for down arrow. Acute/tilde marks and the semicolon dot are explicit procedural cubic pen marks. Membership reuses learned epsilon, and proportionality reuses learned alpha. `condition_label` identifies each donor; `neural-composed-symbol` distinguishes mixed generation from direct neural decoding. These are layout operations, not claims that sparse whole-row targets supplied independently verified isolated glyphs.

The output-space pen is always **0.025 em**, including composed marks, scripts, nested fractions and stretchy delimiters. The original long expression has **163 requested glyphs**, ten one-point `\cdot` marks, and **282 paths all with width 0.025**. Dot position follows the local math axis while thickness stays fixed. Source preview widths remain authentic.

## Final checks and artifacts

- Workspace formatting, 25 Rust tests, all-target Clippy with warnings denied, release build, and `git diff --check` passed.
- Final decoder generation checks: **2,511/2,511 passed**, zero failures, fixed seed 419/variation 0. Repeated `l` geometry RMS is zero. These are generation/AST/provenance contracts, not an OCR score.
- The owner's long formula reproduced byte-identical SVG **and ink JSON** on repeated generation.
- Minus relative chord RMSE improved from m020 `0.0979499` to **0.00118298** (about 83 times smaller); chord/path ratio is **0.999677**. `0` and `O` pass the endpoint-closure diagnostic. Small `o` retains a relative endpoint gap of **0.0651**, so universal perfect closure is not claimed.
- Validation: 251/251 complete rows, zero failures; teacher-forced posterior/prior delta RMSE **0.047029 / 0.048450**, event accuracy **0.927241**. Test was evaluated once after the final mode selection: 251/251, zero failures; posterior/prior RMSE **0.044006 / 0.044901**, event accuracy **0.934455**. No subsequent model selection or tuning used test results. Sequence teacher-forced geometry did not improve over every earlier candidate; it must not be confused with the visibly improved composed glyph output.
- Diagnostic export `exports/m027-candidate.zip` SHA256 **`af5317845099ab73b06e9ed5ceb116fa583935c13c6a1dd3f68ac6f822e6c028`** was unpacked outside the repository. Every bundled checksum passed. Its standalone Rust binary produced byte-identical SVG and ink JSON for the owner's formula without source data, parent-model files, Python, or training libraries.

Review artifacts: `reports/legibility-v19/m027-final/alphabet.svg`, `symbols.svg`, `geometry.json`, `m027-verification/verification.json`, `m027-validation.json`, and `m027-test.json`. The alphabet shows materially improved `b/j/k/q/R/T/5/9`; a human reading review is still stronger evidence for legibility than the numeric geometry contracts. Some Greek forms retain personal retracing/serifs and occasional small openings. No independent second-review labels or broad recognition score exist.

Use the original generation command with **`--model m027 --seed 419 --variation 0`**. Explicit selection is required while this candidate remains unapproved as a default. Further training should branch into a new ID and retain this reviewed decoder/export for comparison.
