> Current source uses experimental sequence v8. Latest full-corpus candidate m033 remains **unreadable and unapproved**. The October 18 owner-authorized GPU trial stayed below ten minutes and preserved old models and the corpus. A separate one-source model learned `x`, demonstrating local learnability only. See [the diagnosis, hashes and remaining gates](squiggle-diagnosis-2026-10-18.md). The v2 evidence below is historical.

# Neural handwriting delivery — 2026-10-15

The exemplar application is retired. A real conditional VAE now trains on the RX 9070 XT through native Rust `tch`/LibTorch ROCm. Rust owns tensor preparation, Adam updates, checkpoint recovery, evaluation, decoder export and inference. Python is not required for this workflow.

## Evidence

The first native CPU run (`m002`) performed 2,078 actual handwriting VAE updates in a three-second training budget. Reconstruction loss fell from 0.157912 to 0.001283. Pure Rust inference matched native decoder output within 4.17e-7.

GPU model `m003` performed 20,000 updates in approximately 32 seconds on the physical RX 9070 XT, PCI `0000:03:00.0`, gfx1201/128 SIMD. Loss fell from 0.159170 to 0.000222. A separate two-second `--resume` run recovered the encoder/decoder and Adam first/second moments, continued another 1,168 updates, and reduced loss to 0.000218. A final one-second verification resume against committed source `ec82ffe` brought the total to 21,729 updates, with final loss 0.0002204. The model remains `m003`; these are not new versions. Pure Rust decoder error was below 5e-7.

Both the native test and real GPU training verify identical checkpoint reconstruction, an actual restored update exactly matching uninterrupted Adam, and changed exported decoder weights. Training verifies no `libpython` mapping. Model manifests and separate run logs retain exact counts, timings and hashes; additional verification resumes may increase the local update count.

## Data and quality

The immutable supplied ZIP SHA256 is `55584a3eac1242cd16899af9573c8b47b241c7f6d998ccaae8d427288f38f715`; canonical corpus SHA256 is `5c33fdf811918ea5466c22a5801bb3c8aef7c63a3ff2ce590b9334fbbe20d862`. Rust import preserves 2,503 samples and 25,960 strokes. The source-grouped 2,001/251/251 split has zero source-hash crossover or duplicate source-stroke ownership.

Only 84 training rows are eligible atomic glyphs: 20 labels, `1–9`, `A C F H P V`, `d w x y`, `\pi`. No phrases or formulas were guessed apart. Missing characters fail explicitly. One validation glyph exceeds eight strokes and is preserved outside the model.

Thirty eligible validation glyphs are evaluated on untouched sources. The first GPU decoder had posterior normalized XY RMSE **0.13734** and stroke-mask accuracy **99.58%**. The final committed-model resume measured XY RMSE 0.138764 with the same 99.58% mask accuracy. These measure reconstruction, not recognizer accuracy or perceptual style. Test has zero atomic glyphs, so atomic test performance is unavailable. The evaluator records content/style as independently unscored, rather than substituting loss for quality. The generated alphabet is visually recognizable, with uneven geometry and limited stylistic diversity; this small corpus does not establish robust handwriting quality.

The archive lacks independent second visual review and has unresolved original page transforms and candidate dispositions. No new transcription/authorship certification or full native-page review is claimed.

## Acceptance and artifacts

Nine Rust tests cover source/split identity, cubic preservation, safe tensor/import contracts, parser/layout, GPU selection, learned stroke conversion, and actual VAE update/checkpoint/Adam/export behavior. Real generation verifies twenty exact-input-unseen formulas for complete labels, finite bounds, neural provenance and fixed-seed reproducibility. The four original attachment examples were not supplied.

Local model: `models/m003/`. Generated alphabet/formulas: `output/neural/` and `reports/neural-verification/`. Export: `exports/personal-handwriting.zip`, with a pinned Nix entrypoint and inference-only Rust binary. Weights, data, outputs, reports, logs and exports remain outside Git. There is no public corpus or exemplar fallback.

Remaining limits: sparse coverage, no atomic test set, no independent style/content scoring, fixed pen width and piecewise linear interpolation of learned ordered points. Curve-preserving source data remains available for a future richer neural decoder without changing corpus evidence.

The [sanitized acceptance receipt](neural-acceptance.json) records final costs, hashes and checks without raw vectors or source sample IDs. Both `nix run .` and the independently unpacked skill generated successfully; unpacked SVG/ink JSON were byte-identical to application output. All export checksums passed. The inference binary links only standard system libraries, with no Torch, ORT or Python dependency.
