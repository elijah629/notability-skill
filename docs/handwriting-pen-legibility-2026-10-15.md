# Pen consistency and controlled legibility experiments — 2026-10-15

Latest saved development candidate: **m023**, recipe v15, unapproved. This continuation supersedes the current-state summary in [the earlier m020 handoff](handwriting-handoff-2026-10-15.md); its corpus authority, supplied-label limitations and holdout rules remain unchanged.

## Renderer

Generated glyphs and structural paths now share one **0.025 em output-space pen**, including nested fractions, scripts and stretchy delimiters. Geometry scales; pen thickness does not. Ink JSON records `pen_width`. Original source widths and source previews remain unchanged.

`\cdot` is a single-point pen mark at the local math axis, 0.4 em above baseline. It bypasses the neural decoder and carries `procedural-symbol` provenance. AST request checks still account for every dot. The regression first failed with width `0.0432` instead of `0.025`; it now passes for nested geometry and decoder bypass.

The owner's long expression is regenerated at `output/handwriting.svg` and `.ink.json` using **m020, seed 419, variation 0**. Its ten multiplication dots are single points; all 268 rendered paths have width 0.025. Original output is preserved under `reports/pen-legibility/original-handwriting.*`.

## Experiments and visual selection

All runs retain 2,001 complete train samples in the sequence branch and 1,617 complete flat-layout rows / 137 conditions in the auxiliary head. No supplied rows were segmented, relabeled or moved between splits.

| Candidate | Parent / recipe | Actual seconds | Sequence / head Adam / refinement updates | Outcome |
| --- | --- | --- | --- | --- |
| m021 | m020 / v13 | 300.208 | 983 / 40,809 / 21,068 | `k` clearer; `b/q/9` and minus unresolved |
| m022 | m021 / v14 | 300.205 | 979 / 40,498 / 20,973 | Fuller outlines but visible zigzags; rejected |
| m023 | explicitly selected m021 / v15 | 300.206 | 977 / 40,744 / 21,234 | Smoother than m022; `k/q` clearer than m020, but `b/9`, minus and some crossbars still fail |

v13 balances short complete contexts for conditions lacking practice anchors and replaces sign-only refinement with bounded RMS-normalized gradients. The initial claim that `b` lacked strong anchors was incorrect: it has repeated-practice anchors, but no singleton. v13's fallback-only contexts therefore did not change its anchor pool.

v14/v15 mix short contexts with practice evidence for conditions without singletons. v14's stronger extremal shape loss introduced zigzags, including in zero-variation decoder means. v15 restores conservative shape weight 0.2 with a small 0.05 extremal term and sharpens prefix occupancy slope from 6 to 20. Continuous raw-count loss remains active. At integer count two, the third slot now receives less than 0.001 training credit instead of approximately 0.047. This reduces mismatch with rounded-count inference.

`train --model m021 --seconds 300` explicitly warm-starts a **new version** with fresh optimizers; `--resume` retains the selected version and restores its optimizer with exact recipe matching. This prevents inheriting a newer failed experiment merely by directory order. Parent compatibility and lineage are checked/recorded. Manifests now record the actual head recipe instead of stale v10 metadata.

Final decoder SHA256:

- m021: `5d22928e0f107c87b96d980c4be20775c52f1ccf2bd654ad9e7cb1c932ad4632`
- m022: `df0b8e37389851ef15bae0f3605b3e231b075a08d61b773a28384a5c0dec8353`
- m023: `77cae4c482e566ed88240e47f20ba10c4d59b9646d4f918c634c738efdc4d8a9`

## Controlled verification

At the owner's subsequent request, all new visual comparisons and `model verify` diagnostics fix **seed 419, variation 0**. Earlier variation-1 artifacts remain historical evidence. Matched m020/m021/m022/m023 alphabet sheets use identical input, renderer and image dimensions. Nonzero variation and cross-seed diversity are deliberately not measured in the new diagnostic receipts. General `generate` defaults remain unchanged; pass `--seed 419 --variation 0` for comparisons.

21 workspace tests, formatting and Clippy with warnings denied pass. Tests cover constant pen width, dot geometry/provenance, complete-row reuse, missing short features, negligible inactive-prefix credit, original cubic preservation, actual native updates, restored Adam equality and Rust/native parity. m023 head Rust/native maximum error is `5.960464477539062e-7`.

m020/m021/m022 passed 2,511 request contracts under the updated renderer. m021 and m023 each passed 2,511/2,511 with zero variation, identical same-seed SVGs and repeated-letter raw-coordinate RMSE zero. m023's final receipt is under `reports/pen-legibility/m023-verification-zero/`; its failed visual review is `m023-quality-review-zero.json`. These checks establish request coverage, finite geometry and determinism, not recognition or visual quality.

Validation evaluated 251/251 whole samples for m021, m022 and m023, with zero failures. Teacher-forced posterior/prior delta RMSE: m021 0.0238855/0.0239711; m022 0.0226807/0.0226390; m023 0.0222849/0.0222465. These sequence measurements do not validate compositional legibility. Test remains untouched because visual selection has not passed.

Evidence: `reports/pen-legibility/` contains matched alphabet sheets, boxed Hello, long formulas, immutable intermediate decoder snapshots, training logs and verification/validation receipts. The final m023 alphabet is `alphabet-m023-zero.svg`; readable text alone does not qualify the full model. Model/data/output artifacts remain gitignored.

## Budget and continuation

Three completed experiments used **900.619 seconds**, approximately 15 minutes. Add this to the earlier conservative 2.5-hour reserve: approximately **2.75 hours spent**, at most **5.25 hours remaining** of the owner's eight-hour authorization. Validation here executes on CPU. No background trainer remains.

Continue using fixed seed/zero variation. Resolve missing/ambiguous features and residual wobble first; raw whole-sequence prior synthesis remains a separate unresolved limitation. No accepted visual review, final independent test evaluation, standalone export or default promotion is claimed.
