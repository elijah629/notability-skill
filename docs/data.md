# Supplied vector data contract

`dataset.zip` is immutable source evidence. Supplied native-vector additions merge into the same corpus and training split, with original source identity retained. Rust import verifies every SHA256 manifest entry, packed stroke ownership, sample offsets, labels, source hashes, original cubic controls and pen metadata. The canonical corpus preserves all 2,524 samples and 26,618 strokes. Coordinates retain float32 round trips; strokes are dots or `3*n+1` cubic controls.

The supplied split is retained: 2,022 train, 251 validation, 251 test. Source groups and shared ink never cross splits. Rejected/unresolved candidates are not supervision. Supplied authorship/transcription annotations retain their review caveats; no independent second pass or page-transform resolution is claimed.

Sequence training uses every whole train sample, including words, formulas and oversized ink. Its relative cubic event representation is reversible with the recorded sample origin and isotropic scale. No sample is silently truncated or segmented. Semantic tokens describe the complete label, not per-stroke annotations. Sequence vocabulary is derived only from train; held-out unknown conditions are reported explicitly.

An auxiliary compositional objective uses complete flat-layout rows and balances whole singleton/repeated-run anchors from train. Its soft vector assignments are internal model variables, not accepted glyph samples. The original row and stroke lineage remain intact. Bounded auxiliary slots/point sampling are disclosed separately; every row still enters the full sequence objective.

Historical v2 training intentionally selects only supported atomic labels of at most eight strokes; its 84 examples/20 labels describe that model, not the dataset's total usable supervision. `ink data coverage --json` distinguishes atomic history, complete-sample use, syntax support and sequence length. `model verify` separately measures generation feasibility; neither grammar coverage nor reconstruction loss establishes content/style quality.

A visually inspected supplied singleton labeled `0` is an open V/U-shaped path. Its original label and vectors remain unchanged and it still enters whole-row supervision; it is explicitly marked in auxiliary lineage and excluded from strong ordered anchors. This is a confidence warning, not a newly certified corrected transcription.
