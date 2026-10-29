# Extreme epoch loss: diagnosis and correction

The reported m029 epoch means near 6,239,500 were real training losses, not a progress formatting error. A frozen-checkpoint GPU replay reproduced **6,239,560.192657**, and teacher-only evaluation also gave **6,239,349.112170**. No optimizer was constructed or weights updated for diagnosis.

## Cause and minimal regression

Sequence normalization formerly used `max(control-bounds extent, 1e-6)`. Two train rows contain a single native point with zero coordinate extent and pen width around 0.79427. Their normalized width became **794,270.8125**. The Gaussian reconstruction loss for their eight-row batch was **1,577,048,960**, dominating the sample-weighted epoch mean. Other top batch losses were at most about 20. This rejected the reporting, KL/duration and scheduled-only explanations.

The two supplied labels are `\text{have}` (`s_41853114b499a216981dd5ab`) and `\text{h}` (`s_d438bd938634642bb4fb4cf8`). A one-point trajectory cannot establish those transcriptions or letter forms. They are retained under supplied-label authority, without relabeling, acceptance certification or invented segmentation. A third nondegenerate row has pen width greater than its control-bounds extent and also receives the corrected scale.

The minimal Rust regression uses an original tap at `[10,20]` with pen width 0.02. Before correction, its normalized width was **20,000**, failing the bounded-width assertion. It now becomes **1** and decodes back to the original coordinate and width. Historical encoding is retained explicitly for legacy evaluation metrics.

## Fix and evidence

Recipe **sequence-cvae-v5** uses `max(control-bounds extent, maximum original pen width, 1e-6)` as its isotropic scale. This bounds normalized widths while preserving reversible original controls, stroke order, pen lifts and widths. The objective, Gaussian variance limits, loss reduction and epoch display are unchanged. No loss is clipped or divided to hide the error. Historical v4 recipe files are unchanged; old checkpoints cannot resume under v5. Explicit transfer maps their learned network weights into a fresh model and optimizer. Approval retains all whole-run quality axes for v5.

A replay using exactly the same m029 checkpoint, seeds, 2,022 train rows and 253 batch boundaries gave:

| Measurement | Before | Corrected inputs |
| --- | ---: | ---: |
| Sample-weighted epoch loss | 6,239,560.192657 | -0.471822 |
| Teacher-only mean | 6,239,349.112170 | -0.878451 |
| Problem batch loss | 1,577,048,960 | 2.073415 |
| Largest normalized width | 794,270.8125 | 1 |

The other 251 batches are unchanged within numerical comparison tolerance; one additional batch changes slightly because of its wider-than-extent source pen. A continuous Gaussian density can exceed one, so its negative log-likelihood and the combined mean may legitimately be negative. These numbers diagnose corrected inputs against frozen old weights; they are not a new trained checkpoint or a quality claim. Long-sequence absolute-path error remains a separate training/quality concern.

Ignored receipts are `reports/loss-diagnosis-20261016/before.json` and `after.json`. The explicitly ignored native diagnostic now asserts a finite, nonexploding epoch mean and unchanged checkpoint hash. The default native-dot test guards zero extent, bounded width and exact source recovery. The real CLI's safe checks passed all 30 default tests and native Clippy without optimizer tests. The native forward-only teacher/recurrent and cache parity check also passed. CLI QA confirmed v5 transfer preflight, v4 resume rejection with a recovery command and the doctor recipe; the complete model inventory, all m027/m028/m029 files and the corpus matched before/after. The comparison/CLI receipt is ignored `reports/loss-diagnosis-20261016/qa.json`.

Use a new version:

```sh
nix run .#training -- train --from m029 --for 30m --dry-run
nix run .#training -- train --from m029 --for 30m
```

The first command performs only preflight. The second starts v5 training with fresh optimizers, up to 64 GPU rows per batch and a new model ID. It does not replace m029. No training was started for this fix.
