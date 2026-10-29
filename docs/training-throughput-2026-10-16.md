# Training throughput — 2026-10-16

The owner requested better use of available GPU compute. The verified RX 9070 XT has 17,095,983,104 bytes of physical VRAM reported by the kernel; approximately 1.2 GiB was in use at inspection, and no `ink` training process was active. No new training run or optimizer update was started for this work.

## Measured bottlenecks and changes

The old sequence trainer fixed batches at eight, rebuilt/padded/transferred inputs for every update, and ran the text encoder twice for each loss. The public wrapper also used an unoptimized development build. Scheduled sampling retains its genuinely recurrent dependence; it is not replaced by independent predictions to improve a benchmark.

The wrapper now uses `cargo run --release`. New GPU models use deterministic length buckets of at most 64 rows and 16,384 padded events. The event budget shrinks batches for long rows; an oversized individual row remains whole. Every train row appears exactly once per epoch. CPU retains a maximum of eight. The current 2,022-row train corpus yields 35 GPU batches, compared with 253 historical batches.

An `execution_plan` in every published manifest overrides the architecture recipe's reference batch size, preserving v4 network/recipe compatibility while recording the actual training configuration. Resume retains that plan and checks the recorded epoch batch count. Historical v4 checkpoints without an execution plan use their original eight-row boundaries, preserving the saved Adam cursor and deterministic order. To adopt larger batches from an existing model, transfer with `train --from latest`, allocating a fresh model and optimizer.

Prepared stroke weights are calculated once. Padded immutable inputs are retained on the GPU when their complete cache is at most 256 MiB; larger datasets use on-demand transfers. Neither activations nor detached encoder results are cached between updates. Text conditioning is shared within a loss so both reconstruction and duration still backpropagate into the encoder. FP32, original controls, event weights, pen lifts, the GRU architecture and inference representation are unchanged.

## Evidence and limits

Release-mode GPU diagnostics computed forward/backward passes on the actual train rows using a disposable random network. No optimizer was constructed. All network tensors were compared before and after and remained identical; no model directory, checkpoint or decoder was written. Only ignored timing reports were saved.

For roughly median-length rows (109–115 events), the baseline eight-row teacher/scheduled paths took 33.42/94.09 ms per batch. At 64 rows they took 27.80/99.91 ms: the GPU handles eight times the examples with similar batch latency. After shared-encoder work, the respective 64-row timings were 26.09/97.51 ms.

The full-corpus comparison used the optimized loss in both arms, deterministic shuffled length buckets and every fourth batch scheduled at warmed-up probabilities. The eight-row arm rebuilt/transferred inputs; the throughput arm cached them. One measured pass over all **2,022 train rows** took:

| Execution | Batches | Seconds | Samples/s |
| --- | ---: | ---: | ---: |
| Historical eight-row profile | 253 | 13.962 | 144.8 |
| GPU throughput profile | 35 | 2.848 | 710.0 |

This is **4.90× forward/backward throughput**, not a measured end-to-end training speedup. It excludes Adam, checkpoint/export validation and the auxiliary phase. The two profiles perform different numbers of updates per epoch; larger batches change stochastic gradients and the examples covered by update-based KL/scheduled-sampling warm-up. Learning rate remains 0.0003. Faster sample throughput does not establish equal convergence or handwriting quality. No GPU utilization percentage or peak-memory claim is inferred from these timings.

Receipts: ignored `reports/training-throughput-20261016/before.json` and `after.json`. The native diagnostic is explicitly ignored by default and requires `INK_PERF_CORPUS`/`INK_PERF_OUT`; `INK_PERF_EPOCH=1` includes the full-corpus comparison. It never constructs Adam. Default checks do not execute it.

The default workspace suite passed 29 tests, including full-coverage bounded batching and legacy/new resume boundaries. Native all-target Clippy passed with warnings denied; the actual release wrapper's safe `check` command passed. The native forward-only test retained teacher/recurrent parity, finite scheduled output and exact cached/rebuilt input/loss equality. Actual CLI transfer and resume dry-runs reported 35 versus 253 batches respectively and `training_started: false`; CPU reported eight rows and 253 batches. Human plan output, doctor and clean JSON passed. All files in m027/m028/m029, the corpus hash and the complete model-directory inventory matched before/after CLI QA, recorded in ignored `reports/training-throughput-20261016/cli.json`. Source changes do not modify or approve existing models.
