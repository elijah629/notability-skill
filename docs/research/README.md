# Research and engine notes

The research records what was tested, what failed and which measurements remain unscored. It includes methods and aggregate results. Personal trajectories, training checkpoints and generated handwriting stay local. The owner-approved m027 inference ZIP is available under `releases/`.

## Start here

- [Vector architecture](../vector-architecture.md) describes runtime boundaries, model versions and provenance.
- [Data contract](../data.md) records the original experiment's corpus constraints. [Use your own data](../your-data.md) is the guide for a fresh checkout.
- [Current repair work](../model-repair-2026-10-18.md) covers learned alignment, em units, sparse pen-lift geometry and unresolved sentence output.
- [Development commands](../development.md) preserves the detailed local workflow that used to be the main README. Model paths there refer to private research artifacts, not files distributed with the source.

## Glyph composition

[m027 training and review](../handwriting-m027-2026-10-15.md) records the controlled glyph baseline. [Export and variation findings](../handwriting-export-2026-10-15.md) explains lossless decoder packaging, the LaTeX PDF experiment and failed natural variation. Those older exports remain local. The current public inference release is documented in [releases](../../releases/README.md).

Earlier work includes [pen legibility](../handwriting-pen-legibility-2026-10-15.md), [handwriting revision](../handwriting-revision-2026-10-15.md) and [quality criteria](../handwriting-quality.md).

## Sequence generation

[Sequence status](../sequence-status.md), [squiggle diagnosis](../squiggle-diagnosis-2026-10-18.md), [loss normalization](../loss-normalization-2026-10-16.md) and [training throughput](../training-throughput-2026-10-16.md) separate execution evidence from actual handwriting quality. The [local handoff](../handwriting-handoff-2026-10-15.md) retains chronology and past compute limits. Historical authorizations do not authorize training on a new checkout.

## Reproduce the framework

Use [the personal-data guide](../your-data.md) with your own source material. [ROCm setup](../rocm-training.md) explains the tested native backend. [ORT notes](../ort-training.md) preserve the earlier backend investigation. [CLI checks](../cli-2026-10-16.md) document command behavior.

Recipes are versioned under [src/recipes](../../src/recipes/). The recurrent text-window design draws on [Graves, Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850). Vector generation also draws on [Ha and Eck, A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477). This implementation does not claim to reproduce either paper's handwriting quality.

## Reading the results

A backend update, lower loss, finite coordinates and matching Rust/native output are different claims from legibility. Teacher-guided reconstruction is different from autonomous text generation. The current whole-sequence work remains experimental. Public-source users must assess their own dataset and decoder.
