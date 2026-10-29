# Repository guide

- The installable public skill is `skills/notability-form-latex/`. Read `INSTALL.md` for installation and `docs/your-data.md` for a fresh user's training setup. Keep skill instructions usable outside this checkout.
- `src/` contains the Rust framework and recipes. `docs/research/README.md` indexes the preserved experiments. The public source includes no personal dataset. The owner explicitly permits the trained m027 inference ZIP in `releases/`; its source samples, checkpoints and generated previews remain private.
- Keep captured writing, unpublished decoders, generated handwriting and completed forms under ignored `.private/`. Public tests use synthetic fixtures. Run `node scripts/check-public.mjs --history` and `node scripts/check-release.mjs` before sharing changes.
- The root repository has a clean public history. Original development history is preserved only under ignored `.private/`. Do not import or push its old refs. Keep that backup and all private files intact.
- `ink` CLI automatically runs in `.private/` when `.private/models` exists. `INK_REPO` identifies the source root for builds and exports; `INK_WORKSPACE` can select another local workspace.
- Installing or testing the PDF skill does not authorize training. Run its model-free smoke test in a fresh temporary directory. Handwriting tests need a decoder authorized for that use.

## Core rules

- Build the personal vector corpus before new model training. Original source notes are read-only. On the owner's existing local corpus, this includes `~/Downloads/Notability`. Account for every note, page and extracted stroke; unresolved sources keep the dataset incomplete.
- Use the Rust CLI for extraction, preparation, review validation, training, evaluation, generation, export and development checks. Python is restricted to offline ONNX graph creation through a separate Nix environment; Rust owns all data and model updates.
- Preserve authentic cubic controls, stroke order, pen lifts and supplied metadata. Active datasets and training use labels plus vector strokes; raster data stays in historical source archives only.
- Each accepted personal sample needs two independent visual reviews agreeing on its label, exact ordered strokes and personal authorship. OCR and historical assistant labels are clues, not ground truth. Quarantine uncertain writing, inferred segmentation and generated ink.
- Keep public pretraining and personal fine-tuning corpora separate. Preserve provenance and licensing; identify public-style output explicitly.
- Group related source versions and shared strokes into one split. Evaluate on untouched held-out sources. Report content, geometry and style separately.
- Store each model under its own `m000`, `m001`, ... directory. Resume keeps its version; new training allocates a new version. Models, data, downloaded resources, run products and exports are gitignored. Git tracks source, recipes and the explicitly approved m027 inference release only.
- Check and commit each coherent completed change. Preserve user work and the source baseline. Prefer tests of meaningful contracts and end-to-end behavior over implementation-mirroring tests.
- Retire replaced implementation into `legacy`; archive before removal. Keep the public CLI small and model recipes versioned instead of exposing experiment switches.
- Claim neural completion only after Rust executes actual weight updates and exports an updated decoder. A successful process or falling loss alone does not establish usable handwriting.

## Context

- Continuing the owner's existing local models, generation quality, CLI/export verification or model promotion: first read [current handoff](docs/handwriting-handoff-2026-10-15.md). It records the owner's ZIP-only authorization, m020/v12 saved state, failed visual cases, GPU budget and exact continuation gates. Keep the current unapproved status until the recorded quality failures are resolved. These historical model paths and authorizations do not apply to a fresh user's dataset or authorize new training.

- Extraction, annotation or dataset acceptance: read [dataset review](docs/vector-review.md).
- Changing runtime boundaries, model lifecycle or CLI behavior: read [target architecture](docs/vector-architecture.md).
- Downloading or exporting resources: read [excluded resources](docs/excluded-resources.md).
