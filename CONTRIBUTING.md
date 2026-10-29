# Contributing

This project distributes one skill, `notability-form-latex`, and one trained archive, `releases/notability.zip`. Other models, historical recipes, source samples and generated handwriting stay local under `.private/`.

## Skill and PDF tools

Edit `skills/notability-form-latex/` and keep its instructions usable outside this repository. Use synthetic fixtures in tests.

```sh
nix run .#test
nix run .#check-release
nix run .#check -- --history
```

Without Nix, run `node scripts/test-tools.mjs`, `node scripts/check-release.mjs` and `node scripts/check-public.mjs --history`.

## Rust engine

Edit `src/`. The executable remains `ink`; there is no root `bin/` wrapper.

```sh
nix develop . --command cargo test --locked --manifest-path src/Cargo.toml
```

Use a compatible local Rust toolchain for the same Cargo command without Nix. Native optimizer changes need the separate `.#rocm` shell and relevant fixture tests. Building or installing the ordinary skill must not start training or fetch LibTorch.

Preserve original curves, source groups and split boundaries. Check actual generated output before making readability claims. A lower training loss is not a content or style score.

## Public files

The source allowlist and release checks reject raw notes, arbitrary archives, model checkpoints, symlinks and unexpected large files. Keep the approved decoder hash pinned unless the owner explicitly selects a new release. Retain dependency notices when changing bundled code.

A new clone can enable the tracked pre-push guard with `git config core.hooksPath .githooks`. Review source prose for personal details too. The owner must choose a project license before offering general redistribution.
