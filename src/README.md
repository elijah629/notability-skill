# Rust engine

The workspace builds `ink`, the command-line program, and `ink-infer`, the small inference runtime used by the PDF helper. Source lives here; model weights and local datasets do not.

```sh
nix run . -- generate 'x^2+1'
nix develop . --command cargo test --locked --manifest-path src/Cargo.toml
```

Run these commands from the repository root. The default flake app uses the decoder from the single skill ZIP without building native training libraries. `nix run .#training -- ...` selects the optional native training build. Cargo users can build `ink-cli` directly; see [installation](../INSTALL.md#without-nix).

`ink-corpus` preserves source curves and split identity. `ink-layout` parses the bounded LaTeX grammar and lays out paths. `ink-neural` runs the learned decoder in Rust. `ink-model` owns private training versions, evaluation and export. Those crate boundaries are internal modules of one framework, not separate agent skills.

The retained [recipes](recipes/README.md) support the current graph and compatibility tests. Historical recipes and all research models stay under ignored `.private/`. [Research notes](../docs/research/README.md) record the methods and unresolved sentence-generation failures.
