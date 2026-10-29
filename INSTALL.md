# Install Notability

## With Nix

From this checkout, on x86_64 Linux:

```sh
nix run .#install -- --trained
```

This installs the one checked skill archive into `~/.agents/skills/notability-form-latex`. It preserves existing installations by refusing an occupied destination. Choose another directory with `--to`:

```sh
nix run .#install -- --trained --to "$HOME/.claude/skills/notability-form-latex"
```

Omit `--trained` for the same skill without its handwriting decoder. Do not keep multiple discovered copies of the same skill name. Start a fresh agent session if the new skill has not appeared, then invoke `$notability-form-latex` in Codex.

The main executable remains `ink`:

```sh
nix run . -- --help
nix run . -- generate 'x^2+1' --out output/first
nix run .#form -- inspect /absolute/input.pdf
```

The default app builds inference and uses the selected decoder in `releases/notability.zip`. It does not build the native training runtime. Run the flake from a checkout for source checks and development commands.

## Without Nix

Use Node.js 18+ to install the skill:

```sh
node scripts/install-skill.mjs --trained
```

The bundled executable targets x86_64 Linux. If it cannot run on your host and Nix is unavailable, build inference with a compatible Rust toolchain. Rust source lives in `src/`; the CLI executable is named `ink`.

```sh
cargo build --release --locked --manifest-path src/Cargo.toml -p ink-cli -p ink-infer
node scripts/install-skill.mjs --trained --to .private/installed-skill
export INK_DEFAULT_MODEL="$PWD/.private/installed-skill/assets/models"
export INK_WORKSPACE="$PWD/.private"
./src/target/release/ink generate 'x^2+1' --out output/first
```

Run ordinary PDF filling directly with Node. For handwriting, point it at the decoder and the inference binary you built:

```sh
export INK_DECODER="$PWD/.private/installed-skill/assets/models/decoder.json"
export INK_BINARY="$PWD/src/target/release/ink-infer"
node skills/notability-form-latex/scripts/form.mjs inspect /absolute/input.pdf
node skills/notability-form-latex/scripts/form.mjs fill /absolute/input.pdf /absolute/answers.json /absolute/completed.pdf
```

`INK_DEFAULT_MODEL` selects the CLI's default decoder directory. `INK_DECODER` and `INK_BINARY` select the PDF helper's runtime. Explicit `--model` arguments override the CLI default. Building these packages does not train a model.

## Check the installed skill

From its installed directory, using fresh test destinations:

```sh
node scripts/verify.mjs
node scripts/smoke.mjs /tmp/notability-pdf-check
node scripts/smoke.mjs /tmp/notability-ink-check --with-ink
```

The first smoke test checks saved field values without a model. The second also checks LaTeX placement with the installed runtime. Render the completed PDF before delivery.

## Use your own decoder

Keep it under `.private/` and pass both files instead of `--trained`:

```sh
nix run .#install -- \
  --to /path/to/your/agent/skills/notability-form-latex \
  --decoder /absolute/path/to/decoder.json \
  --binary /absolute/path/to/ink-infer
```

Without Nix, use the same arguments with `node scripts/install-skill.mjs`. This still installs the same skill. Private research models do not become separate public skills.

## Optional training

The pinned training backend uses native Rust/LibTorch with ROCm. Its verified GPU target is the RX 9070 XT. `--cpu` selects native CPU training but still needs the larger native build.

```sh
nix run .#training -- train --cpu --for 2m --dry-run
```

The dry run requires your own corpus and performs no weight updates. See [use your own data](docs/your-data.md). Without Nix, configure a compatible native LibTorch installation and build with `cargo build --release --locked --manifest-path src/Cargo.toml -p ink-cli --features rocm`. Set `LIBTORCH`, `LIBTORCH_CXX11_ABI` and the platform library path to that installation. The plain inference build does not need those libraries.

## Remove or update

Install an update into a fresh directory and test it before replacing the discovered copy. Keep any private decoder you want to retain. Removing the installed skill directory does not remove this checkout or its training data.
