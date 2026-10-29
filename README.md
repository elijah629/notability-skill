# Notability

One agent skill for PDF forms and handwritten math. Your notes and training runs stay local.

[Download the skill](releases/notability.zip) · [Install](INSTALL.md) · [Train your own style](docs/your-data.md) · [Research](docs/research/README.md)

## Install

With Nix, from a checkout:

```sh
nix run .#install -- --trained
```

This installs one skill, `notability-form-latex`, with the selected trained decoder. It does not install the research models or start training. The installer checks the archive and refuses to overwrite an existing installation.

Without Nix:

```sh
node scripts/install-skill.mjs --trained
```

The PDF tools need Node.js 18+. The download includes an x86_64 Linux executable and a pinned Nix runtime. Other hosts can build inference with Cargo. See [non-flake usage](INSTALL.md#without-nix).

### Let your agent install it

```text
Install the single notability-form-latex skill from this checkout.
Read INSTALL.md and skills/notability-form-latex/SKILL.md first.

Use `nix run .#install -- --trained` when Nix is available. Otherwise follow
INSTALL.md's non-Nix instructions. Preserve an existing installation by choosing
a fresh --to directory. Verify the installed checksums and run the synthetic
PDF test, then the handwriting test if the runtime supports this host.

Use the decoder in releases/notability.zip. Keep other models, source samples,
training runs and generated previews local. Do not train or upload anything.
```

## Use it

In your agent:

```text
Use $notability-form-latex to fill this PDF with my answers.
Place the LaTeX expressions in their answer boxes. Save a new PDF and inspect it.
```

From the flake:

```sh
nix run . -- generate '\frac{x^2+1}{\sqrt{y}}' --out output/formula
nix run .#form -- inspect /path/to/form.pdf
nix run .#form -- fill /path/to/form.pdf /path/to/answers.json /path/to/completed.pdf
```

The executable is still `ink`. `nix run .` and `nix run .#ink` run the same command. The default app uses the selected decoder from the single skill archive. It does not build LibTorch or the training environment.

A fill plan gives ordinary fields their values and handwritten answers their positions:

```json
{
  "values": {"name": "Alex Example", "confirmed": true},
  "ink": [
    {"page": 1, "x": 48, "y": 210, "width": 210, "height": 85,
     "text": "\\frac{x^2+1}{\\sqrt{y}}"}
  ]
}
```

Pages start at 1. Boxes use PDF points measured from the top-left. Set `"ink": []` for ordinary field filling without handwriting. Fields stay editable; handwriting is a vector overlay.

## Flake commands

| Command | Purpose |
| --- | --- |
| `nix run . -- ...` | Run `ink` with the shipped decoder |
| `nix run .#install -- --trained` | Install the one trained skill |
| `nix run .#form -- ...` | Inspect or fill a PDF |
| `nix run .#check -- --history` | Check public files and Git history |
| `nix run .#check-release` | Check the skill archive and model identity |
| `nix run .#test` | Run installer and PDF tool tests |
| `nix run .#training -- ...` | Build and run the optional native training CLI |

`nix develop .` provides Rust and Node tools. `nix develop .#rocm` adds the native training runtime. Training is separate because its LibTorch dependencies are much larger.

## What is shipped

One skill folder and one trained ZIP. The selected decoder composes glyphs and supports fractions, roots, matrices and other bounded LaTeX structures. It does not compile arbitrary TeX documents. Natural connected sentence generation remains unresolved, so inspect each result.

The ZIP includes inference weights, not original handwriting samples, training checkpoints or generated previews. Other research models and historical recipe revisions stay under ignored `.private/`.

## Repository layout

```text
skills/notability-form-latex/  The single installable skill
releases/notability.zip       Its trained download
src/                          Rust engine, layout and training framework
scripts/                      Installation, validation and source export tools
docs/                         Setup, data schema and research methods
flake.nix                     All executable entrypoints
.private/                     Ignored data, models, experiments and history backups
```

There is no root `bin/` directory. Cargo builds still produce the `ink` executable. The source includes only configuration needed by the current engine and compatibility tests, rather than the full experiment inventory.

## Keep local work private

The flake and Cargo CLI use `.private/` when `.private/models` exists. Original notes, unpublished weights, completed forms and training state stay there. `INK_WORKSPACE` can select another local workspace.

Run `nix run .#check -- --history` before creating or pushing a public repository. The tracked pre-push hook runs the same source check when enabled. See [privacy](docs/privacy.md) and [contributing](CONTRIBUTING.md).

No repository has been created or pushed. A project license is still unset; bundled dependencies retain their own notices.
