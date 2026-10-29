# Trained skill download

[Download notability.zip](notability.zip).

This is the owner-approved m027 inference release. It includes the trained decoder, an x86_64 Linux inference binary, the PDF tools, a pinned Nix runtime definition and dependency notices. It contains no original handwriting samples, training checkpoints, generated previews or completed forms.

From a source checkout:

```sh
node scripts/install-skill.mjs --trained
```

Or unpack the ZIP into your agent's skill directory. The ZIP has one top-level folder, `notability-form-latex`. Use a fresh destination. Then run these commands from that folder:

```sh
node scripts/verify.mjs
node scripts/smoke.mjs /tmp/notability-trained-check --with-ink
```

Node.js 18+ is required for PDF filling. The bundled executable targets x86_64 Linux. When its linked runtime is missing, the PDF helper can use the pinned Nix flake on that platform. Other operating systems need a compatible local inference build.

The decoder composes glyphs and supports a bounded LaTeX grammar. Use variation 0 and inspect output. Natural connected sentence generation remains unresolved.

`SHA256SUMS` pins the ZIP. `node scripts/check-release.mjs` from the repository root checks that checksum, every packaged file hash, the m027 decoder identity, archive size and the allowed file list. Other archives remain ignored.
