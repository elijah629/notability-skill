# Public repository and private handwriting

The repository root is prepared for publication. Its public history starts with a clean initial commit. The previous development Git directory and private working files remain on this machine under ignored `.private/`.

## What is public

The agent skill, PDF helpers, Rust framework, versioned recipes and research notes are public source. The owner has also approved `releases/notability.zip` as the downloadable trained inference package. Its checksum is tracked beside it.

The release includes learned weights and a runtime. It does not include original handwriting samples, training checkpoints, generated previews or completed forms. Only this exact archive path is allowed by the public-source check. Other model files and archives remain excluded.

## What stays local

| Path | Contents |
| --- | --- |
| `.private/data/` and `.private/dataset.zip` | Source archive and derived corpus |
| `.private/models/` | Research models, checkpoints and optimizer state |
| `.private/runs/`, `.private/reports/`, `.private/output/` | Training state, diagnostics and generated handwriting |
| `.private/exports/`, `.private/downloads/` | Older private packages and downloaded resources |
| `.private/history-before-public.git/` | Original development history |
| `.private/public-prep/` | Local preservation receipts and pre-existing work |

The `.private/` ignore rule covers every descendant. The older root data/model/output paths are also ignored to catch files created by older tools. Global patterns exclude note archives, corpus files, model checkpoints, images, completed PDFs and ordinary ZIP files. The approved trained ZIP is the explicit exception.

`ink` CLI uses `.private/` as its working directory when `.private/models` exists. Existing relative model and corpus paths still resolve there. `INK_REPO` keeps source builds and exports anchored to the public source root. Set `INK_WORKSPACE` if you need a different local workspace.

## Before pushing

```sh
node scripts/check-public.mjs --history
node scripts/check-release.mjs
```

The source check rejects private paths, unexpected large files and symlinks. It also validates the approved release archive. The history check scans all reachable refs for excluded paths. The repository's local pre-push hook runs that check too.

The release check verifies the ZIP SHA256, its internal file hashes, permitted entry paths, model identity and size. It rejects source samples, previews and training-state files inside the release. These checks do not replace reviewing source text for accidental personal details or credentials.

For a new clone, enable the tracked hook with:

```sh
git config core.hooksPath .githooks
```

Push the current public branch to the repository you own. Do not restore old refs or push the Git backup from `.private/`. No remote is configured by this preparation, and no files have been uploaded.

## Optional source export

`node scripts/export-public.mjs` creates a separate copy of the tracked working files, including the approved trained release. It excludes Git history, raw samples and other ignored local files. Review uncommitted edits first because the exporter copies current working contents for tracked paths.

Your other personal decoders and older exports remain private unless you explicitly choose to publish them.
