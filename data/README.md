# Data stays local

The public repository has no source handwriting samples. Keep original notes read-only and store the corpus under `.private/data/`.

Create `.private/models/` to select the private workspace used by `ink` CLI. Its default corpus then resolves to `.private/data/personal/corpus.jsonl`. Models, runs and outputs stay in that workspace too. Files written to the older root `data/` location are still ignored except for this README.

Read [use your own data](../docs/your-data.md) for the schema, audit commands and training workflow. The ZIP importer expects the framework's specific archive format, not arbitrary note backups or scanned forms.
