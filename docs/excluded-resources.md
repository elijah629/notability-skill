# Resource policy

Git tracks Rust source, locks, versioned recipes and documentation. It excludes `dataset.zip`, `data/` except its README, `legacy/`, `models/`, downloads/resources, runs/reports/output, exports, build products and caches. No Git LFS is used.

All previous datasets, including MathWriting downloads/derivatives, are in `legacy/superseded-20261014/data/`. No public pretraining corpus is active and no public vectors are included in the current decoder/export. Bringing public data back requires verifying provenance and licensing; it must remain separate from personal fine-tuning and require explicit fallback selection.

The detailed private archival inventory is `legacy/superseded-20261014/inventory.json`; the tracked [summary](legacy-inventory.json) records its checksum and archival counts. Original private archives are retained without modification.
