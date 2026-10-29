# Pipeline

Install the PDF skill with `node scripts/install-skill.mjs`. Plain field filling needs no corpus. Handwritten output needs a local decoder. See [installation](../INSTALL.md).

For training, prepare your own labeled vector corpus, audit it, inspect a dry-run plan, then train only when you intend to spend compute. Generate with the exact model ID and inspect actual output before evaluation or export. Keep the original corpus, trained weights and generated writing local. See [use your own data](your-data.md).

The [research index](research/README.md) records older experiments and known failures. A source checkout includes the implementation and recipes, not the private dataset or a quality-approved personal model.
