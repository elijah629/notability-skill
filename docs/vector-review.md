# Vector dataset review

## Sequence

1. Import every `.note` from the supplied source directory through the Rust CLI. Hash sources and preserve extraction failures. Completion requires one manifest entry per source, including empty and duplicate notes.
2. Inspect every extracted page SVG, retaining context and original stroke identities. The source archive is read-only; embedded raster members are never copied into the active corpus. SVG views contain original native paths, not printed glyph substitutes.
3. A reviewing subagent transcribes visible handwriting and selects exact ordered source strokes. A second distinct subagent independently verifies label, stroke membership, sample kind and authorship. Store each review separately. Disagreements require explicit adjudication; uncertain cases remain quarantined.
4. Assign each extracted stroke to an accepted sample, an explicit exclusion or quarantine, with reason. A note closes only after all pages and strokes are accounted for. Parser failures cannot close as reviewed notes.
5. Prepare the canonical personal corpus only after validating review coverage. Link derivatives to immutable source hashes and original stroke identities. Single-glyph supervision requires an actual isolated glyph or reviewed stroke subset; whole-line labels do not imply glyph alignment.
6. Deduplicate source versions and shared ink before deterministic grouped splits. Keep complete source groups in one partition. Insufficient independent groups must be reported, not hidden by splitting glyphs from one source across partitions.
7. Report accepted, excluded, quarantined, unresolved and duplicate counts, plus glyph coverage. Generate targeted writing prompts only after the complete-source review gate passes.

## Authenticity

Native ink is not proof of personal authorship. Exclude known generated notes and printed/typed content; quarantine unknown writers. Existing OCR, worksheet prompts and assistant review labels can assist comparison but cannot substitute for viewing the actual selected strokes. A review record is evidence of work actually performed, never a default filled by software.

## Geometry

Keep original cubic controls and metadata losslessly in extracted source JSON. Use uniform transforms with explicit inverses for canonical geometry. Keep pen lifts separate. Training tensors are derived views; caps must mark samples retrieval-only rather than truncating ink. Missing pressure/timing stays missing.

Some source archives contain short optional style/options buffers or repeated UUIDs. Preserve those bytes, mark uncertain per-stroke values `null`, and use source hash plus native index as identity. Never shift metadata to make lengths agree.

Strokes outside all declared pages remain in `unplaced_stroke_ids` and get a separate `unplaced.svg` review viewport. This is not an invented source page. An annotation page with `page_index: null` reviews those strokes; source closure requires this region to be viewed and accounted for too. Source page count stays unchanged.
