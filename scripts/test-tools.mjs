#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createRequire } from 'node:module';
import { repo, privatePath } from './public-files.mjs';
import { inspectRelease, releasePath } from './release-files.mjs';
import crypto from 'node:crypto';
const exec = promisify(execFile);
const require = createRequire(import.meta.url);
const { PDFDocument, PDFName } = require('../skills/notability-form-latex/vendor/pdf-lib.cjs');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'notability-tools-'));
const testEnv = { ...process.env };
delete testEnv.INK_DECODER;
delete testEnv.INK_BINARY;
const run = args => exec(process.execPath, args, { cwd: repo, env: testEnv });
const installed = path.join(temp, 'skill');
try {
  await run(['scripts/install-skill.mjs', '--to', installed]);
  await run([path.join(installed, 'scripts/verify.mjs')]);
  const before = fs.readFileSync(path.join(installed, 'SKILL.md'));
  await assert.rejects(run(['scripts/install-skill.mjs', '--to', installed]));
  assert.deepEqual(fs.readFileSync(path.join(installed, 'SKILL.md')), before);
  assert(!fs.existsSync(path.join(installed, 'assets/models/decoder.json')));
  console.log('PASS: model-free installation, hashes and overwrite refusal');

  const pdf = await PDFDocument.create();
  const pages = [pdf.addPage(), pdf.addPage()];
  const form = pdf.getForm();
  form.createTextField('first').addToPage(pages[0], { x: 40, y: 600, width: 200, height: 30 });
  form.createTextField('second').addToPage(pages[1], { x: 40, y: 600, width: 200, height: 30 });
  form.createCheckBox('check').addToPage(pages[1], { x: 40, y: 550, width: 15, height: 15 });
  const choice = form.createDropdown('choice');
  choice.addOptions(['one', 'two']);
  choice.addToPage(pages[1], { x: 40, y: 500, width: 200, height: 30 });
  const input = path.join(temp, 'input.pdf'), output = path.join(temp, 'filled.pdf'), plan = path.join(temp, 'plan.json');
  fs.writeFileSync(input, await pdf.save());
  fs.writeFileSync(plan, JSON.stringify({ values: { first: 'Page one', second: 'Page two', check: true, choice: 'two' }, ink: [] }));
  await run([path.join(installed, 'scripts/form.mjs'), 'fill', input, plan, output]);
  const reopened = await PDFDocument.load(fs.readFileSync(output));
  const saved = reopened.getForm();
  assert.equal(saved.getTextField('first').getText(), 'Page one');
  assert.equal(saved.getTextField('second').getText(), 'Page two');
  assert.equal(saved.getCheckBox('check').isChecked(), true);
  assert.deepEqual(saved.getDropdown('choice').getSelected(), ['two']);
  for (const field of saved.getFields()) for (const widget of field.acroField.getWidgets()) {
    assert(widget.dict.get(PDFName.of('Parent')));
    assert(widget.dict.get(PDFName.of('AP')));
  }
  for (const page of reopened.getPages()) assert(page.node.Annots().size() > 0);
  console.log('PASS: multipage values, selection, widget parents and saved appearances');

  for (const [name, spec] of Object.entries({ unknown: { values: { missing: 'x' } }, noModel: { ink: [{ page: 1, x: 40, y: 40, width: 100, height: 40, text: 'x^2' }] }, outside: { ink: [{ page: 1, x: 900, y: 40, width: 100, height: 40, text: 'x' }] } })) {
    const specPath = path.join(temp, `${name}.json`), target = path.join(temp, `${name}.pdf`);
    fs.writeFileSync(specPath, JSON.stringify(spec));
    await assert.rejects(run([path.join(installed, 'scripts/form.mjs'), 'fill', input, specPath, target]));
    assert(!fs.existsSync(target));
  }
  console.log('PASS: missing field, missing model and out-of-page ink leave no output');

  for (const name of ['data/corpus.jsonl', 'models/m000/decoder.json', 'notes.note', '.env.local', 'docs/personal.png', 'skills/notability-form-latex/assets/models/decoder.json']) assert(privatePath(name));
  assert(privatePath('.private/history-before-public.git/config'));
  assert(privatePath('releases/other-model.zip'));
  assert(!privatePath('releases/notability.zip'));
  for (const name of ['README.md', 'docs/research/README.md', 'src/crates/ink-infer/src/main.rs', 'skills/notability-form-latex/SKILL.md']) assert(!privatePath(name));
  console.log('PASS: private artifact path checks');

  const trainedInstall = path.join(temp, 'trained-skill');
  await run(['scripts/install-skill.mjs', '--trained', '--to', trainedInstall]);
  await run([path.join(trainedInstall, 'scripts/verify.mjs')]);
  assert(fs.existsSync(path.join(trainedInstall, 'assets/models/decoder.json')));
  inspectRelease(repo);
  const badRepo = path.join(temp, 'bad-release');
  fs.mkdirSync(path.join(badRepo, 'releases'), { recursive: true });
  const archive = Buffer.from(fs.readFileSync(path.join(repo, releasePath)));
  const oldName = Buffer.from('examples/plan.json'), badName = Buffer.from('examples/leak.note');
  assert.equal(oldName.length, badName.length);
  let cursor = 0, replacements = 0;
  while ((cursor = archive.indexOf(oldName, cursor)) >= 0) { badName.copy(archive, cursor); cursor += badName.length; replacements++; }
  assert(replacements >= 2);
  fs.writeFileSync(path.join(badRepo, releasePath), archive);
  fs.writeFileSync(path.join(badRepo, 'releases/SHA256SUMS'), `${crypto.createHash('sha256').update(archive).digest('hex')}  notability.zip\n`);
  assert.throws(() => inspectRelease(badRepo), /Unapproved release entry/);
  console.log('PASS: trained installation and rejection of a source-note path inside its ZIP');
} finally { fs.rmSync(temp, { recursive: true, force: true }); }
