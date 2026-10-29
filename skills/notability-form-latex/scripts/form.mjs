#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
const require = createRequire(import.meta.url);
const { PDFDocument, PDFTextField, PDFCheckBox, PDFDropdown, PDFOptionList, StandardFonts, rgb } = require('../vendor/pdf-lib.cjs');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [command, input, specFile, output] = process.argv.slice(2);
const fail = message => { throw new Error(message); };
function run(binary, args) {
  return new Promise(resolve => {
    const child = spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stderr = '';
    child.stdout.resume();
    child.stderr.on('data', chunk => { stderr += chunk; });
    child.on('error', error => resolve({ error, stderr }));
    child.on('close', status => resolve({ status, stderr }));
  });
}
async function generate(text, destination) {
  const decoder = path.resolve(process.env.INK_DECODER ?? path.join(root, 'assets/models/decoder.json'));
  const binary = path.resolve(process.env.INK_BINARY ?? path.join(root, 'bin/ink-infer'));
  if (!fs.existsSync(decoder)) fail('No handwriting decoder. Install your own with --decoder and --binary, or set INK_DECODER and INK_BINARY. Plain PDF field filling needs no model.');
  const args = ['--decoder', decoder, '--input', text, '--out', destination, '--seed', '419', '--variation', '0'];
  let result = await run(binary, args);
  if (result.error?.code === 'EACCES') {
    fs.chmodSync(binary, 0o755);
    result = await run(binary, args);
  }
  if (result.error || (result.status !== 0 && /shared librar|dynamic link|interpreter/i.test(result.stderr))) {
    if (fs.existsSync(path.join(root, 'flake.nix')) && !process.env.INK_DECODER && !process.env.INK_BINARY) {
      result = await run('nix', ['run', `path:${root}`, '--', ...args.slice(2)]);
    } else {
      fail(`Cannot run ${binary}. Build ink-infer for this machine or use the trained download's Nix runtime. ${result.error?.message || result.stderr}`);
    }
  }
  if (result.error || result.status !== 0) fail(`Ink generation failed: ${result.error?.message || result.stderr}`);
  return fs.readFileSync(`${destination}.svg`, 'utf8');
}
function fieldValue(field) {
  if (field instanceof PDFTextField) return field.getText() ?? '';
  if (field instanceof PDFCheckBox) return field.isChecked();
  if (field instanceof PDFDropdown || field instanceof PDFOptionList) return field.getSelected();
  return null;
}
function fieldType(field) {
  if (field instanceof PDFTextField) return 'text';
  if (field instanceof PDFCheckBox) return 'checkbox';
  if (field instanceof PDFDropdown) return 'dropdown';
  if (field instanceof PDFOptionList) return 'option-list';
  return 'unsupported';
}
async function main() {
  if (!['inspect', 'fill'].includes(command) || !input || (command === 'fill' && (!specFile || !output))) {
    fail('Usage: node scripts/form.mjs inspect INPUT.pdf | fill INPUT.pdf PLAN.json OUTPUT.pdf');
  }
  const pdf = await PDFDocument.load(fs.readFileSync(input));
  const pages = pdf.getPages();
  const form = pdf.getForm();
  if (command === 'inspect') {
    console.log(JSON.stringify({
      coordinates: 'PDF points; x from left, y from top of displayed unrotated page; pages are 1-based',
      pages: pages.map((page, i) => ({ page: i + 1, ...page.getSize(), rotation: page.getRotation().angle })),
      fields: form.getFields().map(field => ({ name: field.getName(), type: fieldType(field), value: fieldValue(field),
        options: field instanceof PDFDropdown || field instanceof PDFOptionList ? field.getOptions() : undefined,
        widgets: field.acroField.getWidgets().map(widget => ({ rectangle: widget.getRectangle() })) })),
    }, null, 2));
    return;
  }
  if (path.resolve(input) === path.resolve(output)) fail('Use a new output path; preserve the original.');
  if (fs.existsSync(output)) fail('Output already exists; choose a new path.');
  const spec = JSON.parse(fs.readFileSync(specFile, 'utf8'));
  if (spec.flatten) fail('This helper preserves interactive fields; flattening is not supported.');
  const values = spec.values ?? {};
  for (const [name, value] of Object.entries(values)) {
    const field = form.getField(name);
    if (field.isReadOnly()) fail(`Read-only field: ${name}`);
    if (field instanceof PDFTextField && typeof value === 'string') field.setText(value);
    else if (field instanceof PDFCheckBox && typeof value === 'boolean') value ? field.check() : field.uncheck();
    else if (field instanceof PDFDropdown || field instanceof PDFOptionList) field.select(value);
    else fail(`Unsupported field type or value: ${name}`);
  }
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'notability-form-'));
  try {
    for (const [i, item] of (spec.ink ?? []).entries()) {
      const page = pages[item.page - 1];
      if (!page || !Number.isInteger(item.page)) fail(`Invalid page for ink ${i}`);
      if (page.getRotation().angle !== 0) fail('Rotated pages require coordinate normalization before adding ink.');
      const { x, y, width, height, text } = item;
      if (![x, y, width, height].every(Number.isFinite) || x < 0 || y < 0 || width <= 0 || height <= 0 || typeof text !== 'string' || !text.trim()) fail(`Invalid ink box/text ${i}`);
      const { width: pw, height: ph } = page.getSize();
      const crop = page.getCropBox();
      if (crop.x !== 0 || crop.y !== 0 || crop.width !== pw || crop.height !== ph) fail('Non-default crop box requires coordinate normalization.');
      if (x + width > pw || y + height > ph) fail(`Ink box outside page: ${i}`);
      const svg = await generate(text, path.join(temp, `ink-${i}`));
      const view = svg.match(/viewBox="0 0 ([\d.eE+-]+) ([\d.eE+-]+)"/);
      if (!view) fail('Unsupported generated SVG viewBox.');
      const scale = Math.min(width / Number(view[1]), height / Number(view[2]));
      for (const match of svg.matchAll(/<path d="([^"]+)"[^>]*stroke-width="([^"]+)"[^>]*\/>/g)) {
        page.drawSvgPath(match[1], { x, y: ph - y, scale, borderColor: rgb(0.09, 0.125, 0.165), borderWidth: Number(match[2]), borderLineCap: 1 });
      }
      for (const match of svg.matchAll(/<polyline stroke-width="([^"]+)" points="([^"]+)"\/>/g)) {
        const points = match[2].trim().split(/\s+/).map(pair => pair.split(',').map(Number));
        if (points.length === 1) {
          page.drawCircle({ x: x + points[0][0] * scale, y: ph - y - points[0][1] * scale, size: Number(match[1]) * scale / 2, color: rgb(0.09, 0.125, 0.165) });
        } else {
          const line = points.map((point, j) => `${j ? 'L' : 'M'} ${point[0]} ${point[1]}`).join(' ');
          page.drawSvgPath(line, { x, y: ph - y, scale, borderColor: rgb(0.09, 0.125, 0.165), borderWidth: Number(match[1]), borderLineCap: 1 });
        }
      }
      if (!svg.includes('<path d=')) fail('Generated SVG has no ink paths.');
    }
    form.updateFieldAppearances(await pdf.embedFont(StandardFonts.Helvetica));
    const bytes = await pdf.save();
    const reopened = await PDFDocument.load(bytes);
    for (const [name, value] of Object.entries(values)) {
      const actual = fieldValue(reopened.getForm().getField(name));
      const expected = Array.isArray(actual) && typeof value === 'string' ? [value] : value;
      if (JSON.stringify(actual) !== JSON.stringify(expected)) fail(`Saved field mismatch: ${name}`);
    }
    fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
    fs.writeFileSync(output, bytes, { flag: 'wx' });
    console.log(JSON.stringify({ output: path.resolve(output), pages: pages.length, fieldsVerified: Object.keys(values).length, inkPlacements: (spec.ink ?? []).length, seed: 419, variation: 0, interactive: true }));
  } finally { fs.rmSync(temp, { recursive: true, force: true }); }
}
main().catch(error => { console.error(`Error: ${error.message}`); process.exitCode = 1; });
