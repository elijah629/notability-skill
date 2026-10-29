import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
const require = createRequire(import.meta.url);
const { PDFDocument, StandardFonts, rgb } = require('../vendor/pdf-lib.cjs');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dir = path.resolve(process.argv[2] ?? '/tmp/notability-form-check');
const withInk = process.argv.includes('--with-ink');
fs.mkdirSync(dir, { recursive: true });
const pdf = await PDFDocument.create();
const page = pdf.addPage([612, 792]);
const font = await pdf.embedFont(StandardFonts.Helvetica);
page.drawText('Form completion - synthetic test', { x: 48, y: 740, size: 19, font });
page.drawText('Name:', { x: 48, y: 693, size: 12, font });
const form = pdf.getForm();
form.createTextField('name').addToPage(page, { x: 100, y: 680, width: 260, height: 26 });
form.createCheckBox('confirmed').addToPage(page, { x: 48, y: 640, width: 15, height: 15 });
page.drawText('Confirmed', { x: 73, y: 643, size: 12, font });
for (const [x, y, w, h, label] of [[48, 210, 210, 85, 'Fraction and radical'], [320, 210, 210, 85, 'Matrix'], [48, 365, 470, 60, 'Multiplication dot']]) {
  page.drawText(label, { x, y: 792 - y + 12, size: 12, font });
  page.drawRectangle({ x, y: 792 - y - h, width: w, height: h, borderWidth: 0.5, borderColor: rgb(0.75, 0.75, 0.75) });
}
fs.writeFileSync(path.join(dir, 'blank.pdf'), await pdf.save(), { flag: 'wx' });
const output = path.join(dir, 'completed.pdf');
const plan = JSON.parse(fs.readFileSync(path.join(root, 'examples/plan.json'), 'utf8'));
if (!withInk) plan.ink = [];
const planFile = path.join(dir, 'plan.json');
fs.writeFileSync(planFile, JSON.stringify(plan, null, 2), { flag: 'wx' });
await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(root, 'scripts/form.mjs'), 'fill', path.join(dir, 'blank.pdf'), planFile, output], { stdio: 'inherit' });
  child.on('error', reject);
  child.on('close', code => code === 0 ? resolve() : reject(new Error(`Fill failed: ${code}`)));
});
const result = await PDFDocument.load(fs.readFileSync(output));
if (result.getForm().getTextField('name').getText() !== 'Alex Example' || !result.getForm().getCheckBox('confirmed').isChecked()) throw new Error('Fixture fields failed');
console.log(`PASS: saved text and checkbox values${withInk ? '; inspect rendered math before delivery' : '; no handwriting model used'}.`);
