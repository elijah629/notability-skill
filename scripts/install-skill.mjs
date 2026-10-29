#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { inspectRelease } from './release-files.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.join(repo, 'skills/notability-form-latex');
const options = {};
for (let i = 2; i < process.argv.length; i++) {
  const key = process.argv[i];
  if (key === '--trained' && !options[key]) { options[key] = true; continue; }
  if (!['--to', '--decoder', '--binary'].includes(key) || !process.argv[i + 1] || process.argv[i + 1].startsWith('--')) {
    console.error('Usage: node scripts/install-skill.mjs [--to DIRECTORY] [--trained | --decoder FILE --binary FILE]');
    process.exit(1);
  }
  if (options[key]) throw new Error(`Duplicate option: ${key}`);
  options[key] = path.resolve(process.argv[++i]);
}
const destination = options['--to'] ?? path.join(os.homedir(), '.agents/skills/notability-form-latex');
if (fs.existsSync(destination)) throw new Error(`Destination exists; choose a fresh --to directory: ${destination}`);
if (destination === source || destination.startsWith(source + path.sep)) throw new Error('Install outside the source skill directory.');
if (Boolean(options['--decoder']) !== Boolean(options['--binary'])) throw new Error('Provide --decoder and --binary together, or omit both for PDF fields only.');
if (options['--trained'] && options['--decoder']) throw new Error('Choose --trained or your own --decoder and --binary.');
const trained = options['--trained'] ? inspectRelease(repo, { includeFiles: true }).contents : null;
if (options['--decoder']) {
  const decoder = JSON.parse(fs.readFileSync(options['--decoder'], 'utf8'));
  if (!decoder || typeof decoder !== 'object') throw new Error('Decoder must be a JSON object.');
  if (!fs.statSync(options['--binary']).isFile()) throw new Error('Inference binary must be a file.');
}
function files(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name)).flatMap(entry => {
    const full = path.join(directory, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Symlinks are not installable: ${full}`);
    return entry.isDirectory() ? files(full) : [full];
  });
}
const sourceFiles = files(source).filter(file => !/^(?:bin|assets\/models)\//.test(path.relative(source, file)) && path.basename(file) !== 'checksums.json');
fs.mkdirSync(path.dirname(destination), { recursive: true });
// Reserve a new directory. Never replace or merge an existing skill.
fs.mkdirSync(destination);
try {
  for (const file of trained ? [] : sourceFiles) {
    const target = path.join(destination, path.relative(source, file));
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(file, target, fs.constants.COPYFILE_EXCL);
  }
  if (trained) for (const [name, data] of trained) {
    if (name === 'checksums.json') continue;
    const target = path.join(destination, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, data, { flag: 'wx', mode: name === 'bin/ink-infer' ? 0o755 : 0o644 });
  }
  if (options['--decoder']) {
    fs.mkdirSync(path.join(destination, 'assets/models'), { recursive: true });
    fs.mkdirSync(path.join(destination, 'bin'));
    fs.copyFileSync(options['--decoder'], path.join(destination, 'assets/models/decoder.json'));
    fs.copyFileSync(options['--binary'], path.join(destination, 'bin/ink-infer'));
    fs.chmodSync(path.join(destination, 'bin/ink-infer'), 0o755);
  }
  const checks = {};
  for (const file of files(destination)) checks[path.relative(destination, file)] = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  fs.writeFileSync(path.join(destination, 'checksums.json'), JSON.stringify(checks, null, 2) + '\n');
  console.log(JSON.stringify({ installed: destination, mode: trained ? 'PDF fields and bundled m027 handwriting model' : options['--decoder'] ? 'PDF fields and local handwriting model' : 'PDF fields; add your own model for handwriting', verify: `node ${path.join(destination, 'scripts/verify.mjs')}` }, null, 2));
} catch (error) {
  fs.rmSync(destination, { recursive: true, force: true });
  throw error;
}
