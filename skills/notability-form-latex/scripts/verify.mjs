import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifest = path.join(root, 'checksums.json');
if (!fs.existsSync(manifest)) throw new Error('No installation checksums. Run the repository installer first, or test this source checkout with scripts/smoke.mjs.');
const checks = JSON.parse(fs.readFileSync(manifest, 'utf8'));
for (const [name, expected] of Object.entries(checks)) {
  const target = path.resolve(root, name);
  if (!target.startsWith(root + path.sep)) throw new Error(`Invalid entry: ${name}`);
  const actual = crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex');
  if (actual !== expected) throw new Error(`Checksum mismatch: ${name}`);
}
console.log(`PASS: ${Object.keys(checks).length} file checksums`);
