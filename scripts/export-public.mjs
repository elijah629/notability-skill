#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { repo, publicFiles } from './public-files.mjs';
try {
  const destination = path.resolve(process.argv[2] ?? path.join(repo, '.private/exports/public-source'));
  if (fs.existsSync(destination)) throw new Error('Destination exists. Choose a fresh directory.');
  const files = await publicFiles();
  fs.mkdirSync(destination, { recursive: true });
  const checks = {};
  for (const name of files) {
    const source = path.join(repo, name), target = path.join(destination, name);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(target, fs.statSync(source).mode & 0o777);
    checks[name] = crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex');
  }
  fs.writeFileSync(path.join(destination, 'PUBLIC-SOURCE.json'), JSON.stringify({ description: 'Current tracked working files, without Git history or private artifacts. Review source content before publishing.', files: checks }, null, 2) + '\n');
  console.log(`Exported ${files.length} source files to ${destination}. No Git history or personal samples copied. The approved trained release is included.`);
} catch (error) { console.error(`Error: ${error.message}`); process.exitCode = 1; }
