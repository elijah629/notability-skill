#!/usr/bin/env node
import { git, privatePath, publicFiles } from './public-files.mjs';
try {
  const files = await publicFiles();
  console.log(`PASS: ${files.length} tracked source files pass the public path, type and size checks.`);
  if (process.argv.includes('--history')) {
    const history = [...new Set((await git(['log', '--all', '--pretty=format:', '--name-only'])).split('\n').filter(Boolean))];
    const privateFiles = history.filter(privatePath);
    if (privateFiles.length) {
      console.error(`Historical private paths found. Do not publish this checkout's Git history:\n${privateFiles.join('\n')}\nUse scripts/export-public.mjs for a new source tree without history.`);
      process.exitCode = 2;
    } else console.log('PASS: no private paths detected in reachable history. This is not a content or secret audit.');
  }
} catch (error) { console.error(`Error: ${error.message}`); process.exitCode = 1; }
