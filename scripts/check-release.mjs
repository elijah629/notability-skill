#!/usr/bin/env node
import { repo } from './public-files.mjs';
import { inspectRelease } from './release-files.mjs';
try { console.log(JSON.stringify(inspectRelease(repo), null, 2)); }
catch (error) { console.error(`Error: ${error.message}`); process.exitCode = 1; }
