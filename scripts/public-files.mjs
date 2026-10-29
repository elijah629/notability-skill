import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { releasePath, inspectRelease } from './release-files.mjs';
const exec = promisify(execFile);
export const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export async function git(args) {
  return (await exec('git', args, { cwd: repo, maxBuffer: 32 * 1024 * 1024 })).stdout;
}
export function privatePath(name) {
  if (name === releasePath) return false;
  const parts = name.split('/');
  if (parts.some(part => ['.git', '.env', '.aws', '.private', 'node_modules', 'target', 'legacy', 'downloads', 'resources', 'models', 'runs', 'reports', 'output', 'exports', 'private'].includes(part))) return true;
  if (parts[0] === 'data' && name !== 'data/README.md') return true;
  if (parts.some(part => part.startsWith('.env.'))) return true;
  if (/\.(?:note|pdf|png|jpe?g|webp|svg|zip|tar|gz|jsonl|parquet|npy|npz|safetensors|pt|pth|onnx)$/i.test(name)) return true;
  if (/^(decoder|vocabulary|checkpoint)(?:[.-]|$)/i.test(path.basename(name))) return true;
  return false;
}
function allowed(name) {
  if (privatePath(name)) return false;
  if ([releasePath, 'releases/SHA256SUMS', 'releases/README.md', '.githooks/pre-push'].includes(name)) return true;
  if (['README.md', 'INSTALL.md', 'AGENTS.md', 'CONTRIBUTING.md', '.gitignore', 'flake.nix', 'flake.lock', 'data/README.md', 'src/recipes/README.md'].includes(name)) return true;
  if (/^docs\/.+\.(?:md|json|sha256)$/.test(name)) return true;
  if (/^scripts\/[a-z-]+\.mjs$/.test(name)) return true;
  if (/^skills\/notability-form-latex\/(?:SKILL\.md|agents\/openai\.yaml|scripts\/[a-z-]+\.mjs|references\/[a-z-]+\.md|examples\/plan\.json|vendor\/(?:pdf-lib\.cjs|[A-Za-z.-]+))$/.test(name)) return true;
  if (/^src\/(?:Cargo\.(?:toml|lock)|README\.md|export-flake\.nix|nix\/[a-z-]+\.nix|graphs\/[a-z_]+\.py|recipes\/[a-zA-Z0-9-]+\.json|crates\/[a-z-]+\/(?:Cargo\.toml|(?:src|examples)\/[a-z_]+\.rs))$/.test(name)) return true;
  return false;
}
export async function publicFiles() {
  const top = (await git(['rev-parse', '--show-toplevel'])).trim();
  if (fs.realpathSync(top) !== fs.realpathSync(repo)) throw new Error('Run source checks in a Git repository initialized at this project root.');
  const files = (await git(['ls-files', '-z'])).split('\0').filter(Boolean).sort();
  if (!files.length) throw new Error('No tracked source files. Stage the intended public source first.');
  const blocked = files.filter(name => !allowed(name));
  if (blocked.length) throw new Error(`Files need a public/private decision before export:\n${blocked.join('\n')}`);
  for (const name of files) {
    const stat = fs.lstatSync(path.join(repo, name));
    if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Not a regular source file: ${name}`);
    if (stat.size > (name === releasePath ? 25_000_000 : 2_000_000)) throw new Error(`Unexpected large public file: ${name}`);
  }
  if (files.includes(releasePath)) inspectRelease(repo);
  return files;
}
