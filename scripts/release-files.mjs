import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { inflateRawSync } from 'node:zlib';

export const releasePath = 'releases/notability.zip';
const approvedDecoder = '6bccb239ebefcd7202a041101b3255bc1fdbdd556e134574078b98db7634cb5e';
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
// Read the bounded inference ZIP without extracting or executing its contents.
export function inspectRelease(repo, { includeFiles = false } = {}) {
  const archive = fs.readFileSync(path.join(repo, releasePath));
  const expected = fs.readFileSync(path.join(repo, 'releases/SHA256SUMS'), 'utf8').trim();
  if (expected !== `${digest(archive)}  notability.zip`) throw new Error('Release ZIP SHA256 mismatch.');
  let end = -1;
  for (let i = archive.length - 22; i >= Math.max(0, archive.length - 65557); i--) {
    if (archive.readUInt32LE(i) === 0x06054b50 && i + 22 + archive.readUInt16LE(i + 20) === archive.length) { end = i; break; }
  }
  if (end < 0 || archive.readUInt16LE(end + 4) || archive.readUInt16LE(end + 6)) throw new Error('Unsupported ZIP structure.');
  const count = archive.readUInt16LE(end + 10);
  if (count === 65535) throw new Error('ZIP64 release is not supported.');
  const contents = new Map();
  let cursor = archive.readUInt32LE(end + 16), total = 0;
  for (let i = 0; i < count; i++) {
    if (archive.readUInt32LE(cursor) !== 0x02014b50) throw new Error('Invalid central directory.');
    const flags = archive.readUInt16LE(cursor + 8), method = archive.readUInt16LE(cursor + 10);
    const packed = archive.readUInt32LE(cursor + 20), size = archive.readUInt32LE(cursor + 24);
    const nameLength = archive.readUInt16LE(cursor + 28), extra = archive.readUInt16LE(cursor + 30), comment = archive.readUInt16LE(cursor + 32);
    const offset = archive.readUInt32LE(cursor + 42);
    const name = archive.toString('utf8', cursor + 46, cursor + 46 + nameLength);
    cursor += 46 + nameLength + extra + comment;
    if (flags & 1 || ![0, 8].includes(method) || name.includes('\\') || name.split('/').includes('..') || !name.startsWith('notability-form-latex/')) throw new Error(`Unsafe release entry: ${name}`);
    if (name.endsWith('/')) continue;
    const local = name.slice('notability-form-latex/'.length);
    if (!/^(?:SKILL\.md|checksums\.json|flake\.(?:nix|lock)|LICENSES\.md|dependency-licenses\.json|licenses\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+|agents\/openai\.yaml|scripts\/(?:form|smoke|verify)\.mjs|references\/latex\.md|examples\/plan\.json|vendor\/[A-Za-z.-]+|bin\/ink-infer|assets\/models\/(?:decoder|manifest|vocabulary)\.json)$/.test(local)) throw new Error(`Unapproved release entry: ${name}`);
    if (contents.has(local) || size > 25_000_000 || (total += size) >= 25_000_000) throw new Error('Duplicate or oversized release entry.');
    if (archive.readUInt32LE(offset) !== 0x04034b50) throw new Error('Invalid local entry.');
    const localNameLength = archive.readUInt16LE(offset + 26);
    if (archive.toString('utf8', offset + 30, offset + 30 + localNameLength) !== name || archive.readUInt16LE(offset + 8) !== method) throw new Error('ZIP local/central entry mismatch.');
    const start = offset + 30 + archive.readUInt16LE(offset + 26) + archive.readUInt16LE(offset + 28);
    const bytes = archive.subarray(start, start + packed);
    const data = method === 8 ? inflateRawSync(bytes, { maxOutputLength: size }) : bytes;
    if (data.length !== size) throw new Error('ZIP entry length mismatch.');
    contents.set(local, data);
  }
  const checksumBytes = contents.get('checksums.json');
  if (!checksumBytes) throw new Error('Release checksums missing.');
  const checks = JSON.parse(checksumBytes.toString('utf8'));
  if (Object.keys(checks).length !== contents.size - 1) throw new Error('Incomplete release checksums.');
  for (const [name, data] of contents) if (name !== 'checksums.json' && checks[name] !== digest(data)) throw new Error(`Release file checksum mismatch: ${name}`);
  for (const name of ['SKILL.md', 'scripts/form.mjs', 'bin/ink-infer', 'assets/models/decoder.json', 'assets/models/manifest.json', 'assets/models/vocabulary.json']) if (!contents.has(name)) throw new Error(`Required release entry missing: ${name}`);
  const manifest = JSON.parse(contents.get('assets/models/manifest.json').toString('utf8'));
  if (manifest.id !== 'm027' || manifest.decoder_sha256 !== approvedDecoder || approvedDecoder !== digest(contents.get('assets/models/decoder.json'))) throw new Error('Release model identity mismatch.');
  if (manifest.vocabulary_sha256 !== digest(contents.get('assets/models/vocabulary.json'))) throw new Error('Release vocabulary checksum mismatch.');
  return { files: contents.size, uncompressedBytes: total, zipBytes: archive.length, model: manifest.id, sha256: digest(archive), ...(includeFiles ? { contents } : {}) };
}
