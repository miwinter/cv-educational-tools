import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
// Keep a standard dist/ artifact, and an identical version tracked for branch-based Pages.
const html = await readFile('dist/app.html', 'utf8');
await writeFile('dist/index.html', html);
await rm('dist/app.html');
await rm('site', { recursive: true, force: true });
await mkdir('site', { recursive: true });
await cp('dist', 'site', { recursive: true });
