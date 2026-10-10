// Rigenera il CSS di Tailwind e lo inserisce in index.html tra /*TAILWIND:START*/ e /*TAILWIND:END*/,
// poi genera partner.html (demo per educatori e partner) a partire da index.html.
// Uso: cd frontend && npm install && npm run build:css  (da rilanciare dopo ogni modifica a index.html)
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(dir, 'index.html');
const partnerPath = path.join(dir, 'partner.html');
const cli = path.join(dir, 'node_modules', '.bin', 'tailwindcss');

const css = execFileSync(cli, ['-c', 'tailwind.config.cjs', '-i', 'tailwind.input.css', '--minify'], {
    cwd: dir, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'],
}).trim();

const html = fs.readFileSync(htmlPath, 'utf8');
const pattern = /\/\*TAILWIND:START\*\/[\s\S]*?\/\*TAILWIND:END\*\//;
if (!pattern.test(html)) throw new Error('Marcatori TAILWIND non trovati in index.html');
const built = html.replace(pattern, () => `/*TAILWIND:START*/${css.replace(/<\/style/gi, '<\\/style')}/*TAILWIND:END*/`);
fs.writeFileSync(htmlPath, built);
console.log(`CSS inserito: ${(css.length / 1024).toFixed(1)} KB`);

// partner.html: stessa app con la modalità demo attiva, fuori dai motori di ricerca.
function replaceOnce(text, search, replacement) {
    const count = text.split(search).length - 1;
    if (count !== 1) throw new Error(`partner.html: atteso 1 "${search}", trovati ${count}`);
    return text.replace(search, () => replacement);
}
let partner = built;
partner = replaceOnce(partner, 'const PARTNER_MODE = false;', 'const PARTNER_MODE = true;');
partner = replaceOnce(partner, '<meta name="robots" content="index, follow">', '<meta name="robots" content="noindex, nofollow">');
partner = replaceOnce(partner, '<link rel="canonical" href="https://biodog.io/">', '<link rel="canonical" href="https://biodog.io/partner.html">');
partner = replaceOnce(partner, 'data-exclude-hash="true"', 'data-exclude-hash="true" data-tag="partner"');
partner = replaceOnce(partner, '<!DOCTYPE html>', '<!DOCTYPE html>\n<!-- File generato da build-css.mjs a partire da index.html: non modificarlo a mano. -->');
fs.writeFileSync(partnerPath, partner);
console.log('partner.html generata');
