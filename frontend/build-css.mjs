// Rigenera il CSS di Tailwind e lo inserisce in index.html tra /*TAILWIND:START*/ e /*TAILWIND:END*/.
// Uso: cd frontend && npm install && npm run build:css  (da rilanciare dopo aver cambiato classi nell'HTML)
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(dir, 'index.html');
const cli = path.join(dir, 'node_modules', '.bin', 'tailwindcss');

const css = execFileSync(cli, ['-c', 'tailwind.config.cjs', '-i', 'tailwind.input.css', '--minify'], {
    cwd: dir, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'],
}).trim();

const html = fs.readFileSync(htmlPath, 'utf8');
const pattern = /\/\*TAILWIND:START\*\/[\s\S]*?\/\*TAILWIND:END\*\//;
if (!pattern.test(html)) throw new Error('Marcatori TAILWIND non trovati in index.html');
fs.writeFileSync(htmlPath, html.replace(pattern, () => `/*TAILWIND:START*/${css.replace(/<\/style/gi, '<\\/style')}/*TAILWIND:END*/`));
console.log(`CSS inserito: ${(css.length / 1024).toFixed(1)} KB`);
