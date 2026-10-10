// Le classi vengono lette da index.html, escluso il CSS già generato (tra i marcatori TAILWIND).
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8')
    .replace(/\/\*TAILWIND:START\*\/[\s\S]*?\/\*TAILWIND:END\*\//, '');

module.exports = {
    content: [{ raw: html, extension: 'html' }],
};
