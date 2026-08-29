#!/usr/bin/env node
// ═════════════════════════════════════════════════════════════════════
// Синхронизирует версию из package.json в заголовок README.md.
// README — статичный markdown, TypeScript-компилятор его не трогает,
// поэтому версия в нём (в отличие от UI и диалоговых строк, которые
// импортируют src/lib/version.ts) не может подхватываться сама по
// себе. Этот скрипт закрывает именно этот разрыв.
//
// Запускается автоматически перед dev/build (см. "predev"/"prebuild"
// в package.json) — README больше не должен расходиться с реальной
// версией вручную.
// ═════════════════════════════════════════════════════════════════════

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf-8'));
const short = 'v' + pkg.version.split('.').slice(0, 2).join('.');

const readmePath = join(root, 'README.md');
let readme = readFileSync(readmePath, 'utf-8');

const titleRe = /^# Angusht v[\d.]+ — Нейроморфная Когнитивная Архитектура/m;
const newTitle = `# Angusht ${short} — Нейроморфная Когнитивная Архитектура`;

if (!titleRe.test(readme)) {
  console.warn('[sync-version] заголовок README не найден по ожидаемому шаблону — пропущено');
  process.exit(0);
}

const before = readme.match(titleRe)[0];
if (before === newTitle) {
  process.exit(0); // уже актуально, тихо выходим
}

readme = readme.replace(titleRe, newTitle);
writeFileSync(readmePath, readme, 'utf-8');
console.log(`[sync-version] README.md: "${before}" → "${newTitle}"`);
