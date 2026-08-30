// ═══════════════════════════════════════════════════════════════════
// Angusht — единый источник версии.
// Раньше строка "Angusht v2.X" была захардкожена в 8+ местах (README,
// заголовок страницы, приветствие в диалоге, User-Agent веб-поиска,
// версия в persistence-снапшоте) и расходилась при каждом релизе —
// её чинили вручную три раза подряд. Теперь версия читается из
// package.json один раз здесь, а все остальные места импортируют
// эту константу. Бump версии = одна правка в package.json.
// ═══════════════════════════════════════════════════════════════════

import pkg from '../../package.json';

// "2.4.0"
export const ANGUSHT_VERSION: string = pkg.version;

// "v2.4" — короткая форма для заголовков UI и диалоговых реплик
export const ANGUSHT_VERSION_SHORT: string =
  'v' + ANGUSHT_VERSION.split('.').slice(0, 2).join('.');

// "2.4" — используется как версия схемы в persistence-снапшоте
export const ANGUSHT_SCHEMA_VERSION: string = ANGUSHT_VERSION.split('.').slice(0, 2).join('.');
