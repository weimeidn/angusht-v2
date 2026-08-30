// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — Когнитивный пайплайн
// Персистентное самообучение (включая веса сети), параллельный
// веб-поиск, кеш, семантическое понимание текста.
//
// Честная архитектура:
//   Нейросимуляция (LIF, STDP, 6 ядер) → маршрутизация, пороги,
//     уверенность, нейромодуляторы. Не генерирует текст.
//   Текст генерируют: диалог, математика, БЗ, память, веб-поиск.
//   Нейросеть выбирает порядок модулей и регулирует параметры.
//   Понимание смысла запроса (сопоставление с базой знаний и памятью)
//     обеспечивает отдельный лексико-семантический слой
//     (lexicon.ts + semantic-space.ts) — стемминг, синонимы и растущее
//     распределительное пространство на статистике корпуса. Это не
//     нейросеть и не LLM: детерминированная, объяснимая статистика,
//     которая устраняет главный источник неадекватных ответов —
//     сравнение текстов по сырым подстрокам без учёта словоформ.
// ═══════════════════════════════════════════════════════════════════

import { getCluster, processNeuralInput, tickBackground, persistNeuralWeights, resetCluster } from './index';
import { CoreCluster } from './core-cluster';
import { sharedEncoder } from './neural-core';
import { webSearch, searchStats } from './web-search';
import {
  saveSnapshot, loadSnapshot, snapshotExists, clearPersistedWeights,
  type PersistenceSnapshot,
} from './persistence';
import { tokenizeNormalized, expandWithSynonyms } from './lexicon';
import { SemanticSpace } from './semantic-space';
import { AnswerReadout } from './readout';
import { ANGUSHT_VERSION_SHORT, ANGUSHT_SCHEMA_VERSION } from '../version';

// ── Readout: выбор варианта формулировки ответа (см. readout.ts) ──
let answerReadout = new AnswerReadout();

// ── Семантическое пространство ──
// Растёт вместе с памятью: каждый новый сохранённый вопрос-ответ
// индексируется сюда (см. memoryStore). Используется для сопоставления
// запросов с памятью по смыслу, а не по сырым подстрокам. Подробности
// и обоснование — в semantic-space.ts.
let semanticSpace = new SemanticSpace();

// ── Память ──
interface MemoryEntry {
  q: string;
  a: string;
  kw: string[];
  source: string;
  ts: number;
  freq: number;
  neuralFp?: number[];
  confidence: number;
  // URL источника (только у записей, пришедших из веб-поиска) — нужен,
  // чтобы показать пользователю в чате, откуда взят факт, а не только
  // навсегда запомнить его как истину без возможности проверки.
  url?: string;
}

const memory: MemoryEntry[] = [];
const MAX_MEMORY = 1000;

// ── Граф знаний ──
interface KGNode { freq: number; firstSeen: number }
interface KGEdge { from: string; to: string; w: number; type: string }
const kgNodes = new Map<string, KGNode>();
const kgEdges: KGEdge[] = [];

// ── Рабочая память ──
interface WMEntry { q: string; a: string; mode: string; conf: number; ts: number }
const workingMemory: WMEntry[] = [];
const MAX_WM = 20;

// ── Эпизодическая память ──
// layerProfiles/altRoute — задел под будущий обучаемый readout поверх
// нейронных профилей (пока не используются для принятия решений).
interface Episode {
  q: string; a: string; mode: string; conf: number; route: string; ts: number;
  layerProfiles?: { associative: number[]; executive: number[] };
  altRoute?: string | null;
}
const episodes: Episode[] = [];
const MAX_EPISODES = 500;

// ── Отслеживание сдвига маршрутизации от STDP ──
// Если один и тот же запрос со временем начинает получать другой маршрут,
// это значит, что обучение синапсов действительно влияет на решения,
// а не только на уверенность/пороги.
const routeHistory = new Map<string, string>();
let stdpRouteShifts = 0;

function trackRouteChange(query: string, routeDecision: string): boolean {
  const key = query.toLowerCase().trim().slice(0, 200);
  const prev = routeHistory.get(key);
  routeHistory.set(key, routeDecision);
  if (prev && prev !== routeDecision) {
    stdpRouteShifts++;
    return true;
  }
  return false;
}

// ── Нейронный маршрут → приоритет модулей ──
// Decision Core (WTA в ядре 6) выбирает маршрут.
// Здесь мы маппим нейронный маршрут на порядок попытки модулей.
// НЕЙРОНЫ определяют порядок, код только исполняет его.
const MODULE_PRIORITY: Record<string, string[]> = {
  analytical:   ['math', 'knowledge', 'memory'],      // вычисления → факты → память
  encyclopedic: ['knowledge', 'math', 'memory'],      // факты → вычисления → память
  relational:   ['memory', 'knowledge', 'math'],      // память → факты → вычисления
  social:       ['knowledge', 'memory', 'math'],      // для диалога: факты → память → вычисления
};

// ── Статистика самообучения ──
const learningStats = {
  totalQueries: 0,
  memoryHits: 0,
  webSearches: 0,
  newKnowledge: 0,
  selfLearned: 0,
};

// ── Счётчик для периодического сохранения ──
let saveCounter = 0;
const SAVE_INTERVAL = 5; // сохраняем каждые 5 новых знаний

// ── Загрузка при старте ──
function loadFromDisk(): void {
  const snap = loadSnapshot();
  if (!snap) return;
  memory.length = 0;
  memory.push(...snap.memory);
 kgNodes.clear();
  for (const [k, v] of snap.kgNodes) kgNodes.set(k, v);
  kgEdges.length = 0;
  kgEdges.push(...snap.kgEdges);
  Object.assign(learningStats, snap.learningStats);
  searchStats.totalSearches = snap.searchStats.totalSearches;
  searchStats.successfulSearches = snap.searchStats.successfulSearches;
  searchStats.cacheHits = snap.searchStats.cacheHits;
  searchStats.sourcesUsed = new Map(snap.searchStats.sourcesUsed);
  if (snap.semantic) {
    semanticSpace = SemanticSpace.deserialize(snap.semantic);
  }
  if (snap.readout) {
    answerReadout = AnswerReadout.deserialize(snap.readout);
  }
  if (snap.encoder) {
    // Мутируем singleton на месте (см. encoder.ts loadFrom) — на
    // sharedEncoder уже держат ссылку все NeuralCore-инстансы.
    sharedEncoder.loadFrom(snap.encoder);
  }
  console.log(`[Angusht] Loaded from disk: ${memory.length} memories, ${kgNodes.size} concepts, ${semanticSpace.docFreq.size} semantic terms, ${sharedEncoder.vocabularySize} encoder words`);
}

function saveToDisk(): void {
  try {
    const snap: PersistenceSnapshot = {
      version: ANGUSHT_SCHEMA_VERSION,
      savedAt: Date.now(),
      memory: memory.map(m => ({ q: m.q, a: m.a, kw: m.kw, source: m.source, ts: m.ts, freq: m.freq, confidence: m.confidence, neuralFp: m.neuralFp, url: m.url })),
      kgNodes: [...kgNodes.entries()],
      kgEdges: kgEdges.slice(),
      learningStats: { ...learningStats },
      searchStats: {
        totalSearches: searchStats.totalSearches,
        successfulSearches: searchStats.successfulSearches,
        cacheHits: searchStats.cacheHits,
        sourcesUsed: [...searchStats.sourcesUsed.entries()],
      },
      semantic: semanticSpace.serialize(),
      readout: answerReadout.serialize(),
      encoder: sharedEncoder.serialize(),
    };
    saveSnapshot(snap);
    // Веса сети — отдельно, в бинарном виде (см. persistNeuralWeights).
    persistNeuralWeights();
  } catch (e) {
    console.error('[Angusht] save error:', e);
   }
}

// Загружаем при первом импорте модуля
loadFromDisk();

// ── Утилиты ──
const STOP_WORDS = new Set([
  'в', 'на', 'и', 'с', 'о', 'по', 'к', 'из', 'за', 'от', 'для',
  'что', 'это', 'как', 'не', 'а', 'но', 'да', 'или', 'если', 'же',
  'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
  'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
  'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'it',
  'какой', 'какая', 'какие', 'где', 'когда', 'почему', 'зачем',
  'кто', 'чем', 'мне', 'ты', 'вы', 'он', 'она', 'они', 'мы',
  'все', 'этот', 'тот', 'свой', 'мой', 'ваш', 'наш', 'их', 'есть',
  'может', 'был', 'была', 'были', 'будет', 'могут', 'так', 'еще',
]);

// Ацетилхолин (внимание/фокус) расширяет окно извлекаемых ключевых слов.
// Обновляется в начале processChat из текущего состояния нейромодуляторов.
let currentAchLevel = 0.5;

function extractKeywords(text: string, max = 8): string[] {
  const dynamicMax = Math.round(max + currentAchLevel * 4);
  const words = text.toLowerCase().match(/[a-zA-Zа-яА-ЯЁё0-9]{2,}/g) || [];
  return words.filter(w => !STOP_WORDS.has(w) && w.length > 1).slice(0, dynamicMax);
}

// Косинусное сходство между нейронными "отпечатками" (layer profile
// ассоциативного ядра) — используется для content-addressable поиска в памяти.
function cosineSim(a?: number[], b?: number[]): number {
  if (!a || !b || a.length === 0 || b.length === 0) return 0;
  const len = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function sequenceSimilarity(a: string, b: string): number {
  // Биграммное сходство (Jaccard на биграммах) — учитывает порядок символов
  if (a.length < 2 || b.length < 2) {
    // Для коротких строк — посимвольное сравнение
    const longer = a.length > b.length ? a : b;
    const shorter = a.length > b.length ? b : a;
    if (longer.length === 0) return 1.0;
    let matches = 0;
    for (let i = 0; i < shorter.length; i++) {
      if (longer.includes(shorter[i])) matches++;
    }
    return (2 * matches) / (longer.length + shorter.length);
  }
  const bigramsA = new Set<string>();
  for (let i = 0; i < a.length - 1; i++) bigramsA.add(a.slice(i, i + 2));
  const bigramsB = new Set<string>();
  for (let i = 0; i < b.length - 1; i++) bigramsB.add(b.slice(i, i + 2));
  let inter = 0;
  for (const bg of bigramsA) {
    if (bigramsB.has(bg)) inter++;
  }
  const union = bigramsA.size + bigramsB.size - inter;
  return union === 0 ? 1.0 : inter / union;
}

// ── Безопасный математический парсер (recursive descent) ──
// Заменяет Function() — не выполняет произвольный код
type MathToken = { type: 'num' | 'op' | 'lparen' | 'rparen'; value: string };

function tokenize(expr: string): MathToken[] {
  const tokens: MathToken[] = [];
  let i = 0;
  while (i < expr.length) {
    if (expr[i] === ' ') { i++; continue; }
    if (/[0-9.]/.test(expr[i])) {
      let num = '';
      while (i < expr.length && /[0-9.]/.test(expr[i])) num += expr[i++];
      tokens.push({ type: 'num', value: num });
    } else if ('+-*/'.includes(expr[i])) {
      tokens.push({ type: 'op', value: expr[i++] });
    } else if (expr[i] === '(') {
      tokens.push({ type: 'lparen', value: '(' }); i++;
    } else if (expr[i] === ')') {
      tokens.push({ type: 'rparen', value: ')' }); i++;
    } else {
      return []; // недопустимый символ
    }
  }
  return tokens;
}

function parseExpr(tokens: MathToken[], pos: { i: number }): number {
  let left = parseTerm(tokens, pos);
  while (pos.i < tokens.length && tokens[pos.i].type === 'op' && (tokens[pos.i].value === '+' || tokens[pos.i].value === '-')) {
    const op = tokens[pos.i++].value;
    const right = parseTerm(tokens, pos);
    left = op === '+' ? left + right : left - right;
  }
  return left;
}

function parseTerm(tokens: MathToken[], pos: { i: number }): number {
  let left = parsePower(tokens, pos);
  while (pos.i < tokens.length && tokens[pos.i].type === 'op' && (tokens[pos.i].value === '*' || tokens[pos.i].value === '/')) {
    const op = tokens[pos.i++].value;
    const right = parsePower(tokens, pos);
    left = op === '*' ? left * right : (right !== 0 ? left / right : NaN);
  }
  return left;
}

function parsePower(tokens: MathToken[], pos: { i: number }): number {
  let base = parseAtom(tokens, pos);
  if (pos.i < tokens.length && tokens[pos.i].type === 'op' && tokens[pos.i].value === '*') {
    // Проверяем ** (степень)
    if (pos.i + 1 < tokens.length && tokens[pos.i + 1].type === 'op' && tokens[pos.i + 1].value === '*') {
      pos.i += 2; // пропускаем **
      const exp = parseAtom(tokens, pos);
      base = Math.pow(base, exp);
    }
  }
  return base;
}

function parseAtom(tokens: MathToken[], pos: { i: number }): number {
  if (pos.i >= tokens.length) return NaN;
  if (tokens[pos.i].type === 'num') {
    return parseFloat(tokens[pos.i++].value);
  }
  if (tokens[pos.i].type === 'lparen') {
    pos.i++; // пропускаем '('
    const val = parseExpr(tokens, pos);
    if (pos.i < tokens.length && tokens[pos.i].type === 'rparen') pos.i++;
    return val;
  }
  pos.i++;
  return NaN;
}

function safeMathEval(expr: string): number | null {
  // Заменяем ^ на ** для степени
  let normalized = expr.replace(/\^/g, '**');
  const tokens = tokenize(normalized);
  if (tokens.length === 0) return null;
  try {
    const pos = { i: 0 };
    const result = parseExpr(tokens, pos);
    if (pos.i !== tokens.length) return null; // не весь ввод распарсен
    if (typeof result !== 'number' || !isFinite(result)) return null;
    return result;
  } catch {
    return null;
  }
}

// ── Математический движок ──
function mathAnswer(q: string): { answer: string | null; ok: boolean } {
  const low = q.toLowerCase().trim();

  // Простое выражение (только цифры, операторы, скобки, пробелы)
  if (/^[0-9+\-*/().%^\s]+$/.test(low) && /\d/.test(low)) {
    const result = safeMathEval(low);
    if (result !== null) {
      return { answer: `Результат: ${Number.isInteger(result) ? result : result.toFixed(6)}.`, ok: true };
    }
  }

  // Проценты
  const pctMatch = low.match(/(\d+(?:[.,]\d+)?)\s*%\s*от\s*(\d+(?:[.,]\d+)?)/);
  if (pctMatch) {
    const p = parseFloat(pctMatch[1].replace(',', '.'));
    const n = parseFloat(pctMatch[2].replace(',', '.'));
    return { answer: `${p}% от ${n} = ${(p * n / 100).toFixed(2)}.`, ok: true };
  }

  // Квадратный корень
  const sqrtMatch = low.match(/(?:корень|sqrt)\s*(?:из\s*)?(\d+(?:[.,]\d+)?)/);
  if (sqrtMatch) {
    const n = parseFloat(sqrtMatch[1].replace(',', '.'));
    if (n >= 0) {
      const r = Math.sqrt(n);
      return { answer: `Квадратный корень из ${n} = ${Number.isInteger(r) ? r : r.toFixed(6)}.`, ok: true };
    }
  }

  // Степень
  const powMatch = low.match(/(\d+(?:[.,]\d+)?)\s*(?:в степени|\^|\*\*)\s*(\d+(?:[.,]\d+)?)/);
  if (powMatch) {
    const base = parseFloat(powMatch[1].replace(',', '.'));
    const exp = parseFloat(powMatch[2].replace(',', '.'));
    const r = Math.pow(base, exp);
    return { answer: `${base} в степени ${exp} = ${Number.isInteger(r) ? r : r.toFixed(6)}.`, ok: true };
  }

  return { answer: null, ok: false };
}

// ── Расширенная база знаний ──
const KNOWLEDGE: Record<string, string[]> = {
  // Природа и физика
  'огонь': ['Огонь — быстрая окислительная реакция, выделяющая тепло и свет. Требует горючего, окислителя (обычно кислорода) и источника зажигания. Температура пламени может достигать 1400°C для обычного костра и до 3000°C для специальных горелок.'],
  'вода': [
    'Вода (H₂O) — жидкость при стандартных условиях, необходима для всех известных форм жизни. Состоит из водорода и кислорода. Плотность — 1 г/см³ при 4°C. Точка кипения 100°C, замерзания 0°C. Покрывает около 71% поверхности Земли.',
    'H₂O — простое и при этом самое важное соединение на планете: без воды не существует ни одной известной формы жизни. Кипит при 100°C, замерзает при 0°C, а максимальная плотность — при 4°C. Почти три четверти поверхности Земли покрыто водой.',
  ],
  'земля': ['Земля — третья планета от Солнца, единственная известная планета с жизнью. Возраст около 4.5 миллиардов лет. Масса 5.97 × 10²⁴ кг. Атмосфера состоит из 78% азота и 21% кислорода. Имеет один естественный спутник — Луну.'],
  'солнце': ['Солнце — звезда в центре Солнечной системы. Состоит в основном из водорода (73%) и гелия (25%). Температура поверхности около 5500°C, ядра — около 15 млн °C. Возраст — около 4.6 миллиардов лет. Тип — жёлтый карлик (G2V).'],
  'луна': ['Луна — единственный естественный спутник Земли. Диаметр 3474 км (около 1/4 диаметра Земли). Расстояние до Земли в среднем 384 400 км. Поверхность покрыта кратерами, морями и горами. Влияние на приливы.'],
  'гравитация': ['Гравитация — одно из четырёх фундаментальных взаимодействий. Описывается общей теорией относительности Эйнштейна как искривление пространства-времени. Ускорение свободного падения на Земле ≈ 9.81 м/с². Закон всемирного тяготения Ньютона: F = G×m₁×m₂/r².'],
  'квант': ['Квантовая механика — фундаментальная теория в физике, описывающая поведение частиц на микроскопическом уровне. Основные принципы: суперпозиция, квантовая запутанность, принцип неопределённости Гейзенберга, корпускулярно-волновой дуализм.'],
  'свет': ['Свет — электромагнитное излучение, воспринимаемое человеческим глазом. Длина волны видимого света: 380–700 нм. Скорость в вакууме: 299 792 458 м/с. Демонстрирует корпускулярно-волновой дуализм — ведёт себя и как волна, и как поток фотонов.'],
  'атом': ['Атом — наименьшая частица химического элемента, сохраняющая его свойства. Состоит из ядра (протоны + нейтроны) и электронных оболочек. Размер: около 0.1 нм. Открыт Демокритом (понятие), структура доказана Резерфордом (1911).'],
  'электрон': ['Электрон — элементарная частица с отрицательным электрическим зарядом (−1.602 × 10⁻¹⁹ Кл). Масса покоя: 9.109 × 10⁻³¹ кг. Относится к классу лептонов. Один из фундаментальных строительных блоков материи.'],
  'дНК': ['ДНК (дезоксирибонуклеиновая кислота) — носитель генетической информации во всех живых организмах. Двойная спираль, открытая Уотсоном и Криком в 1953 году. Состоит из 4 нуклеотидов: аденин, тимин, гуанин, цитозин.'],
  'молекула': ['Молекула — электрически нейтральная группа из двух или более атомов, соединённых химическими связями. Наименьшая частица вещества, обладающая его химическими свойствами.'],

  // Биология
  'мозг': [
    'Мозг — центральный орган нервной системы позвоночных. Человеческий мозг содержит около 86 миллиардов нейронов и примерно столько же глиальных клеток. Масса около 1.4 кг. Потребляет 20% энергии организма. Отвечает за мышление, память, восприятие, управление движениями.',
    'Человеческий мозг — около 86 миллиардов нейронов при массе всего порядка 1.4 кг, но потребляет он при этом целых 20% энергии всего организма. Именно он отвечает за мышление, память, восприятие и управление движениями — центральный орган нервной системы позвоночных.',
  ],
  'нейрон': ['Нейрон (нервная клетка) — клетка нервной системы, способная принимать, обрабатывать и передавать электрические и химические сигналы. Состоит из тела (сомы), дендритов (входы) и аксона (выход). Человек имеет ~86 млрд нейронов.'],
  'клетка': ['Клетка — элементарная единица строения и жизнедеятельности всех организмов. Содержит ядро с ДНК, цитоплазму с органеллами (митохондрии, рибосомы, ЭПР, аппарат Гольджи). Размеры от 0.1 до 100 мкм.'],
  'эволюция': ['Эволюция — процесс изменения живых организмов в течение времени. Основана на естественном отборе (Дарвин, 1859). Мутации создают вариативность, отбор сохраняет полезные признаки. Доказательства: палеонтология, генетика, сравнительная анатомия.'],
  'фотосинтез': ['Фотосинтез — процесс преобразования световой энергии в химическую, осуществляемый растениями, водорослями и цианобактериями. Уравнение: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂. Происходит в хлоропластах, содержащих хлорофилл.'],

  // Животные
  'кошка': [
    'Кошка (лат. Felis catus) — домашнее животное, хищник семейства кошачьих. Одомашнена около 10 000 лет назад. Отличное зрение (особенно в темноте), слух и рефлексы. Спит 12-16 часов в сутки. Насчитывается около 73 пород.',
    'Кошка была одомашнена около 10 000 лет назад и остаётся хищником по своей природе — отсюда отменное ночное зрение, слух и рефлексы. Большую часть суток (12–16 часов) кошки проводят во сне. Всего известно порядка 73 пород.',
  ],
  'собака': [
    'Собака (лат. Canis familiaris) — домашнее животное, первое одомашненное человеком (15 000–40 000 лет назад). Используется как компаньон, охранник, помощник (поводыри, пастухи, поиск). Около 400 пород.',
    'Собака — первое животное, одомашненное человеком, 15 000–40 000 лет назад. С тех пор она стала компаньоном, охранником и помощником: собаки-поводыри, пастушьи и поисковые породы. Всего насчитывается около 400 пород.',
  ],
  'дельфин': ['Дельфин — морское млекопитающее семейства дельфиновых. Обладает высоким интеллектом, использует эхолокацию. Дышит атмосферным воздухом. Скорость до 55 км/ч. Живёт стаями. Считается одним из самых умных животных.'],

  // Технологии
  'питон': ['Python — высокоуровневый язык программирования, созданный Гвидо ван Россумом в 1991 году. Отличается простым читаемым синтаксисом, динамической типизацией, обширной стандартной библиотекой. Используется в AI, веб-разработке, науке о данных, автоматизации.'],
  'типескрипт': ['TypeScript — типизированный надмножество JavaScript, разработанное Microsoft (2012). Добавляет статическую типизацию, интерфейсы, дженерики. Транслируется в JavaScript. Стал стандартом для крупных проектов.'],
  'javascript': ['JavaScript — динамический язык программирования, созданный Бренданом Айком в 1995 году. Стандарт веб-разработки (frontend + backend через Node.js). Поддерживает объектно-ориентированный, функциональный и событийно-ориентированный стили.'],
  'реакт': ['React — JavaScript-библиотека для построения пользовательских интерфейсов, разработанная Facebook (2013). Основана на компонентном подходе и виртуальном DOM. Использует JSX-синтаксис. Самая популярная UI-библиотека в мире.'],
  'нейросеть': ['Нейронная сеть (искусственная) — математическая модель, вдохновлённая биологическими нейронами. Состоит из слоёв связанных «нейронов» (перцептронов). Обучается на данных методом обратного распространения ошибки. Основа современного AI.'],
  'искусственный интеллект': ['Искусственный интеллект (AI) — область компьютерных наук, изучающая создание систем, способных выполнять задачи, требующие человеческого интеллекта: распознавание речи, изображений, принятие решений, генерация текста. Включает машинное обучение, нейросети, NLP.'],
  'блокчейн': ['Блокчейн — распределённый реестр (база данных), состоящий из последовательных блоков транзакций, связанных криптографическими хешами. Обеспечивает неизменяемость данных, децентрализацию. Основа криптовалют (Bitcoin, Ethereum).'],

  // Математика
  'число пи': ['Число π (пи) — математическая константа, равная отношению длины окружности к её диаметру. π ≈ 3.14159265358979... Иррациональное и трансцендентное число. Известно более 100 триллионов знаков после запятой.'],
  'число е': ['Число e (число Эйлера) — математическая константа, основание натурального логарифма. e ≈ 2.71828182845905... Иррациональное и трансцендентное число. Встречается в задачах роста, процентах, дифференциальных уравнениях.'],
  'фибоначчи': ['Числа Фибоначчи — последовательность, где каждый элемент равен сумме двух предыдущих: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144... Встречаются в природе: спирали раковин, расположение листьев, соцветия подсолнечника.'],
  'теорема пифагора': ['Теорема Пифагора: в прямоугольном треугольнике квадрат гипотенузы равен сумме квадратов катетов: a² + b² = c². Одна из самых известных теорем геометрии. Доказана более 300 способами.'],

  // История и культура
  'вторая мировая': ['Вторая мировая война (1939–1945) — крупнейший вооружённый конфликт в истории. Участники: страны «Оси» (Германия, Япония, Италия) против Антигитлеровской коалиции (СССР, США, Великобритания и др.). Жертвы: 70-85 миллионов человек.'],
  'россия': ['Россия (Российская Федерация) — государство в Восточной Европе и Северной Азии. Площадь: 17.1 млн км² (крупнейшая в мире). Население: ~146 млн. Столица: Москва. Федеративное президентско-парламентское государство.'],
  'москва': ['Москва — столица России, крупнейший город Европы. Население: ~13 млн человек (в агломерации ~21 млн). Основана в 1147 году. Политический, экономический, культурный и научный центр страны. Московский Кремль — резиденция президента.'],
  'париж': ['Париж — столица Франции, один из крупнейших городов Европы. Население: ~2.2 млн (агломерация ~12 млн). Основан в III веке до н.э. Центр моды, искусства, гастрономии. Эйфелева башня (1889), Лувр, Нотр-Дам.'],

  // Наука и космос
  'марс': ['Марс — четвёртая планета от Солнца. «Красная планета» из-за оксида железа на поверхности. Диаметр 6 779 км (53% Земли). Атмосфера: 95% CO₂. Температура: −60°C средняя. Имеются полярные шапки из водяного льда.'],
  'чёрная дыра': ['Чёрная дыра — область пространства-времени с настолько сильным гравитационным притяжением, что даже свет не может покинуть её. Образуется при коллапсе массивных звёзд. Описывается решением Шварцшильда уравнений Эйнштейна.'],
  'большой взрыв': ['Большой взрыв — общепринятая космологическая модель происхождения Вселенной. Произошёл около 13.8 миллиардов лет назад. Вселенная расширяется из сингулярности с бесконечной плотности и температуры. Подтверждается: красное смещение, реликтовое излучение.'],
  'температура': ['Температура — физическая величина, характеризующая степень нагретости тела. Шкалы: Цельсий (°C), Кельвин (K), Фаренгейт (°F). Абсолютный ноль: 0 K = −273.15°C. Измеряется термометрами.'],
  'скорость света': ['Скорость света в вакууме (c) = 299 792 458 м/с. Это предельная скорость в природе согласно СТО Эйнштейна. Свет проходит расстояние от Земли до Луны за 1.3 секунды. На Солнце за 8 минут 20 секунд.'],

  // Общие понятия
  'время': ['Время — фундаментальная физическая величина, измеряемая последовательностью событий и их длительностью. В физике рассматривается как четвёртое измерение пространства-времени. Единицы: секунда (СИ). Необратимо течёт от прошлого к будущему.'],
  'энергия': ['Энергия — скалярная физическая величина, мера способности системы совершать работу. Единица: джоуль (Дж). Закон сохранения энергии: энергия не создаётся и не уничтожается, а лишь преобразуется. Виды: кинетическая, потенциальная, тепловая, электрическая.'],
  'электричество': ['Электричество — совокупность явлений, обусловленных существованием, движением и взаимодействием электрических зарядов. Открыто Фалесом Милетским. Закон Ома: I = U/R. Единицы: вольт (В), ампер (А), ом (Ω).'],
  'интернет': ['Интернет — глобальная система объединённых компьютерных сетей, построенная на использовании протокола IP и маршрутизации пакетов данных. Создан как ARPANET в 1969 году. Основные протоколы: TCP/IP, HTTP, DNS. Насчитывает более 5 миллиардов пользователей.'],
  'программирование': ['Программирование — процесс создания компьютерных программ путём написания исходного кода на языках программирования. Включает: анализ задачи, проектирование алгоритмов, кодирование, тестирование, отладку. Основные парадигмы: императивная, ООП, функциональная.'],
};

// Алиасы для более широкого распознавания
const ALIASES: Record<string, string> = {
  'космос': 'большой взрыв', 'вселенная': 'большой взрыв',
  'мл': 'нейросеть', 'машинное обучение': 'нейросеть', 'ии': 'искусственный интеллект',
  'ai': 'искусственный интеллект', 'криптовалюта': 'блокчейн',
  'биткоин': 'блокчейн', 'js': 'javascript', 'ts': 'типескрипт',
  'вайфай': 'интернет', 'wifi': 'интернет',
  'днк': 'дНК', 'градус цельсия': 'температура', 'кельвин': 'температура',
  'фаренгейт': 'температура', 'электричество': 'электричество',
  'pi': 'число пи', 'число e': 'число е', 'euler': 'число е',
  'эволюция': 'эволюция', 'фотосинтез': 'фотосинтез',
  '重力': 'гравитация', '引力': 'гравитация',
  'программирование': 'программирование', 'химия': 'молекула',
  'биология': 'клетка',
};

// Оценивает, насколько тема (ключ базы знаний или алиас, может быть
// многословной, напр. "число пи", "чёрная дыра") соответствует запросу.
// Взвешенное по длине токена перекрытие + бонус за точное вхождение
// фразы целиком. Возвращает число в диапазоне ~[0, ~1.1].
//
// Раньше здесь была прямая подстрочная проверка (`low.includes(key)`)
// и, отдельно, сравнение `key.includes(low.slice(0, 4))` — то есть
// первые 4 символа ВСЕГО запроса (а не отдельного слова) сверялись с
// произвольным ключом. Из-за перебора Object.entries побеждал первый
// зарегистрированный ключ, а не самый релевантный — отсюда и
// «несуразица»: запрос мог случайно зацепиться не за ту тему.
function scoreKeyAgainstQuery(keyPhrase: string, queryLow: string, queryTokensExpanded: Set<string>): number {
  const keyTokens = tokenizeNormalized(keyPhrase);
  if (keyTokens.length === 0) return 0;

  let matchedWeight = 0;
  let totalWeight = 0;
  for (const kt of keyTokens) {
    const w = Math.min(kt.length, 8);
    totalWeight += w;
    if (queryTokensExpanded.has(kt)) matchedWeight += w;
  }
  let score = totalWeight > 0 ? matchedWeight / totalWeight : 0;

  // Точное вхождение фразы целиком — сильный сигнал, особенно для
  // многословных тем ("чёрная дыра", "теорема пифагора"). ё→е — та же
  // нормализация, что и в tokenizeNormalized, чтобы "черная дыра"
  // (без ё, как чаще всего пишут) тоже засчитывалось.
  if (queryLow.replace(/ё/g, 'е').includes(keyPhrase.replace(/ё/g, 'е'))) {
    score = Math.max(score, 0.85) + 0.1;
  }
  return score;
}

const KNOWLEDGE_MATCH_THRESHOLD = 0.55;

// Возвращает не просто текст, а тему + выбранный вариант формулировки —
// нужно для readout (см. readout.ts): сеть влияет на то, КАКОЙ из
// нескольких заготовленных вариантов вернуть, не только на то, что
// нашёлся ответ вообще.
function knowledgeLookup(
  q: string,
  associativeProfile: number[] | undefined,
  readout: AnswerReadout
): { answer: string; topicKey: string; variantIndex: number } | null {
  const low = q.toLowerCase();
  const queryTokens = new Set(expandWithSynonyms(tokenizeNormalized(low)));

  let best: { key: string; answers: string[]; score: number } | null = null;

  for (const [key, answers] of Object.entries(KNOWLEDGE)) {
    const score = scoreKeyAgainstQuery(key, low, queryTokens);
    if (!best || score > best.score) best = { key, answers, score };
  }
  for (const [alias, target] of Object.entries(ALIASES)) {
    const answers = KNOWLEDGE[target];
    if (!answers) continue;
    const score = scoreKeyAgainstQuery(alias, low, queryTokens);
    if (!best || score > best.score) best = { key: target, answers, score };
  }

  if (!best || best.score < KNOWLEDGE_MATCH_THRESHOLD) return null;

  const variantIndex = readout.select(best.key, best.answers.length, associativeProfile);
  return { answer: best.answers[variantIndex], topicKey: best.key, variantIndex };
}

// ── Диалоговые шаблоны ──
function conversationAnswer(q: string): string | null {
  const low = q.toLowerCase().trim();
  if (/^(привет|здравствуй|хай|hello|hi|добр)/.test(low)) {
    return `Здравствуйте! Я Angusht ${ANGUSHT_VERSION_SHORT} — нейроморфная когнитивная система с 6 специализированными ядрами (1 296 000 LIF-нейронов).

Моя нейросистема управляет обработкой запросов. Decision Core (ядро 6) выбирает маршрут через WTA-конкуренцию групп нейронов с латеральным торможением. Нейромодуляторы (дофамин, серотонин, ацетилхолин, норадреналин) регулируют обучение.

Если не знаю ответ — ищу в интернете (Wikipedia, DuckDuckGo) и запоминаю. Спрашивайте!`;
  }
  if (/^(кто ты|что ты|расскажи о себе|что ты умеешь)/.test(low)) {
    return `Я Angusht ${ANGUSHT_VERSION_SHORT} — нейроморфная когнитивная архитектура с самообучением.

6 специализированных ядер (по 216K LIF-нейронов):
1. Сенсорное — кодирование входного сигнала
2. Ассоциативное — распознавание паттернов, память
3. Аналитическое — логика, вычисления
4. Временное — контекст, последовательности
5. Исполнительное — агрегация сигналов
6. Decision Core — WTA-выбор маршрута (латеральное торможение)

Возможности:
• Математические вычисления
• Встроенная база знаний (100+ тем)
• Веб-поиск через Wikipedia и DuckDuckGo
• Самообучение: каждый найденный ответ сохраняется в нейропамять
• Семантическое пространство: понимает словоформы и синонимы, растёт с каждым новым знанием
• Readout: сеть выбирает вариант формулировки ответа среди нескольких, не только источник
• STDP-обучение формирует устойчивые нейронные пути, веса переживают перезапуск
• Decision Core: НЕЙРОНЫ выбирают маршрут, не код
• 4 нейромодулятора (DA, 5HT, ACh, NE) регулируют обучение

Статистика сессии:
• Запросов обработано: ${learningStats.totalQueries}
• Попаданий в память: ${learningStats.memoryHits}
• Веб-поисков: ${learningStats.webSearches}
• Новых знаний: ${learningStats.newKnowledge}
• Самообучено: ${learningStats.selfLearned}
• Терминов в семантическом пространстве: ${semanticSpace.docFreq.size}`;
  }
  if (/^(спасибо|благодарю)/.test(low)) {
    return 'Пожалуйста! Нейросистема закрепила этот ответ через дофаминовое подкрепление. Чем ещё могу помочь?';
  }
  if (/^(пока|до свидания|прощай)/.test(low)) {
    return 'До связи! Все нейронные пути и знания сохранены в памяти. При повторном обращении система будет отвечать быстрее благодаря STDP.';
  }
  if (/сколько ты помнишь|покажи память|статистика/.test(low)) {
    return `Статистика системы:
• Память: ${memory.length} записей (макс ${MAX_MEMORY})
• Граф знаний: ${kgNodes.size} концептов, ${kgEdges.length} связей
• Эпизодов: ${episodes.length}
• Рабочая память: ${workingMemory.length}
• Веб-поисков: ${learningStats.webSearches} (успешных: ${searchStats.successfulSearches}, из кеша: ${searchStats.cacheHits})
• Самообучено ответов: ${learningStats.selfLearned}
• Сдвигов маршрута от STDP: ${stdpRouteShifts}
• Кеш поиска: ${searchStats.cacheSize()} записей
• Семантическое пространство: ${semanticSpace.docFreq.size} терминов, ${semanticSpace.totalDocs} проиндексированных текстов
• Обучаемый кодировщик входа: ${sharedEncoder.vocabularySize} слов с обученными координатами
• Персистентность: ${snapshotExists() ? 'файл angusht-data/memory.json (' : 'нет сохранённых данных'}${snapshotExists() ? '' : ')'}
• Источники: ${[...searchStats.sourcesUsed.entries()].map(([k,v]) => `${k}: ${v}`).join(', ') || 'пока нет'}`;
  }
  if (/забудь всё|очисти память/.test(low)) {
    memory.length = 0;
    kgNodes.clear();
    kgEdges.length = 0;
    episodes.length = 0;
    workingMemory.length = 0;
    learningStats.selfLearned = 0;
    learningStats.newKnowledge = 0;
    learningStats.memoryHits = 0;
    semanticSpace = new SemanticSpace();
    answerReadout = new AnswerReadout();
    // Обнуляем и обучаемый кодировщик слов (encoder.ts) — иначе
    // координаты, сдрейфовавшие от уже стёртых запросов, продолжали бы
    // влиять на то, какие нейроны стимулируются новыми словами.
    sharedEncoder.loadFrom({ coords: {} });
    // Сбрасываем и веса сети: иначе STDP-пути, обученные на уже
    // стёртых знаниях, продолжали бы влиять на маршрутизацию.
    resetCluster();
    clearPersistedWeights();
    saveToDisk();
    return 'Память полностью очищена: текстовые знания, семантическое пространство и веса нейросети (включая файлы на диске) сброшены. Система стартует заново, как при первом запуске.';
  }
  return null;
}

// ── Поиск в памяти ──
// queryFp: нейронный отпечаток текущего запроса (layer profile ассоциативного
// ядра) — используется как content-addressable сигнал наравне с текстовым сходством.
// kwSim раньше считался как Jaccard по несклеенным словоформам
// (extractKeywords без стемминга) — "кошки" и "кошку" считались
// разными словами и не давали пересечения. Теперь используется
// semanticSpace.overlapScore: токены нормализованы (стем) + расширены
// синонимами, а вес каждого слова взят из IDF растущего корпуса
// (частые общие слова значат меньше, специфичные — больше).
function memorySearch(query: string, threshold: number, queryFp?: number[]): { answer: string; score: number; source?: string; url?: string } | null {
  const qLow = query.toLowerCase();
  const qTokens = expandWithSynonyms(tokenizeNormalized(qLow));
  let bestScore = 0;
  let bestEntry: MemoryEntry | null = null;

  for (const entry of memory) {
    const seqSim = sequenceSimilarity(qLow, entry.q.toLowerCase());
    const eTokens = expandWithSynonyms(tokenizeNormalized(entry.q.toLowerCase()));
    const kwSim = semanticSpace.overlapScore(qTokens, eTokens);
    const neuralSim = cosineSim(queryFp, entry.neuralFp);
    const recency = Math.min(entry.freq * 0.03, 0.15);
    const confBonus = entry.confidence * 0.1;
    const score = seqSim * 0.30 + kwSim * 0.35 + neuralSim * 0.15 + recency + confBonus;
    if (score > bestScore && score >= threshold) {
      bestScore = score;
      bestEntry = entry;
    }
  }

  if (bestEntry) {
    bestEntry.freq++;
    // Прокидываем источник/URL дальше — чтобы ответ, извлечённый из
    // памяти (а не найденный только что), тоже показывал пользователю,
    // откуда изначально взят факт (см. buildResponse/ChatResponse.source).
    return { answer: bestEntry.a, score: bestScore, source: bestEntry.source, url: bestEntry.url };
  }
  return null;
}

// ── Пользовательские коррекции ──
// Отдельный, более строгий поиск (выше порог, чем у memorySearch) —
// ищет ТОЛЬКО среди записей, явно подтверждённых человеком через
// 👎 + поле "правильный ответ". Высокая планка совпадения намеренно:
// ложное срабатывание здесь подставило бы не тот факт под уверенным
// видом "проверено человеком", а не позвало бы неточный, но
// нейтральный ответ, как при обычном memorySearch.
const CORRECTION_MATCH_THRESHOLD = 0.72;

function findUserCorrection(query: string): { answer: string; score: number } | null {
  const qLow = query.toLowerCase();
  const qTokens = expandWithSynonyms(tokenizeNormalized(qLow));
  let best: { entry: MemoryEntry; score: number } | null = null;

  for (const entry of memory) {
    if (entry.source !== 'user-correction') continue;
    const seqSim = sequenceSimilarity(qLow, entry.q.toLowerCase());
    const eTokens = expandWithSynonyms(tokenizeNormalized(entry.q.toLowerCase()));
    const kwSim = semanticSpace.overlapScore(qTokens, eTokens);
    const score = seqSim * 0.5 + kwSim * 0.5;
    if (score >= CORRECTION_MATCH_THRESHOLD && (!best || score > best.score)) {
      best = { entry, score };
    }
  }
  if (!best) return null;
  best.entry.freq++;
  return { answer: best.entry.a, score: best.score };
}

function memoryStore(query: string, answer: string, source: string, confidence: number = 0.5, neuralFp?: number[], url?: string): void {
  const kw = extractKeywords(query);
  const qLow = query.toLowerCase();
  // Проверяем дубликаты
  for (const entry of memory) {
    if (sequenceSimilarity(qLow, entry.q.toLowerCase()) > 0.85) {
      entry.a = answer;
      entry.source = source;
      entry.ts = Date.now();
      entry.freq++;
      entry.kw = kw;
      entry.confidence = Math.max(entry.confidence, confidence);
      if (neuralFp) entry.neuralFp = neuralFp;
      if (url) entry.url = url;
      return;
    }
  }
  memory.push({ q: query, a: answer, kw, source, ts: Date.now(), freq: 1, confidence, neuralFp, url });
  // Растим семантическое пространство новым вопрос-ответом — это и
  // есть "самообучение" на уровне понимания смысла, а не только
  // запоминания готовых ответов.
  semanticSpace.indexText(`${query} ${answer}`);
  if (memory.length > MAX_MEMORY) {
    memory.sort((a, b) => (b.freq * 0.3 + b.confidence * 0.3 + b.ts * 0.4) - (a.freq * 0.3 + a.confidence * 0.3 + a.ts * 0.4));
    memory.splice(0, memory.length - MAX_MEMORY + 100);
  }
  learningStats.newKnowledge++;
  if (source.startsWith('Wikipedia') || source.startsWith('DuckDuckGo')) {
    learningStats.selfLearned++;
  }
  // Периодическое сохранение на диск
  saveCounter++;
  if (saveCounter >= SAVE_INTERVAL) {
    saveCounter = 0;
    saveToDisk();
  }
}

// Обновление графа знаний
function kgUpdate(query: string, answer: string): void {
  const qConcepts = extractKeywords(query, 4);
  const aConcepts = extractKeywords(answer, 6);
  const now = Date.now();
  for (const c of [...qConcepts, ...aConcepts]) {
    if (!kgNodes.has(c)) {
      kgNodes.set(c, { freq: 1, firstSeen: now });
    } else {
      kgNodes.get(c)!.freq++;
    }
  }
  for (const qc of qConcepts.slice(0, 2)) {
    for (const ac of aConcepts.slice(0, 3)) {
      if (qc === ac) continue;
      const existing = kgEdges.find(e => e.from === qc && e.to === ac);
      if (existing) {
        existing.w = Math.min(1, existing.w + 0.1);
      } else if (kgEdges.length < 3000) {
        kgEdges.push({ from: qc, to: ac, w: 0.3, type: 'related' });
      }
    }
  }
}

function wmPush(q: string, a: string, mode: string, conf: number) {
  workingMemory.push({ q, a, mode, conf, ts: Date.now() });
  if (workingMemory.length > MAX_WM) workingMemory.shift();
}

function epiStore(
  q: string, a: string, mode: string, conf: number, route: string,
  layerProfiles?: { associative: number[]; executive: number[] }, altRoute?: string | null
) {
  episodes.push({ q, a: a.slice(0, 500), mode, conf, route, ts: Date.now(), layerProfiles, altRoute });
  if (episodes.length > MAX_EPISODES) episodes.splice(0, episodes.length - 350);
}

// Источники, за которыми стоит реальный внешний URL (или хотя бы
// проверяемое внешнее происхождение) — в отличие от "knowledge"/"math"/
// "conversation"/"user-correction", которые полностью внутренние и
// ссылку показывать не на что.
function isExternalSourceName(source?: string): boolean {
  if (!source) return false;
  return source.startsWith('Wikipedia') || source.startsWith('DuckDuckGo');
}

function computeAltRoute(scores: Record<string, number>, chosen: string): string | null {
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const alt = sorted.find(([k]) => k !== chosen);
  return alt ? alt[0] : null;
}

// ═══════════════════════════════════════════════════════════════════
// ГЛАВНЫЙ ПАЙПЛАЙН — АСИНХРОННЫЙ (веб-поиск)
// ═══════════════════════════════════════════════════════════════════

export interface ChatResponse {
  answer: string;
  mode: string;
  confidence: number;
  // Источник ответа, когда это применимо (веб-поиск, либо память,
  // изначально пополненная веб-поиском) — чтобы пользователь мог сам
  // проверить факт, а не просто получить его на веру. Отсутствует для
  // ответов, порождённых внутри системы (диалог/математика/база знаний/
  // коррекция пользователя) — там проверять по внешней ссылке нечего.
  source: { name: string; url?: string } | null;
  trace: string[];
  neural: {
    routeDecision: string;
    altRoute: string | null;
    neuralConfidence: number;
    familiarity: number;
    neuromodulators: { dopamine: number; serotonin: number; acetylcholine: number; norepinephrine: number };
    coresSummary: Array<{ id: string; name: string; active: number; fired: number; avgWeight: number }>;
    interCoreTraffic: number;
    routeShiftedFromLastTime: boolean;
    stdpRouteShifts: number;
    // Decision Core метрики
    decisionGroupActivities: number[];
    decisionDominance: number;
    moduleOrder: string[];
    // Сходимость каскада (сколько раундов потребовалось ядрам 1-5,
    // чтобы прийти к стабильной активности — см. рекомендацию #2:
    // раньше считалось в snapshot, но нигде не показывалось явно).
    cascadeRounds: number;
    cascadeConverged: boolean;
  };
  core: ReturnType<CoreCluster['snapshot']>;
}

async function processChatInternal(query: string): Promise<ChatResponse> {
  tickBackground();
  const cluster = getCluster();
  const trace: string[] = [];
  const t0 = Date.now();
  learningStats.totalQueries++;

  // ══════════════════════════════════════════
  // ФАЗА 1: НЕЙРОННАЯ ОБРАБОТКА ВХОДА
  // ════════════════════════════════════════════
  trace.push('нейронный вход → сенсорное ядро');
  const neuralResult = processNeuralInput(query);
  trace.push(`сенсорное: ${neuralResult.sensoryIndices.length} нейронов stimulated`);
  trace.push('пропагация: сенсорное → ассоциативное → аналитическое → временное → исполнительное → Decision Core');
  trace.push(`межъядерный трафик: ${cluster.interCoreTraffic} сигналов`);
  const cascadeInfo = cluster.getCascadeInfo();
  trace.push(`каскад: ${cascadeInfo.cascadeRounds}/10 раундов, сошёлся ${cascadeInfo.cascadeConverged ? 'досрочно' : 'НЕТ (упёрся в лимит MAX_ROUNDS)'}`);

  const { routeDecision, routeScores, confidence: neuralConfidence, familiarity } = neuralResult;
  trace.push(`Decision Core: маршрут=${routeDecision} (dominance ${(cluster.lastDecisionDominance * 100).toFixed(0)}% | уверенность ${(neuralConfidence * 100).toFixed(0)}% | знакомство ${(familiarity * 100).toFixed(0)}%)`);
  trace.push(`нейромодуляторы: DA=${cluster.neuromodulators.dopamine.toFixed(2)} 5HT=${cluster.neuromodulators.serotonin.toFixed(2)} ACh=${cluster.neuromodulators.acetylcholine.toFixed(2)} NE=${cluster.neuromodulators.norepinephrine.toFixed(2)}`);

  // ACh расширяет окно ключевых слов для всех модулей этого запроса
  currentAchLevel = cluster.neuromodulators.acetylcholine;

  // Второй по силе маршрут — задел под будущий обучаемый readout
  const altRoute = computeAltRoute(routeScores, routeDecision);

  // STDP: изменился ли маршрут для этого же (или похожего) запроса с прошлого раза?
  const routeShifted = trackRouteChange(query, routeDecision);
  if (routeShifted) {
    trace.push(`STDP: маршрут для этого запроса сместился с прошлого раза (сдвигов всего: ${stdpRouteShifts})`);
  }

  // Нейронный отпечаток запроса (layer profile ассоциативного ядра) —
  // используется для content-addressable поиска в памяти
  const queryFp = neuralResult.layerProfiles['associative'];
  const episodeProfiles = {
    associative: neuralResult.layerProfiles['associative'],
    executive: neuralResult.layerProfiles['executive'],
  };

  // Порог памяти регулируется нейросистемой: знакомство + дофамин
  // (DA = «мне это уже нравилось» → снижает порог, легче вспомнить)
  const daMemBonus = cluster.neuromodulators.dopamine * 0.08;
  const memThreshold = Math.max(0.5, (familiarity > 0.5 ? 0.68 : 0.80) - daMemBonus);
  trace.push(`нейро-порог памяти: ${memThreshold.toFixed(2)} (знакомство ${familiarity.toFixed(2)}, DA-бонус ${daMemBonus.toFixed(2)})`);

  // ── Stage 1: ДИАЛОГ (всегда первый — regex-паттерны) ──
  const convAnswer = conversationAnswer(query);
  if (convAnswer) {
    trace.push('модуль: диалог → найден');
    const conf = Math.min(0.95, neuralConfidence + 0.3);
    cluster.globalReinforce(query, true);
    wmPush(query, convAnswer, 'conversation', conf);
    epiStore(query, convAnswer, 'conversation', conf, routeDecision, episodeProfiles, altRoute);
    kgUpdate(query, convAnswer);
    trace.push('STDP + дофаминовое подкрепление (+0.15)');
    trace.push(`время: ${Date.now() - t0}мс`);
    return buildResponse(convAnswer, 'conversation', conf, trace, neuralResult, cluster, routeShifted);
  }

  // ── Stage 1.5: ПОЛЬЗОВАТЕЛЬСКИЕ КОРРЕКЦИИ ──
  // Приоритет НАД всеми остальными модулями, включая базу знаний.
  // Если человек раньше явно указал правильный ответ на такой же (или
  // очень похожий) вопрос через 👎 + поле коррекции — эта коррекция
  // должна побеждать даже встроенную KNOWLEDGE, которую нельзя изменить
  // во время работы сервера (это статичные данные в исходном коде, а
  // не персистентная память). Без этой проверки MODULE_PRIORITY мог бы
  // снова подсунуть уже опровергнутый ответ из базы знаний раньше, чем
  // очередь дойдёт до memory.
  const correction = findUserCorrection(query);
  if (correction) {
    trace.push(`пользовательская коррекция: найдено (сходство ${(correction.score * 100).toFixed(0)}%) → приоритет над всеми модулями`);
    const conf = Math.min(0.99, 0.9 + correction.score * 0.1);
    cluster.globalReinforce(query, true);
    wmPush(query, correction.answer, 'correction', conf);
    epiStore(query, correction.answer, 'correction', conf, routeDecision, episodeProfiles, altRoute);
    trace.push(`время: ${Date.now() - t0}мс`);
    return buildResponse(correction.answer, 'correction', conf, trace, neuralResult, cluster, routeShifted);
  }

  // ═══════════════════════════════════════════════════════════════
  // Stage 2-4: НЕЙРОННЫЙ ПОРЯДОК МОДУЛЕЙ
  // Decision Core (WTA в ядре 6) выбрал маршрут.
  // MODULE_PRIORITY маппит маршрут на порядок модулей.
  // НЕЙРОНЫ определяют порядок, код только исполняет.
  // ═══════════════════════════════════════════════════════════════
  const moduleOrder = MODULE_PRIORITY[routeDecision] || ['knowledge', 'math', 'memory'];
  trace.push(`Decision Core: маршрут=${routeDecision}, порядок модулей: [${moduleOrder.join(', ')}]`);

  for (const mod of moduleOrder) {
    if (mod === 'math') {
      const { answer: mathA, ok: mathOk } = mathAnswer(query);
      if (mathOk && mathA) {
        trace.push(`модуль: математика (приоритет #${moduleOrder.indexOf('math') + 1}) → решено`);
        cluster.globalReinforce(query, true);
        memoryStore(query, mathA, 'math', 0.95, queryFp);
        kgUpdate(query, mathA);
        const isTop = moduleOrder[0] === 'math';
        const conf = Math.min(0.98, neuralConfidence + 0.35 + (isTop ? 0.08 : 0));
        wmPush(query, mathA, 'math', conf);
        epiStore(query, mathA, 'math', conf, routeDecision, episodeProfiles, altRoute);
        trace.push(isTop ? 'нейро-маршрут совпал → +0.08 к уверенности' : '');
        trace.push(`время: ${Date.now() - t0}мс`);
        return buildResponse(mathA, 'math', conf, trace, neuralResult, cluster, routeShifted);
      }
      trace.push(`математика (приоритет #${moduleOrder.indexOf('math') + 1}) — не подходит`);
    } else if (mod === 'knowledge') {
      const knowResult = knowledgeLookup(query, neuralResult.layerProfiles['associative'], answerReadout);
      if (knowResult) {
        const { answer: knowAnswer, topicKey, variantIndex } = knowResult;
        trace.push(`модуль: база знаний (приоритет #${moduleOrder.indexOf('knowledge') + 1}) → найдено (тема "${topicKey}", вариант формулировки #${variantIndex})`);
        const isTop = moduleOrder[0] === 'knowledge';
        const conf = Math.min(0.88, neuralConfidence + 0.2 + (isTop ? 0.08 : 0));
        cluster.globalReinforce(query, true);
        // Readout: подкрепляем выбранный вариант формулировки тем же
        // сигналом, что и остальную систему (успешный ответ = плюс).
        answerReadout.reinforce(topicKey, variantIndex, KNOWLEDGE[topicKey]?.length || 1, true);
        kgUpdate(query, knowAnswer);
        memoryStore(query, knowAnswer, 'knowledge', conf, queryFp);
        wmPush(query, knowAnswer, 'knowledge', conf);
        epiStore(query, knowAnswer, 'knowledge', conf, routeDecision, episodeProfiles, altRoute);
        trace.push('ассоциативное ядро: паттерн закреплён (STDP)');
        trace.push(`время: ${Date.now() - t0}мс`);
        return buildResponse(knowAnswer, 'knowledge', conf, trace, neuralResult, cluster, routeShifted);
      }
      trace.push(`база знаний (приоритет #${moduleOrder.indexOf('knowledge') + 1}) — не найдено`);
    } else if (mod === 'memory') {
      trace.push(`поиск в памяти (приоритет #${moduleOrder.indexOf('memory') + 1}, порог: ${memThreshold.toFixed(2)})`);
      const memResult = memorySearch(query, memThreshold, queryFp);
      if (memResult) {
        trace.push(`память: найдено (сходство ${(memResult.score * 100).toFixed(0)}%)`);
        learningStats.memoryHits++;
        const isTop = moduleOrder[0] === 'memory';
        const conf = Math.min(0.9, memResult.score + (isTop ? 0.08 : 0));
        cluster.globalReinforce(query, true);
        kgUpdate(query, memResult.answer);
        wmPush(query, memResult.answer, 'memory', conf);
        epiStore(query, memResult.answer, 'memory', conf, routeDecision, episodeProfiles, altRoute);
        trace.push('ассоциативное ядро: путь усилен через STDP');
        trace.push(`время: ${Date.now() - t0}мс`);
        // Если этот ответ изначально пришёл из внешнего источника —
        // прокидываем его в UI, чтобы факт из памяти оставался
        // проверяемым, а не превращался в анонимную "истину системы".
        const memSourceInfo = isExternalSourceName(memResult.source)
          ? { name: memResult.source!, url: memResult.url }
          : null;
        return buildResponse(memResult.answer, 'memory', conf, trace, neuralResult, cluster, routeShifted, memSourceInfo);
      }
      trace.push('память — не найдено');
    }
  }

  // ════════════════════════════════════════════
  // ФАЗА 2: ВЕБ-ПОИСК (самообучение!)
  // Если нет ответа в системе — ищем в интернете
  // NE регулирует скорость (короче таймауты при высоком arousal)
  // ════════════════════════════════════════════
  trace.push('внутренние модули исчерпаны → веб-поиск');
  learningStats.webSearches++;
  searchStats.totalSearches++;
  searchStats.lastSearchTime = Date.now();

  try {
    const webResult = await webSearch(query);
    if (webResult && webResult.answer.length > 30) {
      trace.push(`веб-поиск: найден ответ (${webResult.source})`);
      searchStats.successfulSearches++;
      searchStats.sourcesUsed.set(webResult.source, (searchStats.sourcesUsed.get(webResult.source) || 0) + 1);

      const answer = webResult.answer;
      const conf = Math.min(0.85, webResult.confidence);

      // ════════════════════════════════════════════
      // САМООБУЧЕНИЕ: сохраняем в нейропамять!
      // Следующий раз ответим мгновенно из памяти
      // ════════════════════════════════════════════
      memoryStore(query, answer, webResult.source, conf, queryFp, webResult.url || undefined);
      kgUpdate(query, answer);

      // Сильное нейронное подкрепление (новое знание!)
      cluster.globalReinforce(query, true);
      cluster.neuromodulators.dopamine = Math.min(1, cluster.neuromodulators.dopamine + 0.2);
      trace.push('САМООБУЧЕНИЕ: ответ сохранён в память + сильное дофаминовое подкрепление');
      trace.push(`время: ${Date.now() - t0}мс`);
      wmPush(query, answer, 'web', conf);
      epiStore(query, answer, 'web', conf, routeDecision, episodeProfiles, altRoute);
      // Источник виден пользователю в чате — самообучение больше не
      // превращает веб-факт в анонимную "истину системы" без ссылки.
      return buildResponse(answer, 'web', conf, trace, neuralResult, cluster, routeShifted, {
        name: webResult.source,
        url: webResult.url || undefined,
      });
    }
  } catch (e: any) {
    trace.push(`веб-поиск: ошибка — ${e.message}`);
  }
  trace.push('веб-поиск: ничего не найдено');

  // ── Stage 6: Fallback ──
  const confidence = Math.max(0.2, neuralConfidence * 0.5);
  const fallback = `Я обработал ваш запрос через 6 нейроморфных ядер (1.3M нейронов), но не нашёл точного ответа ни во внутренней базе знаний, ни в памяти, ни в интернете.

Нейросистема запомнила этот запрос — при повторе паттерн распознается быстрее благодаря STDP.

Попробуйте:
• Математическое выражение (2+3*4, sqrt(144), 15% от 200)
• Вопрос о науке, технике, истории, природе
• «кто ты» — о системе
• «статистика» — что система запомнила

Система ищет ответы в Wikipedia и DuckDuckGo при каждом новом вопросе.`;
  wmPush(query, fallback, 'fallback', confidence);
  epiStore(query, fallback, 'fallback', confidence, routeDecision, episodeProfiles, altRoute);
  trace.push(`время: ${Date.now() - t0}мс`);

  cluster.globalReinforce(query, false);
  trace.push('анти-Хеббовское ослабление нераспознанных путей');

  return buildResponse(fallback, 'fallback', confidence, trace, neuralResult, cluster, routeShifted);
}

// ── Сериализация обработки чата ──
// getCluster() отдаёт общий на весь процесс singleton (index.ts).
// processChatInternal делает `await webSearch(...)` в середине
// обработки одного запроса; пока он ждёт ответ от Wikipedia/DuckDuckGo,
// event loop свободен — и второй вызов (вторая вкладка браузера, два
// быстрых сообщения подряд) мог бы начать СВОЙ нейронный каскад над тем
// же мутируемым состоянием ядер (activeSet, potentials, веса), а затем,
// при возврате из await, ошибочно подкрепить STDP по чужой, уже
// перезаписанной активности — тихая порча весов сети, которую было бы
// очень трудно потом диагностировать. Простая FIFO-очередь на промисах
// гарантирует, что весь processChatInternal одного запроса выполняется
// целиком, прежде чем начнётся следующий.
let chatQueue: Promise<unknown> = Promise.resolve();

export function processChat(query: string): Promise<ChatResponse> {
  const result = chatQueue.then(() => processChatInternal(query));
  // Ошибка одного запроса не должна останавливать очередь для всех
  // последующих — цепочка продолжается независимо от исхода.
  chatQueue = result.then(() => undefined, () => undefined);
  return result;
}

function buildResponse(
  answer: string, mode: string, confidence: number, trace: string[],
  neuralResult: ReturnType<CoreCluster['processInput']>,
  cluster: CoreCluster,
  routeShifted: boolean = false,
  sourceInfo: { name: string; url?: string } | null = null
): ChatResponse {
  const snap = cluster.snapshot();
  const modOrder = MODULE_PRIORITY[neuralResult.routeDecision] || ['knowledge', 'math', 'memory'];
  return {
    answer,
    mode,
    confidence,
    source: sourceInfo,
    trace,
    neural: {
      routeDecision: neuralResult.routeDecision,
      altRoute: computeAltRoute(neuralResult.routeScores, neuralResult.routeDecision),
      neuralConfidence: neuralResult.confidence,
      familiarity: neuralResult.familiarity,
      neuromodulators: { ...cluster.neuromodulators },
      coresSummary: snap.cores.map(c => ({
        id: c.id, name: c.name, active: c.active,
        fired: c.firedThisCycle, avgWeight: c.avgWeight,
      })),
      interCoreTraffic: snap.interCoreTraffic,
      routeShiftedFromLastTime: routeShifted,
      stdpRouteShifts,
      decisionGroupActivities: snap.decisionGroupActivities,
      decisionDominance: snap.decisionDominance,
      moduleOrder: modOrder,
      cascadeRounds: snap.cascadeRounds,
      cascadeConverged: snap.cascadeConverged,
    },
    core: snap,
  };
}

// ═══════════════════════════════════════════════════════════════════
// ОБРАТНАЯ СВЯЗЬ ОТ ЧЕЛОВЕКА (👍/👎 + коррекция в UI)
// ═══════════════════════════════════════════════════════════════════
// До этого момента ЕДИНСТВЕННЫЙ сигнал подкрепления во всей системе
// был "нашёлся ли хоть какой-то ответ", а не "был ли ответ верным" —
// cluster.globalReinforce(query, true) вызывался при любом успешном
// попадании любого модуля, false — только при полном провале (когда
// не нашлось вообще ничего). Это подкрепление никогда не проверяло
// фактическую правильность. submitFeedback — первое место в системе,
// где эту роль исполняет реальный человек, а не факт наличия ответа.
export function submitFeedback(
  query: string,
  answer: string,
  mode: string,
  positive: boolean,
  correction?: string
): { stored: boolean; correctionStored: boolean } {
  const cluster = getCluster();
  const qLow = query.toLowerCase();
  let stored = false;
  let correctionStored = false;

  // Настоящее подкрепление по факту правильности — а не по факту
  // "что-то нашлось". Раньше false вызывалось только при полном
  // провале; здесь false может прийти на КОНКРЕТНЫЙ неверный ответ,
  // который система была уверена, что нашла правильно.
  cluster.globalReinforce(query, positive);

  if (positive) {
    // Подтверждённый человеком ответ — закрепляем в памяти с
    // максимальной уверенностью, независимо от исходного источника
    // (база знаний, память, веб-поиск), чтобы повторные такие же
    // вопросы отвечались быстро и с полным доверием.
    let found = false;
    for (const entry of memory) {
      if (entry.a === answer && sequenceSimilarity(qLow, entry.q.toLowerCase()) > 0.85) {
        entry.confidence = 1.0;
        entry.freq++;
        found = true;
        break;
      }
    }
    if (!found) {
      memoryStore(query, answer, `${mode}-confirmed`, 1.0);
    }
    stored = true;
  } else {
    // Убираем именно ЭТОТ неверный ответ из памяти — иначе memorySearch
    // продолжит его предлагать при следующем похожем вопросе.
    for (let i = memory.length - 1; i >= 0; i--) {
      if (memory[i].a === answer && sequenceSimilarity(qLow, memory[i].q.toLowerCase()) > 0.85) {
        memory.splice(i, 1);
      }
    }
    if (correction && correction.trim().length > 0) {
      // Раз у нас теперь есть подтверждённая человеком истина — убираем
      // ЛЮБЫЕ другие кэшированные ответы на этот же вопрос (не только
      // тот конкретный неверный текст), чтобы коррекция не конкурировала
      // сама с собой при повторном memoryStore.
      for (let i = memory.length - 1; i >= 0; i--) {
        if (sequenceSimilarity(qLow, memory[i].q.toLowerCase()) > 0.85) {
          memory.splice(i, 1);
        }
      }
      memoryStore(query, correction.trim(), 'user-correction', 1.0);
      correctionStored = true;
    }
  }

  // Человеческая обратная связь — сохраняем немедленно, не дожидаясь
  // счётчика SAVE_INTERVAL: это подтверждённые факты, терять их при
  // случайном рестарте до следующего автосохранения недопустимо.
  saveToDisk();

  return { stored, correctionStored };
}
