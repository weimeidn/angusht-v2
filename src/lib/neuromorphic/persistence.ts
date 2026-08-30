// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — Персистентность
// Сохраняет memory, граф знаний, семантическое пространство и
// статистику на диск (JSON) — плюс, отдельно, веса нейросети
// (внутрисинаптические + межъядерные проекции) в бинарном виде.
// При перезапуске — загружает всё обратно.
// ═══════════════════════════════════════════════════════════════════
//
// Почему веса — не в JSON: это массивы из сотен тысяч float-чисел
// (216 000 нейронов × 6 направлений × 6 ядер + 15 проекций по 216 000
// весов). JSON.stringify превращает каждое число в текст — файл
// раздувается в разы, а синхронная запись всего файла целиком на
// каждое автосохранение начинает заметно блокировать сервер. Сырые
// Float32Array пишутся как есть, без кодирования.

import {
  readFileSync,
  writeFileSync,
  existsSync,
  mkdirSync,
  rmSync,
  renameSync,
} from 'fs';
import { join } from 'path';
import { ANGUSHT_SCHEMA_VERSION } from '../version';
import type { EncoderData } from './encoder';

const DATA_DIR = join(process.cwd(), 'angusht-data');
const SNAPSHOT_FILE = join(DATA_DIR, 'memory.json');
const WEIGHTS_DIR = join(DATA_DIR, 'weights');

// ── Атомарная запись ──
// Раньше writeFileSync писал прямо в целевой файл. Если процесс падал
// (kill, сбой питания, OOM) посреди записи — файл оставался обрезанным
// или повреждённым, и при следующем запуске вся текстовая память
// (или веса сети) терялись разом, без возможности восстановления.
// Пишем во временный файл в той же директории (важно — тот же диск/раздел,
// чтобы rename был атомарным на уровне ОС) и переименовываем поверх
// целевого файла только после успешной записи. rename в POSIX и Windows
// атомарен: наблюдатель либо видит старый файл целиком, либо новый
// целиком — промежуточного "битого" состояния не бывает.
function atomicWriteFileSync(targetPath: string, data: string | NodeJS.ArrayBufferView): void {
  const tmpPath = `${targetPath}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  writeFileSync(tmpPath, data as any);
  renameSync(tmpPath, targetPath);
}

// Сериализуемые данные (текстовая часть)
export interface PersistenceSnapshot {
  version: string;
  savedAt: number;
  memory: Array<{
    q: string;
    a: string;
    kw: string[];
    source: string;
    ts: number;
    freq: number;
    confidence: number;
    neuralFp?: number[];
    // URL источника (только у записей из веб-поиска) — для отображения
    // ссылки на источник в чате (см. cognitive-pipeline.ts).
    url?: string;
  }>;
  kgNodes: Array<[string, { freq: number; firstSeen: number }]>;
  kgEdges: Array<{ from: string; to: string; w: number; type: string }>;
  learningStats: {
    totalQueries: number;
    memoryHits: number;
    webSearches: number;
    newKnowledge: number;
    selfLearned: number;
  };
  searchStats: {
    totalSearches: number;
    successfulSearches: number;
    cacheHits: number;
    sourcesUsed: Array<[string, number]>;
  };
  // Семантическое пространство (docFreq + co-occurrence), см. semantic-space.ts.
  // Опционально — старые снапшоты (2.2/2.3) без этого поля остаются валидными,
  // пространство просто стартует пустым и накопится заново.
  semantic?: {
    totalDocs: number;
    docFreq: Record<string, number>;
    cooc: Record<string, Record<string, number>>;
  };
  // Веса readout — выбор варианта формулировки ответа (см. readout.ts).
  // Опционально: старые снапшоты без этого поля остаются валидными,
  // readout стартует с нейтральными весами и накапливается заново.
  readout?: {
    weights: Record<string, number[]>;
  };
  // Обученные координаты слов в решётке (см. encoder.ts, v2.4.3).
  // Опционально — старые снапшоты (до 2.4.3) без этого поля остаются
  // валидными, кодировщик просто стартует пустым и обучится заново
  // (выпадение в чистый детерминированный хеш-бутстрап, а не ошибка).
  encoder?: EncoderData;
}

function ensureDataDir(): void {
  if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });
}

function ensureWeightsDir(): void {
  if (!existsSync(WEIGHTS_DIR)) mkdirSync(WEIGHTS_DIR, { recursive: true });
}

export function saveSnapshot(data: PersistenceSnapshot): void {
  try {
    ensureDataDir();
    data.savedAt = Date.now();
    atomicWriteFileSync(SNAPSHOT_FILE, JSON.stringify(data, null, 2));
  } catch (e) {
    console.error('[Persistence] save error:', e);
  }
}

export function loadSnapshot(): PersistenceSnapshot | null {
  try {
    if (!existsSync(SNAPSHOT_FILE)) return null;
    const raw = readFileSync(SNAPSHOT_FILE, 'utf-8');
    const data = JSON.parse(raw) as PersistenceSnapshot;
    // Схема снапшота меняется реже, чем версия приложения — принимаем
    // текущую версию (из package.json, через version.ts) плюс список
    // более старых схем, совместимых по формату полей.
    const LEGACY_COMPATIBLE_SCHEMAS = ['2.2', '2.3'];
    if (data.version !== ANGUSHT_SCHEMA_VERSION && !LEGACY_COMPATIBLE_SCHEMAS.includes(data.version)) {
      console.warn(`[Persistence] version mismatch: ${data.version}, starting fresh`);
      return null;
    }
    return data;
  } catch (e) {
    console.error('[Persistence] load error:', e);
    return null;
  }
}

export function snapshotExists(): boolean {
  return existsSync(SNAPSHOT_FILE);
}

// ── Бинарные веса ──

export function saveCoreWeights(coreId: string, buffer: ArrayBuffer): void {
  try {
    ensureWeightsDir();
    atomicWriteFileSync(join(WEIGHTS_DIR, `core-${coreId}.bin`), Buffer.from(buffer));
  } catch (e) {
    console.error(`[Persistence] saveCoreWeights(${coreId}) error:`, e);
  }
}

export function loadCoreWeights(coreId: string): ArrayBuffer | null {
  try {
    const p = join(WEIGHTS_DIR, `core-${coreId}.bin`);
    if (!existsSync(p)) return null;
    const buf = readFileSync(p);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  } catch (e) {
    console.error(`[Persistence] loadCoreWeights(${coreId}) error:`, e);
    return null;
  }
}

export function saveProjectionWeights(index: number, buffer: ArrayBuffer): void {
  try {
    ensureWeightsDir();
    atomicWriteFileSync(join(WEIGHTS_DIR, `proj-${index}.bin`), Buffer.from(buffer));
  } catch (e) {
    console.error(`[Persistence] saveProjectionWeights(${index}) error:`, e);
  }
}

export function loadProjectionWeights(index: number): ArrayBuffer | null {
  try {
    const p = join(WEIGHTS_DIR, `proj-${index}.bin`);
    if (!existsSync(p)) return null;
    const buf = readFileSync(p);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  } catch (e) {
    console.error(`[Persistence] loadProjectionWeights(${index}) error:`, e);
    return null;
  }
}

export function weightsExist(): boolean {
  return existsSync(WEIGHTS_DIR);
}

// Полная очистка весов на диске (используется командой «забудь всё» —
// иначе STDP-обученные пути к удалённым знаниям продолжали бы жить
// в весах, даже когда текстовая память уже стёрта).
export function clearPersistedWeights(): void {
  try {
    if (existsSync(WEIGHTS_DIR)) {
      rmSync(WEIGHTS_DIR, { recursive: true, force: true });
    }
  } catch (e) {
    console.error('[Persistence] clearPersistedWeights error:', e);
  }
}
