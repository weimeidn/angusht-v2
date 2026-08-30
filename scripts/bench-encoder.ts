// ═══════════════════════════════════════════════════════════════════
// Angusht — проверка TrainableEncoder (п.7 приоритетного списка)
// ═══════════════════════════════════════════════════════════════════
// Это не бенчмарк производительности (как bench-neural-core.ts), а
// содержательная проверка: обучаемое кодирование действительно
// ОБУЧАЕТСЯ, а не просто заменяет один хеш на другой.
//
// Проверяет три вещи:
//   1) Бутстрап координаты нового слова ПОБИТОВО совпадает со старой
//      формулой FNV-1a → [l,r,c] (neural-core.ts до v2.4.3) — то есть
//      до какого-либо обучения система ведёт себя так же, как раньше,
//      это не случайный разброс "с нуля".
//   2) Слова, которые часто встречаются РЯДОМ в текстах, со временем
//      физически СБЛИЖАЮТСЯ в координатах решётки (расстояние падает
//      монотонно и существенно за N предъявлений).
//   3) Слова, которые НИКОГДА не встречаются рядом, не сближаются
//      (расстояние остаётся на уровне случайного хеша — обучение не
//      "размазывает" всё пространство в одну точку).
//
// Запуск: npx tsx scripts/bench-encoder.ts
// ═══════════════════════════════════════════════════════════════════

import { TrainableEncoder } from '../src/lib/neuromorphic/encoder';

const GRID = 60;

function fnv1aRef(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function wrapRef(x: number, g: number): number {
  return ((x % g) + g) % g;
}

// Старая формула из neural-core.ts (до v2.4.3) БЕЗ поправки на знак —
// именно так когда-то вычислялись baseR/baseC перед вызовом idx().
function refCoordsRaw(stem: string): [number, number, number] {
  const h = fnv1aRef(stem);
  return [h % GRID, (h >> 6) % GRID, (h >> 12) % GRID];
}

let failures = 0;
function check(name: string, cond: boolean, detail?: string): void {
  const mark = cond ? 'OK ' : 'FAIL';
  console.log(`[${mark}] ${name}${detail ? ' — ' + detail : ''}`);
  if (!cond) failures++;
}

// ── 1) Бутстрап согласован со старым хешем — с точностью до тороидальной
//        нормализации (см. находку ниже) ──
{
  const enc = new TrainableEncoder(GRID);
  const words = ['кошка', 'собака', 'нейрон', 'python', 'гравитация', 'квант'];
  let allMatch = true;
  let foundNegative = false;
  for (const w of words) {
    const got = enc.getIntCoords(w);
    const raw = refCoordsRaw(w);
    if (raw[1] < 0 || raw[2] < 0) foundNegative = true;
    const want: [number, number, number] = [wrapRef(raw[0], GRID), wrapRef(raw[1], GRID), wrapRef(raw[2], GRID)];
    if (got[0] !== want[0] || got[1] !== want[1] || got[2] !== want[2]) {
      allMatch = false;
      console.log(`  mismatch ${w}: got=${got} want(wrapped)=${want} raw=${raw}`);
    }
  }
  check('бутстрап = старая FNV-1a формула по модулю (после нормализации знака)', allMatch);
  // ПОБОЧНАЯ НАХОДКА: (h >> 6) — это ЗНАКОВЫЙ сдвиг 32-битного числа в JS.
  // Если у хеша установлен старший бит, (h>>6)%GRID может быть ОТРИЦАТЕЛЬНЫМ.
  // Старый код (neural-core.ts до v2.4.3) передавал это напрямую в
  // idx(l,r,c) = (l*GRID+r)*GRID+c БЕЗ нормализации — при достаточно
  // отрицательном r индекс мог уйти в отрицательную область и запись в
  // Float32Array молча терялась (no-op на отрицательный индекс), то есть
  // часть стимуляции слова могла просто исчезать в никуда. Новый
  // TrainableEncoder всегда нормализует координату через wrap() — этот
  // класс багов для входного кодирования исчезает как побочный эффект
  // перехода на обучаемые координаты.
  console.log(`  [находка] среди проверенных слов встретился отрицательный "сырой" компонент: ${foundNegative ? 'да — старый код мог терять часть стимуляции; новый корректно нормализует' : 'нет в этой выборке'}`);
}

// ── 2) Совместно встречающиеся слова сближаются ──
{
  const enc = new TrainableEncoder(GRID);
  const before = enc.distance('кошка', 'диван');
  const CORPUS = [
    ['кошка', 'спит', 'на', 'диван'],
    ['кошка', 'лежит', 'на', 'диван'],
    ['кошка', 'снова', 'на', 'диван'],
    ['моя', 'кошка', 'любит', 'диван'],
    ['кошка', 'и', 'диван', 'весь', 'день'],
    ['кошка', 'спит', 'на', 'диван'],
    ['кошка', 'опять', 'на', 'диван'],
    ['кошка', 'дремлет', 'на', 'диван'],
  ];
  const distances: number[] = [before];
  for (const sentence of CORPUS) {
    enc.coOccurLearn(sentence);
    distances.push(enc.distance('кошка', 'диван'));
  }
  const after = distances[distances.length - 1];
  console.log(`  distance(кошка, диван): ${distances.map(d => d.toFixed(2)).join(' → ')}`);
  check('расстояние сближающихся слов заметно упало', after < before * 0.5,
    `${before.toFixed(2)} → ${after.toFixed(2)}`);
  // Монотонность не требуем строго (возможны локальные колебания из-за
  // разных пар в предложении), но общий тренд должен быть убывающим:
  // сравниваем среднее первой половины прогонов со средней второй.
  const mid = Math.floor(distances.length / 2);
  const firstHalfAvg = distances.slice(0, mid).reduce((s, v) => s + v, 0) / mid;
  const secondHalfAvg = distances.slice(mid).reduce((s, v) => s + v, 0) / (distances.length - mid);
  check('тренд сближения устойчив (вторая половина < первой)', secondHalfAvg < firstHalfAvg,
    `${firstHalfAvg.toFixed(2)} → ${secondHalfAvg.toFixed(2)}`);
}

// ── 3) Несвязанные слова НЕ сближаются от чужого обучения ──
{
  const enc = new TrainableEncoder(GRID);
  const before = enc.distance('квант', 'фотосинтез'); // ни разу не встречаются вместе
  const CORPUS = [
    ['кошка', 'спит', 'на', 'диван'],
    ['собака', 'гуляет', 'в', 'парке'],
    ['python', 'язык', 'программирования'],
    ['гравитация', 'притягивает', 'тела'],
  ];
  for (const s of CORPUS) enc.coOccurLearn(s);
  // «квант» и «фотосинтез» ни разу не встретились ни в одном предложении
  // и даже не были инициализированы — расстояние должно остаться
  // ТОЧНО таким же, как при чистом бутстрапе (обучение не затрагивает
  // слова, которых не было в тексте).
  const after = enc.distance('квант', 'фотосинтез');
  check('несвязанные слова не сдвигаются от чужого обучения', Math.abs(after - before) < 1e-9,
    `${before.toFixed(4)} → ${after.toFixed(4)}`);
}

// ── 4) Персистентность: serialize → deserialize восстанавливает состояние ──
{
  const enc = new TrainableEncoder(GRID);
  enc.coOccurLearn(['альфа', 'бета', 'гамма']);
  enc.coOccurLearn(['альфа', 'бета']);
  const before = enc.distance('альфа', 'бета');
  const vocabBefore = enc.vocabularySize;
  const data = enc.serialize();
  const restored = TrainableEncoder.deserialize(data, GRID);
  const after = restored.distance('альфа', 'бета');
  check('serialize/deserialize сохраняет обученные координаты', after === before,
    `${before} vs ${after}`);
  check('serialize/deserialize сохраняет размер словаря', restored.vocabularySize === vocabBefore,
    `${vocabBefore} vs ${restored.vocabularySize}`);
}

// ── 5) Стресс: долгая эксплуатация не уводит координаты за границы
//        решётки и не порождает NaN/Infinity (важно — это состояние
//        накапливается ГОДАМИ работы сервера, без перезапуска) ──
{
  const enc = new TrainableEncoder(GRID);
  const VOCAB = ['альфа', 'бета', 'гамма', 'дельта', 'эпсилон', 'дзета', 'эта', 'тета'];
  function seededRandom(seed: number) {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0xffffffff; };
  }
  const rnd = seededRandom(2024);
  let ok = true;
  for (let round = 0; round < 20000; round++) {
    // Случайное "предложение" из 2-5 слов словаря — имитация многолетней
    // эксплуатации с постоянно повторяющимися сочетаниями.
    const n = 2 + Math.floor(rnd() * 4);
    const sentence: string[] = [];
    for (let i = 0; i < n; i++) sentence.push(VOCAB[Math.floor(rnd() * VOCAB.length)]);
    enc.coOccurLearn(sentence);
    for (const w of sentence) {
      const c = enc.getIntCoords(w);
      for (const v of c) {
        if (!Number.isFinite(v) || v < 0 || v >= GRID) { ok = false; break; }
      }
    }
    if (!ok) break;
  }
  check('20000 раундов обучения: координаты остаются в [0,GRID) и конечны', ok);
}

console.log(failures === 0 ? '\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ✓' : `\n${failures} ПРОВЕРОК ПРОВАЛЕНО ✗`);
process.exitCode = failures === 0 ? 0 : 1;
