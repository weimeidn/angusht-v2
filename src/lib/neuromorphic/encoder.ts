// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4.3 — TrainableEncoder: обучаемое (не хеш-based) кодирование
// слов в координаты нейронной решётки.
// ═══════════════════════════════════════════════════════════════════
// Было (до v2.4.3): stimulateText() хешировал стем слова через FNV-1a
// в координаты [l,r,c] решётки 60×60×60. Хеш детерминирован, но
// ЗАФИКСИРОВАН НАВСЕГДА: "кошка" и "собака" занимают одни и те же
// (псевдослучайные) координаты хоть в первый день работы системы, хоть
// через год — расположение слова в пространстве нейронов никак не
// связано с тем, как оно реально употребляется. Ручной список
// синонимов (lexicon.ts) частично компенсирует это, но покрывает
// только заранее размеченные пары — координата слова сама по себе
// ничему не учится.
//
// Стало: координата слова — не хеш, а ОБУЧАЕМОЕ состояние:
//   1) При первой встрече слова координата бутстрапится тем же FNV-1a,
//      что и раньше — детерминированный, воспроизводимый старт (а не
//      случайный разброс с нуля, который был бы недетерминирован
//      между запусками и не имел бы вообще никакой начальной
//      структуры до первого обучения).
//   2) При каждом запросе слова, встретившиеся ВМЕСТЕ в одном тексте,
//      слегка притягиваются друг к другу в координатах решётки —
//      self-organizing-map: движение по кратчайшему пути тороидальной
//      (закольцованной) решётки, чтобы направление сближения не
//      зависело от того, с какой стороны решётки хеш изначально
//      разместил слово.
//   3) Learning rate убывает с числом встреч слова (annealing, как в
//      SOM/Kohonen): часто встречающиеся слова стабилизируются и
//      перестают «убегать», редкие остаются подвижными.
// Результат: слова, которые реально употребляются рядом в запросах
// пользователя, со временем физически сближаются в решётке — их
// стимуляция начинает перекрываться в общих нейронах — а не только
// через заранее вручную прописанный список синонимов. Это по-прежнему
// не градиентный спуск и не нейросетевые эмбеддинги (word2vec/BERT) —
// честная, объяснимая, устойчиво воспроизводимая (при одинаковой
// последовательности запросов) адаптация вместо статичного хеша.
// Персистентность (persistence.ts) сохраняет обученные координаты —
// они переживают перезапуск сервера так же, как и веса синапсов.
// ═══════════════════════════════════════════════════════════════════

const ATTRACT_LR = 0.15;   // базовый шаг притяжения (доля кратчайшего расстояния)
const LR_DECAY = 0.05;     // чем больше встреч слова — тем меньше эффективный lr

export interface EncoderData {
  // стем → [l, r, c, счётчик встреч]
  coords: Record<string, [number, number, number, number]>;
}

function wrap(x: number, g: number): number {
  return ((x % g) + g) % g;
}

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export class TrainableEncoder {
  private coords: Map<string, [number, number, number]> = new Map();
  private seenCount: Map<string, number> = new Map();
  private readonly grid: number;

  constructor(grid: number) {
    this.grid = grid;
  }

  // Детерминированный бутстрап координаты — используется ОДИН РАЗ, при
  // первой встрече слова, как начальная точка для дальнейшего обучения
  // (не как постоянная хеш-функция, как было раньше).
  private bootstrap(stem: string): [number, number, number] {
    const h = fnv1a(stem);
    return [h % this.grid, (h >> 6) % this.grid, (h >> 12) % this.grid];
  }

  private getFloatCoords(stem: string): [number, number, number] {
    let c = this.coords.get(stem);
    if (!c) {
      c = this.bootstrap(stem);
      this.coords.set(stem, c);
      this.seenCount.set(stem, 0);
    }
    return c;
  }

  // Целочисленные координаты для индексации нейрона — округление
  // текущего (возможно дробного из-за обучения) положения слова.
  getIntCoords(stem: string): [number, number, number] {
    const c = this.getFloatCoords(stem);
    const g = this.grid;
    return [
      wrap(Math.round(c[0]), g),
      wrap(Math.round(c[1]), g),
      wrap(Math.round(c[2]), g),
    ];
  }

  // Обучение на совместной встречаемости слов одного запроса: каждая
  // пара стемов из текста слегка притягивается друг к другу.
  coOccurLearn(stems: string[]): void {
    const unique = Array.from(new Set(stems));
    for (const s of unique) this.getFloatCoords(s); // гарантируем наличие координат
    for (let i = 0; i < unique.length; i++) {
      for (let j = i + 1; j < unique.length; j++) {
        this.attract(unique[i], unique[j]);
      }
    }
    for (const s of unique) {
      this.seenCount.set(s, (this.seenCount.get(s) ?? 0) + 1);
    }
  }

  private attract(a: string, b: string): void {
    const g = this.grid;
    const ca = this.coords.get(a)!;
    const cb = this.coords.get(b)!;
    const na = this.seenCount.get(a) ?? 0;
    const nb = this.seenCount.get(b) ?? 0;
    const lrA = ATTRACT_LR / (1 + na * LR_DECAY);
    const lrB = ATTRACT_LR / (1 + nb * LR_DECAY);
    const nextA: [number, number, number] = [0, 0, 0];
    const nextB: [number, number, number] = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      // Кратчайший путь по закольцованной решётке (тороид): если прямой
      // путь длиннее половины решётки, идём в обход через "край".
      let d = cb[k] - ca[k];
      const half = g / 2;
      if (d > half) d -= g;
      if (d < -half) d += g;
      nextA[k] = wrap(ca[k] + d * lrA, g);
      nextB[k] = wrap(cb[k] - d * lrB, g);
    }
    this.coords.set(a, nextA);
    this.coords.set(b, nextB);
  }

  // Тороидальное евклидово расстояние между текущими координатами двух
  // стемов. Используется диагностикой/регрессионным тестом, чтобы
  // ОБЪЕКТИВНО показать: слова, встречающиеся рядом, реально сближаются
  // (а не просто "код скомпилировался").
  distance(a: string, b: string): number {
    const g = this.grid;
    const ca = this.getFloatCoords(a);
    const cb = this.getFloatCoords(b);
    let sumSq = 0;
    for (let k = 0; k < 3; k++) {
      let d = Math.abs(cb[k] - ca[k]);
      if (d > g / 2) d = g - d;
      sumSq += d * d;
    }
    return Math.sqrt(sumSq);
  }

  get vocabularySize(): number {
    return this.coords.size;
  }

  serialize(): EncoderData {
    const coords: Record<string, [number, number, number, number]> = {};
    for (const [stem, c] of this.coords) {
      coords[stem] = [c[0], c[1], c[2], this.seenCount.get(stem) ?? 0];
    }
    return { coords };
  }

  static deserialize(data: EncoderData, grid: number): TrainableEncoder {
    const enc = new TrainableEncoder(grid);
    enc.loadFrom(data);
    return enc;
  }

  // Загрузка обученного состояния В СУЩЕСТВУЮЩИЙ экземпляр — нужно для
  // sharedEncoder (neural-core.ts): это модульный singleton, на который
  // другие модули уже держат ссылку по значению переменной, поэтому его
  // состояние нужно мутировать на месте, а не подменять новым объектом
  // (см. NeuralCore.loadWeights — тот же паттерн для весов синапсов).
  loadFrom(data: EncoderData): void {
    this.coords.clear();
    this.seenCount.clear();
    for (const [stem, [l, r, c, n]] of Object.entries(data.coords)) {
      this.coords.set(stem, [l, r, c]);
      this.seenCount.set(stem, n);
    }
  }
}
