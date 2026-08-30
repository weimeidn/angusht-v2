// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — SemanticSpace: растущее распределительное семантическое
// пространство, построенное на статистике корпуса, а не на готовой
// предобученной модели.
// ═══════════════════════════════════════════════════════════════════
// Что это: каждый новый вопрос-ответ, сохранённый в память, индексируется
// сюда. Пространство накапливает:
//   - document frequency (docFreq) — в скольких сохранённых текстах
//     встретилось слово → TF-IDF-подобное взвешивание (редкие,
//     специфичные слова значат больше, чем частицы и общие слова).
//   - co-occurrence (cooc) — какие слова встречаются рядом друг с
//     другом в пределах окна ±3 токена → грубая, но настоящая,
//     дистрибутивная семантика ("если два слова часто стоят рядом
//     в текстах системы — они, вероятно, по смыслу связаны").
// Это НЕ нейросетевые эмбеддинги (word2vec/BERT и т.п.) — здесь нет
// обученной на больших данных модели и градиентного спуска. Но это
// честная, работающая и практичная замена: пространство растёт вместе
// с памятью системы (тот же принцип самообучения, что и остальной
// системе), не требует внешних зависимостей и весов, и переживает
// перезапуск сервера, так как персистентность (persistence.ts)
// сохраняет docFreq/cooc в memory.json вместе с остальным.
// ═══════════════════════════════════════════════════════════════════

import { tokenizeNormalized } from './lexicon';

const WINDOW = 3;
const MAX_VOCAB = 4000;
const MAX_NEIGHBORS_PER_WORD = 80;

export interface SemanticSpaceData {
  totalDocs: number;
  docFreq: Record<string, number>;
  cooc: Record<string, Record<string, number>>;
}

export class SemanticSpace {
  totalDocs = 0;
  docFreq: Map<string, number> = new Map();
  cooc: Map<string, Map<string, number>> = new Map();

  // Индексирует один текст (например "вопрос + ответ") в пространство.
  // Вызывается при каждом новом сохранении в память — это и есть
  // "рост" семантического пространства по мере самообучения системы.
  indexText(text: string): void {
    const tokens = tokenizeNormalized(text);
    if (tokens.length === 0) return;
    this.totalDocs++;

    const uniq = new Set(tokens);
    for (const t of uniq) {
      this.docFreq.set(t, (this.docFreq.get(t) || 0) + 1);
    }

    for (let i = 0; i < tokens.length; i++) {
      const wi = tokens[i];
      const from = Math.max(0, i - WINDOW);
      const to = Math.min(tokens.length - 1, i + WINDOW);
      for (let j = from; j <= to; j++) {
        if (j === i) continue;
        this.bump(wi, tokens[j]);
      }
    }

    this.prune();
  }

  private bump(a: string, b: string): void {
    let m = this.cooc.get(a);
    if (!m) {
      m = new Map();
      this.cooc.set(a, m);
    }
    m.set(b, (m.get(b) || 0) + 1);
    if (m.size > MAX_NEIGHBORS_PER_WORD * 1.5) {
      // Периодически подрезаем самых редких соседей, чтобы карта
      // не росла неограниченно для очень частых слов.
      const sorted = [...m.entries()].sort((x, y) => y[1] - x[1]);
      m.clear();
      for (const [k, v] of sorted.slice(0, MAX_NEIGHBORS_PER_WORD)) m.set(k, v);
    }
  }

  // Ограничивает словарь по общему числу термов, отбрасывая самые
  // редкие — чтобы файл на диске и память процесса не росли бесконечно.
  private prune(): void {
    if (this.docFreq.size <= MAX_VOCAB) return;
    const entries = [...this.docFreq.entries()].sort((a, b) => a[1] - b[1]);
    const toDrop = entries.slice(0, entries.length - MAX_VOCAB);
    for (const [w] of toDrop) {
      this.docFreq.delete(w);
      this.cooc.delete(w);
    }
  }

  idf(term: string): number {
    const df = this.docFreq.get(term) || 0;
    return Math.log((this.totalDocs + 1) / (df + 1)) + 1;
  }

  neighbors(term: string, topK = 5): string[] {
    const m = this.cooc.get(term);
    if (!m) return [];
    return [...m.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([w]) => w);
  }

  // Взвешенное (по IDF) пересечение двух наборов токенов — направленная
  // мера "насколько targetTokens покрывает то, что важно в queryTokens".
  // Используется вместо сырого Jaccard по несклеенным словоформам.
  private directionalOverlap(queryTokens: string[], targetTokens: string[]): number {
    if (queryTokens.length === 0 || targetTokens.length === 0) return 0;
    const targetSet = new Set(targetTokens);
    let matched = 0;
    let total = 0;
    for (const t of queryTokens) {
      const w = this.idf(t);
      total += w;
      if (targetSet.has(t)) matched += w;
    }
    return total > 0 ? matched / total : 0;
  }

  // Симметричная версия (среднее двух направлений) — устойчивее к
  // разнице в длине сравниваемых текстов.
  overlapScore(tokensA: string[], tokensB: string[]): number {
    const ab = this.directionalOverlap(tokensA, tokensB);
    const ba = this.directionalOverlap(tokensB, tokensA);
    return (ab + ba) / 2;
  }

  serialize(): SemanticSpaceData {
    const cooc: Record<string, Record<string, number>> = {};
    for (const [k, m] of this.cooc) cooc[k] = Object.fromEntries(m);
    return {
      totalDocs: this.totalDocs,
      docFreq: Object.fromEntries(this.docFreq),
      cooc,
    };
  }

  static deserialize(data: SemanticSpaceData): SemanticSpace {
    const s = new SemanticSpace();
    s.totalDocs = data.totalDocs || 0;
    s.docFreq = new Map(Object.entries(data.docFreq || {}));
    s.cooc = new Map();
    for (const [k, v] of Object.entries(data.cooc || {})) {
      s.cooc.set(k, new Map(Object.entries(v)));
    }
    return s;
  }
}
