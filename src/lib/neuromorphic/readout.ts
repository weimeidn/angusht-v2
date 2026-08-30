// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — AnswerReadout: обучаемый выбор ВАРИАНТА формулировки
// ═══════════════════════════════════════════════════════════════════
// Честная архитектура (README) до сих пор гласила: "нейросеть не
// генерирует текст, только маршрутизацию". Это оставалось потолком
// возможностей — сеть решала, КАКОЙ модуль отвечает и в каком порядке,
// но не влияла на то, КАКОЙ именно текст из уже готового набора
// вернуть. Многие темы в KNOWLEDGE хранят несколько формулировок
// (answers: string[]), но код всегда брал answers[0] — вторая и
// третья формулировки существовали только на бумаге.
//
// AnswerReadout — простой линейный readout поверх layerProfile
// ассоциативного ядра: для каждой темы хранится обучаемый вектор
// предпочтений по вариантам (весы), который заводится нейронным
// "затравочным" сигналом (какие слои ассоциативного ядра активны) и
// подстраивается через reinforcement (тот же принцип, что и остальная
// система: успешный ответ = положительное подкрепление).
//
// Это НЕ значит, что сеть теперь "пишет текст" — она по-прежнему
// выбирает из готового, заранее написанного набора. Но выбор ЭТОГО
// готового варианта теперь — функция нейронной активности и обучения,
// а не всегда одна и та же захардкоженная строка. Это реальный, хоть
// и скромный шаг за пределы "сеть только маршрутизирует".
// ═══════════════════════════════════════════════════════════════════

export interface ReadoutData {
  weights: Record<string, number[]>;
}

const REINFORCE_POSITIVE = 0.08;
const REINFORCE_NEGATIVE = 0.05;
const NEURAL_BIAS_SCALE = 0.3;

export class AnswerReadout {
  weights: Map<string, Float32Array> = new Map();

  private getWeights(topic: string, n: number): Float32Array {
    let w = this.weights.get(topic);
    if (!w || w.length !== n) {
      w = new Float32Array(n).fill(0);
      this.weights.set(topic, w);
    }
    return w;
  }

  // Выбирает индекс варианта формулировки (0..n-1) для темы topic.
  // associativeProfile — layerProfile ассоциативного ядра текущего
  // запроса (60 значений, активность по слоям). Используется как
  // нейронная "затравка": делит профиль на n сегментов, суммарная
  // активность сегмента i задаёт начальный вес варианта i — пока
  // readout ещё не обучен, выбор всё равно грунтуется в реальной
  // нейронной активности, а не в случайности или всегда answers[0].
  select(topic: string, n: number, associativeProfile: number[] | undefined): number {
    if (n <= 1) return 0;
    const w = this.getWeights(topic, n);

    const neuralBias = new Float32Array(n);
    if (associativeProfile && associativeProfile.length > 0) {
      const bucket = Math.max(1, Math.floor(associativeProfile.length / n));
      for (let i = 0; i < n; i++) {
        let s = 0;
        const from = i * bucket;
        const to = i === n - 1 ? associativeProfile.length : Math.min((i + 1) * bucket, associativeProfile.length);
        for (let j = from; j < to; j++) s += associativeProfile[j];
        neuralBias[i] = s;
      }
      let maxB = 0;
      for (let i = 0; i < n; i++) if (neuralBias[i] > maxB) maxB = neuralBias[i];
      if (maxB > 0) {
        for (let i = 0; i < n; i++) neuralBias[i] = (neuralBias[i] / maxB) * NEURAL_BIAS_SCALE;
      }
    }

    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < n; i++) {
      const score = w[i] + neuralBias[i];
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  // Подкрепление выбранного варианта. Как и остальная система, сигнал —
  // "ответ успешно выдан", а не "пользователь подтвердил, что доволен"
  // (у Angusht нет явной обратной связи от пользователя) — та же
  // эпистемическая граница, что и у остального STDP-подкрепления в
  // системе, честно фиксируем её здесь же, а не только в README.
  reinforce(topic: string, variantIndex: number, n: number, positive: boolean): void {
    if (n <= 1 || variantIndex < 0 || variantIndex >= n) return;
    const w = this.getWeights(topic, n);
    w[variantIndex] += positive ? REINFORCE_POSITIVE : -REINFORCE_NEGATIVE;
  }

  serialize(): ReadoutData {
    const weights: Record<string, number[]> = {};
    for (const [k, v] of this.weights) weights[k] = Array.from(v);
    return { weights };
  }

  static deserialize(data: ReadoutData): AnswerReadout {
    const r = new AnswerReadout();
    for (const [k, v] of Object.entries(data.weights || {})) {
      r.weights.set(k, Float32Array.from(v));
    }
    return r;
  }
}
