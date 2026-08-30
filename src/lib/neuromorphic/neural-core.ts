// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — Neural Core: 60×60×60 решётка с реалистичными LIF-нейронами
// Типы: 80% возбуждающих, 20% тормозных
// STDP, рефрактерный период, spike-frequency adaptation, гомеостаз
// Lateral inhibition для Decision Core (WTA конкуренция групп)
// ═══════════════════════════════════════════════════════════════════

import {
  NeuronType, LIFParams, createDefaultParams,
} from './lif-neuron';
import { stemWord, expandWithSynonyms, STOP_WORDS } from './lexicon';
import { TrainableEncoder } from './encoder';

export const GRID = 60;

// ── Обучаемое кодирование слов (см. encoder.ts, v2.4.3) ──
// Единый на процесс экземпляр — используется ВСЕМИ ядрами (см.
// stimulateText ниже), чтобы координата слова означала одно и то же
// "семантическое место" решётки в КАЖДОМ из 6 ядер, как это было и при
// чистом хеше раньше (одна и та же FNV-1a формула, общий GRID=60 для
// всех CORE_CONFIGS). Персистентность (persistence.ts) сохраняет и
// восстанавливает обученное состояние этого объекта между запусками.
export const sharedEncoder = new TrainableEncoder(GRID);
export const TOTAL_NEURONS = GRID * GRID * GRID; // 216 000
export const DIRECTIONS: number[][] = [
  [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
];
export const NDIR = DIRECTIONS.length;
export const INHIBITORY_RATIO = 0.2;

export interface CoreConfig {
  id: string;
  name: string;
  role: CoreRole;
  color: string;
  params: {
    excitatory: Partial<LIFParams>;
    inhibitory: Partial<LIFParams>;
  };
}

export enum CoreRole {
  Sensory = 'sensory',
  Associative = 'associative',
  Analytical = 'analytical',
  Temporal = 'temporal',
  Executive = 'executive',
  Modulatory = 'modulatory',
}

export const CORE_CONFIGS: CoreConfig[] = [
  {
    id: 'sensory', name: 'Sensory Core', role: CoreRole.Sensory, color: '#f59e0b',
    params: {
      excitatory: { threshold: 0.85, decay: 0.82, refractoryPeriod: 2, stdpLtpRate: 0.05, stdpWindow: 25 },
      inhibitory: { threshold: 0.75, decay: 0.78, refractoryPeriod: 1 },
    },
  },
  {
    id: 'associative', name: 'Associative Core', role: CoreRole.Associative, color: '#10b981',
    params: {
      excitatory: { threshold: 0.9, decay: 0.88, refractoryPeriod: 3, stdpLtpRate: 0.06, stdpWindow: 30, adaptationRate: 0.015 },
      inhibitory: { threshold: 0.8, decay: 0.80, refractoryPeriod: 2 },
    },
  },
  {
    id: 'analytical', name: 'Analytical Core', role: CoreRole.Analytical, color: '#3b82f6',
    params: {
      excitatory: { threshold: 1.1, decay: 0.90, refractoryPeriod: 4, stdpLtpRate: 0.03, stdpWindow: 20, adaptationRate: 0.025 },
      inhibitory: { threshold: 1.0, decay: 0.85, refractoryPeriod: 3 },
    },
  },
  {
    id: 'temporal', name: 'Temporal Core', role: CoreRole.Temporal, color: '#8b5cf6',
    params: {
      excitatory: { threshold: 0.95, decay: 0.92, refractoryPeriod: 3, stdpLtpRate: 0.04, stdpWindow: 35, adaptationRate: 0.01, adaptationDecay: 0.97 },
      inhibitory: { threshold: 0.85, decay: 0.87, refractoryPeriod: 2 },
    },
  },
  {
    id: 'executive', name: 'Executive Core', role: CoreRole.Executive, color: '#ef4444',
    params: {
      excitatory: { threshold: 0.8, decay: 0.83, refractoryPeriod: 2, stdpLtpRate: 0.035, stdpWindow: 22 },
      inhibitory: { threshold: 0.7, decay: 0.76, refractoryPeriod: 2 },
    },
  },
  {
    id: 'modulatory', name: 'Decision Core', role: CoreRole.Modulatory, color: '#ec4899',
    params: {
      // Decision Core: включено STDP для обучения маршрутов.
      // Порог ниже → чувствительнее к проекциям от ядер 1-5.
      // Decay ниже → активность затухает быстрее, чётче WTA.
      excitatory: { threshold: 0.85, decay: 0.85, refractoryPeriod: 3, stdpLtpRate: 0.07, stdpWindow: 25, adaptationRate: 0.015, adaptationDecay: 0.95 },
      inhibitory: { threshold: 0.75, decay: 0.80, refractoryPeriod: 2 },
    },
  },
];

export interface SpikeEvent {
  tick: number;
  coreId: string;
  layer: number;
  row: number;
  col: number;
  strength: number;
  type: NeuronType;
}

export interface CoreSnapshot {
  id: string;
  name: string;
  role: string;
  color: string;
  total: number;
  active: number;
  events: number;
  tick: number;
  avgWeight: number;
  strongSynapses: number;
  firedThisCycle: number;
  inhibitoryFired: number;
  avgPotential: number;
  layerProfile: number[];
  lastSpikes: SpikeEvent[];
}

export class NeuralCore {
  readonly id: string;
  readonly name: string;
  readonly role: CoreRole;
  readonly color: string;
  readonly total = TOTAL_NEURONS;

  // Нейродинамика
  potentials: Float32Array;          // мембранные потенциалы
  refractories: Uint8Array;          // счётчики рефрактерного периода
  adaptations: Float32Array;         // уровень адаптации
  lastSpikes: Int32Array;            // время последнего спайка
  spikeCounts: Uint32Array;          // общее число спайков
  neuronTypes: Uint8Array;           // 0=excitatory, 1=inhibitory

  // Синапсы: weights[neuron_idx * NDIR + direction]
  weights: Float32Array;
  weightSum = 0;
  strongCount = 0;
  readonly WMIN = 0.01;
  readonly WMAX = 1.5;

  // Состояние симуляции
  tick = 0;
  events = 0;
  activeSet = new Set<number>();
  lastSpikeEvents: SpikeEvent[] = [];

  // ── Разреженное обновление (v2.4.2) ──
  // Инвариант: если индекс НЕ входит в liveSet, то для него
  // potentials[i] === 0, adaptations[i] === 0 и refractories[i] === 0.
  // Раньше simulateTick() на каждый тик проходил по ВСЕМ 216 000
  // нейронам ядра дважды (Фаза 1 и Фаза 3), даже если реально
  // затронута была пара сотен — расширение ядра (больше ядер/крупнее
  // решётка/больше раундов каскада) масштабировалось от общего числа
  // нейронов, а не от активности. liveSet хранит только нейроны,
  // которые физически МОГУТ измениться на следующем тике (см.
  // simulateTick — вход, ненулевой потенциал/адаптация, рефрактерность);
  // всё остальное гарантированно находится в состоянии покоя и его
  // незачем обходить. Поддерживается в stimulate()/receiveInput()
  // (внешние источники входа) и в самом simulateTick().
  liveSet = new Set<number>();

  // Параметры
  excParams: LIFParams;
  inhParams: LIFParams;

  // Для STDP: храним времена спайков предсинаптических нейронов
  private pendingInputs: Map<number, number> = new Map();

  // Гомеостаз
  private homeostaticWindow = 100;
  private homeostaticCounters: Uint32Array;

  constructor(config: CoreConfig) {
    this.id = config.id;
    this.name = config.name;
    this.role = config.role;
    this.color = config.color;

    const n = TOTAL_NEURONS;
    this.potentials = new Float32Array(n);
    this.refractories = new Uint8Array(n);
    this.adaptations = new Float32Array(n);
    this.lastSpikes = new Int32Array(n).fill(-100);
    this.spikeCounts = new Uint32Array(n);
    this.neuronTypes = new Uint8Array(n);
    this.homeostaticCounters = new Uint32Array(n);

    // Назначаем типы: 80% возбуждающих, 20% тормозных
    // Используем детерминированный паттерн на основе позиции
    for (let i = 0; i < n; i++) {
      const l = Math.floor(i / 3600);
      const r = Math.floor((i % 3600) / 60);
      const c = i % 60;
      // Чередующийся паттерн с пространственной структурой
      this.neuronTypes[i] = ((l + r * 3 + c * 7) % 5 === 0) ? 1 : 0;
    }

    // Инициализация весов
    const numSyn = n * NDIR;
    this.weights = new Float32Array(numSyn);
    for (let i = 0; i < numSyn; i++) {
      this.weights[i] = 0.15 + Math.random() * 0.2;
    }
    this.recalcWeightStats();

    // Параметры
    this.excParams = createDefaultParams(NeuronType.Excitatory);
    this.inhParams = createDefaultParams(NeuronType.Inhibitory);
    if (config.params.excitatory) Object.assign(this.excParams, config.params.excitatory);
    if (config.params.inhibitory) Object.assign(this.inhParams, config.params.inhibitory);

    // Сохраняем базовые пороги для delta-модели нейромодуляции
    this.baseExcThreshold = this.excParams.threshold;
    this.baseInhThreshold = this.inhParams.threshold;
    this.baseExcDecay = this.excParams.decay;
    this.baseInhDecay = this.inhParams.decay;
  }

  private idx(l: number, r: number, c: number): number {
    return (l * GRID + r) * GRID + c;
  }

  // Нормализация координаты в [0, GRID) — см. находку про знаковый сдвиг
  // (h >> 6)/(h >> 12) выше: без неё idx(l,r,c) мог уйти в отрицательный
  // индекс, который Float32Array молча "принимает" как обычное свойство
  // объекта (не как элемент буфера) — запись в никуда, а если такой
  // индекс попадает в liveSet/activeSet, чтение назад даёт undefined,
  // и любая арифметика с ним (undefined + x) даёт NaN, заражающее
  // totalPotential в snapshot() на весь срок жизни процесса.
  private wrapCoord(x: number, mod: number = GRID): number {
    return ((x % mod) + mod) % mod;
  }

  private pos(i: number): [number, number, number] {
    return [
      Math.floor(i / 3600),
      Math.floor((i % 3600) / 60),
      i % 60,
    ];
  }

  private neighbor(i: number, d: number): number | null {
    if (d < 0 || d >= DIRECTIONS.length) return null;
    const dir = DIRECTIONS[d];
    if (!dir) return null;
    const dl = dir[0], dr = dir[1], dc = dir[2];
    const [l, r, c] = this.pos(i);
    const nl = l + dl, nr = r + dr, nc = c + dc;
    if (nl >= 0 && nl < GRID && nr >= 0 && nr < GRID && nc >= 0 && nc < GRID) {
      return this.idx(nl, nr, nc);
    }
    return null;
  }

  private recalcWeightStats() {
    this.weightSum = 0;
    this.strongCount = 0;
    const strongThresh = 0.6;
    for (let i = 0; i < this.weights.length; i++) {
      this.weightSum += this.weights[i];
      if (this.weights[i] >= strongThresh) this.strongCount++;
    }
  }

  // Стимуляция нейрона
  stimulate(neuronIdx: number, strength: number = 1.05): void {
    this.potentials[neuronIdx] += strength;
    this.liveSet.add(neuronIdx);
    this.events++;
  }

  // Приём сигнала от ДРУГОГО ядра (межъядерная проекция). Раньше
  // CoreCluster писал прямо в core.potentials[idx] и добавлял idx в
  // core.activeSet в обход внутреннего состояния ядра. С разреженным
  // обновлением (liveSet выше) такая прямая запись "теряла" бы нейрон —
  // Фаза 1/3 следующего тика просто не узнала бы о нём. Поэтому любой
  // внешний источник входа (CoreCluster) обязан идти через этот метод,
  // а не писать в potentials напрямую.
  receiveInput(neuronIdx: number, amount: number): void {
    this.potentials[neuronIdx] += amount;
    this.liveSet.add(neuronIdx);
    this.activeSet.add(neuronIdx);
  }

  // Стимуляция текстом: FNV-1a хеш → координаты → стимуляция.
  //
  // До v2.4 слова хешировались как есть: "кошка" и "кошки" давали
  // РАЗНЫЕ координаты нейронов — ядро не имело никакой семантической
  // структуры на входе, только случайное (но детерминированное)
  // распределение. Теперь:
  //   1) слово приводится к основе (стему) ПЕРЕД хешированием — все
  //      словоформы одного слова стимулируют одни и те же нейроны;
  //   2) дополнительно, слабее, стимулируются координаты слов из той
  //      же синонимической группы (lexicon.ts) — "гравитация" и
  //      "тяготение" теперь частично перекрываются в пространстве
  //      нейронов, а не активируют полностью несвязанные зоны.
  // Это по-прежнему не обученные embeddings — а размеченная вручную,
  // но настоящая (не случайная) семантическая структура кодирования.
  stimulateText(text: string): number[] {
    // weight: индекс нейрона → сила стимуляции. Используем Map вместо
    // немедленного вызова stimulate() на каждый источник, чтобы, как и
    // в исходной версии, один и тот же нейрон (случайная коллизия хешей
    // биграммы/стема/синонима) стимулировался один раз — на максимум из
    // предложенных весов, а не накопительно от каждого источника.
    const weight = new Map<number, number>();
    const bump = (idx: number, w: number) => {
      const cur = weight.get(idx) ?? 0;
      if (w > cur) weight.set(idx, w);
    };

    const low = text.toLowerCase().trim();

    // Биграммы → нейроны. Символьный уровень намеренно НЕ стеммируется:
    // это отдельный, более "сырой" канал, устойчивый к опечаткам и
    // неизвестным словам, дополняющий смысловой канал ниже, а не
    // заменяющий его.
    for (let i = 0; i < low.length - 1; i++) {
      const bg = low.slice(i, i + 2);
      const h = this.fnv1a(bg);
      // (h >> 6) / (h >> 12) — ЗНАКОВЫЙ сдвиг 32-битного числа в JS. Если
      // у хеша установлен старший бит, результат может быть отрицательным
      // — без нормализации idx(l,r,c) уходил в отрицательный "теневой"
      // индекс за пределами Float32Array (см. wrapCoord — та же находка
      // и тот же фикс, что и в encoder.ts, TrainableEncoder.bootstrap).
      const l = h % GRID;
      const r = this.wrapCoord(h >> 6);
      const c = this.wrapCoord(h >> 12);
      const base = this.idx(l, r, c);
      bump(base, 1.05);
      // neighbor() уже отдельно защищён от d<0 (см. её реализацию), но
      // нормализуем и здесь — иначе "сырое" отрицательное d просто
      // молча теряло направление облака (nb === null), а не падало.
      const d = this.wrapCoord(h >> 18, NDIR);
      const nb = this.neighbor(base, d);
      if (nb !== null) bump(nb, 1.05);
    }

    // Слова → кластеры нейронов, через стем (не сырое слово).
    // "кошка" и "кошки" теперь дают ОДИНАКОВЫЕ координаты — раньше
    // (до v2.4) хешировалось сырое слово, и словоформы одного слова
    // случайно разлетались по несвязанным зонам решётки.
    const rawWords = low.match(/[a-zA-Zа-яА-ЯЁё0-9]{2,}/g) || [];
    const stems = new Set<string>();
    for (const w of rawWords) {
      if (STOP_WORDS.has(w)) continue; // частицы/предлоги не несут смысла — не шумим ими
      stems.add(stemWord(w));
    }

    for (const stem of stems) {
      // Координата стема больше не хеш, а ОБУЧАЕМОЕ состояние (см.
      // encoder.ts, v2.4.3) — при первой встрече слова бутстрапится тем
      // же FNV-1a, что и раньше, но затем дрейфует от совместной
      // встречаемости слов в реальных запросах.
      const [baseL, baseR, baseC] = sharedEncoder.getIntCoords(stem);
      bump(this.idx(baseL, baseR, baseC), 1.05);
      // «Облако» нейронов вокруг базового — распределённое представление
      for (const offset of [7, 13, 23, 31, 41, 53]) {
        const l2 = (baseL + offset) % GRID;
        bump(this.idx(l2, baseR, baseC), 1.05);
      }
    }

    // Синонимы стемов из текста — слабее, без "облака", только базовая
    // точка. Даёт ЧАСТИЧНОЕ (не полное) перекрытие активации для слов,
    // родственных по смыслу, но разных по написанию — "гравитация" и
    // "тяготение" теперь делят часть нейронов, а не активируют полностью
    // несвязанные зоны, как было при хешировании сырых слов. Это ручное,
    // заранее размеченное перекрытие (lexicon.ts) остаётся как есть —
    // обучаемый энкодер ниже дополняет его перекрытием, которое возникает
    // САМО, из статистики реальных запросов, а не только из словаря.
    const expanded = expandWithSynonyms([...stems]);
    for (const stem of expanded) {
      if (stems.has(stem)) continue; // уже простимулировано выше на полную силу
      const [baseL, baseR, baseC] = sharedEncoder.getIntCoords(stem);
      bump(this.idx(baseL, baseR, baseC), 0.4);
    }

    // Обучение кодировщика: слова, встретившиеся ВМЕСТЕ в этом тексте,
    // слегка притягиваются друг к другу в координатах решётки — для
    // БУДУЩИХ стимуляций (эта, текущая, уже посчитана выше по прежним
    // координатам). Синонимы намеренно не включены — это ручной,
    // отдельный от статистического обучения канал (см. комментарий выше).
    sharedEncoder.coOccurLearn([...stems]);

    for (const [idx, w] of weight) {
      this.stimulate(idx, w);
    }
    return Array.from(weight.keys());
  }

  private fnv1a(s: string): number {
    let h = 0x811C9DC5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  // Один тик симуляции — ОБЯЗАТЕЛЬНО асинхронный стиль
  //
  // v2.4.2: разреженное обновление. Раньше Фаза 1 и Фаза 3 проходили
  // по ВСЕМ this.total (216 000) нейронам ядра на каждый тик, а
  // `inputs` был Float32Array(this.total), аллоцируемым и обнуляемым
  // заново на каждый тик — то есть постоянная O(N) стоимость на тик
  // независимо от того, сколько нейронов реально было затронуто.
  // Теперь Фаза 1/3 обходят только `candidates` — liveSet (нейроны,
  // которые ещё "не улеглись" с прошлого тика: ненулевой потенциал/
  // адаптация/рефрактерность) плюс нейроны, получившие вход в этом
  // тике. Нейрон вне этого множества по инварианту liveSet (см. выше)
  // имеет potentials[i]=0, adaptations[i]=0, refractories[i]=0 и не
  // получает входа — его состояние физически не может измениться,
  // обходить его незачем. Результат симуляции (какие нейроны спайкают,
  // как меняются веса) идентичен прежней плотной версии — см.
  // scripts/bench-neural-core.ts, который сверяет это детерминированной
  // фикстурой при любом изменении этого файла.
  simulateTick(): { fired: number[]; events: SpikeEvent[] } {
    const fired: number[] = [];
    const spikeEvents: SpikeEvent[] = [];
    this.tick++;

    // Разреженные входы этого тика: индекс нейрона → накопленный вход.
    const inputs = new Map<number, number>();
    for (const [idx, val] of this.pendingInputs) {
      inputs.set(idx, (inputs.get(idx) ?? 0) + val);
    }
    this.pendingInputs.clear();

    // Кандидаты на обработку в этом тике.
    const candidates = new Set<number>(this.liveSet);
    for (const idx of inputs.keys()) candidates.add(idx);

    // Фаза 1: определяем, какие нейроны спайкают — среди кандидатов.
    const willFire = new Set<number>();
    for (const i of candidates) {
      if (this.refractories[i] > 0) continue;
      const isExc = this.neuronTypes[i] === 0;
      const params = isExc ? this.excParams : this.inhParams;
      const effectiveThreshold = params.threshold + this.adaptations[i];
      if (this.potentials[i] >= effectiveThreshold) {
        willFire.add(i);
      }
    }

    // Фаза 2: обрабатываем спайки
    for (const i of willFire) {
      const isExc = this.neuronTypes[i] === 0;
      const params = isExc ? this.excParams : this.inhParams;
      const output = isExc ? 1.0 : -1.0;

      // Записываем спайк
      fired.push(i);
      this.spikeCounts[i]++;
      this.lastSpikes[i] = this.tick;
      this.refractories[i] = params.refractoryPeriod;
      this.potentials[i] = params.resetPotential;
      this.adaptations[i] += params.adaptationRate;
      if (this.adaptations[i] > 0.5) this.adaptations[i] = 0.5;
      this.activeSet.add(i);
      this.events++;

      const [l, r, c] = this.pos(i);
      spikeEvents.push({
        tick: this.tick,
        coreId: this.id,
        layer: l + 1,
        row: r + 1,
        col: c + 1,
        strength: Math.abs(this.potentials[i]) + 1.0,
        type: isExc ? NeuronType.Excitatory : NeuronType.Inhibitory,
      });

      // Рассылаем сигнал по синапсам. Соседи становятся кандидатами
      // Фазы 3 этого же тика, даже если раньше были в состоянии покоя.
      for (let d = 0; d < NDIR; d++) {
        const j = this.neighbor(i, d);
        if (j === null) continue;
        const wi = i * NDIR + d;
        const w = this.weights[wi];
        inputs.set(j, (inputs.get(j) ?? 0) + output * w);
        candidates.add(j);
      }
    }

    // Фаза 3: обновляем потенциалы несработавших кандидатов и строим
    // liveSet для следующего тика (см. инвариант в объявлении поля).
    const nextLive = new Set<number>();
    for (const i of candidates) {
      if (willFire.has(i)) {
        // Только что спайкнул → вошёл в рефрактерный период → жив.
        nextLive.add(i);
        continue;
      }

      // Затухание рефрактерного периода
      if (this.refractories[i] > 0) {
        this.refractories[i]--;
        this.potentials[i] = this.potentials[i] * 0.5;
        nextLive.add(i); // ещё в рефрактерном периоде или потенциал не обнулился
        continue;
      }

      const isExc = this.neuronTypes[i] === 0;
      const params = isExc ? this.excParams : this.inhParams;

      // Применяем вход + затухание
      const inputVal = inputs.get(i) ?? 0;
      this.potentials[i] = this.potentials[i] * params.decay + inputVal;

      // Затухание адаптации
      this.adaptations[i] *= params.adaptationDecay;
      if (Math.abs(this.adaptations[i]) < 1e-4) this.adaptations[i] = 0;

      // Убираем negligible потенциалы
      if (Math.abs(this.potentials[i]) < 1e-4) this.potentials[i] = 0;

      if (this.potentials[i] !== 0 || this.adaptations[i] !== 0) {
        nextLive.add(i);
      }
    }
    this.liveSet = nextLive;

    // Фаза 4: STDP — обновляем веса на основе спайков
    this.applySTDP(fired, willFire);

    // Фаза 5: Геббовское обучение для сработавших путей
    for (const i of fired) {
      for (let d = 0; d < NDIR; d++) {
        const j = this.neighbor(i, d);
        if (j === null || !willFire.has(j)) continue;
        // Оба нейрона спайкали → усилить связь
        const wi = i * NDIR + d;
        const oldW = this.weights[wi];
        const newW = Math.min(this.WMAX, oldW + this.excParams.stdpLtpRate);
        if (oldW !== newW) {
          this.weightSum += (newW - oldW);
          if (oldW < 0.6 && newW >= 0.6) this.strongCount++;
          this.weights[wi] = newW;
        }
      }
    }

    // Очистка
    if (this.activeSet.size > 5000) {
      const arr = Array.from(this.activeSet);
      this.activeSet = new Set(arr.slice(-3000));
    }

    this.lastSpikeEvents = spikeEvents.slice(-60);
    return { fired, events: spikeEvents };
  }

  // STDP: обновление весов для пар спайков
  // willFireSet: Set<number> из Фазы 1 для O(1) проверки в LTD
  private applySTDP(fired: number[], willFireSet: Set<number>): void {
    // Для каждого спайка проверяем недавние спайки пресинаптических нейронов
    for (const postIdx of fired) {
      for (let d = 0; d < NDIR; d++) {
        const preIdx = this.neighbor(postIdx, d);
        if (preIdx === null) continue;
        // Проверяем, спайкал ли пресинаптический нейрон недавно
        const preTime = this.lastSpikes[preIdx];
        const postTime = this.tick;
        const dt = postTime - preTime;
        if (dt <= 0 || dt > this.excParams.stdpWindow) continue;

        const wi = preIdx * NDIR + d;
        const isExc = this.neuronTypes[postIdx] === 0;
        const params = isExc ? this.excParams : this.inhParams;
        if (!params.stdpEnabled) continue;

        // LTP: пресинаптический спайк перед постсинаптическим
        const ltpDelta = params.stdpLtpRate * Math.exp(-dt / (params.stdpWindow * 0.4));
        const oldW = this.weights[wi];
        const newW = Math.min(this.WMAX, oldW + ltpDelta);
        if (oldW !== newW) {
          this.weightSum += (newW - oldW);
          if (oldW < 0.6 && newW >= 0.6) this.strongCount++;
          else if (oldW >= 0.6 && newW < 0.6) this.strongCount--;
          this.weights[wi] = newW;
        }
      }
    }

    // LTD: для спайков без последующих постсинаптических спайков
    for (const preIdx of fired) {
      for (let d = 0; d < NDIR; d++) {
        const postIdx = this.neighbor(preIdx, d);
        if (postIdx === null) continue;
        if (willFireSet.has(postIdx)) continue; // O(1) вместо O(n) через .includes()
        const wi = preIdx * NDIR + d;
        const isExc = this.neuronTypes[preIdx] === 0;
        const params = isExc ? this.excParams : this.inhParams;
        if (!params.stdpEnabled) continue;

        // LTD: постсинаптический не спайкнул → ослабить
        const oldW = this.weights[wi];
        const newW = Math.max(this.WMIN, oldW - params.stdpLtdRate * 0.15);
        if (oldW !== newW) {
          this.weightSum += (newW - oldW);
          if (oldW >= 0.6 && newW < 0.6) this.strongCount--;
          this.weights[wi] = newW;
        }
      }
    }
  }

  // Много тиков симуляции
  propagate(maxTicks: number = 10): SpikeEvent[] {
    const allEvents: SpikeEvent[] = [];
    let totalFired = 0;
    for (let t = 0; t < maxTicks; t++) {
      const { fired, events } = this.simulateTick();
      allEvents.push(...events);
      totalFired += fired.length;
      if (fired.length === 0 && this.pendingInputs.size === 0) break;
      if (totalFired > 6000) break; // предел на вызов
    }
    return allEvents;
  }

  // Базовые пороги и decay (для delta-модели нейромодуляции)
  // Публичные, т.к. CoreCluster читает их при нейромодуляции
  baseExcThreshold: number;
  baseInhThreshold: number;
  baseExcDecay: number;
  baseInhDecay: number;

  // Гомеостатическая нормализация (вызывается периодически)
  // Исправлено: вместо мутации общего порога каждым нейроном (баг),
  // вычисляем среднюю ошибку и корректируем порог ядра один раз.
  applyHomeostasis(): void {
    let totalError = 0;
    let counted = 0;
    const window = Math.max(this.tick, 1);

    for (let i = 0; i < this.total; i++) {
      if (this.spikeCounts[i] === 0) continue;
      const isExc = this.neuronTypes[i] === 0;
      const targetRate = isExc ? 0.05 : 0.1;
      const actualRate = this.spikeCounts[i] / window;
      totalError += (actualRate - targetRate);
      counted++;
    }

    if (counted === 0) return;
    const avgError = totalError / counted;
    if (Math.abs(avgError) > 0.02) {
      const correction = avgError * 0.005;
      this.baseExcThreshold = Math.max(0.3, Math.min(2.0, this.baseExcThreshold + correction));
      this.baseInhThreshold = Math.max(0.3, Math.min(2.0, this.baseInhThreshold + correction));
    }
  }

  // Обратная связь: подкрепить (positive) или ослабить (negative) путь
  reinforce(text: string, positive: boolean): void {
    const indices = this.stimulateText(text);
    const savedLtp = this.excParams.stdpLtpRate;
    const savedLtd = this.excParams.stdpLtdRate;

    if (positive) {
      this.excParams.stdpLtpRate = savedLtp * 3.0;
      this.propagate(12);
    } else {
      this.excParams.stdpLtdRate = savedLtd * 2.5;
      const savedThresh = this.excParams.threshold;
      this.excParams.threshold = 0.5;
      this.propagate(12);
      this.excParams.threshold = savedThresh;
    }

    this.excParams.stdpLtpRate = savedLtp;
    this.excParams.stdpLtdRate = savedLtd;
  }

  // ── Decision Core: латеральное торможение между группами нейронов ──
  // Группы определены диапазонами слоёв. Активные нейроны в самой сильной
  // группе подавляют нейроны в более слабых группах → winner-take-all.
  // Это механизм, благодаря которому НЕЙРОНЫ, а не код, выбирают маршрут.
  applyLateralInhibition(
    groups: Array<{ from: number; to: number }>,
    strength: number = 0.35
  ): void {
    if (this.activeSet.size === 0) return;

    // 1. Считаем суммарный потенциал (активность) каждой группы
    const groupActivities = new Float64Array(groups.length);
    const neuronToGroup = new Map<number, number>();

    for (const idx of this.activeSet) {
      const layer = Math.floor(idx / 3600);
      for (let g = 0; g < groups.length; g++) {
        if (layer >= groups[g].from && layer < groups[g].to) {
          groupActivities[g] += Math.abs(this.potentials[idx]);
          neuronToGroup.set(idx, g);
          break;
        }
      }
    }

    // 2. Находим максимальную активность
    let maxActivity = 0;
    for (let g = 0; g < groups.length; g++) {
      if (groupActivities[g] > maxActivity) maxActivity = groupActivities[g];
    }
    if (maxActivity <= 0) return;

    // 3. Подавляем нейроны в группах, активность которых < 70% от максимума
    const threshold = maxActivity * 0.7;
    for (const [idx, g] of neuronToGroup) {
      if (groupActivities[g] < threshold) {
        this.potentials[idx] *= (1 - strength);
      }
    }
  }

  // ── Decision Core: чтение активности по группам (для определения маршрута) ──
  // Возвращает массив суммарных активностей для каждой группы.
  // Побеждает группа с максимальной активностью — это и есть нейронное решение.
  getGroupActivity(groups: Array<{ from: number; to: number }>): number[] {
    const result = new Array(groups.length).fill(0);
    if (this.activeSet.size === 0) return result;

    for (const idx of this.activeSet) {
      const layer = Math.floor(idx / 3600);
      for (let g = 0; g < groups.length; g++) {
        if (layer >= groups[g].from && layer < groups[g].to) {
          result[g] += Math.abs(this.potentials[idx]);
          break;
        }
      }
    }
    return result;
  }

  // Профиль активности по слоям (для маршрутизации и анализа)
  getLayerProfile(): number[] {
    const profile = new Array(GRID).fill(0);
    for (const idx of this.activeSet) {
      profile[Math.floor(idx / 3600)]++;
    }
    return profile;
  }

  // Снимок состояния
  snapshot(): CoreSnapshot {
    const n = this.weights.length;
    let totalPotential = 0;
    let inhibitoryFired = 0;
    // По инварианту liveSet (см. объявление поля) все нейроны ВНЕ
    // liveSet имеют потенциал ровно 0 — раньше здесь был проход по
    // всем 216 000 нейронам ядра, вызываемый на каждый /api/neuro/status
    // (раз в 2 секунды) и на каждый ответ чата; сумма по liveSet даёт
    // тот же результат на порядки дешевле.
    for (const i of this.liveSet) {
      totalPotential += Math.abs(this.potentials[i]);
    }
    for (const ev of this.lastSpikeEvents) {
      if (ev.type === NeuronType.Inhibitory) inhibitoryFired++;
    }
    return {
      id: this.id,
      name: this.name,
      role: this.role,
      color: this.color,
      total: this.total,
      active: this.activeSet.size,
      events: this.events,
      tick: this.tick,
      avgWeight: n > 0 ? this.weightSum / n : 0,
      strongSynapses: this.strongCount,
      firedThisCycle: this.lastSpikeEvents.length,
      inhibitoryFired,
      avgPotential: this.total > 0 ? totalPotential / this.total : 0,
      layerProfile: this.getLayerProfile(),
      lastSpikes: this.lastSpikeEvents.slice(-30),
    };
  }

  // Сброс активности (не трогает веса)
  reset(): void {
    this.potentials.fill(0);
    this.refractories.fill(0);
    this.adaptations.fill(0);
    this.pendingInputs.clear();
    this.activeSet.clear();
    this.liveSet.clear();
    this.lastSpikeEvents = [];
    this.events = 0;
    this.tick = 0;
  }

  // Сериализация весов
  serializeWeights(): ArrayBuffer {
    return this.weights.buffer.slice(0) as ArrayBuffer;
  }

  // Загрузка весов
  loadWeights(buffer: ArrayBuffer): void {
    if (buffer.byteLength === this.weights.byteLength) {
      this.weights.set(new Float32Array(buffer));
      this.recalcWeightStats();
    }
  }
}