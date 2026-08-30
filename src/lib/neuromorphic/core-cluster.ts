// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — CoreCluster: 6 специализированных ядер
// ═══════════════════════════════════════════════════════════════════
// Рекурсивная нейронная цепочка (каскадная обработка).
// Ядра 1-5 обмениваются спайками через ОБУЧАЕМЫЕ межъядерные проекции.// Код НЕ решает, кому что передавать и сколько. STDP на проекциях
// обучает, какие пути усиливать, а какие ослаблять.
// Ядро 6 (Decision Core): WTA-конкуренция групп, нейронный выбор маршрута.
// ═══════════════════════════════════════════════════════════════════

import { NeuralCore, CORE_CONFIGS, SpikeEvent, GRID, TOTAL_NEURONS, NDIR } from './neural-core';
import { NeuronType } from './lif-neuron';

// ── Нейромодуляторы: глобальные химические сигналы ──
export interface NeuromodulatorState {
  dopamine: number;      // 0..1 — reward signal, усиливает LTP
  serotonin: number;     // 0..1 — mood/stability, снижает пороги
  acetylcholine: number;  // 0..1 — attention/focus, увеличивает чувствительность
  norepinephrine: number; // 0..1 — arousal, увеличивает decay
}

// ── Статическая топология связей (используется для построения проекций) ──
export interface InterCoreLink {
  from: string;
  to: string;
  strength: number;
  active: boolean;
}

// ── Обучаемая межъядерная проекция ──
// Каждый нейрон-источник имеет 1 вес и 1 предвычисленный целевой индекс.
// STDP обучает веса: если источник спайкал и цель ответила — LTP,
// если цель промолчала — LTD.
export interface CoreProjection {
  from: string;
  to: string;
  weights: Float32Array;      // вес проекции для каждого нейрона-источника
  targetIndices: Int32Array;  // предвычисленный индекс нейрона-приёмника
  ltpRate: number;
  ltdRate: number;
}

// ── Топология связей (определяет какие ядра соединены) ──
const INTER_CORE_TOPOLOGY: InterCoreLink[] = [
  // Сенсорный → все остальные (входной поток)
  { from: 'sensory', to: 'associative', strength: 1.2, active: true },
  { from: 'sensory', to: 'analytical', strength: 0.8, active: true },
  { from: 'sensory', to: 'temporal', strength: 1.0, active: true },
  { from: 'sensory', to: 'executive', strength: 0.6, active: true },

  // Ассоциативный → аналитический и временной (обработка)
  { from: 'associative', to: 'analytical', strength: 1.0, active: true },
  { from: 'associative', to: 'temporal', strength: 0.9, active: true },
  { from: 'associative', to: 'executive', strength: 0.7, active: true },

  // Аналитический → исполнительный (решения)
  { from: 'analytical', to: 'executive', strength: 1.1, active: true },
  { from: 'analytical', to: 'associative', strength: 0.5, active: true }, // обратная связь

  // Временный → ассоциативный и исполнительный (контекст)
  { from: 'temporal', to: 'associative', strength: 0.8, active: true },
  { from: 'temporal', to: 'executive', strength: 0.7, active: true },

  // Исполнительный → сенсорный (attentional feedback)
  { from: 'executive', to: 'sensory', strength: 0.5, active: true },

  // ── Проекции В Decision Core (ядро 6) от всех обработчиков ──
  { from: 'sensory', to: 'modulatory', strength: 0.9, active: true },
  { from: 'associative', to: 'modulatory', strength: 1.2, active: true },
  { from: 'analytical', to: 'modulatory', strength: 1.0, active: true },
  { from: 'temporal', to: 'modulatory', strength: 0.8, active: true },
  { from: 'executive', to: 'modulatory', strength: 1.3, active: true },
];

// ── Decision Core: 4 группы маршрутов (WTA конкуренция) ──
// Каждая группа занимает 15 слоёв (9000 нейронов) из 60.
export const DECISION_GROUPS: Array<{ from: number; to: number; route: string }> = [
  { from: 0,  to: 15, route: 'analytical' },   // Слои 0-14:   вычисления, логика
  { from: 15, to: 30, route: 'encyclopedic' },  // Слои 15-29:  знания, факты
  { from: 30, to: 45, route: 'relational' },    // Слои 30-44:  память, ассоциации
  { from: 45, to: 60, route: 'social' },        // Слои 45-59:  диалог, общение
];

// Маппинг: номер группы → индекс проекции modulatory (для STDP подкрепления)
const MODULATORY_PROJ_INDEX: Record<string, number> = {};

export interface ClusterSnapshot {
  cores: ReturnType<NeuralCore['snapshot']>[];
  neuromodulators: NeuromodulatorState;
  interCoreLinks: InterCoreLink[];   // сохранено для UI-совместимости
  globalTick: number;
  totalNeurons: number;
  totalSynapses: number;
  totalActive: number;
  interCoreTraffic: number;
  decisionGroupActivities: number[];
  decisionDominance: number;
  // Метрики обучаемых проекций
  projectionCount: number;
  avgProjectionWeight: number;
  strongProjectionCount: number;
  cascadeRounds: number;           // сколько раундов потребовалось
  cascadeConverged: boolean;        // каскад сошёлся досрочно?
}

export class CoreCluster {
  cores: Map<string, NeuralCore> = new Map();
  links: InterCoreLink[];  // статическая топология (для UI)
  neuromodulators: NeuromodulatorState;
  globalTick = 0;
  interCoreTraffic = 0;

  // Обучаемые межъядерные проекции
  projections: CoreProjection[] = [];
  private projectionBySource: Map<string, CoreProjection[]> = new Map();

  // Decision Core кэш
  lastDecisionGroupActivities: number[] = [];
  lastDecisionDominance = 0;

  // Кэши для быстрого доступа
  private coreList: NeuralCore[];
  private processingCores: NeuralCore[];  // ядра 1-5 (без Decision Core)

  constructor() {
    for (const config of CORE_CONFIGS) {
      this.cores.set(config.id, new NeuralCore(config));
    }
    this.coreList = Array.from(this.cores.values());
    this.processingCores = this.coreList.filter(c => c.id !== 'modulatory');
    this.links = INTER_CORE_TOPOLOGY.map(l => ({ ...l }));

    this.buildProjections();

    this.neuromodulators = {
      dopamine: 0.5,
      serotonin: 0.5,
      acetylcholine: 0.5,
      norepinephrine: 0.3,
    };

    // Применяем начальную нейромодуляцию
    this.applyNeuromodulation();
  }

  // ── Построение обучаемых проекций из топологии ──
  // Для каждой InterCoreLink создаём CoreProjection с:
  //   - targetIndices: FNV-1a хеш → детерминированный маппинг нейронов
  //   - weights: инициализация из link.strength + случайный компонент
  private buildProjections(): void {
    for (let li = 0; li < INTER_CORE_TOPOLOGY.length; li++) {
      const link = INTER_CORE_TOPOLOGY[li];
      if (!link.active) continue;

      const proj: CoreProjection = {
        from: link.from,
        to: link.to,
        weights: new Float32Array(TOTAL_NEURONS),
        targetIndices: new Int32Array(TOTAL_NEURONS),
        ltpRate: 0.008,
        ltdRate: 0.003,
      };

      for (let i = 0; i < TOTAL_NEURONS; i++) {
        const l = Math.floor(i / 3600);
        const r = Math.floor((i % 3600) / 60);
        const c = i % 60;
        const h = this.fnv1a(`${link.from}:${l}:${r}:${c}`);
        const tL = h % GRID;
        const tR = this.wrapCoord(h >> 6);
        const tC = this.wrapCoord(h >> 12);
        proj.targetIndices[i] = (tL * GRID + tR) * GRID + tC;
        // Начальный вес: пропорционален силе связи + случайность
        proj.weights[i] = link.strength * (0.12 + Math.random() * 0.18);
      }

      // Проекции в Decision Core: повышенная обучаемость
      if (link.to === 'modulatory') {
        proj.ltpRate = 0.012;
        proj.ltdRate = 0.004;
        MODULATORY_PROJ_INDEX[link.from] = this.projections.length;
      }

      this.projections.push(proj);

      if (!this.projectionBySource.has(link.from)) {
        this.projectionBySource.set(link.from, []);
      }
      this.projectionBySource.get(link.from)!.push(proj);
    }
  }

  getCore(id: string): NeuralCore {
    return this.cores.get(id)!;
  }

  // ═══════════════════════════════════════════════════════════════════
  // ГЛАВНЫЙ МЕТОД: обработка входного текста
  // ═══════════════════════════════════════════════════════════════════
  // ФАЗА 1: Каскадная нейронная цепочка (ядра 1-5)
  //   Все ядра тикают ОДНОВРЕМЕННО. Спайки текут через обучаемые
  //   проекции. Код НЕ решает, кому что передавать и сколько.
  //   STDP на проекциях обучает, какие пути усиливать.
  //
  // ФАЗА 2: Нейромодуляторы обновляются из активности каскада.
  //
  // ФАЗА 3: Decision Core — проекции от ядер 1-5,
  //   латеральное торможение, WTA → маршрут.
  // ═══════════════════════════════════════════════════════════════════
  processInput(text: string): {
    sensoryIndices: number[];
    allEvents: SpikeEvent[];
    layerProfiles: Record<string, number[]>;
    routeDecision: string;
    routeScores: Record<string, number>;
    confidence: number;
    familiarity: number;
  } {
    // ── Стимуляция сенсорного ядра ──
    const sensory = this.getCore('sensory');
    const sensoryIndices = sensory.stimulateText(text);
    const allEvents: SpikeEvent[] = [];
    const layerProfiles: Record<string, number[]> = {};

    // ══════════════════════════════════════════════════════════════
    // ФАЗА 1: НЕЙРОННАЯ КАСКАДНАЯ ЦЕПОЧКА (ядра 1-5)
    // ══════════════════════════════════════════════════════════════
    // Все ядра тикают одновременно каждый раунд.
    // Спайки передаются через обучаемые проекции (не через .slice()).
    // STDP на проекциях: источник спайкал в R, цель ответила в R+1 → LTP.
    // ══════════════════════════════════════════════════════════════

    const MAX_ROUNDS = 10;
    let prevFiredByCore = new Map<string, Set<number>>();
    let stableRounds = 0;
    let cascadeRounds = 0;
    let cascadeConverged = false;

    for (let round = 0; round < MAX_ROUNDS; round++) {
      cascadeRounds++;
      const currFiredByCore = new Map<string, Set<number>>();
      let roundFiredTotal = 0;

      // 1A: все ядра 1-5 — один тик simultaneously
      for (const core of this.processingCores) {
        const { fired, events } = core.simulateTick();
        allEvents.push(...events);
        roundFiredTotal += fired.length;
        currFiredByCore.set(core.id, new Set(fired));
      }

      // 1B: межъядерная передача через обучаемые проекции
      for (const core of this.processingCores) {
        const outgoing = this.projectionBySource.get(core.id);
        if (!outgoing) continue;
        const srcFired = currFiredByCore.get(core.id);
        if (!srcFired || srcFired.size === 0) continue;

        for (const proj of outgoing) {
          // Проекции в Decision Core обрабатываются отдельно в Фазе 3
          if (proj.to === 'modulatory') continue;
          const targetCore = this.getCore(proj.to);

          for (const srcIdx of srcFired) {
            const tgtIdx = proj.targetIndices[srcIdx];
            const w = proj.weights[srcIdx];
            const isExc = core.neuronTypes[srcIdx] === 0;
            const signal = isExc ? w : -w * 0.6;
            // receiveInput(), а не прямая запись в .potentials/.activeSet —
            // с разреженным обновлением ядра (см. neural-core.ts, v2.4.2)
            // цель обязана попасть в liveSet ядра, иначе на следующем
            // тике она была бы просто пропущена как "спящая".
            targetCore.receiveInput(tgtIdx, signal);
            this.interCoreTraffic++;
          }
        }
      }

      // 1C: STDP на проекциях между ядрами 1-5
      // Если источник спайкал в (round-1), а цель спайкала в round → LTP
      // Если источник спайкал, а цель промолчала → LTD
      if (prevFiredByCore.size > 0) {
        const daFactor = 1 + this.neuromodulators.dopamine * 0.5;

        for (let pi = 0; pi < this.projections.length; pi++) {
          const proj = this.projections[pi];
          // Только проекции между ядрами 1-5
          if (proj.to === 'modulatory' || proj.from === 'modulatory') continue;

          const prevSrcFired = prevFiredByCore.get(proj.from);
          if (!prevSrcFired || prevSrcFired.size === 0) continue;
          const currTgtFired = currFiredByCore.get(proj.to);

          for (const srcIdx of prevSrcFired) {
            const tgtIdx = proj.targetIndices[srcIdx];
            if (currTgtFired && currTgtFired.has(tgtIdx)) {
              // LTP: источник спайкал в прошлом раунде, цель ответила
              proj.weights[srcIdx] = Math.min(1.5, proj.weights[srcIdx] + proj.ltpRate * daFactor);
            } else {
              // LTD: источник спайкал, цель не ответила
              proj.weights[srcIdx] = Math.max(0.01, proj.weights[srcIdx] - proj.ltdRate);
            }
          }
        }
      }

      prevFiredByCore = currFiredByCore;

      // Проверка сходимости
      if (roundFiredTotal < 5) {
        stableRounds++;
        if (stableRounds >= 2) {
          cascadeConverged = true;
          break;
        }
      } else {
        stableRounds = 0;
      }
    }

    // Собираем layer profiles ядер 1-5
    for (const core of this.processingCores) {
      layerProfiles[core.id] = core.getLayerProfile();
    }

    // ══════════════════════════════════════════════════════════════
    // ФАЗА 2: ОБНОВЛЕНИЕ НЕЙРОМОДУЛЯТОРОВ
    // Нейромодуляторы вычисляются из активности каскада.
    // Применяются к Decision Core перед его обработкой.
    // ══════════════════════════════════════════════════════════════
    this.updateNeuromodulators(allEvents);
    this.applyNeuromodulation();

    // ══════════════════════════════════════════════════════════════
    // ФАЗА 3: DECISION CORE — нейронный выбор маршрута
    // Получает текущее состояние ядер 1-5 через обучаемые проекции,
    // латеральное торможение между 4 группами, WTA.
    // ══════════════════════════════════════════════════════════════
    const decisionCore = this.getCore('modulatory');

    // 3a: Передаём текущее состояние ядер 1-5 в Decision Core
    //     через их обучаемые проекции (веса, выученные STDP)
    for (const core of this.processingCores) {
      const outgoing = this.projectionBySource.get(core.id);
      if (!outgoing) continue;

      for (const proj of outgoing) {
        if (proj.to !== 'modulatory') continue;

        // Передаём только активные нейроны с ненулевым потенциалом
        for (const idx of core.activeSet) {
          const pot = Math.abs(core.potentials[idx]);
          if (pot < 0.01) continue;
          const tgtIdx = proj.targetIndices[idx];
          const w = proj.weights[idx];
          const isExc = core.neuronTypes[idx] === 0;
          const signal = (isExc ? w : -w * 0.6) * pot * 0.5;
          decisionCore.receiveInput(tgtIdx, signal);
          this.interCoreTraffic++;
        }
      }
    }

    // 3b: Латеральное торможение между 4 группами
    const groupRanges = DECISION_GROUPS.map(g => ({ from: g.from, to: g.to }));
    decisionCore.applyLateralInhibition(groupRanges, 0.4);

    // 3c: Пропагация через Decision Core
    const decisionEvents = decisionCore.propagate(4);
    allEvents.push(...decisionEvents);
    layerProfiles['modulatory'] = decisionCore.getLayerProfile();

    // 3d: Повторное латеральное торможение (усиление WTA)
    decisionCore.applyLateralInhibition(groupRanges, 0.3);

    // 3e: Чтение решения — какая группа победила?
    const groupActivities = decisionCore.getGroupActivity(groupRanges);
    this.lastDecisionGroupActivities = groupActivities;

    const totalGroupActivity = groupActivities.reduce((s, v) => s + v, 0);
    const maxGroupActivity = Math.max(...groupActivities);
    const winnerIdx = groupActivities.indexOf(maxGroupActivity);
    const routeDecision = DECISION_GROUPS[winnerIdx].route;
    const dominance = totalGroupActivity > 0 ? maxGroupActivity / totalGroupActivity : 0;
    this.lastDecisionDominance = dominance;

    // 3f: STDP на проекциях → Decision Core
    //     Усиливаем проекции, чьи цели попали в победившую группу.
    //     Ослабляем те, что попали в проигравшие.
    const winnerRange = DECISION_GROUPS[winnerIdx];
    const daFactor = 1 + this.neuromodulators.dopamine * 0.5;

    for (const core of this.processingCores) {
      const projIdx = MODULATORY_PROJ_INDEX[core.id];
      if (projIdx === undefined) continue;
      const proj = this.projections[projIdx];

      for (const idx of core.activeSet) {
        const tgtIdx = proj.targetIndices[idx];
        const tgtLayer = Math.floor(tgtIdx / 3600);
        if (tgtLayer >= winnerRange.from && tgtLayer < winnerRange.to) {
          // LTP: проекция попала в победившую группу
          proj.weights[idx] = Math.min(1.5, proj.weights[idx] + proj.ltpRate * daFactor * 2);
        } else if (tgtLayer < 60) {
          // LTD: проекция попала в проигравшую группу
          proj.weights[idx] = Math.max(0.01, proj.weights[idx] - proj.ltdRate);
        }
      }
    }

    // Формируем scores для совместимости
    const routeScores: Record<string, number> = {};
    for (let i = 0; i < DECISION_GROUPS.length; i++) {
      routeScores[DECISION_GROUPS[i].route] = totalGroupActivity > 0
        ? groupActivities[i] / totalGroupActivity
        : 0;
    }

    this.globalTick++;

    const confidence = this.computeConfidence(layerProfiles, dominance);
    const familiarity = this.computeFamiliarity(layerProfiles);

    // Сохраняем метрики каскада для snapshot
    this._lastCascadeRounds = cascadeRounds;
    this._lastCascadeConverged = cascadeConverged;

    return {
      sensoryIndices,
      allEvents,
      layerProfiles,
      routeDecision,
      routeScores,
      confidence,
      familiarity,
    };
  }

  // Кэш метрик каскада (для snapshot)
  private _lastCascadeRounds = 0;
  private _lastCascadeConverged = false;

  // Лёгкий доступ к сходимости каскада без полного snapshot() (который
  // проходит по всем нейронам всех ядер и заметно дороже).
  getCascadeInfo(): { cascadeRounds: number; cascadeConverged: boolean } {
    return { cascadeRounds: this._lastCascadeRounds, cascadeConverged: this._lastCascadeConverged };
  }

  // ── Обновление нейромодуляторов на основе активности ──
  private updateNeuromodulators(events: SpikeEvent[]): void {
    const excSpikes = events.filter(e => e.type === NeuronType.Excitatory).length;
    const inhSpikes = events.filter(e => e.type === NeuronType.Inhibitory).length;
    const ratio = excSpikes / Math.max(excSpikes + inhSpikes, 1);

    // Дофамин: высокий при сбалансированной активности
    this.neuromodulators.dopamine = Math.max(0, Math.min(1,
      this.neuromodulators.dopamine * 0.8 + ratio * 0.4
    ));

    // Серотонин: стабилизируется при умеренной активности
    const totalActivity = events.length / (6 * 50);
    this.neuromodulators.serotonin = Math.max(0, Math.min(1,
      this.neuromodulators.serotonin * 0.85 + (1 - Math.abs(totalActivity - 0.5) * 2) * 0.3
    ));

    // Ацетилхолин: зависит от количества активных ядер
    const activeCores = this.coreList.filter(c => c.activeSet.size > 100).length;
    this.neuromodulators.acetylcholine = Math.max(0, Math.min(1,
      this.neuromodulators.acetylcholine * 0.9 + (activeCores / 6) * 0.2
    ));

    // Норадреналин: arousal, зависит от общей интенсивности
    this.neuromodulators.norepinephrine = Math.max(0, Math.min(1,
      this.neuromodulators.norepinephrine * 0.85 + Math.min(totalActivity, 1) * 0.3
    ));
  }

  // ── Применить нейромодуляторы ко всем ядрам ──
  // Delta-модель: пороги и decay вычисляются от base + neuromod_delta.
  private applyNeuromodulation(): void {
    const { dopamine, serotonin, norepinephrine, acetylcholine } = this.neuromodulators;

    for (const core of this.coreList) {
      // Дофамин усиливает STDP rates
      const baseLtp = createBaseLtp(core.id);
      core.excParams.stdpLtpRate = baseLtp * (1 + dopamine * 0.5);

      // Серотонин снижает пороги (повышает возбудимость)
      const seroFactor = 1 - serotonin * 0.1;
      core.excParams.threshold = core.baseExcThreshold * seroFactor;
      core.inhParams.threshold = core.baseInhThreshold * seroFactor;

      // Ацетилхолин: фокус на исполнительном ядре
      if (core.id === 'executive') {
        core.excParams.threshold *= (1 - acetylcholine * 0.08);
      }

      // Норадреналин: decay ближе к 1 = медленнее затухание
      const baseDecay = createBaseDecay(core.id);
      let decay = baseDecay + norepinephrine * 0.03;

      // Ацетилхолин: обостряет кодирование в сенсорном и ассоциативном
      if (core.id === 'sensory' || core.id === 'associative') {
        decay -= acetylcholine * 0.03;
      }

      core.excParams.decay = Math.max(0.5, Math.min(0.99, decay));
    }
  }

  // ── Уверенность из нейронной активности + Decision Core dominance ──
  private computeConfidence(profiles: Record<string, number[]>, decisionDominance: number): number {
    const totalActivity = Object.values(profiles).reduce(
      (sum, profile) => sum + profile.reduce((s, v) => s + v, 0), 0
    );
    const numActiveCores = Object.values(profiles).filter(
      profile => profile.reduce((s, v) => s + v, 0) > 50
    ).length;

    let maxPeak = 0;
    for (const profile of Object.values(profiles)) {
      for (const v of profile) {
        if (v > maxPeak) maxPeak = v;
      }
    }

    const focus = totalActivity > 0 ? maxPeak / totalActivity : 0;
    const coreSpread = numActiveCores / 6;
    const neuromodFactor = this.neuromodulators.dopamine * 0.3 +
                          this.neuromodulators.serotonin * 0.2;
    const decisionFactor = decisionDominance * 0.3;

    return Math.min(1, Math.max(0,
      focus * 0.3 + coreSpread * 0.15 + neuromodFactor + decisionFactor + 0.1
    ));
  }

  // ── «Знакомство» с запросом ──
  private computeFamiliarity(profiles: Record<string, number[]>): number {
    const assocProfile = profiles['associative'] || [];
    const assocTotal = assocProfile.reduce((s, v) => s + v, 0);
    if (assocTotal < 10) return 0;

    const nonZero = assocProfile.filter(v => v > 0).length;
    const concentration = assocTotal / Math.max(nonZero, 1);
    const assocCore = this.getCore('associative');
    const strongRatio = assocCore.strongCount / (TOTAL_NEURONS * NDIR);

    return Math.min(1, concentration * 0.01 + strongRatio * 5);
  }

  // ── Спонтанная стимуляция (фоновая активность) ──
  // Мини-каскад на 3 раунда для поддержания тонуса проекций.
  spontaneousStimulation(count: number = 100, coreId?: string | null): void {
    const targets = coreId ? [this.getCore(coreId)] : this.processingCores;
    for (let i = 0; i < count; i++) {
      const core = targets[Math.floor(Math.random() * targets.length)];
      const neuronIdx = Math.floor(Math.random() * TOTAL_NEURONS);
      core.stimulate(neuronIdx, 0.6 + Math.random() * 0.7);
    }
    // Мини-каскад: 3 раунда одновременных тиков + проекции
    for (let round = 0; round < 3; round++) {
      for (const core of this.processingCores) {
        core.simulateTick();
      }
      // Передача через проекции (без STDP — фоновая активность)
      for (const core of this.processingCores) {
        const outgoing = this.projectionBySource.get(core.id);
        if (!outgoing) continue;
        for (const proj of outgoing) {
          if (proj.to === 'modulatory') continue;
          const targetCore = this.getCore(proj.to);
          for (const idx of core.activeSet) {
            if (Math.abs(core.potentials[idx]) < 0.05) continue;
            const tgtIdx = proj.targetIndices[idx];
            const w = proj.weights[idx];
            const isExc = core.neuronTypes[idx] === 0;
            targetCore.receiveInput(tgtIdx, (isExc ? w : -w * 0.6) * 0.3);
          }
        }
      }
    }
  }

  // ── Обучение с подкреплением через ВСЕ ядра + проекции ──
  globalReinforce(text: string, positive: boolean): void {
    // Внутреннее подкрепление ядер (STDP на внутренних синапсах)
    for (const core of this.coreList) {
      core.reinforce(text, positive);
    }

    // Подкрепление проекций: усиливаем/ослабляем веса активных нейронов
    for (const proj of this.projections) {
      const sourceCore = this.getCore(proj.from);
      if (sourceCore.activeSet.size === 0) continue;

      for (const idx of sourceCore.activeSet) {
        if (Math.abs(sourceCore.potentials[idx]) < 0.01) continue;
        if (positive) {
          proj.weights[idx] = Math.min(1.5, proj.weights[idx] + 0.003);
        } else {
          proj.weights[idx] = Math.max(0.01, proj.weights[idx] - 0.002);
        }
      }
    }

    // Нейромодуляторы
    if (positive) {
      this.neuromodulators.dopamine = Math.min(1, this.neuromodulators.dopamine + 0.15);
    } else {
      this.neuromodulators.dopamine = Math.max(0, this.neuromodulators.dopamine - 0.1);
      this.neuromodulators.serotonin = Math.max(0, this.neuromodulators.serotonin - 0.05);
    }
  }

  // ── Snapshot (состояние кластера для UI) ──
  snapshot(): ClusterSnapshot {
    const coreSnapshots = this.coreList.map(c => c.snapshot());

    // Статистика проекций
    let totalProjWeight = 0;
    let strongProjCount = 0;
    let projCount = 0;
    for (const proj of this.projections) {
      for (let i = 0; i < proj.weights.length; i++) {
        totalProjWeight += proj.weights[i];
        if (proj.weights[i] >= 0.6) strongProjCount++;
      }
      projCount++;
    }

    return {
      cores: coreSnapshots,
      neuromodulators: { ...this.neuromodulators },
      interCoreLinks: this.links.map(l => ({ ...l })),  // UI-совместимость
      globalTick: this.globalTick,
      totalNeurons: TOTAL_NEURONS * 6,
      totalSynapses: TOTAL_NEURONS * NDIR * 6,
      totalActive: coreSnapshots.reduce((s, c) => s + c.active, 0),
      interCoreTraffic: this.interCoreTraffic,
      decisionGroupActivities: this.lastDecisionGroupActivities,
      decisionDominance: this.lastDecisionDominance,
      projectionCount: projCount,
      avgProjectionWeight: projCount > 0 ? totalProjWeight / (projCount * TOTAL_NEURONS) : 0,
      strongProjectionCount: strongProjCount,
      cascadeRounds: this._lastCascadeRounds,
      cascadeConverged: this._lastCascadeConverged,
    };
  }

  // ── Сброс всей активности (не трогает веса) ──
  reset(): void {
    for (const core of this.coreList) {
      core.reset();
    }
    this.globalTick = 0;
    this.interCoreTraffic = 0;
  }

  // ── Периодическая гомеостатическая нормализация ──
  homeostasis(): void {
    for (const core of this.coreList) {
      core.applyHomeostasis();
    }
  }

  // ── Сериализация весов проекций (для персистентности) ──
  serializeProjections(): ArrayBuffer[] {
    return this.projections.map(p => p.weights.buffer.slice(0) as ArrayBuffer);
  }

  // ── Загрузка весов проекций ──
  loadProjections(buffers: ArrayBuffer[]): void {
    for (let i = 0; i < this.projections.length && i < buffers.length; i++) {
      if (buffers[i].byteLength === this.projections[i].weights.byteLength) {
        this.projections[i].weights.set(new Float32Array(buffers[i]));
      }
    }
  }

  // ── FNV-1a хеш ──
  private fnv1a(s: string): number {
    let h = 0x811C9DC5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  // Нормализация координаты в [0, GRID) — (h >> 6)/(h >> 12) это ЗНАКОВЫЙ
  // сдвиг 32-битного числа в JS: при установленном старшем бите хеша
  // результат может быть отрицательным. Без нормализации targetIndices[i]
  // (см. buildProjections) уходил в отрицательный "теневой" индекс —
  // Float32Array молча трактует его как обычное свойство объекта, а не
  // элемент буфера; receiveInput() на такой индекс добавляет его в
  // liveSet/activeSet ядра-приёмника, а чтение потенциала такого
  // "нейрона" назад даёт undefined → NaN при первой же арифметике,
  // заражающее totalPotential в NeuralCore.snapshot() на весь срок
  // жизни процесса (симптом: avgPotential вместо числа — null в JSON,
  // падение UI на .toFixed()). Та же находка и тот же фикс, что и в
  // neural-core.ts (wrapCoord) и encoder.ts (TrainableEncoder.bootstrap).
  private wrapCoord(x: number): number {
    return ((x % GRID) + GRID) % GRID;
  }
}

// ── Базовые параметры для нейромодуляции ──
function createBaseLtp(coreId: string): number {
  const bases: Record<string, number> = {
    sensory: 0.05, associative: 0.06, analytical: 0.03,
    temporal: 0.04, executive: 0.035,
    modulatory: 0.07,
  };
  return bases[coreId] || 0.04;
}

function createBaseDecay(coreId: string): number {
  const bases: Record<string, number> = {
    sensory: 0.82, associative: 0.88, analytical: 0.90,
    temporal: 0.92, executive: 0.83, modulatory: 0.94,
  };
  return bases[coreId] || 0.85;
}
