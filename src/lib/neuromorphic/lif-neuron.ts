// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — Реалистичная модель LIF-нейрона
// Рефрактерный период, STDP, spike-frequency adaptation, гомеостаз
// ═══════════════════════════════════════════════════════════════════

export enum NeuronType {
  Excitatory = 'excitatory',
  Inhibitory = 'inhibitory',
  Modulatory = 'modulatory',
}

export interface NeuronState {
  potential: number;
  refractory: number;
  adaptation: number;
  lastSpike: number;
  spikeCount: number;
  totalInput: number;
}

export interface LIFParams {
  threshold: number;       // порог срабатывания
  decay: number;           // затухание мембранного потенциала (tau_m)
  refractoryPeriod: number;// длительность рефрактерного периода (тики)
  resetPotential: number;  // потенциал сброса после спайка
  restPotential: number;   // потенциал покоя
  adaptationRate: number;  // скорость адаптации частоты спайков
  adaptationDecay: number; // затухание адаптации
  // STDP параметры
  stdpEnabled: boolean;
  stdpLtpRate: number;     // long-term potentiation rate
  stdpLtdRate: number;     // long-term depression rate
  stdpWindow: number;      // временное окно STDP (тики)
}

// Параметры по умолчанию для разных типов нейронов
export const DEFAULT_PARAMS: Record<NeuronType, Partial<LIFParams>> = {
  [NeuronType.Excitatory]: {
    threshold: 1.0,
    decay: 0.85,
    refractoryPeriod: 3,
    resetPotential: 0.0,
    restPotential: 0.0,
    adaptationRate: 0.02,
    adaptationDecay: 0.95,
    stdpEnabled: true,
    stdpLtpRate: 0.04,
    stdpLtdRate: 0.025,
    stdpWindow: 20,
  },
  [NeuronType.Inhibitory]: {
    threshold: 0.9,
    decay: 0.75,
    refractoryPeriod: 2,
    resetPotential: -0.1,
    restPotential: -0.05,
    adaptationRate: 0.01,
    adaptationDecay: 0.97,
    stdpEnabled: true,
    stdpLtpRate: 0.03,
    stdpLtdRate: 0.02,
    stdpWindow: 15,
  },
  [NeuronType.Modulatory]: {
    threshold: 1.2,
    decay: 0.90,
    refractoryPeriod: 5,
    resetPotential: 0.0,
    restPotential: 0.0,
    adaptationRate: 0.005,
    adaptationDecay: 0.99,
    stdpEnabled: false,
    stdpLtpRate: 0.0,
    stdpLtdRate: 0.0,
    stdpWindow: 0,
  },
};

export function createDefaultParams(type: NeuronType): LIFParams {
  const base: LIFParams = {
    threshold: 1.0,
    decay: 0.85,
    refractoryPeriod: 3,
    resetPotential: 0.0,
    restPotential: 0.0,
    adaptationRate: 0.02,
    adaptationDecay: 0.95,
    stdpEnabled: true,
    stdpLtpRate: 0.04,
    stdpLtdRate: 0.025,
    stdpWindow: 20,
  };
  return { ...base, ...DEFAULT_PARAMS[type] };
}

// LIF-динамика одного нейрона
export function lifStep(
  state: NeuronState,
  input: number,
  params: LIFParams,
  type: NeuronType,
  globalTick: number
): { spiked: boolean; newState: NeuronState; output: number } {
  let { potential, refractory, adaptation, lastSpike, spikeCount, totalInput } = state;

  // В рефрактерном периоде — нейрон молчит
  if (refractory > 0) {
    refractory--;
    // Медленное возвращение к потенциалу покоя даже в рефрактерном периоде
    potential = potential * 0.5 + params.restPotential * 0.5;
    return {
      spiked: false,
      newState: { potential, refractory, adaptation, lastSpike, spikeCount, totalInput },
      output: 0,
    };
  }

  // Spike-frequency adaptation: снижает чувствительность после серии спайков
  const effectiveThreshold = params.threshold + adaptation;

  // Накопление входного сигнала
  potential = potential * params.decay + input + params.restPotential * (1 - params.decay);
  totalInput += Math.abs(input);

  // Проверка порога
  if (potential >= effectiveThreshold) {
    // СПАЙК!
    const spiked = true;
    spikeCount++;
    lastSpike = globalTick;

    // Сброс потенциала и вход в рефрактерный период
    potential = params.resetPotential;
    refractory = params.refractoryPeriod;

    // Увеличение адаптации
    adaptation += params.adaptationRate;
    if (adaptation > 0.5) adaptation = 0.5;

    // Выходной сигнал: возбуждающий (+1), тормозной (-1), модуляторный (+0.5)
    const output = type === NeuronType.Inhibitory ? -1.0 : type === NeuronType.Modulatory ? 0.5 : 1.0;

    return {
      spiked,
      newState: { potential, refractory, adaptation, lastSpike, spikeCount, totalInput },
      output,
    };
  }

  // Затухание адаптации
  adaptation *= params.adaptationDecay;
  if (Math.abs(adaptation) < 1e-4) adaptation = 0;

  // Удаление negligible потенциала
  if (Math.abs(potential) < 1e-4) potential = params.restPotential;

  return {
    spiked: false,
    newState: { potential, refractory, adaptation, lastSpike, spikeCount, totalInput },
    output: 0,
  };
}

// STDP: обновление веса синапса на основе временной разницы спайков
export function stdpUpdate(
  preSpikeTime: number,
  postSpikeTime: number,
  currentWeight: number,
  params: LIFParams,
  currentTick: number,
  minW: number,
  maxW: number
): number {
  if (!params.stdpEnabled) return currentWeight;

  const dt = postSpikeTime - preSpikeTime;
  if (Math.abs(dt) > params.stdpWindow) return currentWeight;

  let delta: number;
  if (dt > 0 && dt <= params.stdpWindow) {
    // Пресинаптический спайк ПЕРЕД постсинаптическим -> LTP (усиление)
    delta = params.stdpLtpRate * Math.exp(-dt / (params.stdpWindow * 0.4));
  } else if (dt < 0 && Math.abs(dt) <= params.stdpWindow) {
    // Пресинаптический спайк ПОСЛЕ постсинаптического -> LTD (ослабление)
    delta = -params.stdpLtdRate * Math.exp(dt / (params.stdpWindow * 0.4));
  } else {
    return currentWeight;
  }

  return Math.max(minW, Math.min(maxW, currentWeight + delta));
}

// Гомеостатическая пластичность: нормализация активности нейрона
export function homeostaticUpdate(
  spikeCount: number,
  targetRate: number,
  windowSize: number,
  currentThreshold: number,
  params: LIFParams
): number {
  const actualRate = spikeCount / windowSize;
  const error = actualRate - targetRate;
  // Если нейрон спайкит слишком часто — повысить порог
  // Если слишком редко — понизить порог
  const newThreshold = currentThreshold + error * 0.01;
  return Math.max(0.3, Math.min(2.0, newThreshold));
}
