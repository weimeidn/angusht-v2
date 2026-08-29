// ═══════════════════════════════════════════════════════════════════
// Angusht — регрессионный бенчмарк для neural-core.ts
// ════════════════════════════════════════════════════════════════════
// В проекте не было ни одного теста. Это не полноценный юнит-тест
// (нет фреймворка), но даёт ровно то, чего не хватало для безопасного
// рефакторинга симуляции (см. п.6/7 приоритетного списка):
//
//  1) детерминированный "отпечаток" поведения ядра (fixture) —
//     сумма потенциалов, сумма весов, число спайков по раундам;
//  2) число (мс), которое можно честно сравнивать до/после изменения.
//
// ВАЖНО (с v2.4.3): текущий bench/baseline.json зафиксирован ПОСЛЕ
// внедрения TrainableEncoder (п.7) — кодирование слов в координаты
// решётки теперь обучаемое состояние (encoder.ts), а не чистый хеш.
// Для ЭТОГО скрипта это не проблема: FIXTURE_TEXTS не содержат
// повторяющихся стемов между предложениями, поэтому каждое слово
// стимулируется по своей (пока не обученной) начальной координате
// внутри одного прогона, а сам прогон полностью детерминирован (без
// Math.random в encoder.ts) — фикстура воспроизводима бит-в-бит между
// независимыми запусками. Значит: --compare с этим baseline по-прежнему
// осмысленно проверяет "поведение симуляции не изменилось" для ЛЮБОГО
// будущего рефакторинга (производительности или иного), который не
// должен трогать результат. Но если вы specifически меняете сам
// encoder.ts (алгоритм обучения координат) — расхождение ОЖИДАЕМО и
// означает, что нужно заново отревьюить и пересохранить baseline
// (см. scripts/bench-encoder.ts — тот проверяет корректность обучения
// содержательно, а не через сравнение с фиксированным числом).
//
// Запуск:
//   npx tsx scripts/bench-neural-core.ts --save bench/baseline.json
//   npx tsx scripts/bench-neural-core.ts --compare bench/baseline.json
// ═══════════════════════════════════════════════════════════

import { writeFileSync, readFileSync } from 'fs';
import { NeuralCore, CORE_CONFIGS } from '../src/lib/neuromorphic/neural-core';

function seededRandom(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function round(x: number): number {
  return Math.round(x * 1e6) / 1e6;
}

const FIXTURE_TEXTS = [
  'кошка спит на диване весь день',
  'собака гуляет в парке с хозяином',
  'нейрон передаёт электрический сигнал',
  'python используется для машинного обучения',
  'гравитация притягивает все тела с массой',
];

function buildFixture() {
  const originalRandom = Math.random;
  Math.random = seededRandom(42);
  const core = new NeuralCore(CORE_CONFIGS[1]); // associative — реалистичные параметры ядра
  Math.random = originalRandom;

  const spikeLog: number[] = [];
  let totalFired = 0;

  for (const text of FIXTURE_TEXTS) {
    core.stimulateText(text);
    const events = core.propagate(15);
    totalFired += events.length;
    spikeLog.push(events.length);
  }

  let potentialSum = 0;
  for (let i = 0; i < core.total; i++) potentialSum += core.potentials[i];
  let weightSum = 0;
  for (let i = 0; i < core.weights.length; i++) weightSum += core.weights[i];
  let refractoryCount = 0;
  for (let i = 0; i < core.total; i++) if (core.refractories[i] > 0) refractoryCount++;
  let adaptationSum = 0;
  for (let i = 0; i < core.total; i++) adaptationSum += core.adaptations[i];

  return {
    tick: core.tick,
    events: core.events,
    totalFired,
    spikeLog,
    potentialSum: round(potentialSum),
    weightSum: round(weightSum),
    adaptationSum: round(adaptationSum),
    strongCount: core.strongCount,
    refractoryCount,
  };
}

const BENCH_TEXTS = [
  'кошка спит на диване весь день напролёт',
  'собака гуляет в парке с хозяином каждый вечер',
  'нейрон передаёт электрический сигнал через синапс',
  'python используется для машинного обучения и анализа данных',
  'гравитация притягивает все тела с массой друг к другу',
  'квантовая механика описывает поведение частиц на микроуровне',
  'фотосинтез превращает световую энергию в химическую',
  'блокчейн это распределённый реестр транзакций',
];

function benchTiming(): number {
  const originalRandom = Math.random;
  Math.random = seededRandom(7);
  const core = new NeuralCore(CORE_CONFIGS[1]);
  Math.random = originalRandom;

  const ROUNDS = 40;
  const start = performance.now();
  for (let r = 0; r < ROUNDS; r++) {
    for (const text of BENCH_TEXTS) {
      core.stimulateText(text);
      core.propagate(10);
    }
  }
  return performance.now() - start;
}

function main() {
  const args = process.argv.slice(2);
  const fixture = buildFixture();
  // Берём медиану из трёх прогонов таймингЯ единичный замер шумный.
  const timings = [benchTiming(), benchTiming(), benchTiming()].sort((a, b) => a - b);
  const timeMs = timings[1];

  const result = { fixture, timeMs, allTimings: timings };
  console.log(JSON.stringify(result, null, 2));

  const saveIdx = args.indexOf('--save');
  if (saveIdx >= 0) {
    writeFileSync(args[saveIdx + 1], JSON.stringify(result, null, 2));
    console.error(`\nСохранено в ${args[saveIdx + 1]}`);
  }

  const compareIdx = args.indexOf('--compare');
  if (compareIdx >= 0) {
    const baseline = JSON.parse(readFileSync(args[compareIdx + 1], 'utf-8'));
    const same = JSON.stringify(baseline.fixture) === JSON.stringify(fixture);
    console.error(`\nСверка с ${args[compareIdx + 1]}: ${same ? 'ИДЕНТИЧНО ✓ (поведение симуляции не изменилось)' : 'РАСХОЖДЕНИЕ ✗ (поведение изменилось!)'}`);
    console.error(`Время: baseline=${baseline.timeMs.toFixed(1)}мс, сейчас=${timeMs.toFixed(1)}мс, ускорение×${(baseline.timeMs / timeMs).toFixed(2)}`);
    if (!same) {
      console.error('baseline.fixture:', JSON.stringify(baseline.fixture));
      console.error('current.fixture: ', JSON.stringify(fixture));
      process.exitCode = 1;
    }
  }
}

main();
