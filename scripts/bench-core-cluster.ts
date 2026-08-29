// ═══════════════════════════════════════════════════════════════════
// Angusht — регрессионный бенчмарк для CoreCluster (полный каскад 6 ядер)
// ═══════════════════════════════════════════════════════════════════
// То же самое, что bench-neural-core.ts, но на уровне всего кластера —
// даёт честное число (мс) на уровне полного processInput() (ближе к
// реальной стоимости одного чат-запроса) и детерминированный отпечаток
// каскада (маршрутизацию, нейромодуляторы, веса проекций).
//
// ВАЖНО (с v2.4.3): в отличие от bench-neural-core.ts, здесь
// FIXTURE_TEXTS СОДЕРЖАТ повторяющиеся стемы ("кошка", "нейрон") — то
// есть этот фикстур-тест реально проходит через обучение
// TrainableEncoder (encoder.ts, п.7) внутри одного прогона, а не только
// через бутстрап. Это по-прежнему бит-в-бит детерминировано между
// независимыми запусками (в encoder.ts нет Math.random — только
// арифметика над стемами), но означает, что bench/cluster-baseline.json
// зависит от алгоритма обучения координат, а не только от симуляции
// нейронов. Расхождение при --compare после правки core-cluster.ts или
// neural-core.ts (не трогающей encoder.ts) — настоящая регрессия.
// Расхождение после правки encoder.ts — ожидаемо, пересохраните
// baseline и объективно проверьте обучение через bench-encoder.ts.
//
// Запуск:
//   npx tsx scripts/bench-core-cluster.ts --save bench/cluster-baseline.json
//   npx tsx scripts/bench-core-cluster.ts --compare bench/cluster-baseline.json
// ═══════════════════════════════════════════════════════════════════

import { writeFileSync, readFileSync } from 'fs';
import { CoreCluster } from '../src/lib/neuromorphic/core-cluster';

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
  'кошка снова спит на диване',
  '2+2',
  'что такое нейрон',
];

function buildFixture() {
  const originalRandom = Math.random;
  Math.random = seededRandom(123);
  const cluster = new CoreCluster();
  Math.random = originalRandom;

  const routeLog: string[] = [];
  const confidenceLog: number[] = [];

  for (const text of FIXTURE_TEXTS) {
    const r = cluster.processInput(text);
    routeLog.push(r.routeDecision);
    confidenceLog.push(round(r.confidence));
  }

  let potentialSum = 0;
  let weightSum = 0;
  for (const core of cluster.cores.values()) {
    for (let i = 0; i < core.total; i++) potentialSum += core.potentials[i];
    for (let i = 0; i < core.weights.length; i++) weightSum += core.weights[i];
  }
  let projWeightSum = 0;
  for (const proj of cluster.projections) {
    for (let i = 0; i < proj.weights.length; i++) projWeightSum += proj.weights[i];
  }

  return {
    globalTick: cluster.globalTick,
    interCoreTraffic: cluster.interCoreTraffic,
    routeLog,
    confidenceLog,
    neuromodulators: {
      dopamine: round(cluster.neuromodulators.dopamine),
      serotonin: round(cluster.neuromodulators.serotonin),
      acetylcholine: round(cluster.neuromodulators.acetylcholine),
      norepinephrine: round(cluster.neuromodulators.norepinephrine),
    },
    decisionGroupActivities: cluster.lastDecisionGroupActivities.map(round),
    decisionDominance: round(cluster.lastDecisionDominance),
    potentialSum: round(potentialSum),
    weightSum: round(weightSum),
    projWeightSum: round(projWeightSum),
  };
}

const BENCH_TEXTS = [
  'кошка спит на диване весь день напролёт',
  'собака гуляет в парке с хозяином каждый вечер',
  'нейрон передаёт электрический сигнал через синапс',
  'python используется для машинного обучения и анализа данных',
  'гравитация притягивает все тела с массой друг к другу',
  '2*(15+3)^2',
  'что такое чёрная дыра',
  'кто ты',
];

function benchTiming(): number {
  const originalRandom = Math.random;
  Math.random = seededRandom(99);
  const cluster = new CoreCluster();
  Math.random = originalRandom;

  const ROUNDS = 15;
  const start = performance.now();
  for (let r = 0; r < ROUNDS; r++) {
    for (const text of BENCH_TEXTS) {
      cluster.processInput(text);
    }
  }
  return performance.now() - start;
}

function main() {
  const args = process.argv.slice(2);
  const fixture = buildFixture();
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
    console.error(`\nСверка с ${args[compareIdx + 1]}: ${same ? 'ИДЕНТИЧНО ✓ (поведение каскада не изменилось)' : 'РАСХОЖДЕНИЕ ✗ (поведение изменилось!)'}`);
    console.error(`Время: baseline=${baseline.timeMs.toFixed(1)}мс, сейчас=${timeMs.toFixed(1)}мс, ускорение×${(baseline.timeMs / timeMs).toFixed(2)}`);
    if (!same) {
      console.error('baseline.fixture:', JSON.stringify(baseline.fixture));
      console.error('current.fixture: ', JSON.stringify(fixture));
      process.exitCode = 1;
    }
  }
}

main();
