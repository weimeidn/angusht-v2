// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — Singleton инстанс нейроморфного кластера
// Глобальное состояние, сохранение/загрузка (включая веса сети)
// ═══════════════════════════════════════════════════════════════════

import { CoreCluster } from './core-cluster';
import { NeuralCore, TOTAL_NEURONS, NDIR } from './neural-core';
import {
  saveCoreWeights,
  loadCoreWeights,
  saveProjectionWeights,
  loadProjectionWeights,
} from './persistence';

// Глобальный singleton
let _cluster: CoreCluster | null = null;
let _homeostasisCounter = 0;
let _weightsRestoredOnCreate = false;

export function resetCluster(): void {
  _cluster = null;
  _weightsRestoredOnCreate = false;
}

// Восстанавливает веса ядер и межъядерных проекций из angusht-data/weights/
// в уже созданный кластер. Возвращает true, если что-то реально
// загрузилось (иначе кластер остаётся со свежеинициализированными
// случайными весами — это нормальный "первый запуск").
function restoreNeuralWeightsInto(cluster: CoreCluster): boolean {
  let restoredAny = false;

  for (const core of cluster.cores.values()) {
    const buf = loadCoreWeights(core.id);
    if (buf) {
      core.loadWeights(buf);
      restoredAny = true;
    }
  }

  const projBuffers: ArrayBuffer[] = [];
  let i = 0;
  while (true) {
    const buf = loadProjectionWeights(i);
    if (!buf) break;
    projBuffers.push(buf);
    i++;
  }
  if (projBuffers.length > 0) {
    cluster.loadProjections(projBuffers);
    restoredAny = true;
  }

  if (restoredAny) {
    console.log('[Angusht] Neural weights restored from disk (core synapses + inter-core projections)');
  }
  return restoredAny;
}

export function getCluster(): CoreCluster {
  if (!_cluster) {
    _cluster = new CoreCluster();
    if (!_weightsRestoredOnCreate) {
      restoreNeuralWeightsInto(_cluster);
      _weightsRestoredOnCreate = true;
    }
  }
  return _cluster;
}

// Сохраняет текущие веса ядер и проекций на диск. Дорогая по вводу-выводу
// операция (несколько МБ бинарных данных) — вызывается вместе с
// сохранением текстовой памяти (saveToDisk в cognitive-pipeline.ts),
// а не на каждый тик.
export function persistNeuralWeights(): void {
  const cluster = getCluster();
  for (const core of cluster.cores.values()) {
    saveCoreWeights(core.id, core.serializeWeights());
  }
  const projBuffers = cluster.serializeProjections();
  projBuffers.forEach((buf, i) => saveProjectionWeights(i, buf));
}

// Периодическая фоновая активность (вызывается из API)
export function tickBackground(): void {
  const cluster = getCluster();
  _homeostasisCounter++;

  // Каждые 50 глобальных тиков — гомеостаз
  if (_homeostasisCounter % 50 === 0) {
    cluster.homeostasis();
  }

  // Спонтанная активность для поддержания тонуса
  if (_homeostasisCounter % 10 === 0) {
    cluster.spontaneousStimulation(15);
  }
}

// Полная обработка входа (используется из /api/chat)
export function processNeuralInput(text: string) {
  const cluster = getCluster();
  return cluster.processInput(text);
}

export type { ClusterSnapshot } from './core-cluster';
export type { CoreSnapshot, SpikeEvent } from './neural-core';
export { CORE_CONFIGS, CoreRole, GRID, TOTAL_NEURONS, NDIR } from './neural-core';
export { NeuronType } from './lif-neuron';
