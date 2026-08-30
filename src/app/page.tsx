'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { CORE_CONFIGS } from '@/lib/neuromorphic/neural-core';
import type { ClusterSnapshot } from '@/lib/neuromorphic/core-cluster';
import { ANGUSHT_VERSION_SHORT } from '@/lib/version';
import { AngushtLogo } from '@/components/AngushtLogo';

// ═══════════════════════════════════════════════════════════════
// Angusht v2.4 — Нейроморфная когнитивная архитектура
// 6 ядер × 216K LIF-нейронов = 1 296 000 нейронов
// ═══════════════════════════════════════════════════════════════

interface Message {
  role: 'user' | 'assistant';
  text: string;
  mode?: string;
  confidence?: number;
  trace?: string[];
  neural?: any;
  // Источник ответа (веб-поиск, либо память, изначально пополненная
  // веб-поиском) — чтобы пользователь мог сам проверить факт, а не
  // получить его на веру от системы, которая "самообучилась" на нём.
  source?: { name: string; url?: string } | null;
  // Раскрыт ли блок "почему такой ответ" (trace + нейро-метрики) —
  // эти данные система считает на каждый запрос, но раньше нигде не
  // показывала в чате, только в общей боковой панели.
  detailsOpen?: boolean;
  // Для сообщений ассистента: исходный запрос пользователя — нужен,
  // чтобы отправить обратную связь (👍/👎 + коррекция) с правильным
  // контекстом. У сообщений без query (например, текст ошибки) блок
  // обратной связи не показывается.
  query?: string;
  feedback?: 'positive' | 'negative' | null;
  correcting?: boolean;
  correctionDraft?: string;
}

export default function AngushtPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [snapshot, setSnapshot] = useState<ClusterSnapshot | null>(null);
  // 'thinking' сразу при отправке, 'web' — если ответ не пришёл за
  // ~900мс (локальная нейронная обработка обычно укладывается в
  // десятки-сотни миллисекунд; дольше — почти всегда значит, что
  // система ушла в веб-поиск, и это стоит показать явно, а не держать
  // один и тот же неизменный индикатор для миллисекундной и
  // многосекундной операции).
  const [loadingPhase, setLoadingPhase] = useState<'thinking' | 'web' | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'cores' | 'topology'>('chat');
  const [selectedCore, setSelectedCore] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Polling нейро-статуса
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch('/api/neuro/status');
        const data = await r.json();
        setSnapshot(data);
      } catch {}
    };
    poll();
    const iv = setInterval(poll, 2000);
    return () => clearInterval(iv);
  }, []);

  // Автопрокрутка чата
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = useCallback(async () => {
    const msg = input.trim();
    if (!msg || loading) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: msg }]);
    setLoading(true);
    setLoadingPhase('thinking');
    const webHintTimer = setTimeout(() => setLoadingPhase('web'), 900);

    try {
      const r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      });
      const data = await r.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer,
        mode: data.mode,
        confidence: data.confidence,
        trace: data.trace,
        neural: data.neural,
        source: data.source ?? null,
        query: msg,
        feedback: null,
      }]);
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', text: `Ошибка: ${e.message}` }]);
    }
    clearTimeout(webHintTimer);
    setLoadingPhase(null);
    setLoading(false);
  }, [input, loading]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  // ── Обратная связь по ответу: 👍/👎 + опциональная коррекция ──
  // Единственное место в системе, где подкрепление зависит от того,
  // был ли ответ ФАКТИЧЕСКИ верным, а не только от того, что система
  // вообще что-то нашла (см. submitFeedback в cognitive-pipeline.ts).
  const sendFeedback = useCallback(async (index: number, positive: boolean, correction?: string) => {
    const m = messages[index];
    if (!m || m.role !== 'assistant' || !m.query) return;
    try {
      await fetch('/api/chat/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: m.query, answer: m.text, mode: m.mode, positive, correction }),
      });
    } catch {}
    setMessages(prev => prev.map((mm, i) =>
      i === index ? { ...mm, feedback: positive ? 'positive' : 'negative', correcting: false } : mm
    ));
  }, [messages]);

  const openCorrection = useCallback((index: number) => {
    setMessages(prev => prev.map((mm, i) => i === index ? { ...mm, correcting: true } : mm));
  }, []);

  const closeCorrection = useCallback((index: number) => {
    setMessages(prev => prev.map((mm, i) => i === index ? { ...mm, correcting: false } : mm));
  }, []);

  // Раскрыть/скрыть блок "почему такой ответ" (trace + нейро-метрики
  // конкретного сообщения) — данные уже вычисляются сервером на каждый
  // запрос (ChatResponse.trace/.neural), просто раньше нигде не
  // показывались в чате.
  const toggleDetails = useCallback((index: number) => {
    setMessages(prev => prev.map((mm, i) => i === index ? { ...mm, detailsOpen: !mm.detailsOpen } : mm));
  }, []);

  const stimulate = async (coreId?: string) => {
    try {
      await fetch('/api/neuro/stimulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 80, coreId: coreId || null }),
      });
    } catch {}
  };

  const resetAll = async () => {
    // Обнуляет всю текущую активность нейронной сети (не веса на диске,
    // но состояние ядер) — раньше срабатывало по одному клику без
    // подтверждения, и случайное нажатие ничем не отличалось от
    // намеренного сброса.
    const confirmed = window.confirm(
      'Сбросить текущую активность всех 6 ядер сети?\n\nВеса STDP на диске (обученные пути) не удаляются, но всё текущее состояние возбуждения ядер обнулится.'
    );
    if (!confirmed) return;
    try {
      await fetch('/api/neuro/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'reset' }),
      });
    } catch {}
  };

  const totalNeurons = 1296000;
  const totalSynapses = snapshot?.totalSynapses || 7776000;
  const totalActive = snapshot?.totalActive || 0;
  const totalStrong = snapshot?.cores.reduce((s, c) => s + c.strongSynapses, 0) || 0;

  return (
    <div className="h-screen overflow-hidden bg-neutral-950 text-neutral-100 flex flex-col">
      {/* ═══ HEADER ═══ */}
      <header className="border-b border-neutral-800 px-4 py-3 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg border border-amber-500/50 flex items-center justify-center overflow-hidden">
            <AngushtLogo className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-wide">Angusht {ANGUSHT_VERSION_SHORT}</h1>
            <p className="text-[10px] text-neutral-500">Нейроморфная когнитивная архитектура</p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-[11px]">
          <div className="text-neutral-400">
            <span className="text-amber-400 font-medium">{totalNeurons.toLocaleString('ru-RU')}</span> нейронов
          </div>
          <div className="text-neutral-400">
            <span className="text-emerald-400 font-medium">{totalSynapses.toLocaleString('ru-RU')}</span> синапсов
          </div>
          <div className="text-neutral-400">
            Активных: <span className="text-red-400 font-medium">{totalActive.toLocaleString('ru-RU')}</span>
          </div>
          <div className="text-neutral-400">
            Устойч. путей: <span className="text-cyan-400 font-medium">{totalStrong.toLocaleString('ru-RU')}</span>
          </div>
        </div>
      </header>

      {/* ═══ TABS ═══ */}
      <div className="flex border-b border-neutral-800 px-4 flex-shrink-0">
        {(['chat', 'cores', 'topology'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-xs font-medium transition-colors ${activeTab === tab ? 'text-amber-400 border-b-2 border-amber-400' : 'text-neutral-500 hover:text-neutral-300'}`}
          >
            {tab === 'chat' ? 'Чат' : tab === 'cores' ? '6 Ядер' : 'Топология'}
          </button>
        ))}
      </div>

      {/* ═══ MAIN CONTENT ═══ */}
      <main className="flex-1 flex min-h-0">
        {/* LEFT: Chat panel */}
        <div className={`${activeTab === 'chat' ? 'flex-1' : 'hidden'} flex flex-col min-w-0`}>
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center py-16">
                <div className="mb-4 flex justify-center opacity-30">
                  <AngushtLogo className="w-12 h-12" />
                </div>
                <h2 className="text-xl font-semibold mb-2">Нейроморфная когнитивная система</h2>
                <p className="text-neutral-500 text-sm max-w-lg mx-auto mb-8">
                  6 специализированных ядер (1.3M LIF-нейронов) с реальным STDP, рефрактерным периодом, нейромодуляторами.
                  Нейросистема управляет маршрутизацией, памятью и уверенностью.
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 max-w-xl mx-auto">
                  {[
                    ['2*(15+3)^2', 'Математика'],
                    ['Что такое нейрон?', 'Знания'],
                    ['кто ты', 'О системе'],
                    ['2+3*4-1', 'Арифметика'],
                  ].map(([q, label]) => (
                    <button key={q} onClick={() => { setInput(q); }}
                      className="bg-neutral-900 border border-neutral-800 rounded-lg p-2.5 text-left hover:border-neutral-600 transition-colors">
                      <div className="text-[10px] text-neutral-500 mb-1">{label}</div>
                      <div className="text-xs font-medium text-neutral-300">{q}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}>
                {m.role === 'assistant' && (
                  <div className="w-7 h-7 rounded-md border border-amber-500/30 flex items-center justify-center flex-shrink-0">
                    <AngushtLogo className="w-4 h-4" />
                  </div>
                )}
                <div className={`max-w-2xl ${m.role === 'user' ? 'bg-neutral-800 rounded-2xl rounded-tr-sm px-4 py-2.5' : ''}`}>
                  <div className={`text-sm whitespace-pre-wrap leading-relaxed ${m.role === 'user' ? '' : 'text-neutral-200'}`}>{m.text}</div>

                  {/* Источник ответа (если он пришёл из веб-поиска или
                      из памяти, изначально пополненной веб-поиском) +
                      разворачиваемые детали "почему такой ответ"
                      (trace + нейро-метрики конкретного сообщения). */}
                  {m.role === 'assistant' && (m.source || (m.trace && m.trace.length > 0)) && (
                    <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
                      {m.source && (
                        m.source.url ? (
                          <a
                            href={m.source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 hover:underline"
                          >
                            🔗 Источник: {m.source.name}
                          </a>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-neutral-500">
                            🔗 Источник: {m.source.name}
                          </span>
                        )
                      )}
                      {m.trace && m.trace.length > 0 && (
                        <button
                          onClick={() => toggleDetails(i)}
                          className="text-neutral-500 hover:text-neutral-300 transition-colors"
                        >
                          {m.detailsOpen ? '▾ Скрыть детали' : '▸ Почему такой ответ?'}
                        </button>
                      )}
                    </div>
                  )}

                  {m.detailsOpen && m.neural && (
                    <div className="mt-1.5 max-w-md bg-neutral-900 border border-neutral-800 rounded-lg p-2.5 space-y-1.5 text-[11px] text-neutral-400">
                      <div>
                        Маршрут: <span className="text-neutral-200">{m.neural.routeDecision}</span>
                        {m.neural.altRoute && <span> (альт.: {m.neural.altRoute})</span>}
                        {' · '}доминирование {(m.neural.decisionDominance * 100).toFixed(0)}%
                      </div>
                      <div>
                        Уверенность сети: {(m.neural.neuralConfidence * 100).toFixed(0)}%
                        {' · '}знакомство: {(m.neural.familiarity * 100).toFixed(0)}%
                      </div>
                      <div>
                        Каскад: {m.neural.cascadeRounds}/10 раундов {m.neural.cascadeConverged ? '(сошёлся досрочно)' : '(упёрся в лимит)'}
                      </div>
                      <div>
                        DA {m.neural.neuromodulators.dopamine.toFixed(2)} · 5HT {m.neural.neuromodulators.serotonin.toFixed(2)} · ACh {m.neural.neuromodulators.acetylcholine.toFixed(2)} · NE {m.neural.neuromodulators.norepinephrine.toFixed(2)}
                      </div>
                      <div className="pt-1.5 border-t border-neutral-800 space-y-0.5 max-h-40 overflow-y-auto">
                        {m.trace?.filter(Boolean).map((t, ti) => (
                          <div key={ti} className="text-neutral-500">· {t}</div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Обратная связь: 👍/👎 + коррекция. Единственный в
                      системе источник подкрепления, завязанный на
                      реальную правильность, а не на факт "что-то нашлось". */}
                  {m.role === 'assistant' && m.query && (
                    <div className="mt-2">
                      {!m.feedback && !m.correcting && (
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => sendFeedback(i, true)}
                            title="Ответ верный"
                            className="w-6 h-6 rounded-md flex items-center justify-center text-xs bg-neutral-900 border border-neutral-800 hover:border-emerald-500/50 hover:bg-emerald-500/10 transition-colors"
                          >
                            <span className="text-emerald-500">✓</span>
                          </button>
                          <button
                            onClick={() => openCorrection(i)}
                            title="Ответ неверный"
                            className="w-6 h-6 rounded-md flex items-center justify-center text-xs bg-neutral-900 border border-neutral-800 hover:border-red-500/50 hover:bg-red-500/10 transition-colors"
                          >
                            <span className="text-red-500">✕</span>
                          </button>
                        </div>
                      )}

                      {m.correcting && (
                        <div className="mt-1.5 max-w-md bg-neutral-900 border border-red-500/20 rounded-lg p-2.5 space-y-2">
                          <div className="text-[11px] text-neutral-400">Какой ответ правильный? (необязательно)</div>
                          <textarea
                            value={m.correctionDraft || ''}
                            onChange={e => {
                              const val = e.target.value;
                              setMessages(prev => prev.map((mm, ii) => ii === i ? { ...mm, correctionDraft: val } : mm));
                            }}
                            placeholder="Введите правильный ответ..."
                            rows={2}
                            className="w-full bg-neutral-950 border border-neutral-800 rounded-md px-2 py-1.5 text-xs outline-none focus:border-red-500/40 placeholder:text-neutral-600 resize-none"
                          />
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => sendFeedback(i, false, m.correctionDraft)}
                              className="bg-red-500/90 hover:bg-red-500 text-white text-[11px] font-medium rounded-md px-2.5 py-1 transition-colors"
                            >
                              {m.correctionDraft?.trim() ? 'Отправить исправление' : 'Отметить как неверное'}
                            </button>
                            <button
                              onClick={() => closeCorrection(i)}
                              className="text-[11px] text-neutral-500 hover:text-neutral-300 transition-colors"
                            >
                              Отмена
                            </button>
                          </div>
                        </div>
                      )}

                      {m.feedback === 'positive' && (
                        <div className="text-[11px] text-emerald-500/80">✓ Отмечено как верное — сеть подкреплена, память закреплена</div>
                      )}
                      {m.feedback === 'negative' && (
                        <div className="text-[11px] text-red-500/80">✕ Отмечено как неверное — сеть ослаблена{m.correctionDraft?.trim() ? ', коррекция сохранена и будет приоритетной для этого вопроса' : ''}</div>
                      )}
                    </div>
                  )}
                </div>
                {m.role === 'user' && (
                  <div className="w-7 h-7 rounded-md bg-neutral-700 flex items-center justify-center text-xs flex-shrink-0">Вы</div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-md border border-amber-500/30 flex items-center justify-center">
                  <AngushtLogo className="w-4 h-4" />
                </div>
                <div className="text-sm text-neutral-500 animate-pulse">
                  {loadingPhase === 'web'
                    ? 'Ищу в интернете (Wikipedia / DuckDuckGo)...'
                    : 'Обработка через 6 ядер...'}
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div className="border-t border-neutral-800 p-3 flex-shrink-0">
            <div className="flex gap-2 max-w-3xl mx-auto">
              <input
                value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey} placeholder="Задайте вопрос..."
                className="flex-1 bg-neutral-900 border border-neutral-700 rounded-xl px-4 py-2.5 text-sm outline-none focus:border-amber-500/50 transition-colors placeholder:text-neutral-600"
                disabled={loading}
              />
              <button onClick={send} disabled={loading || !input.trim()}
                className="bg-amber-500 hover:bg-amber-400 disabled:bg-neutral-700 text-neutral-900 font-medium rounded-xl px-4 py-2.5 text-sm transition-colors">
                {loading ? '...' : '↑'}
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT: Neural dashboard (always visible) */}
        <div className={`${activeTab === 'chat' ? 'w-80' : 'flex-1'} border-l border-neutral-800 overflow-y-auto p-3 space-y-3 flex-shrink-0`}>
          {/* Neuromodulators */}
          {snapshot && (
            <div className="space-y-2">
              <div className="text-[10px] tracking-widest text-neutral-500 font-medium">НЕЙРОМОДУЛЯТОРЫ</div>
              <div className="grid grid-cols-2 gap-2">
                {([
                  ['dopamine', 'Дофамин', '#f59e0b'],
                  ['serotonin', 'Серотонин', '#10b981'],
                  ['acetylcholine', 'Ацетилхолин', '#3b82f6'],
                  ['norepinephrine', 'Норадреналин', '#ef4444'],
                ] as const).map(([key, label, color]) => (
                  <div key={key} className="bg-neutral-900 rounded-lg p-2">
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-neutral-400">{label}</span>
                      <span style={{ color }} className="font-medium">{(snapshot.neuromodulators[key] * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${snapshot.neuromodulators[key] * 100}%`, backgroundColor: color }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 6 Cores Grid */}
          <div className="text-[10px] tracking-widest text-neutral-500 font-medium">6 ЯДЕР</div>
          {snapshot?.cores.map(core => (
            <div
              key={core.id}
              onClick={() => setSelectedCore(selectedCore === core.id ? null : core.id)}
              className={`bg-neutral-900 rounded-lg p-3 cursor-pointer hover:bg-neutral-800/80 transition-colors border ${selectedCore === core.id ? 'border-amber-500/50' : 'border-transparent'}`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: core.color }} />
                  <span className="text-xs font-medium">{core.name}</span>
                </div>
                <div className="flex gap-2 text-[10px]">
                  <span className="text-neutral-500">{core.active.toLocaleString('ru-RU')} акт.</span>
                  <span className="text-neutral-600">{core.firedThisCycle} спайков</span>
                </div>
              </div>
              {/* Activity bar */}
              <div className="h-1 bg-neutral-800 rounded-full overflow-hidden mb-1.5">
                <div className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min(100, (core.active / 500) * 100)}%`,
                    backgroundColor: core.color,
                  }} />
              </div>
              {/* Layer profile (mini chart) */}
              {selectedCore === core.id && core.layerProfile && (
                <div className="mt-2 flex gap-px h-8 items-end">
                  {core.layerProfile.map((v, i) => (
                    <div key={i} className="flex-1 rounded-t-sm transition-all"
                      style={{
                        height: `${Math.min(32, Math.max(1, v * 0.5))}px`,
                        backgroundColor: core.color,
                        opacity: 0.3 + Math.min(0.7, v / 20),
                      }} />
                  ))}
                </div>
              )}
              <div className="flex justify-between text-[10px] text-neutral-600">
                <span>Ср. вес: {core.avgWeight.toFixed(3)}</span>
                <span>Устойч.: {core.strongSynapses.toLocaleString('ru-RU')}</span>
              </div>
            </div>
          ))}

          {/* Inter-core traffic */}
          {snapshot && (
            <div className="text-[10px] text-neutral-600">
              Межъядерный трафик: {snapshot.interCoreTraffic.toLocaleString('ru-RU')} сигналов
              <br />Тик: {snapshot.globalTick}
              <br />Каскад: {snapshot.cascadeRounds}/10 раундов
              {' '}
              <span className={snapshot.cascadeConverged ? 'text-emerald-500' : 'text-amber-500'}>
                {snapshot.cascadeConverged ? '(сошёлся досрочно)' : '(упёрся в лимит)'}
              </span>
            </div>
          )}

          {/* Controls */}
          <div className="flex gap-2 flex-wrap">
            <button onClick={() => stimulate()} className="bg-neutral-800 hover:bg-neutral-700 text-xs px-3 py-1.5 rounded-lg transition-colors">
              ⚡ Стимулировать
            </button>
            <button onClick={() => resetAll()} className="bg-neutral-800 hover:bg-neutral-700 text-xs px-3 py-1.5 rounded-lg transition-colors">
              ↻ Сбросить
            </button>
          </div>
        </div>

        {/* CORES TAB: full view */}
        {activeTab === 'cores' && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-lg font-semibold mb-4">6 специализированных ядер</h2>
              <p className="text-neutral-500 text-sm mb-6">
                Каждое ядро — 60×60×60 решётка LIF-нейронов (216 000). 80% возбуждающих, 20% тормозных.
                STDP, рефрактерный период, spike-frequency adaptation.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {snapshot?.cores.map(core => (
                  <div key={core.id} className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: core.color }} />
                      <h3 className="text-sm font-semibold">{core.name}</h3>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-neutral-400">Активных:</div>
                      <div className="font-medium text-right">{core.active.toLocaleString('ru-RU')}</div>
                      <div className="text-neutral-400">Спайков:</div>
                      <div className="font-medium text-right">{core.firedThisCycle}</div>
                      <div className="text-neutral-400">Ср. вес:</div>
                      <div className="font-medium text-right">{(core.avgWeight ?? 0).toFixed(4)}</div>
                      <div className="text-neutral-400">Устойч. синапсов:</div>
                      <div className="font-medium text-right">{core.strongSynapses.toLocaleString('ru-RU')}</div>
                      <div className="text-neutral-400">Тормозных спайков:</div>
                      <div className="font-medium text-right">{core.inhibitoryFired}</div>
                      <div className="text-neutral-400">Ср. потенциал:</div>
                      {/* ?? 0 — защитный фолбэк: null здесь означал бы NaN на
                          бэкенде (JSON.stringify(NaN) === "null"), см.
                          устранённую находку про знаковый сдвиг хеша в
                          neural-core.ts/core-cluster.ts (v2.4.4). UI не
                          должен падать, даже если где-то ещё всплывёт
                          похожий на первый взгляд безобидный edge case. */}
                      <div className="font-medium text-right">{(core.avgPotential ?? 0).toFixed(4)}</div>
                    </div>
                    {/* Layer profile chart */}
                    <div className="mt-3 flex gap-px h-12 items-end">
                      {core.layerProfile.map((v, i) => (
                        <div key={i} className="flex-1 rounded-t-sm"
                          style={{
                            height: `${Math.max(1, Math.min(48, v * 0.8))}px`,
                            backgroundColor: core.color,
                            opacity: 0.2 + Math.min(0.8, v / 15),
                          }} />
                      ))}
                    </div>
                    <div className="flex justify-between mt-2">
                      <button onClick={() => stimulate(core.id)} className="text-[10px] bg-neutral-800 hover:bg-neutral-700 px-2 py-1 rounded transition-colors">
                        Стимулировать
                      </button>
                      <span className="text-[10px] text-neutral-600">Слой 1 → 60</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TOPOLOGY TAB */}
        {activeTab === 'topology' && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-lg font-semibold mb-4">Топология межъядерных связей</h2>
              <p className="text-neutral-500 text-sm mb-6">
                15 направленных связей между 6 ядрами. Модуляторное ядро управляет всеми через нейромодуляторы.
              </p>
              {/* Topology visualization */}
              <div className="relative bg-neutral-900 rounded-xl border border-neutral-800 p-8 min-h-[400px]">
                <svg viewBox="0 0 600 400" className="w-full h-full">
                  {/* Links */}
                  {snapshot?.interCoreLinks.filter(l => l.active).map((link, i) => {
                    const from = CORE_CONFIGS.find(c => c.id === link.from);
                    const to = CORE_CONFIGS.find(c => c.id === link.to);
                    if (!from || !to) return null;
                    const positions: Record<string, [number, number]> = {
                      sensory: [300, 60], associative: [150, 160],
                      analytical: [450, 160], temporal: [100, 300],
                      executive: [300, 340], modulatory: [500, 300],
                    };
                    const [x1, y1] = positions[link.from];
                    const [x2, y2] = positions[link.to];
                    return (
                      <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                        stroke={to.color} strokeWidth={link.strength * 1.5}
                        strokeOpacity={0.3} />
                    );
                  })}
                  {/* Nodes */}
                  {CORE_CONFIGS.map(core => {
                    const positions: Record<string, [number, number]> = {
                      sensory: [300, 60], associative: [150, 160],
                      analytical: [450, 160], temporal: [100, 300],
                      executive: [300, 340], modulatory: [500, 300],
                    };
                    const [cx, cy] = positions[core.id];
                    const snap = snapshot?.cores.find(c => c.id === core.id);
                    const r = 20 + (snap?.active || 0) * 0.01;
                    return (
                      <g key={core.id}>
                        <circle cx={cx} cy={cy} r={Math.min(35, r)}
                          fill={core.color} fillOpacity={0.15}
                          stroke={core.color} strokeWidth={1.5} />
                        <text x={cx} y={cy - 4} textAnchor="middle" fill={core.color}
                          fontSize="9" fontWeight="600">{core.id.slice(0, 4).toUpperCase()}</text>
                        <text x={cx} y={cy + 8} textAnchor="middle" fill="#999" fontSize="7">
                          {snap ? `${snap.active} akt` : ''}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
              {/* Links table */}
              <div className="mt-4 space-y-1">
                {snapshot?.interCoreLinks.map((link, i) => (
                  <div key={i} className={`flex items-center gap-3 text-xs py-1.5 px-3 rounded ${link.active ? 'bg-neutral-900' : 'bg-neutral-900/50 opacity-50'}`}>
                    <span className="text-neutral-400 w-24 truncate">{link.from}</span>
                    <span className="text-neutral-600">→</span>
                    <span className="text-neutral-400 w-24 truncate">{link.to}</span>
                    <div className="flex-1 h-1 bg-neutral-800 rounded-full">
                      <div className="h-full bg-amber-500/50 rounded-full" style={{ width: `${link.strength * 60}%` }} />
                    </div>
                    <span className="text-neutral-500 w-12 text-right">{link.strength.toFixed(1)}</span>
                    <span className={`w-2 h-2 rounded-full ${link.active ? 'bg-emerald-500' : 'bg-neutral-600'}`} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}