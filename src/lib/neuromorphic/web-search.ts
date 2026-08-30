// ═══════════════════════════════════════════════════════════════════
// Angusht v2.4 — Модуль веб-поиска
// PARALLEL: все движки запускаются одновременно через Promise.allSettled
// CACHE + RATE-LIMIT: кеш ответов, мин. интервал между запросами
// ═══════════════════════════════════════════════════════════════════

export interface SearchResult {
  answer: string;
  source: string;
  url: string;
  confidence: number;
}

import { ANGUSHT_VERSION_SHORT } from '../version';
const USER_AGENT = `Angusht-${ANGUSHT_VERSION_SHORT}-Bot/1.0 (neuromorphic-cognitive-system)`;

// ── Кеш ответов (TTL 24 часа) ──
const CACHE_TTL = 24 * 60 * 60 * 1000;
const searchCache = new Map<string, { result: SearchResult; ts: number }>();
const MAX_CACHE = 500;

function cacheKey(query: string): string {
  return query.toLowerCase().trim().slice(0, 200);
}

function cacheGet(query: string): SearchResult | null {
  const k = cacheKey(query);
  const entry = searchCache.get(k);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL) {
    searchCache.delete(k);
    return null;
  }
  return entry.result;
}

function cacheSet(query: string, result: SearchResult): void {
  const k = cacheKey(query);
  if (searchCache.size >= MAX_CACHE) {
    // Удаляем самые старые
    const oldest = [...searchCache.entries()].sort((a, b) => a[1].ts - b[1].ts);
    for (let i = 0; i < 50; i++) searchCache.delete(oldest[i][0]);
  }
  searchCache.set(k, { result, ts: Date.now() });
}

// ── Rate limiting (per-domain: не чаще 1 запроса в 2 сек на домен) ──
const domainLastRequest = new Map<string, number>();
const MIN_REQUEST_INTERVAL = 2000;

function extractDomain(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return '_unknown';
  }
}

async function rateLimitedFetch(url: string, init?: RequestInit): Promise<Response> {
  const domain = extractDomain(url);
  const now = Date.now();
  const lastReq = domainLastRequest.get(domain) || 0;
  const wait = MIN_REQUEST_INTERVAL - (now - lastReq);
  if (wait > 0) await new Promise(r => setTimeout(r, wait));
  domainLastRequest.set(domain, Date.now());
  return fetch(url, init);
}

// ── Утилиты ──
function stripHtml(html: string): string {
  return html
    .replace(/<br\s*\/?/gi, '\n')
    .replace(/<li[^>]*>/gi, '\n- ')
    .replace(/<[^>]+>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&mdash;/g, '—')
    .replace(/&ndash;/g, '–')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function truncate(s: string, max: number = 1500): string {
  if (s.length <= max) return s;
  return s.slice(0, max).replace(/\s+\S*$/, '') + '…';
}

// ── Единый Wikipedia поиск (RU и EN) ──
// Рефакторинг: убрано дублирование кода между RU и EN версиями
interface WikiSearchResult {
  title: string;
  snippet: string;
}

async function searchWikipedia(query: string, lang: 'ru' | 'en'): Promise<SearchResult | null> {
  const baseUrl = `https://${lang}.wikipedia.org/w/api.php`;
  const sourceLabel = lang === 'ru' ? 'Wikipedia (RU)' : 'Wikipedia (EN)';
  const baseConfidence = lang === 'ru' ? 0.85 : 0.75;
  const snippetConfidence = lang === 'ru' ? 0.7 : 0.6;

  const searchUrl = `${baseUrl}?${new URLSearchParams({
    action: 'query', list: 'search', srsearch: query,
    format: 'json', utf8: '1', srlimit: '3',
  })}`;

  let result: WikiSearchResult | null = null;

  try {
    const r = await rateLimitedFetch(searchUrl, {
      headers: { 'User-Agent': USER_AGENT },
      signal: AbortSignal.timeout(5000),
    });
    const data = await r.json();
    const results = data?.query?.search;
    if (!results || results.length === 0) return null;

    const title = results[0].title;
    const snippet = stripHtml(results[0].snippet);
    result = { title, snippet };
  } catch {
    return null;
  }

  const { title, snippet } = result!;
  const articleUrl = `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, '_'))}`;

  // Пытаемся получить полный extract
  try {
    const extractUrl = `${baseUrl}?${new URLSearchParams({
      action: 'query', titles: title, prop: 'extracts',
      exintro: 'true', explaintext: 'true', format: 'json',
    })}`;
    const r2 = await rateLimitedFetch(extractUrl, {
      headers: { 'User-Agent': USER_AGENT },
      signal: AbortSignal.timeout(5000),
    });
    const data2 = await r2.json();
    const pages = data2?.query?.pages;
    if (pages) {
      const pageId = Object.keys(pages)[0];
      const extract = pages[pageId]?.extract;
      if (extract && extract.length > snippet.length) {
        return { answer: truncate(extract), source: sourceLabel, url: articleUrl, confidence: baseConfidence };
      }
    }
  } catch {}

  return { answer: snippet, source: sourceLabel, url: articleUrl, confidence: snippetConfidence };
}

// ── DuckDuckGo Instant Answer API ──
async function searchDuckDuckGoAPI(query: string): Promise<SearchResult | null> {
  const url = `https://api.duckduckgo.com/?${new URLSearchParams({
    q: query, format: 'json', no_html: '1', skip_disambig: '1',
  })}`;
  const r = await rateLimitedFetch(url, { signal: AbortSignal.timeout(4000) });
  const data = await r.json();

  if (data.Abstract) {
    return { answer: truncate(data.Abstract), source: 'DuckDuckGo', url: data.AbstractURL || '', confidence: 0.8 };
  }
  if (data.Answer) {
    return { answer: stripHtml(data.Answer), source: 'DuckDuckGo', url: data.AbstractURL || '', confidence: 0.75 };
  }
  if (data.Definition) {
    return { answer: stripHtml(data.Definition), source: 'DuckDuckGo', url: '', confidence: 0.75 };
  }
  if (data.RelatedTopics && data.RelatedTopics.length > 0) {
    const texts = data.RelatedTopics
      .filter((t: { Text?: string }) => t.Text)
      .map((t: { Text: string }) => t.Text)
      .slice(0, 3);
    if (texts.length > 0) {
      return { answer: truncate(texts.join('\n\n')), source: 'DuckDuckGo', url: data.RelatedTopics[0]?.FirstURL || '', confidence: 0.6 };
    }
  }
  return null;
}

// ── DuckDuckGo HTML (хрупкий, но полезный) ──
async function searchDuckDuckGoHTML(query: string): Promise<SearchResult | null> {
  const url = `https://html.duckduckgo.com/html/?${new URLSearchParams({ q: query })}`;
  const r = await rateLimitedFetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (compatible; Angusht-bot/1.0)' },
    signal: AbortSignal.timeout(5000),
  });
  const html = await r.text();

  const results: { title: string; snippet: string }[] = [];
  const resultRegex = /class="result__a"[^>]*>([^<]+)<\/a>[\s\S]*?class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
  let match;
  while ((match = resultRegex.exec(html)) !== null && results.length < 3) {
    results.push({ title: stripHtml(match[1]), snippet: stripHtml(match[2]) });
  }
  if (results.length === 0) {
    const altRegex = /<a rel="nofollow" class="result__a"[^>]*>([^<]+)<\/a>[\s\S]*?<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
    while ((match = altRegex.exec(html)) !== null && results.length < 3) {
      results.push({ title: stripHtml(match[1]), snippet: stripHtml(match[2]) });
    }
  }

  if (results.length > 0) {
    const combined = results.map((r, i) => `${i + 1}. ${r.title}\n${r.snippet}`).join('\n\n');
    return { answer: truncate(combined, 2000), source: 'DuckDuckGo Web', url: '', confidence: 0.55 };
  }
  return null;
}

// ═══════════════════════════════════════════════════════════════════
// ГЛАВНАЯ ФУНКЦИЯ ПОИСКА
// PARALLEL: все 4 движка запускаются одновременно
// Возвращаем лучший результат (max confidence)
// ═══════════════════════════════════════════════════════════════════

export async function webSearch(query: string): Promise<SearchResult | null> {
  // Проверяем кеш
  const cached = cacheGet(query);
  if (cached) {
    searchStats.cacheHits++;
    return cached;
  }

  // Запускаем все движки параллельно
  const engines = [
    { name: 'Wikipedia RU', fn: () => searchWikipedia(query, 'ru') },
    { name: 'DuckDuckGo API', fn: () => searchDuckDuckGoAPI(query) },
    { name: 'DuckDuckGo HTML', fn: () => searchDuckDuckGoHTML(query) },
    { name: 'Wikipedia EN', fn: () => searchWikipedia(query, 'en') },
  ];

  const settled = await Promise.allSettled(
    engines.map(e => e.fn().catch(() => null))
  );

  // Собираем все успешные результаты, берём лучший
  let bestResult: SearchResult | null = null;
  for (let i = 0; i < settled.length; i++) {
    const s = settled[i];
    if (s.status === 'fulfilled' && s.value && s.value.answer.length > 20) {
      if (!bestResult || s.value.confidence > bestResult.confidence) {
        bestResult = s.value;
      }
    }
  }

  // Кешируем результат
  if (bestResult) {
    cacheSet(query, bestResult);
  }

  return bestResult;
}

// Статистика поиска
export const searchStats = {
  totalSearches: 0,
  successfulSearches: 0,
  cacheHits: 0,
  sourcesUsed: new Map<string, number>(),
  lastSearchTime: 0,
  cacheSize: () => searchCache.size,
};
