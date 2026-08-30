// ═══════════════════════════════════════════════════════════════════
// Минимальная защита управляющих нейро-эндпоинтов (control/stimulate/
// reinforce) от постороннего вызова.
//
// Это НЕ полноценная авторизация — просто same-origin проверка:
// запрос должен приходить со страницы, открытой с того же хоста
// (Origin, либо Referer как запасной вариант, совпадает с Host
// запроса). Она не защищает от человека, который открыл DevTools на
// самой странице Angusht, — это требовало бы настоящей аутентификации,
// которой в проекте пока нет. Но она перекрывает самый дешёвый вектор:
// сторонний сайт/скрипт в другой вкладке браузера, посылающий fetch на
// эти эндпоинты (CSRF-подобная атака), и случайный доступ из той же
// сети, если порт когда-нибудь будет проброшен наружу без отдельного
// прокси/авторизации.
// ═══════════════════════════════════════════════════════════════════

import { NextRequest } from 'next/server';

export function isSameOriginRequest(req: NextRequest): boolean {
  const host = req.headers.get('host');
  if (!host) return false;

  const origin = req.headers.get('origin');
  if (origin) {
    try {
      return new URL(origin).host === host;
    } catch {
      return false;
    }
  }

  // Часть браузеров/сценариев не шлёт Origin на same-origin POST —
  // подстраховываемся Referer, если он есть.
  const referer = req.headers.get('referer');
  if (referer) {
    try {
      return new URL(referer).host === host;
    } catch {
      return false;
    }
  }

  // Ни Origin, ни Referer нет — типично для прямого curl/скрипта извне,
  // не для браузерного fetch со страницы Angusht. Отклоняем.
  return false;
}
