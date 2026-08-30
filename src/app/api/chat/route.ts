// ═══════════════════════════════════════════════════════════════════
// POST /api/chat — основной эндпоинт чата (АСИНХРОННЫЙ)
// Нейросистема + веб-поиск + самообучение
// ═══════════════════════════════════════════════════════════════════

import { NextRequest, NextResponse } from 'next/server';
import { processChat } from '@/lib/neuromorphic/cognitive-pipeline';

export async function POST(req: NextRequest) {
  try {
    const data = await req.json().catch(() => ({}));
    const message = String(data.message || '').trim();
    if (!message) {
      return NextResponse.json({ error: 'message required' }, { status: 400 });
    }

    // processChat теперь асинхронный (веб-поиск)
    const result = await processChat(message);
    return NextResponse.json(result);
  } catch (e: any) {
    console.error('[Chat API]', e);
    return NextResponse.json({
      answer: `Внутренняя ошибка: ${e.message}`,
      mode: 'error', confidence: 0,
      trace: [`ошибка: ${e.message}`],
      neural: null, core: null,
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
