// ═══════════════════════════════════════════════════════════════════
// POST /api/chat/feedback — обратная связь человека по конкретному
// ответу: 👍 (подтверждение) или 👎 (неверно, опционально с полем
// правильного ответа). См. submitFeedback в cognitive-pipeline.ts —
// это первое место в системе, где подкрепление зависит от реальной
// правильности, а не только от факта "что-то нашлось".
// ═══════════════════════════════════════════════════════════════════

import { NextRequest, NextResponse } from 'next/server';
import { submitFeedback } from '@/lib/neuromorphic/cognitive-pipeline';

export async function POST(req: NextRequest) {
  try {
    const data = await req.json().catch(() => ({}));
    const query = String(data.query || '').trim();
    const answer = String(data.answer || '').trim();
    const mode = String(data.mode || 'unknown');
    const positive = Boolean(data.positive);
    const correctionRaw = data.correction;
    const correction = typeof correctionRaw === 'string' && correctionRaw.trim().length > 0
      ? correctionRaw.trim()
      : undefined;

    if (!query || !answer) {
      return NextResponse.json({ error: 'query and answer required' }, { status: 400 });
    }

    const result = submitFeedback(query, answer, mode, positive, correction);
    return NextResponse.json({ ok: true, ...result });
  } catch (e: any) {
    console.error('[Feedback API]', e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
