import { NextRequest, NextResponse } from 'next/server';
import { getCluster, resetCluster } from '@/lib/neuromorphic/index';
import { isSameOriginRequest } from '@/lib/same-origin';

export async function POST(req: NextRequest) {
  // Минимальная защита: этот эндпоинт обнуляет всю текущую активность
  // сети и должен вызываться только со страницы Angusht, а не любым
  // сторонним скриптом/сайтом, знающим адрес (см. same-origin.ts).
  if (!isSameOriginRequest(req)) {
    return NextResponse.json({ error: 'forbidden: cross-origin request' }, { status: 403 });
  }
  try {
    const data = await req.json().catch(() => ({}));
    const action = String(data.action || '');

    if (action === 'reset') {
      resetCluster();
      return NextResponse.json({ ok: true, action: 'reset' });
    }

    return NextResponse.json({ error: 'unknown action' }, { status: 400 });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
