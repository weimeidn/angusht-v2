import { NextRequest, NextResponse } from 'next/server';
import { getCluster } from '@/lib/neuromorphic/index';
import { isSameOriginRequest } from '@/lib/same-origin';

export async function POST(req: NextRequest) {
  // Минимальная защита: см. same-origin.ts — эндпоинт напрямую двигает
  // веса STDP и не должен отвечать на посторонние кросс-доменные запросы.
  if (!isSameOriginRequest(req)) {
    return NextResponse.json({ error: 'forbidden: cross-origin request' }, { status: 403 });
  }
  try {
    const data = await req.json().catch(() => ({}));
    const text = String(data.text || '');
    const positive = Boolean(data.positive);
    if (!text) {
      return NextResponse.json({ error: 'text required' }, { status: 400 });
    }
    const cluster = getCluster();
    cluster.globalReinforce(text, positive);
    return NextResponse.json({ ok: true, text, positive });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
