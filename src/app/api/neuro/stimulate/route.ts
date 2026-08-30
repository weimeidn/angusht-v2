import { NextRequest, NextResponse } from 'next/server';
import { getCluster } from '@/lib/neuromorphic/index';
import { isSameOriginRequest } from '@/lib/same-origin';

export async function POST(req: NextRequest) {
  // Минимальная защита: см. same-origin.ts — эндпоинт двигает
  // внутреннюю активность сети и не должен отвечать на посторонние
  // кросс-доменные запросы.
  if (!isSameOriginRequest(req)) {
    return NextResponse.json({ error: 'forbidden: cross-origin request' }, { status: 403 });
  }
  try {
    const data = await req.json().catch(() => ({}));
    const count = Number(data.count) || 50;
    const coreId = (data.coreId as string) || null;
    const cluster = getCluster();
    cluster.spontaneousStimulation(count, coreId);
    return NextResponse.json({ ok: true, stimulated: count, coreId });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
