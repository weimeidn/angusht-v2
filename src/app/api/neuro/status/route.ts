import { NextResponse } from 'next/server';
import { getCluster, tickBackground } from '@/lib/neuromorphic/index';

export async function GET() {
  tickBackground();
  const cluster = getCluster();
  return NextResponse.json(cluster.snapshot());
}

export const dynamic = 'force-dynamic';
