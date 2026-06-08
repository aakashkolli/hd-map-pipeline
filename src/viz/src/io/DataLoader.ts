import type { GeoJSONFeatureCollection, QAAnnotation, QAReport } from '../types/spatial.js';

export type PointCloudData = {
  positions: Float32Array;
  intensities: Float32Array;
  /** Centroid of the loaded point cloud in world ENU. */
  centroid: [number, number, number];
  /** Approximate radius of the bounding sphere in meters. */
  extent: number;
};

/**
 * Load a binary point cloud written by run_pipeline.py.
 *
 * Wire format (little-endian):
 *   bytes 0-3       uint32  N point count
 *   bytes 4..N*12+4 float32 xyz positions  FRAME: world ENU
 *   bytes N*12+4..  float32 intensities [0, 1]
 */
export async function loadPointCloudBin(url: string): Promise<PointCloudData | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const buf = await res.arrayBuffer();
    if (buf.byteLength < 4) return null;

    const n = new DataView(buf).getUint32(0, true);
    const expectedBytes = 4 + n * 3 * 4 + n * 4;
    if (buf.byteLength < expectedBytes) return null;

    // Float32Array views into the same backing buffer — zero copy.
    const positions = new Float32Array(buf, 4, n * 3);
    const intensities = new Float32Array(buf, 4 + n * 12, n);

    // Bounding box center — more stable than mean centroid when background
    // points cluster near the origin and distort the weighted average.
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const extent = Math.max((maxX - minX) / 2, (maxY - minY) / 2, 1);

    return { positions, intensities, centroid: [cx, cy, cz], extent };
  } catch {
    return null;
  }
}

export async function loadFeatures(url: string): Promise<GeoJSONFeatureCollection | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as GeoJSONFeatureCollection;
  } catch {
    return null;
  }
}

export async function loadQAReport(url: string): Promise<QAReport | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as QAReport;
  } catch {
    return null;
  }
}

export type PipelineStatus = 'idle' | 'running' | 'done' | 'error';

/** Start a pipeline run on the backend. Returns the current status. */
export async function triggerPipeline(): Promise<PipelineStatus> {
  try {
    const res = await fetch('/api/run-pipeline', { method: 'POST' });
    if (!res.ok) return 'error';
    const body = await res.json() as { status: PipelineStatus };
    return body.status;
  } catch {
    return 'error';
  }
}

/**
 * Poll /api/status every intervalMs until the status is 'done' or 'error'.
 * Calls onDone or onError once, then stops.
 */
export function pollPipelineStatus(
  onDone: () => void,
  onError: (msg: string) => void,
  intervalMs = 1500,
): () => void {
  let stopped = false;
  const tick = async () => {
    if (stopped) return;
    try {
      const res = await fetch('/api/status');
      if (!res.ok) { onError('API unreachable'); stopped = true; return; }
      const body = await res.json() as { status: PipelineStatus; error?: string | null };
      if (body.status === 'done') { stopped = true; onDone(); return; }
      if (body.status === 'error') { stopped = true; onError(body.error ?? 'unknown error'); return; }
    } catch {
      onError('API unreachable'); stopped = true; return;
    }
    setTimeout(tick, intervalMs);
  };
  setTimeout(tick, intervalMs);
  return () => { stopped = true; };
}

export function buildQAAnnotations(
  report: QAReport,
  features: GeoJSONFeatureCollection | null,
): QAAnnotation[] {
  const annotations: QAAnnotation[] = [];

  for (let i = 0; i < report.missed_gt_features.length; i++) {
    const geom = report.missed_gt_features[i];
    if (Array.isArray(geom) && geom.length >= 2) {
      annotations.push({
        id: `missed_gt_${i}`,
        kind: 'missed_gt',
        geometry: geom,
        metrics: { completeness: report.completeness },
      });
    }
  }

  if (features) {
    const fpSet = new Set(report.false_positive_ids);
    features.features.forEach((feat, i) => {
      const id = String(i);
      if (fpSet.has(id)) {
        annotations.push({
          id: `false_positive_${i}`,
          kind: 'false_positive',
          geometry: feat.geometry.coordinates,
          metrics: { false_positive_rate: report.false_positive_rate },
        });
      }
    });
  }

  return annotations;
}
