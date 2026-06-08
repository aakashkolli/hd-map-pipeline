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

    // Compute centroid and bounding sphere extent.
    let sumX = 0, sumY = 0, sumZ = 0;
    for (let i = 0; i < n; i++) {
      sumX += positions[i * 3];
      sumY += positions[i * 3 + 1];
      sumZ += positions[i * 3 + 2];
    }
    const cx = sumX / n, cy = sumY / n, cz = sumZ / n;

    let maxDist = 1;
    for (let i = 0; i < n; i++) {
      const dx = positions[i * 3] - cx;
      const dy = positions[i * 3 + 1] - cy;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > maxDist) maxDist = d;
    }

    return { positions, intensities, centroid: [cx, cy, cz], extent: maxDist };
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
