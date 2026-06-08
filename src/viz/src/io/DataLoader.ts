import type { GeoJSONFeatureCollection, QAAnnotation, QAReport } from '../types/spatial.js';

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
