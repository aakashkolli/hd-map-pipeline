export type GeoJSONLineFeature = {
  geometry: {
    type: 'LineString';
    coordinates: number[][];
  };
  properties?: {
    feature_type?: string;
    confidence?: number;
    source?: string;
  };
};

export type GeoJSONFeatureCollection = {
  type: 'FeatureCollection';
  features: GeoJSONLineFeature[];
};

export type QAReport = {
  scene_id: string;
  completeness: number;
  positional_accuracy_p50: number;
  positional_accuracy_p95: number;
  false_positive_rate: number;
  classification_accuracy: number;
  per_class_completeness: Record<string, number>;
  missed_gt_features: number[][][];
  false_positive_ids: string[];
};

export type QAAnnotationKind = 'false_positive' | 'missed_gt';

export type QAAnnotation = {
  id: string;
  kind: QAAnnotationKind;
  geometry: number[][];
  metrics: Record<string, number | string>;
};

export type ColorMode = 'intensity' | 'height';
export type ViewMode = 'perspective' | 'bev';
