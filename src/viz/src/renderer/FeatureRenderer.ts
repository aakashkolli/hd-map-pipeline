import * as THREE from 'three';

type GeoJSONLineFeature = {
  geometry: {
    type: 'LineString';
    coordinates: number[][];
  };
  properties?: {
    feature_type?: string;
  };
};

type GeoJSONFeatureCollection = {
  type: 'FeatureCollection';
  features: GeoJSONLineFeature[];
};

const FEATURE_COLORS: Record<string, number> = {
  lane_line: 0x38bdf8,
  curb: 0xa3e635,
  default: 0xe5e7eb,
};

export class FeatureRenderer {
  private readonly group = new THREE.Group();
  public renderedFeatureCount = 0;

  constructor(private readonly scene: THREE.Scene) {
    this.scene.add(this.group);
  }

  /**
   * Render GeoJSON LineString features in world ENU coordinates.
   *
   * collection: FeatureCollection with xyz coordinates. FRAME: world ENU.
   */
  load(collection: GeoJSONFeatureCollection): void {
    this.clear();
    const { features } = collection;
    for (const feature of features) {
      const line = this.createLine(feature);
      this.group.add(line);
    }
    this.renderedFeatureCount = features.length;
  }

  clear(): void {
    for (const child of this.group.children) {
      const line = child as THREE.Line;
      line.geometry.dispose();
      const material = line.material as THREE.Material;
      material.dispose();
    }
    this.group.clear();
    this.renderedFeatureCount = 0;
  }

  dispose(): void {
    this.clear();
    this.scene.remove(this.group);
  }

  private createLine(feature: GeoJSONLineFeature): THREE.Line {
    const coordinates = feature.geometry.coordinates;
    const positions = new Float32Array(coordinates.length * 3);
    coordinates.forEach((coordinate, index) => {
      positions[index * 3] = coordinate[0];
      positions[index * 3 + 1] = coordinate[1];
      positions[index * 3 + 2] = coordinate[2] ?? 0;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const color = FEATURE_COLORS[feature.properties?.feature_type ?? 'default'] ?? FEATURE_COLORS.default;
    const material = new THREE.LineBasicMaterial({ color });
    return new THREE.Line(geometry, material);
  }
}
