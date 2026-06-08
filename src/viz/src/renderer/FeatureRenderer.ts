import * as THREE from 'three';
import type { GeoJSONFeatureCollection, GeoJSONLineFeature } from '../types/spatial.js';

const FEATURE_COLORS: Record<string, number> = {
  lane_line: 0x38bdf8,
  curb:      0xa3e635,
  default:   0xe5e7eb,
};

const SELECTED_COLOR = 0xffffff;

export class FeatureRenderer {
  private readonly group = new THREE.Group();
  private selectedIndex = -1;
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
    for (let i = 0; i < collection.features.length; i++) {
      const line = this.createLine(collection.features[i], i);
      this.group.add(line);
    }
    this.renderedFeatureCount = collection.features.length;
  }

  setSelected(index: number): void {
    for (let i = 0; i < this.group.children.length; i++) {
      const line = this.group.children[i] as THREE.Line;
      const material = line.material as THREE.LineBasicMaterial;
      const feature = line.userData.feature as GeoJSONLineFeature;
      const baseColor = FEATURE_COLORS[feature.properties?.feature_type ?? 'default'] ?? FEATURE_COLORS.default;
      material.color.setHex(i === index ? SELECTED_COLOR : baseColor);
      material.linewidth = i === index ? 2 : 1;
    }
    this.selectedIndex = index;
  }

  clearSelected(): void {
    for (const child of this.group.children) {
      const line = child as THREE.Line;
      const material = line.material as THREE.LineBasicMaterial;
      const feature = line.userData.feature as GeoJSONLineFeature;
      const baseColor = FEATURE_COLORS[feature.properties?.feature_type ?? 'default'] ?? FEATURE_COLORS.default;
      material.color.setHex(baseColor);
    }
    this.selectedIndex = -1;
  }

  getObjects(): THREE.Object3D[] {
    return this.group.children;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  get visible(): boolean {
    return this.group.visible;
  }

  clear(): void {
    for (const child of this.group.children) {
      const line = child as THREE.Line;
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    }
    this.group.clear();
    this.renderedFeatureCount = 0;
    this.selectedIndex = -1;
  }

  dispose(): void {
    this.clear();
    this.scene.remove(this.group);
  }

  private createLine(feature: GeoJSONLineFeature, index: number): THREE.Line {
    const coords = feature.geometry.coordinates;
    const positions = new Float32Array(coords.length * 3);
    for (let i = 0; i < coords.length; i++) {
      positions[i * 3]     = coords[i][0];
      positions[i * 3 + 1] = coords[i][1];
      positions[i * 3 + 2] = coords[i][2] ?? 0;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const color = FEATURE_COLORS[feature.properties?.feature_type ?? 'default'] ?? FEATURE_COLORS.default;
    const material = new THREE.LineBasicMaterial({ color });
    const line = new THREE.Line(geometry, material);
    line.userData.feature = feature;
    line.userData.index = index;
    return line;
  }
}
