import * as THREE from 'three';

export type QAAnnotationKind = 'false_positive' | 'missed_gt';

export type QAAnnotation = {
  id: string;
  kind: QAAnnotationKind;
  geometry: number[][];
  metrics: Record<string, number | string>;
};

const QA_COLORS: Record<QAAnnotationKind, number> = {
  false_positive: 0xf59e0b,
  missed_gt: 0xef4444,
};

export class QAAnnotationRenderer {
  private readonly group = new THREE.Group();

  constructor(
    private readonly scene: THREE.Scene,
    private readonly onAnnotationSelected: (annotation: QAAnnotation) => void,
  ) {
    this.scene.add(this.group);
  }

  /**
   * Render QA annotations as world ENU line overlays.
   *
   * annotations: QA polylines. FRAME: world ENU.
   */
  load(annotations: QAAnnotation[]): void {
    this.clear();
    for (const annotation of annotations) {
      this.group.add(this.createAnnotationLine(annotation));
    }
  }

  handleClick(object: THREE.Object3D): void {
    const annotation = object.userData.annotation as QAAnnotation | undefined;
    if (annotation) {
      this.onAnnotationSelected(annotation);
    }
  }

  clear(): void {
    for (const child of this.group.children) {
      const line = child as THREE.Line;
      line.geometry.dispose();
      const material = line.material as THREE.Material;
      material.dispose();
    }
    this.group.clear();
  }

  dispose(): void {
    this.clear();
    this.scene.remove(this.group);
  }

  private createAnnotationLine(annotation: QAAnnotation): THREE.Line {
    const positions = new Float32Array(annotation.geometry.length * 3);
    annotation.geometry.forEach((coordinate, index) => {
      positions[index * 3] = coordinate[0];
      positions[index * 3 + 1] = coordinate[1];
      positions[index * 3 + 2] = coordinate[2] ?? 0;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({
      color: QA_COLORS[annotation.kind],
    });
    const line = new THREE.Line(geometry, material);
    line.userData.annotation = annotation;
    return line;
  }
}
