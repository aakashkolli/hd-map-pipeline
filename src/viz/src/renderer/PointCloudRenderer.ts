import * as THREE from 'three';

export type PointColorMode = 'intensity' | 'height';

export class PointCloudRenderer {
  private geometry: THREE.BufferGeometry;
  private material: THREE.PointsMaterial;
  private mesh: THREE.Points;

  constructor(private readonly scene: THREE.Scene, pointSizeMeters: number) {
    this.geometry = new THREE.BufferGeometry();
    this.material = new THREE.PointsMaterial({
      size: pointSizeMeters,
      vertexColors: true,
      sizeAttenuation: true,
    });
    this.mesh = new THREE.Points(this.geometry, this.material);
    this.scene.add(this.mesh);
  }

  /**
   * Load world ENU point positions and intensities into GPU buffers.
   *
   * positions: Float32Array of xyz triples. FRAME: world ENU.
   * intensities: Float32Array, one value per point.
   */
  load(positions: Float32Array, intensities: Float32Array): void {
    this.geometry.dispose();
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(this.intensityToColors(intensities), 3),
    );
    this.geometry.computeBoundingSphere();
    this.mesh.geometry = this.geometry;
  }

  setColorMode(
    mode: PointColorMode,
    positions: Float32Array,
    intensities: Float32Array,
  ): void {
    const colors =
      mode === 'intensity'
        ? this.intensityToColors(intensities)
        : this.heightToColors(positions);
    const colorAttribute = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colorAttribute.array = colors;
    colorAttribute.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.scene.remove(this.mesh);
  }

  private intensityToColors(intensities: Float32Array): Float32Array {
    const colors = new Float32Array(intensities.length * 3);
    for (let index = 0; index < intensities.length; index += 1) {
      const value = intensities[index];
      colors[index * 3] = value;
      colors[index * 3 + 1] = value;
      colors[index * 3 + 2] = value;
    }
    return colors;
  }

  private heightToColors(positions: Float32Array): Float32Array {
    let minZ = Number.POSITIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;
    for (let index = 2; index < positions.length; index += 3) {
      minZ = Math.min(minZ, positions[index]);
      maxZ = Math.max(maxZ, positions[index]);
    }
    const range = maxZ - minZ || 1;
    const colors = new Float32Array(positions.length);

    for (let index = 0; index < positions.length / 3; index += 1) {
      const normalized = (positions[index * 3 + 2] - minZ) / range;
      colors[index * 3] = Math.max(0, Math.min(1, 1.5 - Math.abs(normalized - 1.0) * 2));
      colors[index * 3 + 1] = Math.max(
        0,
        Math.min(1, 1.5 - Math.abs(normalized - 0.5) * 2),
      );
      colors[index * 3 + 2] = Math.max(0, Math.min(1, 1.5 - Math.abs(normalized) * 2));
    }
    return colors;
  }
}
