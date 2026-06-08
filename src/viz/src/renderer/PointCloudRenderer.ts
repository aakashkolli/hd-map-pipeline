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

  setVisible(visible: boolean): void {
    this.mesh.visible = visible;
  }

  get visible(): boolean {
    return this.mesh.visible;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.scene.remove(this.mesh);
  }

  // 4-stop perceptual ramp: dark-blue → cyan → yellow → red
  // Each entry: [t, r, g, b]
  private static readonly INTENSITY_STOPS: ReadonlyArray<[number, number, number, number]> = [
    [0.00, 0.05, 0.05, 0.55],
    [0.40, 0.00, 0.85, 0.90],
    [0.70, 1.00, 0.88, 0.00],
    [1.00, 1.00, 0.08, 0.08],
  ];

  private intensityToColors(intensities: Float32Array): Float32Array {
    const colors = new Float32Array(intensities.length * 3);
    for (let i = 0; i < intensities.length; i++) {
      const [r, g, b] = PointCloudRenderer.sampleRamp(
        Math.max(0, Math.min(1, intensities[i])),
        PointCloudRenderer.INTENSITY_STOPS,
      );
      colors[i * 3]     = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
    return colors;
  }

  private static sampleRamp(
    t: number,
    stops: ReadonlyArray<[number, number, number, number]>,
  ): [number, number, number] {
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const prev = stops[i - 1];
        const next = stops[i];
        const a = (t - prev[0]) / (next[0] - prev[0]);
        return [
          prev[1] + a * (next[1] - prev[1]),
          prev[2] + a * (next[2] - prev[2]),
          prev[3] + a * (next[3] - prev[3]),
        ];
      }
    }
    const last = stops[stops.length - 1];
    return [last[1], last[2], last[3]];
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
