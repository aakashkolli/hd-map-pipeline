import * as THREE from 'three';
import { PointCloudRenderer } from './renderer/PointCloudRenderer';
import { FeatureRenderer } from './renderer/FeatureRenderer';
import { QAAnnotationRenderer } from './renderer/QAAnnotationRenderer';
import rawConfig from '../../../configs/viz.json';

const scene = new THREE.Scene();

type ViewerConfig = {
  backgroundColorHex: string;
  pointSizeMeters: number;
  cameraFovDegrees: number;
  nearClipMeters: number;
  farClipMeters: number;
  cameraPositionWorld: [number, number, number];
  cameraLookAtWorld: [number, number, number];
  antialias: boolean;
  demoFeatureLineWorld: number[][];
  demoPointCount: number;
  syntheticPointCount: number;
  benchmarkFrameCount: number;
  syntheticExtentMeters: number;
  syntheticHeightMeters: number;
  syntheticCenterOffset: number;
  intensityAverageDivisor: number;
};

type ViewerMetrics = {
  benchmarkEnabled: boolean;
  pointCount: number;
  benchmarkFrameCount: number;
  framesRendered: number;
  averageFps: number;
};

declare global {
  interface Window {
    __HD_MAP_VIEWER_METRICS__: ViewerMetrics;
  }
}

const viewerConfig = rawConfig.viewer as ViewerConfig;
scene.background = new THREE.Color(viewerConfig.backgroundColorHex);

const camera = new THREE.PerspectiveCamera(
  viewerConfig.cameraFovDegrees,
  window.innerWidth / window.innerHeight,
  viewerConfig.nearClipMeters,
  viewerConfig.farClipMeters,
);
camera.position.set(...viewerConfig.cameraPositionWorld);
camera.lookAt(...viewerConfig.cameraLookAtWorld);

const renderer = new THREE.WebGLRenderer({ antialias: viewerConfig.antialias });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const pointCloudRenderer = new PointCloudRenderer(scene, viewerConfig.pointSizeMeters);
const featureRenderer = new FeatureRenderer(scene);
const qaRenderer = new QAAnnotationRenderer(scene, (annotation) => {
  console.log('Selected QA annotation', annotation.id, annotation.metrics);
});

const benchmarkEnabled = new URLSearchParams(window.location.search).has('benchmark');
const pointCount = benchmarkEnabled
  ? viewerConfig.syntheticPointCount
  : viewerConfig.demoPointCount;
const { positions, intensities } = generateSyntheticPointCloud(pointCount, viewerConfig);

pointCloudRenderer.load(positions, intensities);
featureRenderer.load({
  type: 'FeatureCollection',
  features: [
    {
      geometry: { type: 'LineString', coordinates: viewerConfig.demoFeatureLineWorld },
      properties: { feature_type: 'lane_line' },
    },
  ],
});
qaRenderer.load([]);

const metricsElement = document.createElement('pre');
metricsElement.id = 'viewer-metrics';
document.body.appendChild(metricsElement);

const startTimeMs = performance.now();
window.__HD_MAP_VIEWER_METRICS__ = {
  benchmarkEnabled,
  pointCount,
  benchmarkFrameCount: viewerConfig.benchmarkFrameCount,
  framesRendered: 0,
  averageFps: 0,
};

function generateSyntheticPointCloud(
  count: number,
  config: ViewerConfig,
): { positions: Float32Array; intensities: Float32Array } {
  const positions = new Float32Array(count * 3);
  const intensities = new Float32Array(count);
  const gridWidth = Math.ceil(Math.sqrt(count));
  const extent = config.syntheticExtentMeters;
  const heightScale = config.syntheticHeightMeters;

  for (let index = 0; index < count; index += 1) {
    const xIndex = index % gridWidth;
    const yIndex = Math.floor(index / gridWidth);
    const xNorm = xIndex / gridWidth;
    const yNorm = yIndex / gridWidth;
    const offset = index * 3;
    positions[offset] = (xNorm - config.syntheticCenterOffset) * extent;
    positions[offset + 1] = (yNorm - config.syntheticCenterOffset) * extent;
    positions[offset + 2] = Math.sin(xNorm * Math.PI) * Math.cos(yNorm * Math.PI) * heightScale;
    intensities[index] = (xNorm + yNorm) / config.intensityAverageDivisor;
  }

  return { positions, intensities };
}

function updateMetrics(): void {
  const metrics = window.__HD_MAP_VIEWER_METRICS__;
  const elapsedSeconds = (performance.now() - startTimeMs) / 1000;
  metrics.framesRendered += 1;
  metrics.averageFps = metrics.framesRendered / elapsedSeconds;
  metricsElement.textContent = JSON.stringify(metrics);
}

function animate(): void {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
  updateMetrics();
}

animate();
