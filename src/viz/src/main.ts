import * as THREE from 'three';
import { PointCloudRenderer } from './renderer/PointCloudRenderer.js';
import { FeatureRenderer } from './renderer/FeatureRenderer.js';
import { QAAnnotationRenderer } from './renderer/QAAnnotationRenderer.js';
import { CameraController } from './controls/CameraController.js';
import { Sidebar } from './controls/Sidebar.js';
import { loadFeatures, loadQAReport, loadPointCloudBin, buildQAAnnotations } from './io/DataLoader.js';
import rawConfig from '../../../configs/viz.json';

/* ─── Config ─────────────────────────────────────────────────── */

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
const SIDEBAR_WIDTH = 240;

/* ─── Scene ──────────────────────────────────────────────────── */

const scene = new THREE.Scene();
scene.background = new THREE.Color(viewerConfig.backgroundColorHex);

/* ─── Renderer ───────────────────────────────────────────────── */

const container = document.getElementById('canvas-container')!;
const canvasW = () => window.innerWidth - SIDEBAR_WIDTH;
const canvasH = () => window.innerHeight;

const renderer = new THREE.WebGLRenderer({ antialias: viewerConfig.antialias });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(canvasW(), canvasH());
container.appendChild(renderer.domElement);

/* ─── Camera controller ──────────────────────────────────────── */

const cameraCtrl = new CameraController(
  renderer,
  viewerConfig.syntheticExtentMeters / 2,
  viewerConfig.cameraPositionWorld,
  viewerConfig.cameraLookAtWorld,
  viewerConfig.cameraFovDegrees,
  viewerConfig.nearClipMeters,
  viewerConfig.farClipMeters,
);

/* ─── Feature/QA renderers ───────────────────────────────────── */

const pointCloudRenderer = new PointCloudRenderer(scene, viewerConfig.pointSizeMeters);
const featureRenderer = new FeatureRenderer(scene);
const qaRenderer = new QAAnnotationRenderer(scene, (annotation) => {
  sidebar.showSelectedAnnotation(annotation);
});

/* ─── Sidebar ────────────────────────────────────────────────── */

// State that survives color mode switches
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let currentPositions: Float32Array<any> = new Float32Array(0);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let currentIntensities: Float32Array<any> = new Float32Array(0);

const sidebar = new Sidebar({
  onLayerToggle(layer, enabled) {
    switch (layer) {
      case 'pointCloud': pointCloudRenderer.setVisible(enabled); break;
      case 'features':   featureRenderer.setVisible(enabled);   break;
      case 'qa':         qaRenderer.setVisible(enabled);        break;
    }
  },
  onColorMode(mode) {
    if (currentPositions.length > 0) {
      pointCloudRenderer.setColorMode(mode, currentPositions, currentIntensities);
    }
  },
  onViewToggle() {
    const mode = cameraCtrl.toggleMode();
    sidebar.setViewMode(mode);
  },
});

/* ─── Raycasting ─────────────────────────────────────────────── */

const raycaster = new THREE.Raycaster();
raycaster.params.Line = { threshold: 0.8 };
const mouse = new THREE.Vector2();

renderer.domElement.addEventListener('click', (event) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, cameraCtrl.camera);

  // Check QA annotations first (they overlay features)
  const qaHits = raycaster.intersectObjects(qaRenderer.getObjects(), false);
  if (qaHits.length > 0) {
    const annotation = qaHits[0].object.userData.annotation;
    if (annotation) {
      sidebar.showSelectedAnnotation(annotation);
      return;
    }
  }

  // Check feature lines
  const featureHits = raycaster.intersectObjects(featureRenderer.getObjects(), false);
  if (featureHits.length > 0) {
    const { feature, index } = featureHits[0].object.userData;
    if (feature) {
      featureRenderer.setSelected(index);
      sidebar.showSelectedFeature(feature, index);
      return;
    }
  }

  // Miss — clear selection
  featureRenderer.clearSelected();
  sidebar.clearSelection();
});

/* ─── Keyboard shortcuts ─────────────────────────────────────── */

window.addEventListener('keydown', (event) => {
  switch (event.key) {
    case '1': sidebar.setLayer('pointCloud', !pointCloudRenderer.visible); break;
    case '2': sidebar.setLayer('features',   !featureRenderer.visible);   break;
    case '3': sidebar.setLayer('qa',         !qaRenderer.visible);        break;
    case 'i':
    case 'I': sidebar.setColorMode('intensity'); break;
    case 'h':
    case 'H': sidebar.setColorMode('height'); break;
    case 'v':
    case 'V': {
      const mode = cameraCtrl.toggleMode();
      sidebar.setViewMode(mode);
      break;
    }
  }
});

/* ─── Resize ─────────────────────────────────────────────────── */

window.addEventListener('resize', () => {
  renderer.setSize(canvasW(), canvasH());
  cameraCtrl.onResize();
});

/* ─── Synthetic fallback (benchmark + no-data mode) ─────────── */

function generateSyntheticPointCloud(
  count: number,
): { positions: Float32Array; intensities: Float32Array } {
  const positions = new Float32Array(count * 3);
  const intensities = new Float32Array(count);
  const gridWidth = Math.ceil(Math.sqrt(count));
  const extent = viewerConfig.syntheticExtentMeters;
  const heightScale = viewerConfig.syntheticHeightMeters;

  for (let i = 0; i < count; i++) {
    const xi = i % gridWidth;
    const yi = Math.floor(i / gridWidth);
    const xn = xi / gridWidth;
    const yn = yi / gridWidth;
    const off = i * 3;
    positions[off]     = (xn - viewerConfig.syntheticCenterOffset) * extent;
    positions[off + 1] = (yn - viewerConfig.syntheticCenterOffset) * extent;
    positions[off + 2] = Math.sin(xn * Math.PI) * Math.cos(yn * Math.PI) * heightScale;
    intensities[i] = (xn + yn) / viewerConfig.intensityAverageDivisor;
  }

  return { positions, intensities };
}

/* ─── Scene loading ──────────────────────────────────────────── */

const benchmarkEnabled = new URLSearchParams(window.location.search).has('benchmark');

async function loadSceneData(): Promise<void> {
  // Benchmark mode: always use the full synthetic cloud, skip data/ fetch.
  if (benchmarkEnabled) {
    const { positions, intensities } = generateSyntheticPointCloud(viewerConfig.syntheticPointCount);
    currentPositions = positions;
    currentIntensities = intensities;
    pointCloudRenderer.load(positions, intensities);
    sidebar.updatePointCount(viewerConfig.syntheticPointCount);
    featureRenderer.load({ type: 'FeatureCollection', features: [] });
    qaRenderer.load([]);
    return;
  }

  // Parallel-load everything the pipeline writes to data/outputs/.
  const [cloud, features, report] = await Promise.all([
    loadPointCloudBin('/data/points.bin'),
    loadFeatures('/data/features.geojson'),
    loadQAReport('/data/qa_report.json'),
  ]);

  // Point cloud — use pipeline output or fall back to small synthetic demo.
  if (cloud) {
    currentPositions = cloud.positions;
    currentIntensities = cloud.intensities;
    pointCloudRenderer.load(cloud.positions, cloud.intensities);
    sidebar.updatePointCount(cloud.positions.length / 3);
    cameraCtrl.recenterOn(...cloud.centroid, cloud.extent);
  } else {
    const { positions, intensities } = generateSyntheticPointCloud(viewerConfig.demoPointCount);
    currentPositions = positions;
    currentIntensities = intensities;
    pointCloudRenderer.load(positions, intensities);
    sidebar.updatePointCount(viewerConfig.demoPointCount);
  }

  // Feature lines — use pipeline output or fall back to demo line.
  featureRenderer.load(
    features ?? {
      type: 'FeatureCollection',
      features: [
        {
          geometry: { type: 'LineString', coordinates: viewerConfig.demoFeatureLineWorld },
          properties: { feature_type: 'lane_line' },
        },
      ],
    },
  );

  // QA report.
  if (report) {
    sidebar.showQAReport(report);
    qaRenderer.load(buildQAAnnotations(report, features));
  } else {
    qaRenderer.load([]);
  }
}

loadSceneData();

/* ─── Benchmark metrics ──────────────────────────────────────── */

const metricsEl = document.getElementById('viewer-metrics')!;
if (benchmarkEnabled) metricsEl.style.display = 'block';

const startTimeMs = performance.now();
window.__HD_MAP_VIEWER_METRICS__ = {
  benchmarkEnabled,
  pointCount: benchmarkEnabled ? viewerConfig.syntheticPointCount : viewerConfig.demoPointCount,
  benchmarkFrameCount: viewerConfig.benchmarkFrameCount,
  framesRendered: 0,
  averageFps: 0,
};

/* ─── Render loop ────────────────────────────────────────────── */

function animate(): void {
  requestAnimationFrame(animate);
  cameraCtrl.update();
  renderer.render(scene, cameraCtrl.camera);

  const metrics = window.__HD_MAP_VIEWER_METRICS__;
  const elapsedS = (performance.now() - startTimeMs) / 1000;
  metrics.framesRendered += 1;
  metrics.averageFps = metrics.framesRendered / elapsedS;

  sidebar.updateFPS(metrics.averageFps);

  if (benchmarkEnabled) {
    metricsEl.textContent = JSON.stringify(metrics);
  }
}

animate();
