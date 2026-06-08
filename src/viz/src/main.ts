import * as THREE from 'three';
import { PointCloudRenderer } from './renderer/PointCloudRenderer.js';
import { FeatureRenderer } from './renderer/FeatureRenderer.js';
import { QAAnnotationRenderer } from './renderer/QAAnnotationRenderer.js';
import { CameraController } from './controls/CameraController.js';
import { Sidebar } from './controls/Sidebar.js';
import { loadFeatures, loadQAReport, buildQAAnnotations } from './io/DataLoader.js';
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

const cfg = rawConfig.viewer as ViewerConfig;
const SIDEBAR_WIDTH = 240;

/* ─── Scene ──────────────────────────────────────────────────── */

const scene = new THREE.Scene();
scene.background = new THREE.Color(cfg.backgroundColorHex);

/* ─── Renderer ───────────────────────────────────────────────── */

const container = document.getElementById('canvas-container')!;
const canvasW = () => window.innerWidth - SIDEBAR_WIDTH;
const canvasH = () => window.innerHeight;

const renderer = new THREE.WebGLRenderer({ antialias: cfg.antialias });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(canvasW(), canvasH());
container.appendChild(renderer.domElement);

/* ─── Camera controller ──────────────────────────────────────── */

const cameraCtrl = new CameraController(
  renderer,
  cfg.syntheticExtentMeters / 2,
  cfg.cameraPositionWorld,
  cfg.cameraLookAtWorld,
  cfg.cameraFovDegrees,
  cfg.nearClipMeters,
  cfg.farClipMeters,
);

/* ─── Feature/QA renderers ───────────────────────────────────── */

const pointCloudRenderer = new PointCloudRenderer(scene, cfg.pointSizeMeters);
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

/* ─── Point cloud generation ─────────────────────────────────── */

const benchmarkEnabled = new URLSearchParams(window.location.search).has('benchmark');
const pointCount = benchmarkEnabled ? cfg.syntheticPointCount : cfg.demoPointCount;
const { positions, intensities } = generateSyntheticPointCloud(pointCount, cfg);
currentPositions = positions;
currentIntensities = intensities;

pointCloudRenderer.load(positions, intensities);
sidebar.updatePointCount(pointCount);

function generateSyntheticPointCloud(
  count: number,
  config: ViewerConfig,
): { positions: Float32Array; intensities: Float32Array } {
  const positions = new Float32Array(count * 3);
  const intensities = new Float32Array(count);
  const gridWidth = Math.ceil(Math.sqrt(count));
  const extent = config.syntheticExtentMeters;
  const heightScale = config.syntheticHeightMeters;

  for (let i = 0; i < count; i++) {
    const xi = i % gridWidth;
    const yi = Math.floor(i / gridWidth);
    const xn = xi / gridWidth;
    const yn = yi / gridWidth;
    const off = i * 3;
    positions[off]     = (xn - config.syntheticCenterOffset) * extent;
    positions[off + 1] = (yn - config.syntheticCenterOffset) * extent;
    positions[off + 2] = Math.sin(xn * Math.PI) * Math.cos(yn * Math.PI) * heightScale;
    intensities[i] = (xn + yn) / config.intensityAverageDivisor;
  }

  return { positions, intensities };
}

/* ─── Async data loading ─────────────────────────────────────── */

async function loadSceneData(): Promise<void> {
  const [features, report] = await Promise.all([
    loadFeatures('/data/features.geojson'),
    loadQAReport('/data/qa_report.json'),
  ]);

  // Features
  const featureCollection = features ?? {
    type: 'FeatureCollection' as const,
    features: [
      {
        geometry: { type: 'LineString' as const, coordinates: cfg.demoFeatureLineWorld },
        properties: { feature_type: 'lane_line' },
      },
    ],
  };
  featureRenderer.load(featureCollection);

  // QA report
  if (report) {
    sidebar.showQAReport(report);
    const annotations = buildQAAnnotations(report, features);
    qaRenderer.load(annotations);
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
  pointCount,
  benchmarkFrameCount: cfg.benchmarkFrameCount,
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
