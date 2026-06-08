import * as THREE from 'three';
import { PointCloudRenderer } from './renderer/PointCloudRenderer';
import { FeatureRenderer } from './renderer/FeatureRenderer';
import { QAAnnotationRenderer } from './renderer/QAAnnotationRenderer';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(20, -30, 20);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const pointCloudRenderer = new PointCloudRenderer(scene);
const featureRenderer = new FeatureRenderer(scene);
const qaRenderer = new QAAnnotationRenderer(scene, (annotation) => {
  console.log('Selected QA annotation', annotation.id, annotation.metrics);
});

const positions = new Float32Array([0, 0, 0, 10, 0, 0, 20, 0, 0]);
const intensities = new Float32Array([0.2, 0.8, 1.0]);
pointCloudRenderer.load(positions, intensities);
featureRenderer.load({
  type: 'FeatureCollection',
  features: [
    {
      geometry: { type: 'LineString', coordinates: [[0, 0, 0], [20, 0, 0]] },
      properties: { feature_type: 'lane_line' },
    },
  ],
});
qaRenderer.load([]);

function animate(): void {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

animate();

