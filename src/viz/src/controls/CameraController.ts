import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { ViewMode } from '../types/spatial.js';

const SIDEBAR_WIDTH = 240;

export class CameraController {
  private readonly perspCamera: THREE.PerspectiveCamera;
  private readonly orthoCamera: THREE.OrthographicCamera;
  private readonly perspControls: OrbitControls;
  private readonly orthoControls: OrbitControls;
  private mode: ViewMode = 'perspective';
  private orthoScale: number;

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    private readonly sceneHalfExtent: number,
    perspPosition: [number, number, number],
    lookAt: [number, number, number],
    fov: number,
    near: number,
    far: number,
  ) {
    this.orthoScale = sceneHalfExtent;

    const w = window.innerWidth - SIDEBAR_WIDTH;
    const h = window.innerHeight;
    const aspect = w / h;

    this.perspCamera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this.perspCamera.position.set(...perspPosition);
    this.perspCamera.lookAt(...lookAt);

    this.perspControls = new OrbitControls(this.perspCamera, renderer.domElement);
    this.perspControls.enableDamping = true;
    this.perspControls.dampingFactor = 0.05;
    this.perspControls.target.set(...lookAt);

    const hs = this.orthoScale;
    this.orthoCamera = new THREE.OrthographicCamera(
      -hs * aspect, hs * aspect, hs, -hs, near, far,
    );
    this.orthoCamera.position.set(lookAt[0], lookAt[1], far * 0.5);
    this.orthoCamera.lookAt(lookAt[0], lookAt[1], 0);
    this.orthoCamera.up.set(0, 1, 0);

    this.orthoControls = new OrbitControls(this.orthoCamera, renderer.domElement);
    this.orthoControls.enableRotate = false;
    this.orthoControls.enableDamping = true;
    this.orthoControls.dampingFactor = 0.05;
    this.orthoControls.screenSpacePanning = true;
    this.orthoControls.target.set(lookAt[0], lookAt[1], 0);
    this.orthoControls.enabled = false;
  }

  get camera(): THREE.Camera {
    return this.mode === 'perspective' ? this.perspCamera : this.orthoCamera;
  }

  get currentMode(): ViewMode {
    return this.mode;
  }

  toggleMode(): ViewMode {
    if (this.mode === 'perspective') {
      this.mode = 'bev';
      this.perspControls.enabled = false;
      this.orthoControls.enabled = true;
    } else {
      this.mode = 'perspective';
      this.perspControls.enabled = true;
      this.orthoControls.enabled = false;
    }
    return this.mode;
  }

  update(): void {
    if (this.mode === 'perspective') {
      this.perspControls.update();
    } else {
      this.orthoControls.update();
    }
  }

  onResize(): void {
    const w = window.innerWidth - SIDEBAR_WIDTH;
    const h = window.innerHeight;
    const aspect = w / h;

    this.perspCamera.aspect = aspect;
    this.perspCamera.updateProjectionMatrix();

    const hs = this.orthoScale;
    this.orthoCamera.left = -hs * aspect;
    this.orthoCamera.right = hs * aspect;
    this.orthoCamera.top = hs;
    this.orthoCamera.bottom = -hs;
    this.orthoCamera.updateProjectionMatrix();
  }

  dispose(): void {
    this.perspControls.dispose();
    this.orthoControls.dispose();
  }
}
