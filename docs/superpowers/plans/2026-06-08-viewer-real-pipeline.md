# Viewer Real Pipeline + UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "Run Pipeline" process real KITTI LiDAR data and display it with a polished viewer (turbo colormap, loading overlay, camera reset, scene metadata).

**Architecture:** Replace the synthetic smoke-test in `run_pipeline.py`'s `full` stage with a real pipeline (ingest → RANSAC ground separation → voxel filter → lane extraction). Add four self-contained UI improvements to the Three.js viewer, each touching only the relevant renderer/control file.

**Tech Stack:** Python 3 / NumPy / pandas / scikit-learn (pipeline), TypeScript / Three.js / Vite (viewer), FastAPI / uvicorn (API)

---

## File Map

| File | Change |
|---|---|
| `scripts/run_pipeline.py` | Replace `_run_full_smoke` with `_run_full_kitti`; add missing imports |
| `src/api/server.py` | Change `CONFIG` from `default.yaml` → `kitti.yaml` so the API uses dataset paths |
| `configs/kitti.yaml` | Override `n_frames_accumulate: 5` and `seed_xy_radius: 200.0` for batch mode |
| `src/viz/src/renderer/PointCloudRenderer.ts` | Replace grayscale intensity map with 4-stop turbo-style colormap |
| `src/viz/index.html` | Add `#loading-overlay` element + CSS |
| `src/viz/src/controls/CameraController.ts` | Add `resetToLastRecenter()` method |
| `src/viz/src/controls/Sidebar.ts` | Add "Reset camera" button; add `setSceneMeta()` method |
| `src/viz/src/main.ts` | Wire loading overlay, camera reset shortcut, scene metadata updates |

---

## Task 1 — Fix config: kitti.yaml overrides for batch ingest

**Files:**
- Modify: `configs/kitti.yaml`

The API server triggers `run_pipeline.py --stage full` against `kitti.yaml`. Two config values need overriding for batch use:
- `n_frames_accumulate: 5` — only 5 KITTI frames exist in `data/raw/kitti/`; the default of 30 will throw a `ValueError`.
- `seed_xy_radius: 200.0` — the default `20.0 m` only covers the first accumulated frame. In world ENU after multi-frame accumulation the vehicle has moved away from the ENU origin, so later frames' ground points fall outside the seed radius and RANSAC degrades. 200 m safely covers the full accumulated extent.

- [ ] **Step 1: Update kitti.yaml**

Replace the file contents with:

```yaml
base: "configs/default.yaml"

dataset:
  scene: "data/raw/kitti/2011_09_26_drive_0005_sync"
  calib: "data/raw/kitti/2011_09_26_calib"

pipeline:
  n_frames_accumulate: 5      # only 5 frames exist in the sample dataset

filters:
  ransac:
    seed_xy_radius: 200.0     # cover full multi-frame ENU extent for batch RANSAC
```

- [ ] **Step 2: Verify config merges correctly**

```bash
cd /Users/kolli/hd-map-pipeline
.venv/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from scripts.run_pipeline import _load_yaml
from pathlib import Path
cfg = _load_yaml(Path('configs/kitti.yaml'))
print('n_frames:', cfg['pipeline']['n_frames_accumulate'])    # expect 5
print('seed_r:', cfg['filters']['ransac']['seed_xy_radius'])  # expect 200.0
print('scene:', cfg['dataset']['scene'])
"
```

Expected output:
```
n_frames: 5
seed_r: 200.0
scene: data/raw/kitti/2011_09_26_drive_0005_sync
```

- [ ] **Step 3: Commit**

```bash
git add configs/kitti.yaml
git commit -m "config: override n_frames and seed_xy_radius for kitti batch pipeline"
```

---

## Task 2 — Wire real KITTI pipeline in run_pipeline.py

**Files:**
- Modify: `scripts/run_pipeline.py`

Replace the synthetic `_run_full_smoke()` with `_run_full_kitti()` that runs: ingest → RANSAC ground separation → voxel downsample → extract lane boundaries → compute QA → write outputs.

- [ ] **Step 1: Add missing imports at the top of run_pipeline.py**

After the existing imports block (after line `from src.pipeline.qa import QAConfig, compute_qa_metrics`), add:

```python
import pandas as pd

from src.filters.ground_plane import ransac_ground_plane, RansacConfig
from src.filters.voxel import voxel_downsample
```

- [ ] **Step 2: Add `_run_full_kitti()` function**

Add this function just above `_run_full_smoke()`:

```python
def _run_full_kitti(config: dict[str, Any], output: Path) -> None:
    """Run the full pipeline on real KITTI data.

    Stages: ingest → RANSAC ground separation → voxel downsample →
    lane extraction → QA → write viewer outputs.

    Output FRAME: all written coordinates are world ENU.
    """
    dataset = config.get("dataset", {})
    repo_root = Path(__file__).resolve().parents[1]

    scene_path = Path(dataset["scene"])
    calib_path = Path(dataset["calib"])
    if not scene_path.is_absolute():
        scene_path = repo_root / scene_path
    if not calib_path.is_absolute():
        calib_path = repo_root / calib_path

    n_frames = int(config["pipeline"]["n_frames_accumulate"])

    # Stage 1: ingest — accumulate LiDAR frames in world ENU
    acc_path = output / "accumulated.parquet"
    accumulate_kitti_frames(
        scene_dir=scene_path,
        calib_dir=calib_path,
        output_path=acc_path,
        n_frames=n_frames,
    )
    print(f"[ingest] wrote {acc_path}")

    # Stage 2: load accumulated cloud
    df = pd.read_parquet(acc_path)
    points_4 = df[["x", "y", "z", "intensity"]].to_numpy(dtype=np.float32)
    print(f"[load]   {len(points_4):,} points in world ENU")

    # Stage 3: RANSAC ground separation
    ransac_cfg = RansacConfig(**config["filters"]["ransac"])
    ground_result = ransac_ground_plane(points_4[:, :3], ransac_cfg)
    ground_4 = points_4[ground_result.ground_mask]
    print(f"[ransac] inlier_ratio={ground_result.inlier_ratio:.3f}, "
          f"{len(ground_4):,} ground points")

    # Stage 4: voxel downsample ground for extraction
    ground_voxed = voxel_downsample(
        ground_4, voxel_size=float(config["filters"]["voxel_size"])
    )
    print(f"[voxel]  {len(ground_voxed):,} points after downsampling")

    # Stage 5: geometric lane extraction
    extraction_cfg = ExtractionConfig(**config["extraction"])
    features = extract_lane_boundaries(ground_voxed, extraction_cfg)
    print(f"[extract] {len(features)} lane boundary features")

    # Stage 6: QA — no external GT for KITTI scene 0005; report completeness=NaN
    qa_cfg = QAConfig(max_gt_match_distance=config["qa"]["max_gt_match_distance"])
    report = compute_qa_metrics(features, [], qa_cfg, scene_id="kitti_0005")

    # Stage 7: write outputs for the viewer
    _write_points_bin(output / "points.bin", points_4)
    _write_geojson(output / "features.geojson", features)
    (output / "qa_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    print(f"[done]   wrote {output / 'points.bin'} ({len(points_4):,} pts), "
          f"{output / 'features.geojson'} ({len(features)} features)")
```

- [ ] **Step 3: Change the `full` stage dispatch to call `_run_full_kitti`**

In `run_pipeline()`, replace:

```python
    if stage == "full":
        _run_full_smoke(config, output)
        return
```

with:

```python
    if stage == "full":
        if config.get("dataset"):
            _run_full_kitti(config, output)
        else:
            _run_full_smoke(config, output)
        return
```

This keeps the smoke test available when no dataset is configured (e.g., CI), while using real data when `kitti.yaml` is active.

- [ ] **Step 4: Run the pipeline manually to verify it works end-to-end**

```bash
cd /Users/kolli/hd-map-pipeline
.venv/bin/python3 scripts/run_pipeline.py \
  --config configs/kitti.yaml \
  --stage full \
  --output data/outputs
```

Expected output (approximately):
```
[ingest] wrote data/outputs/accumulated.parquet
[load]   461327 points in world ENU
[ransac] inlier_ratio=0.XXX, XXXXX ground points
[voxel]  XXXXX points after downsampling
[extract] XX lane boundary features
[done]   wrote data/outputs/points.bin (461327 pts), data/outputs/features.geojson (XX features)
```

Key checks:
- No Python exception
- `data/outputs/points.bin` size > 1 MB (real data, not 94 KB synthetic)
- `data/outputs/features.geojson` has features with coordinates outside the range [-2, 52] (synthetic was [0,50] × [-1.5,1.5])

```bash
python3 -c "
import os, struct
sz = os.path.getsize('data/outputs/points.bin')
with open('data/outputs/points.bin','rb') as f:
    n = struct.unpack('<I', f.read(4))[0]
print(f'points.bin: {sz/1024:.0f} KB, {n:,} points')
assert sz > 1_000_000, 'Too small — still synthetic data'
print('OK')
"
```

- [ ] **Step 5: Update API server to use kitti.yaml**

In `src/api/server.py`, change line 21:

```python
CONFIG = ROOT / "configs" / "default.yaml"
```

to:

```python
CONFIG = ROOT / "configs" / "kitti.yaml"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_pipeline.py src/api/server.py
git commit -m "pipeline: wire real KITTI ingest/RANSAC/extract in full stage"
```

---

## Task 3 — Turbo colormap for intensity rendering

**Files:**
- Modify: `src/viz/src/renderer/PointCloudRenderer.ts`

Replace the grayscale `(v, v, v)` intensity mapping with a 4-stop perceptual ramp (dark-blue → cyan → yellow → red). This makes high-intensity lane marking returns glow orange-red against a dark blue background — the standard appearance in LiDAR visualization tools.

- [ ] **Step 1: Replace `intensityToColors` and add `sampleColorRamp` in PointCloudRenderer.ts**

Replace the entire `intensityToColors` method and add the static stops + helper. The file from line 67 onwards currently reads:

```typescript
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
```

Replace it with:

```typescript
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
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/kolli/hd-map-pipeline/src/viz
npm run build 2>&1 | tail -5
```

Expected: no TypeScript errors, build succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/viz/src/renderer/PointCloudRenderer.ts
git commit -m "viz: turbo-style 4-stop intensity colormap (blue→cyan→yellow→red)"
```

---

## Task 4 — Loading overlay during data fetch

**Files:**
- Modify: `src/viz/index.html`
- Modify: `src/viz/src/main.ts`

Show a `#loading-overlay` while `loadSceneData()` awaits its three parallel fetches. Without this the viewer is blank and unresponsive for 1–3 seconds on first load with real data.

- [ ] **Step 1: Add overlay element and CSS to index.html**

Inside `<body>`, after the `<div id="canvas-container">` closing tag and before `<script>`, add:

```html
    <div id="loading-overlay">
      <div class="lo-spinner"></div>
      <span class="lo-label">loading scene…</span>
    </div>
```

Add to the `<style>` block (after the `#scene-flash` rule block):

```css
      /* ─── Loading overlay ───────────────────────────────────── */
      #loading-overlay {
        position: absolute;
        inset: 0;
        left: 240px; /* sidebar width */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
        background: #0d1117cc;
        z-index: 20;
        pointer-events: none;
      }

      #loading-overlay[hidden] { display: none; }

      .lo-spinner {
        width: 28px;
        height: 28px;
        border: 3px solid #30363d;
        border-top-color: #58a6ff;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
      }

      @keyframes spin { to { transform: rotate(360deg); } }

      .lo-label {
        font-size: 11px;
        color: #8b949e;
        font-family: ui-monospace, 'SFMono-Regular', monospace;
      }
```

- [ ] **Step 2: Wire the overlay in main.ts**

In `main.ts`, add a helper function right before `loadSceneData`:

```typescript
function setLoadingVisible(visible: boolean): void {
  const el = document.getElementById('loading-overlay')!;
  el.hidden = !visible;
}
```

Then wrap `loadSceneData` body so loading shows at entry and hides at exit. The current function opens with `if (benchmarkEnabled)` — change the function to:

```typescript
async function loadSceneData(): Promise<void> {
  setLoadingVisible(true);
  try {
    if (benchmarkEnabled) {
      // ... keep existing benchmark block unchanged ...
      return;
    }

    // ... keep existing parallel-load block unchanged ...

  } finally {
    setLoadingVisible(false);
  }
}
```

Important: the initial page load has `hidden` NOT set on `#loading-overlay`, so it shows immediately. After the first `loadSceneData()` resolves, `setLoadingVisible(false)` sets `hidden`. When `onRunPipeline` calls `loadSceneData()` again after pipeline completion, the overlay reappears.

- [ ] **Step 3: Set overlay to initially visible in index.html**

Make sure the `#loading-overlay` div does NOT have `hidden` attribute — it should be visible at page load and hidden once data arrives. Verify the div you added in Step 1 reads:

```html
    <div id="loading-overlay">
```

(No `hidden` attribute.)

- [ ] **Step 4: TypeScript compile check**

```bash
cd /Users/kolli/hd-map-pipeline/src/viz
npm run build 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add src/viz/index.html src/viz/src/main.ts
git commit -m "viz: loading overlay during point cloud fetch"
```

---

## Task 5 — Camera reset + scene metadata HUD

**Files:**
- Modify: `src/viz/src/controls/CameraController.ts`
- Modify: `src/viz/src/controls/Sidebar.ts`
- Modify: `src/viz/index.html`
- Modify: `src/viz/src/main.ts`

Add a "Reset view" button ([R] key) that repositions the camera to the last `recenterOn()` position, and a metadata chip at the bottom of the canvas showing the current scene's point and feature counts.

### CameraController changes

- [ ] **Step 1: Add reset state and `resetToLastRecenter()` to CameraController.ts**

After the `private mode: ViewMode = 'perspective';` line, add:

```typescript
  private lastCentroid: [number, number, number] = [0, 0, 0];
  private lastExtent = 50;
```

At the end of `recenterOn()`, save the parameters (insert before the closing `}`):

```typescript
    this.lastCentroid = [cx, cy, cz];
    this.lastExtent = extent;
```

Add the new method after `recenterOn()`:

```typescript
  resetToLastRecenter(): void {
    this.recenterOn(...this.lastCentroid, this.lastExtent);
  }
```

### Sidebar changes

- [ ] **Step 2: Add reset camera callback and `setSceneMeta()` to Sidebar.ts**

In `SidebarCallbacks`, add:

```typescript
  onResetCamera: () => void;
```

In `buildDOM()`, inside the `sb-section` for **View**, replace the existing view section HTML:

```html
      <div class="sb-section">
        <div class="sb-section-label">View</div>
        <button class="sb-view-btn" id="view-toggle">
          <span id="view-label">Perspective</span>
          <span class="sb-key">[V]</span>
        </button>
        <button class="sb-view-btn" id="camera-reset" style="margin-top:4px">
          <span>Reset camera</span>
          <span class="sb-key">[R]</span>
        </button>
      </div>
```

In `bindEvents()`, after the view-toggle listener, add:

```typescript
    document.getElementById('camera-reset')!.addEventListener('click', () => {
      this.callbacks.onResetCamera();
    });
```

Add the public method after `clearSelection()`:

```typescript
  setSceneMeta(points: number, features: number): void {
    const el = document.getElementById('scene-meta');
    if (!el) return;
    const pStr = points >= 1000 ? `${(points / 1000).toFixed(0)}K pts` : `${points} pts`;
    el.textContent = `KITTI 0005 · ${pStr} · ${features} features`;
    el.hidden = false;
  }
```

### index.html and main.ts changes

- [ ] **Step 3: Add `#scene-meta` chip to index.html**

Inside `#canvas-container` div, after `#scene-flash`, add:

```html
      <div id="scene-meta" hidden class="scene-meta-chip"></div>
```

Add to the `<style>` block:

```css
      /* ─── Scene metadata chip ───────────────────────────────── */
      .scene-meta-chip {
        position: absolute;
        bottom: 12px;
        left: 12px;
        font-family: ui-monospace, 'SFMono-Regular', monospace;
        font-size: 10px;
        color: #8b949e;
        pointer-events: none;
        background: #0d111799;
        padding: 3px 7px;
        border-radius: 3px;
        border: 1px solid #21262d;
      }
```

- [ ] **Step 4: Wire reset camera in main.ts**

In the `Sidebar` constructor call, add the `onResetCamera` callback:

```typescript
  onResetCamera() {
    cameraCtrl.resetToLastRecenter();
  },
```

In the `keydown` handler, add the `r`/`R` case:

```typescript
    case 'r':
    case 'R': cameraCtrl.resetToLastRecenter(); break;
```

In `loadSceneData()`, after the point cloud loads (in the `if (cloud)` branch, after `cameraCtrl.recenterOn(...)`), add:

```typescript
      sidebar.setSceneMeta(
        cloud.positions.length / 3,
        features ? features.features.length : 0,
      );
```

- [ ] **Step 5: TypeScript compile check**

```bash
cd /Users/kolli/hd-map-pipeline/src/viz
npm run build 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 6: Commit**

```bash
git add src/viz/src/controls/CameraController.ts \
        src/viz/src/controls/Sidebar.ts \
        src/viz/index.html \
        src/viz/src/main.ts
git commit -m "viz: camera reset button [R], scene metadata chip"
```

---

## Task 6 — End-to-end smoke test

Verify the full flow: start the dev server, trigger the pipeline via the API, confirm the viewer shows real data.

- [ ] **Step 1: Start the dev server**

```bash
cd /Users/kolli/hd-map-pipeline/src/viz
npm run dev
```

The `npm run dev` script (via `concurrently`) starts both the FastAPI server on port 8000 and the Vite dev server on port 5173.

- [ ] **Step 2: Confirm API is reachable**

In a second terminal:

```bash
curl -s http://localhost:8000/api/status
```

Expected: `{"status":"idle","error":null}`

- [ ] **Step 3: Trigger a pipeline run via the API**

```bash
curl -s -X POST http://localhost:8000/api/run-pipeline
```

Expected: `{"status":"running","error":null}`

- [ ] **Step 4: Poll until done**

```bash
until [ "$(curl -s http://localhost:8000/api/status | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")" != "running" ]; do
  echo "still running..."; sleep 3
done
curl -s http://localhost:8000/api/status
```

Expected final status: `{"status":"done","error":null}`

- [ ] **Step 5: Confirm outputs are real**

```bash
python3 -c "
import struct, os, json
sz = os.path.getsize('data/outputs/points.bin')
with open('data/outputs/points.bin','rb') as f:
    n = struct.unpack('<I', f.read(4))[0]
print(f'points.bin: {sz/1024:.0f} KB, {n:,} points (expect >1000 KB and >100K pts)')
feat = json.load(open('data/outputs/features.geojson'))
print(f'features: {len(feat[\"features\"])} (expect >0)')
qa = json.load(open('data/outputs/qa_report.json'))
print(f'QA scene_id: {qa[\"scene_id\"]} (expect kitti_0005)')
"
```

- [ ] **Step 6: Visual check in the browser**

Open `http://localhost:5173`.

Expected:
1. Loading spinner appears briefly on initial load
2. A real LiDAR point cloud appears — scattered 3D points with intensity coloring visible as blue/cyan/yellow/red gradient (not two flat white lines)
3. "KITTI 0005 · NNNK pts · M features" chip shows at bottom-left of canvas
4. Click "Run Pipeline" → button shows "Running…" → after ~30s shows "loaded ✓" → scene flashes and reloads with fresh data
5. Press [R] → camera snaps back to framing the point cloud
6. Press [I]/[H] to toggle colormap; [V] for BEV mode; [1][2][3] to toggle layers

- [ ] **Step 7: Commit final state**

```bash
git add -p  # stage any unstaged fixes found during smoke test
git commit -m "viz: end-to-end smoke test verified with real KITTI data"
```

---

## Self-Review

**Spec coverage:**
- ✅ "Run Pipeline does nothing" — Task 2 replaces smoke test with real KITTI pipeline; Task 1 fixes config so n_frames=5 is valid and RANSAC seed radius covers the full accumulated extent
- ✅ "Viewer only shows two lines" — Task 2 writes real point cloud and real extracted features to data/outputs/
- ✅ Turbo colormap — Task 3
- ✅ Loading overlay — Task 4
- ✅ Camera reset — Task 5
- ✅ Scene metadata — Task 5

**Placeholder scan:** None found.

**Type consistency:**
- `resetToLastRecenter()` defined in Task 5 Step 1, called in Task 5 Step 4 — matches.
- `setSceneMeta(points: number, features: number)` defined in Task 5 Step 2, called in Task 5 Step 4 — matches.
- `onResetCamera` added to `SidebarCallbacks` in Task 5 Step 2, provided in main.ts in Task 5 Step 4 — matches.
- `sampleRamp` is `private static` in Task 3 — called as `PointCloudRenderer.sampleRamp(...)` which is correct for static methods.
