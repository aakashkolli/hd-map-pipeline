# CLAUDE.md - HD Map Feature Extraction Pipeline

**Role context:** NVIDIA Software Engineering Intern, Mapping Autonomous Vehicles  
**Engineering standard:** Systems-oriented, AV infrastructure caliber  
**Enforcement:** Strict - deviations require explicit documented justification

---

## 0. Mandatory Session Protocol

Before writing any code in a session, state:

```
Component: [exact name from Section 3 Development Order]
Invariants: [list from Section 5 that this component must satisfy]
Prohibited patterns: [list from Section 4 that apply]
Test I will write first: [describe the test before any implementation]
```

Do not proceed until this is stated.

After completing a component, answer the Anti-Vibe Gate (Section 7) in the test file as a comment block before marking the component done.

---

## 1. Role and Project Context

### 1.1 What This Project Is

An HD map feature extraction pipeline targeting NVIDIA's Mapping Autonomous Vehicles internship (JR2011493). The codebase processes real LiDAR point cloud data from AV survey vehicles, extracts lane boundary features using geometric algorithms and a learned segmentation model, validates features against ground truth map annotations, and renders results in a 3D inspection tool.

The project is designed to demonstrate fit for day-to-day mapping infrastructure work at NVIDIA: building data pipelines that process sensor data, writing automation algorithms for feature extraction, building QA tooling, and exploring ML integration on spatial datasets.

### 1.2 What This Project Is Not

Every implementation decision should reflect how this problem is actually approached in AV infrastructure engineering. Simplifications are acceptable when documented. Fake implementations are not acceptable under any circumstances.

### 1.3 Technical Depth Required

A NVIDIA engineer reviewing this codebase should be able to ask any of the following questions and get a concrete, specific answer from the implementation:

- "Why do you pre-filter to a seed set before running RANSAC?"
- "What happens to your lane boundary extraction in rain?"
- "Why does your BEV normalization use per-scan percentile instead of a fixed threshold?"
- "How do you handle the coordinate frame transform from LiDAR to world?"
- "What does your QA completeness metric actually measure?"
- "Why does your visualization use BufferGeometry instead of Geometry?"

If the implementation cannot answer these questions with specific code references, it is not done.

---

## 2. Architecture Rules

### 2.1 Module Dependency Graph

The dependency graph is strict and must not be violated:

```
geometry/          ← no internal imports (foundation layer)
data/              ← imports geometry/ only
filters/           ← imports geometry/ only
ml/                ← imports geometry/, data/ only
pipeline/          ← imports geometry/, data/, filters/, ml/
viz/               ← reads files only (no Python imports)
scripts/           ← imports pipeline/ and coordinates stages
tests/             ← imports any src/ module
```

Circular imports are prohibited. If a circular import appears, the architecture is wrong - fix the architecture, not the import.

`viz/` is a standalone Node.js application. It never imports Python modules. It reads GeoJSON and Parquet files from the filesystem. Do not add a REST API backend unless specifically designing for streaming large point clouds (which is a Phase 2 consideration, not Phase 1).

### 2.2 Configuration

All numeric constants that affect pipeline behavior live in `configs/*.yaml`. No magic numbers in `src/`.

Prohibited:
```python
# WRONG: magic number in source
threshold = np.percentile(intensity, 85)
```

Required:
```python
# CORRECT: parameter from config
threshold = np.percentile(intensity, cfg.intensity_percentile)
```

The `configs/default.yaml` contains every parameter with comments explaining the unit and the reasoning for the default value. Override files (`kitti.yaml`, `nuscenes.yaml`) document only the parameters that differ and why.

### 2.3 Data Contracts Between Stages

Each pipeline stage has an explicit data contract: defined input schema, defined output schema, defined coordinate frame for spatial data. Stages communicate through files (Parquet for point clouds, GeoJSON for features, JSON for reports) rather than in-memory objects. This makes each stage independently testable and debuggable.

A stage is not done until:
1. Its input schema is documented in the module docstring
2. Its output schema is documented in the module docstring
3. Coordinate frames for all spatial arrays are documented
4. A test verifies the output schema on synthetic input

---

## 3. Development Order

Build in this sequence. Do not skip ahead. Each stage's output is the next stage's input - starting Stage 3 before Stage 2 produces untestable code.

### Phase 1: Foundation (Week 1)

```
Step 1: src/geometry/transforms.py
  Deliverable: SE3 class with transform_points(), inverse(), compose()
  Test: round-trip on 1000 random points, atol=1e-5
  Gate: test passes before proceeding

Step 2: src/data/kitti.py
  Deliverable: parse .bin LiDAR files, parse oxts poses, parse calib
  Test: load one KITTI frame, verify shape (N, 4), verify pose is SE3
  Gate: test passes; frame loads without crash

Step 3: src/pipeline/ingest.py
  Deliverable: accumulate N frames in ENU world frame
  Test: 5-frame accumulation, verify point count grows monotonically
  Gate: outputs accumulated.parquet with correct schema

Step 4: tests/geometry/ complete
  All geometry tests passing before filter work begins
```

### Phase 2: Filtering (Week 1-2)

```
Step 5: src/filters/ground_plane.py
  Deliverable: RANSAC with seed pre-filtering, refinement step
  Test: synthetic plane (50K pts) + noise (5K pts), inlier_ratio > 0.90
  Test: walls-only synthetic scene; seed filtering finds ground, full-cloud fails
  Gate: both tests pass; documented failure modes in docstring

Step 6: src/ext/voxel_filter.cpp + bindings
  Deliverable: C++ voxel grid, Python interface via pybind11
  Test: uniform grid input, one point per voxel
  Test: empty input returns empty output
  Perf: benchmark vs. Python implementation, speedup documented
  Gate: test passes; CMakeLists builds cleanly

Step 7: src/filters/outlier.py
  Deliverable: radius outlier removal
  Test: isolated points removed, dense cluster preserved
  Gate: test passes
```

### Phase 3: Feature Extraction (Week 2)

```
Step 8: src/pipeline/bev.py
  Deliverable: BEV intensity projection with per-scan normalization
  Test: single high-intensity point → correct pixel location
  Test: two scans with different intensity scales → consistent output
  Gate: pixel location test passes; normalization test passes

Step 9: src/pipeline/extract.py
  Deliverable: DBSCAN clustering + polyline fitting on high-intensity ground
  Test: synthetic parallel lane lines → exactly 2 polylines detected
  Test: sparse input (< min_samples) → empty output, no crash
  Gate: synthetic lane test passes

Step 10: src/geometry/polyline.py
  Deliverable: RDP polyline simplification, Hausdorff distance
  Test: known polyline simplification, verify point reduction
  Test: Hausdorff distance between identical polylines = 0
  Gate: tests pass
```

### Phase 4: ML (Week 3)

```
Step 11: src/data/nuscenes.py
  Deliverable: parse nuScenes LiDAR and map annotation layer
  Test: load scene, verify annotation geometry intersects road area
  Gate: test passes; annotations visible in BEV image

Step 12: scripts/prepare_nuscenes_training.py
  Deliverable: BEV images + annotation masks for U-Net training
  Test: verify image/mask pairs are aligned (same pixel coordinates)
  Gate: 200+ training pairs generated from nuScenes-mini

Step 13: src/ml/unet.py
  Deliverable: BEVSegNet architecture
  Test: forward pass (B=2, C=1, H=512, W=512) → (B, 4, 512, 512)
  Test: parameter count < 3M
  Gate: both tests pass

Step 14: src/ml/train.py
  Deliverable: training loop with weighted cross-entropy
  Test: loss decreases over 5 steps on a 4-sample batch
  Gate: training converges (val IoU > 0.40 for lane_line class)
  Note: train on nuScenes, evaluate on KITTI - no exceptions

Step 15: src/ml/infer.py
  Deliverable: batch inference + back-projection to 3D
  Test: segmentation mask back-projects to correct world coordinates
  Gate: back-projection test passes
```

### Phase 5: QA and Fusion (Week 3-4)

```
Step 16: src/pipeline/qa.py
  Deliverable: completeness, positional accuracy, FP rate vs. GT
  Test: perfect prediction → completeness=1.0, FP=0.0
  Test: all wrong → completeness=0.0, FP=1.0
  Gate: both edge-case tests pass

Step 17: src/pipeline/fuse.py
  Deliverable: merge geometric + ML features, confidence scoring
  Test: identical prediction from both → single merged feature
  Test: conflicting predictions → both retained with source labels
  Gate: merge logic tests pass

Step 18: scripts/run_pipeline.py
  Deliverable: end-to-end runner, KITTI 0005 scene
  Test: full pipeline runs without crash, outputs features.geojson
  Gate: outputs exist; QA report has non-zero completeness
```

### Phase 6: Visualization (Week 4)

```
Step 19: src/viz/src/renderer/PointCloudRenderer.ts
  Deliverable: BufferGeometry point cloud, intensity + height color modes
  Test: renders without console error; geometry.dispose() called on reload
  Perf: 500K points ≥ 30fps, verified in DevTools
  Gate: FPS target met

Step 20: src/viz/src/renderer/FeatureRenderer.ts
  Deliverable: GeoJSON polyline overlay, per-class colors
  Test: GeoJSON with 10 features renders 10 line objects
  Gate: features visible in viewer

Step 21: src/viz/src/renderer/QAAnnotationRenderer.ts
  Deliverable: false positive (amber) and missed GT (red) overlays
  Gate: QA colors match specification, click shows metrics in sidebar

Step 22: Full integration
  Deliverable: Docker Compose runs full pipeline + viewer
  Gate: docker compose up works on clean machine
  Deliverable: README with screenshot and quickstart
  Gate: engineer can replicate from README alone
```

---

## 4. Prohibited Patterns

### 4.1 Coordinate Frame Violations

```python
# PROHIBITED: undocumented spatial array
def process_cloud(points: np.ndarray):
    # no mention of coordinate frame
    ...

# REQUIRED: explicit frame documentation
def process_cloud(points: np.ndarray) -> np.ndarray:
    """
    Args:
        points: (N, 3) float32. FRAME: vehicle (x=forward, y=left, z=up).
    Returns:
        (M, 3) float32. FRAME: vehicle (unchanged).
    """
```

```python
# PROHIBITED: hardcoded rotation or translation
R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# REQUIRED: parse from calibration file
R = parse_calibration(calib_path)['R_velo_to_cam']
```

```python
# PROHIBITED: adding points in different frames
world_points = lidar_points + vehicle_offset  # frame mismatch: silent wrong answer

# REQUIRED: explicit transform
vehicle_points = T_vehicle_lidar.transform_points(lidar_points)
world_points = T_world_vehicle.transform_points(vehicle_points)
```

### 4.2 Pipeline Correctness Violations

```python
# PROHIBITED: RANSAC on full cloud
labels = ransac_plane_fit(all_points)  # will find walls, not ground

# REQUIRED: seed-set pre-filtering
ground_mask, obstacle_mask, plane = ransac_ground_plane(points, cfg)
# (seed filtering is inside the function per the spec)
```

```python
# PROHIBITED: fixed intensity threshold
markings = points[points[:, 3] > 0.7]  # arbitrary constant, fails in rain

# REQUIRED: per-scan percentile threshold
threshold = np.percentile(points[:, 3], cfg.intensity_percentile)
markings = points[points[:, 3] >= threshold]
```

```python
# PROHIBITED: store features in pixel coordinates
feature.geometry = [(px1, py1), (px2, py2), ...]  # pixel coords in GeoJSON = wrong

# REQUIRED: world-frame coordinates
feature.geometry = [(x1_enu, y1_enu, z1_enu), ...]  # world frame (ENU)
```

```python
# PROHIBITED: train and evaluate on same dataset
train_loader = DataLoader(nuscenes_dataset, ...)
val_loader = DataLoader(nuscenes_dataset, ...)  # in-distribution, meaningless

# REQUIRED: cross-dataset evaluation
train_loader = DataLoader(NuScenesDataset(...), ...)
eval_loader = DataLoader(KITTIDataset(...), ...)   # out-of-distribution
```

```python
# PROHIBITED: global normalization constant in BEV projection
normalized = intensity / 255.0  # sensor-specific constant, wrong for different LiDAR

# REQUIRED: per-scan normalization
max_i = intensity.max()
normalized = intensity / max_i if max_i > 0 else intensity
```

### 4.3 Visualization Violations

```typescript
// PROHIBITED: legacy Three.js geometry API
const geometry = new THREE.Geometry();
points.forEach(p => geometry.vertices.push(new THREE.Vector3(p[0], p[1], p[2])));

// REQUIRED: BufferGeometry with typed arrays
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(points.length * 3);
// fill positions...
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
```

```typescript
// PROHIBITED: no geometry disposal on scene change
function loadNewCloud(newData: Float32Array) {
  // previous geometry still in GPU memory
  this.geometry = new THREE.BufferGeometry();
}

// REQUIRED: explicit disposal
function loadNewCloud(newData: Float32Array) {
  this.geometry.dispose();
  this.geometry = new THREE.BufferGeometry();
  this.mesh.geometry = this.geometry;
}
```

```typescript
// PROHIBITED: rebuilding geometry for color mode change
function setColorMode(mode: ColorMode) {
  this.geometry.dispose();
  this.geometry = new THREE.BufferGeometry();
  // re-upload positions + new colors (wasteful)
}

// REQUIRED: update color buffer attribute in place
function setColorMode(mode: ColorMode) {
  const colors = this.computeColors(mode);
  const colorAttr = this.geometry.getAttribute('color') as THREE.BufferAttribute;
  colorAttr.array = colors;
  colorAttr.needsUpdate = true;
  // positions unchanged - no geometry rebuild
}
```

### 4.4 General Anti-Patterns

```python
# PROHIBITED: Python for-loop over point array
colors = []
for point in points:  # 200K iterations in Python = 2+ seconds
    colors.append(intensity_to_color(point[3]))

# REQUIRED: vectorized numpy
intensities = points[:, 3]
colors = np.stack([intensities, intensities, intensities], axis=1)
```

```python
# PROHIBITED: float64 for point arrays
points = np.zeros((n, 4), dtype=np.float64)  # doubles memory, unnecessary

# REQUIRED: float32 for spatial data
points = np.zeros((n, 4), dtype=np.float32)
# Use float64 only in SE3 math where precision matters
```

```python
# PROHIBITED: magic number in source, no config hook
eps = 0.15  # where did this come from?
labels = DBSCAN(eps=eps).fit_predict(pts)

# REQUIRED: parameter from config with unit annotation
# In configs/default.yaml:
# extraction:
#   dbscan_eps: 0.15  # meters, cluster radius for road marking grouping
labels = DBSCAN(eps=cfg.extraction.dbscan_eps).fit_predict(pts)
```

```python
# PROHIBITED: assert without message
assert result.shape == (n, 3)

# REQUIRED: assert with diagnostic context
assert result.shape == (n, 3), \
    f"Expected ({n}, 3) in vehicle frame, got {result.shape}. " \
    f"Check that input was filtered to ground points before calling."
```

```python
# PROHIBITED: silent failure on edge case
def extract_features(points):
    labels = DBSCAN(...).fit_predict(points)
    # no check: if points is empty, this crashes downstream

# REQUIRED: explicit early return with log
def extract_features(points: np.ndarray) -> list[LaneBoundaryFeature]:
    if len(points) < cfg.dbscan_min_samples:
        logger.warning(f"Insufficient points for extraction: {len(points)}")
        return []
```

```python
# PROHIBITED: in-place modification of input arrays
def filter_cloud(points):
    points[:, 3] /= points[:, 3].max()  # modifies caller's data
    return points

# REQUIRED: operate on copy or clearly document mutation
def normalize_intensity(points: np.ndarray) -> np.ndarray:
    """Returns new array; input is not modified."""
    result = points.copy()
    max_i = result[:, 3].max()
    if max_i > 0:
        result[:, 3] /= max_i
    return result
```

---

## 5. Invariant Enforcement

These are checked via the Anti-Vibe Gate before each component is marked complete. Tests must verify each applicable invariant.

```
INV-1: Coordinate frame documented in every spatial function.
  Check: grep for functions accepting np.ndarray arguments;
         verify docstring contains "FRAME:" annotation.

INV-2: SE3 round-trip passes.
  Check: test_se3_round_trip() in tests/geometry/test_transforms.py
         must pass. This test is never deleted or skipped.

INV-3: Visualization uses ENU world frame.
  Check: PointCloudRenderer receives world-frame coordinates.
         Verify in ingest.py that accumulated cloud is in ENU before
         writing to Parquet.

INV-4: No hardcoded rotation/translation floats.
  Check: grep for np.array([[.*,.*,.*]]) in src/ - any 3x3 float
         array literal is a violation unless it's a test fixture.

INV-5: RANSAC uses seed-set pre-filtering.
  Check: ransac_ground_plane() contains seed_mask computation.
         Test: walls-only synthetic scene finds ground, not walls.

INV-6: BEV uses per-scan intensity normalization.
  Check: bev.py references max_intensity computed from the input
         array, not from a config constant.

INV-7: GeoJSON output contains world-frame coordinates.
  Check: test_geojson_coordinates() verifies that output coordinates
         are in the expected range for ENU (not pixel range 0–2000).

INV-8: U-Net trained on nuScenes, evaluated on KITTI.
  Check: train.py loads NuScenesDataset. eval section loads KITTIDataset.
         No shared split of the same dataset.

INV-9: BEV inference uses per-image normalization.
  Check: infer.py normalizes each image independently before forward pass.

INV-10: QA metrics computed against external GT annotations.
  Check: qa.py accepts ground_truth parameter from nuScenes annotation,
         not from pipeline's own geometric output.

INV-11: THREE.BufferGeometry used for all point cloud rendering.
  Check: grep for "THREE.Geometry(" in src/viz/ - must return no results.

INV-12: Previous geometry disposed on cloud reload.
  Check: loadNewCloud or equivalent method calls this.geometry.dispose()
         before constructing new geometry.
```

---

## 6. Validation Checkpoints

At each milestone tag (see prd.md Section 10.4), run and document these checks.

### Checkpoint v0.1 (Ingest)

```bash
# SE3 round-trip test
pytest tests/geometry/test_transforms.py::test_se3_round_trip -v

# KITTI loader test
pytest tests/data/test_kitti.py -v

# Ingest pipeline (requires KITTI download)
python scripts/run_pipeline.py \
  --config configs/kitti.yaml \
  --stage ingest \
  --n_frames 5 \
  --output /tmp/test_output
ls /tmp/test_output/accumulated.parquet  # must exist
python -c "
import pandas as pd
df = pd.read_parquet('/tmp/test_output/accumulated.parquet')
print(df.columns.tolist())  # must include x,y,z,intensity,timestamp,frame_id
print(df.dtypes)             # x,y,z,intensity must be float32
print(len(df), 'points')     # sanity check
"
```

### Checkpoint v0.2 (Filtering)

```bash
pytest tests/filters/ -v

# Ground plane test on synthetic data
python -c "
import numpy as np
from src.filters.ground_plane import ransac_ground_plane
from configs import load_config
cfg = load_config('configs/default.yaml')

# Flat ground: 50K points at z=0±0.05m
rng = np.random.default_rng(42)
ground = rng.standard_normal((50000, 3)).astype(np.float32)
ground[:, 2] = rng.normal(0, 0.05, 50000).astype(np.float32)
noise = rng.uniform(-5, 5, (5000, 3)).astype(np.float32)
noise[:, 2] = rng.uniform(0.5, 3.0, 5000).astype(np.float32)
points = np.vstack([ground, noise])

result = ransac_ground_plane(points, cfg.filters.ransac)
print(f'Inlier ratio: {result.inlier_ratio:.3f}')  # expect > 0.88
assert result.inlier_ratio > 0.88, 'RANSAC inlier ratio too low'
print('Ground plane test PASSED')
"

# Voxel filter benchmark
python scripts/benchmark.py --stage voxel
```

### Checkpoint v0.3 (Feature Extraction)

```bash
pytest tests/pipeline/test_bev.py tests/pipeline/test_extract.py -v

# BEV pixel location test
pytest tests/pipeline/test_bev.py::test_bev_single_point -v

# Synthetic lane extraction
python -c "
import numpy as np
from src.pipeline.extract import extract_lane_boundaries
from configs import load_config
cfg = load_config('configs/default.yaml')

# Two parallel 50m lane lines, 3m apart, high intensity
line1_x = np.linspace(0, 50, 500)
line1_y = np.full(500, -1.5)
line1 = np.column_stack([line1_x, line1_y, np.zeros(500), np.ones(500)])

line2_x = np.linspace(0, 50, 500)
line2_y = np.full(500, 1.5)
line2 = np.column_stack([line2_x, line2_y, np.zeros(500), np.ones(500)])

# Background: low intensity
bg = np.random.default_rng(42).uniform(0, 0.3, (5000, 4)).astype(np.float32)
bg[:, 2] = 0
points = np.vstack([line1, line2, bg]).astype(np.float32)

features = extract_lane_boundaries(points, cfg.extraction)
print(f'Detected {len(features)} lane boundaries')  # expect 2
assert len(features) == 2, f'Expected 2 lane boundaries, got {len(features)}'
print('Lane extraction test PASSED')
"
```

### Checkpoint v0.4 (ML)

```bash
# Model architecture test
python -c "
import torch
from src.ml.unet import BEVSegNet
model = BEVSegNet(num_classes=4)
x = torch.randn(2, 1, 512, 512)
y = model(x)
assert y.shape == (2, 4, 512, 512), f'Wrong output shape: {y.shape}'
param_count = sum(p.numel() for p in model.parameters())
print(f'Parameters: {param_count:,}')  # expect < 3M
assert param_count < 3_000_000, f'Model too large: {param_count}'
print('Architecture test PASSED')
"

# Training loss should decrease
python -c "
import torch
import torch.nn as nn
from src.ml.unet import BEVSegNet

model = BEVSegNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x = torch.randn(4, 1, 512, 512)
y = torch.randint(0, 4, (4, 512, 512))
losses = []
for _ in range(5):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print('Losses:', [f'{l:.4f}' for l in losses])
assert losses[-1] < losses[0], 'Loss did not decrease - training is broken'
print('Training convergence test PASSED')
"

# Cross-dataset evaluation (requires both datasets)
python scripts/evaluate_model.py \
  --model data/models/bev_segmentation_v1.pt \
  --dataset kitti \
  --config configs/kitti.yaml
# Output: per-class IoU table; lane_line IoU must be > 0.35
```

### Checkpoint v0.5 (QA)

```bash
pytest tests/pipeline/test_qa.py -v

# Perfect prediction edge case
python -c "
from src.pipeline.qa import compute_qa_metrics
from src.data.types import LaneBoundaryFeature, LaneType
import numpy as np

# Identical predicted and GT features
gt = [LaneBoundaryFeature(
    geometry=[[0,0,0],[10,0,0],[20,0,0]],
    feature_type=LaneType.LANE_LINE, confidence=1.0)]
pred = [LaneBoundaryFeature(
    geometry=[[0.05,0,0],[10.05,0,0],[20.05,0,0]],  # 5cm offset
    feature_type=LaneType.LANE_LINE, confidence=0.9)]

from configs import load_config
report = compute_qa_metrics(pred, gt, load_config('configs/default.yaml').qa)
print(f'Completeness: {report.completeness:.3f}')   # expect 1.0
print(f'FP rate: {report.false_positive_rate:.3f}') # expect 0.0
assert report.completeness == 1.0
assert report.false_positive_rate == 0.0
print('QA edge case test PASSED')
"
```

### Checkpoint v1.0 (Full Pipeline + Visualization)

```bash
# Full pipeline end-to-end
docker compose up pipeline
# Expect: outputs/features_kitti_0005.geojson exists
# Expect: outputs/qa_report_kitti_0005.json exists with completeness > 0

# Visualization performance
docker compose up viz
# Navigate to http://localhost:5173
# Load kitti_0005 scene
# Open Chrome DevTools > Performance > Record 5s of interaction
# Verify: frame rate ≥ 30fps during orbit; no frames > 33ms

# README quickstart test
# Delete all processed/ data
# Follow README from scratch (a colleague should be able to do this)
```

---

## 7. Anti-Vibe Gate

Before marking any component complete, add this comment block to the component's test file. All five questions must have specific, implementation-referencing answers. Vague answers indicate the implementation is not done.

```python
# ANTI-VIBE GATE - [component name, e.g., "ground_plane.py"]
#
# 1. COORDINATE FRAME CONTRACT
#    What frame do inputs arrive in? What frame do outputs leave in?
#    Where is the transform documented?
#    Example answer: "Input: vehicle frame (documented in docstring).
#    Output: vehicle frame (unchanged). No transform applied.
#    Vehicle frame required because the seed Z threshold is relative
#    to the vehicle height above ground."
#
# 2. SILENT FAILURE MODE
#    What input causes this component to produce incorrect output
#    without raising an exception?
#    Example answer: "Banked road > 5 degrees cross-slope.
#    RANSAC will fit a plane to the banked road but the normal vector
#    will be tilted, causing nearby obstacle points to be classified
#    as ground. Detection is: inlier_ratio suddenly drops below 0.3
#    on a known flat road segment."
#
# 3. VECTORIZATION STRATEGY
#    What numpy operations replace Python for-loops?
#    No Python iteration over point arrays is permitted.
#    Example answer: "Seed mask computed as boolean index:
#    seed_mask = (points[:,2] <= z_thresh) & (xy_dist <= radius).
#    Distance computation: xy_dist = np.linalg.norm(points[:,:2], axis=1).
#    Both fully vectorized. No per-point loop."
#
# 4. KNOWN LIMITATIONS
#    What does this component get wrong, and is it documented?
#    Example answer: "Fails on roads with >5° cross-slope (banked turns).
#    Documented in ground_plane.py module docstring with suggested
#    production solution (terrain model). Limitation also in README
#    known limitations section."
#
# 5. OBSERVABILITY CHECK
#    What does correct output look like in the 3D viewer?
#    What would you look at in the viewer to confirm this works?
#    Example answer: "After ground separation, the viewer should show
#    obstacle points (vehicles, buildings) in the obstacle layer
#    with no road surface points. Toggle to ground layer: should show
#    flat road surface with lane marking points clearly visible as
#    bright spots. If curbs appear in ground layer, seed radius is too large."
```

---

## 8. Debugging Expectations

When a test fails or output is incorrect, the debugging process is:

1. **Visualize before fixing.** Load the intermediate output in the viewer or matplotlib. Do not guess at the fix without seeing the data.

2. **Check coordinate frames first.** The majority of spatial data bugs are coordinate frame errors. Before investigating algorithm parameters, verify that the input to the failing stage is in the expected frame by printing a few example points and checking whether they are in the expected range.

3. **Test on synthetic data.** If KITTI output is wrong, reproduce on the synthetic fixture. If the synthetic test passes but KITTI fails, the issue is data-specific (calibration parsing, dataset format). If the synthetic test also fails, the algorithm is wrong.

4. **Profile before optimizing.** If a stage is too slow, profile with `cProfile` or `py-spy` before modifying the implementation. Print the timing breakdown. Optimization without profiling produces slower code.

```bash
# Profile a specific pipeline stage
python -m cProfile -s cumulative -o profile_output \
  scripts/run_pipeline.py --stage ransac --config configs/kitti.yaml

# Visualize profile (requires snakeviz)
pip install snakeviz
snakeviz profile_output
```

5. **Log intermediate values.** Every pipeline stage logs: input point count, output count, and the key diagnostic metric (inlier ratio for RANSAC, cluster count for DBSCAN, loss for training). These appear in the pipeline output and allow diagnosis without re-running.

6. **Do not patch around the bug.** If RANSAC is finding walls instead of ground, the fix is the seed pre-filter, not a post-processing step that removes high-Z points from the output.

---

## 9. Profiling Requirements

Profiling is not optional. Performance claims in `docs/benchmarks.md` must be backed by measured data.

```bash
# Run the benchmark script after implementing each stage
python scripts/benchmark.py --stage <name> --n_iterations 10

# Expected output format:
# Stage: ransac
# Hardware: Apple M1 Pro, 8-core CPU, 16GB RAM
# Input: 200,000 points
# Mean: 287ms | Std: 12ms | Min: 271ms | Max: 318ms
# Target: < 400ms
# Status: PASS
```

Every time a performance target is missed:
1. Profile to find the bottleneck
2. Fix the bottleneck (vectorize, add C++ extension, reduce iterations)
3. Re-benchmark and document the change

The benchmark results in `docs/benchmarks.md` are committed at each milestone tag. Do not update benchmark claims without running the benchmark.

---

## 10. Git Commit Standards

Commit after each completed step in Section 3, not at the end of a phase. A week of work in one commit is unacceptable. Each commit should represent one coherent, tested change.

```bash
# Example commit sequence for Phase 1
git commit -m "Add SE3 rigid body transform with round-trip test"
git commit -m "Add KITTI LiDAR .bin parser and pose loader"
git commit -m "Add SE3 composition and inverse tests"
git commit -m "Add 30-frame accumulation in ENU world frame"
git commit -m "Add frame convention documentation"
git commit -m "Ingest and coordinate frames complete"
```

Bad commit messages that will not be accepted:
- Any commit that touches more than 2 pipeline stages simultaneously
- A commit that breaks tests (tests must pass before commit to main)

The commit history is part of what an NVIDIA engineer might review. It should tell the story of the project being built incrementally, with clear technical decisions at each step.

---

## 11. Files Not to Commit

`prd.md` and `claude.md` remain in the working directory but are not committed to the repository. They are not added to `.gitignore`. The repository is public-facing - these files are working documents for the engineer building the project and should not appear in the public commit history.

The repository must stand on its own without these files. The README, inline docstrings, and `docs/` directory are the public-facing documentation. If an engineer who did not have access to `prd.md` or `claude.md` could not understand the architecture from the repository alone, the documentation is insufficient.

---

## 12. What Done Actually Means

A component is done when all of the following are true:

```
□ The implementation matches the spec in prd.md Section 5
□ All tests for this component pass (pytest -v, no skips)
□ Anti-Vibe Gate answered in the test file
□ No prohibited patterns present (review Section 4 against diff)
□ Performance target met if applicable (benchmark run and logged)
□ Module docstring documents input/output schema and coordinate frames
□ Config parameters used (no magic numbers in source)
□ Commit created with correct message format
```

The project is done when:

```
□ All unit and integration tests pass on clean clone
□ Full pipeline runs end-to-end on KITTI 0005 via docker compose
□ Three.js viewer renders at ≥30fps with 500K points
□ All QA metrics produce non-trivial values (completeness > 0)
□ Benchmarks documented in docs/benchmarks.md
□ README: quickstart, screenshot, architecture diagram, limitations
□ Milestone tags v0.1 through v1.0 present on main branch
□ Repository cloneable and runnable by someone following README alone
```

---

## 13. Cleanroom Code Generation Policy

When generating source code or comments:
* **NO Meta-References**: Do not include comment references to `CLAUDE.md`, `PRD.md`, "Anti-Vibe Gate," "invariant(s)", "prohibited patterns" in file headers or inline comments.
* **Professional Standards**: Comments should describe *why* a geometric or algorithmic choice was made, not *why* it satisfies a project constraint.
* **Documentation Scope**: Use concise and standard docstrings (Google or NumPy style). Do not mention specific repository development roles or project identifiers in the code.
* **Maintainability**: Ensure code is self-documenting for a general-purpose engineer who does not have access to these internal project setup files.

---


## 14. Resume Project Bullet Format

When documenting this project for a resume (output in chat not in a file), format each project entry exactly as:

```
Descriptive Project Name | Languages  Technologies
- bullet 1 (30 words maximum, following XYZ/Google format: "Accomplished X as measured by Y, leading to Z")
- bullet 2 (30 words maximum, following XYZ/Google format)
```

### 14.1 Example (derived from this project)

```
HD Map Feature Extraction for Autonomous Vehicles | Python, C, PyTorch, Three.js
- Built a production‑oriented pipeline that processes LiDAR point clouds from survey vehicles, extracting lane boundary features with 92% completeness and 0.08 false positives per meter when evaluated against ground truth map annotations.
- Implemented a RANSAC ground plane with seed pre‑filtering (improving robustness on banked roads by 40% over baseline), integrated a U‑Net BEV segmentation model, and delivered a Three.js QA viewer that renders 500K points at ≥30 fps.
```

### 14.2 Required Content for This Project

The two bullets must draw from the actual achievements of the implemented pipeline as verified by the validation checkpoints (Section 6). At minimum, they should reference:

- **First bullet**: End‑to‑end pipeline accomplishment – measurable improvement in extraction quality (completeness, precision, or runtime) over a naive baseline, or a novel combination of geometric  learned methods that enables a new capability (e.g., working across different LiDAR sensors).
- **Second bullet**: Concrete technical implementation detail – seed‑filtered RANSAC, per‑scan BEV normalization, cross‑dataset generalization (trained on nuScenes, evaluated on KITTI), or visualization performance characteristics.

### 14.3 Enforcement

- Before marking the project as complete (Section 12), produce the two resume bullets in a file `docs/resume_bullets.md` (manually git untracked for github repo purposes, not in gitignore).
- Each bullet must be exactly ≤30 words. Count words; if exceeded, rewrite.
- Each bullet must follow the XYZ/Google format: an active verb, a metric, and a business/impact outcome.
- Do not include vague claims like "improved accuracy" without a number. Do not list responsibilities ("responsible for…") – state achievements.

### 14.4 Word‑Counting and Validation

```bash
# Quick word count for a bullet
echo "Bullet text here" | wc -w
```

If a bullet exceeds 30 words, shorten it. The 30‑word limit is strict – it forces prioritisation of the most impactful information.

The resume bullets are the final deliverable that connects the technical work in this codebase to a hiring manager’s review at NVIDIA. Treat them with the same rigour as any invariant or test.
