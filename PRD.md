# PRD: HD Map Feature Extraction and Validation Pipeline

**Project Class:** Infrastructure / Spatial Data Engineering  
**Target Role:** Software Engineering Intern, Mapping Autonomous Vehicles - NVIDIA  
**Requisition:** JR2011493, Santa Clara CA  
**Author:** Aakash Kolli 
**Status:** Implementation-ready

---

## 1. Role Alignment

### 1.1 Internship Context

NVIDIA's HD Mapping team builds survey-grade centimeter-accuracy maps for the DRIVE platform. These are not navigation maps - they are geometric representations of road geometry that autonomous vehicles use for localization, perception priors, and path planning. The team ingests raw sensor data from survey vehicles (LiDAR, cameras, GPS/IMU), extracts structured map features (lane boundaries, road markings, curbs, signs), and publishes validated map artifacts consumed by perception, simulation, and planning stacks.

The intern role sits on the critical path between raw sensor data and usable map features. Based on the JD, day-to-day work includes:

- Building and debugging data processing pipelines that operate over LiDAR and camera data
- Writing automation algorithms for feature extraction and quality assessment
- Building visualization tools used by map QA engineers to inspect extracted features
- Exploring ML integration for feature detection, completion, and simulation augmentation
- Interfacing with downstream consumers: perception, infra, simulation

### 1.2 What This Project Demonstrates

This project is designed to answer the technical questions an NVIDIA map team engineer would ask in an interview:

| Interview Question | Project Evidence |
|---|---|
| How do you transform points between coordinate frames? | SE3 transform implementation with documented frame contracts |
| How do you separate ground from obstacles in a point cloud? | RANSAC plane fitting with seed-set pre-filtering |
| How do you detect lane markings from LiDAR? | BEV intensity projection + DBSCAN clustering + polyline fitting |
| Have you worked with ML on spatial data? | Lightweight U-Net trained on BEV images, cross-dataset evaluated |
| How would you validate extracted map features? | QA module: completeness, positional accuracy, FP rate vs. ground truth |
| How do map engineers inspect data? | Three.js 3D viewer with feature toggle, QA annotation, BEV mode |
| What are the failure modes? | RANSAC on banked roads, intensity thresholding in rain, documented |

The project does not claim to be a production mapping system. It claims to demonstrate understanding of the problem space, the data characteristics, the algorithmic approaches, and the engineering discipline required to build reliable spatial data pipelines.

### 1.3 Signal Coverage Against JD Requirements

**Required:**
- Python/C++: pipeline in Python, performance-critical filter as C++ pybind11 extension
- Robotics/AI coursework: RANSAC, coordinate frames, sensor fusion, ML segmentation
- Mapping/3D data: LiDAR point clouds, BEV projection, GeoJSON feature output

**Differentiating:**
- Data pipeline + automation algorithms: five-stage pipeline with defined boundaries
- Visualization tools: Three.js 3D viewer designed for map QA workflows
- ML + mapping dataset: U-Net segmentation on BEV intensity images
- Cloud/Docker: containerized pipeline with reproducible execution

---

## 2. Project Definition

### 2.1 Problem Statement

Raw LiDAR scans from survey vehicles contain all the information needed to build HD maps, but the information is implicit - lane boundaries exist as high-intensity returns on the ground plane, curbs exist as height discontinuities, road markings exist as geometric patterns when viewed from above. Extracting these features reliably, validating them against reference data, and giving engineers tools to inspect failures is the core engineering challenge of HD mapping.

This project implements a five-stage pipeline that takes raw LiDAR point cloud data, processes it through geometric and learned algorithms, produces structured map features in GeoJSON format, scores them against ground truth annotations, and renders the results in an interactive 3D viewer.

### 2.2 Scope Decisions

**Intentionally simplified vs. production:**

| This Project | Production System |
|---|---|
| Single-vehicle scan accumulation | Multi-vehicle aggregation with cross-session alignment |
| GPS/IMU pose from dataset | Full SLAM with loop closure and pose graph optimization |
| RANSAC plane fitting | Learned ground segmentation with terrain modeling |
| Intensity threshold + DBSCAN | Multi-modal fusion (LiDAR + camera + radar) |
| Lightweight U-Net | Large-scale perception models with multi-task heads |
| Local GeoJSON output | Distributed map database with tile-based access |
| Three.js browser viewer | Native C++ map inspection tools |

These simplifications are documented, defensible, and do not compromise the core engineering signals the project is designed to demonstrate.

**Correctness is non-negotiable for:**
- Coordinate frame transforms (silent errors here are catastrophic)
- Pose accumulation (drift here corrupts all downstream features)
- QA metric computation (incorrect metrics defeat the purpose)

**Approximation is acceptable for:**
- U-Net training data quality (nuScenes mini is limited but sufficient)
- Polyline fitting precision (cm-level accuracy not required for demo)
- Voxel grid resolution (5cm is demonstrably reasonable, not optimal)

---

## 3. Dataset Specification

### 3.1 Data Sources

**KITTI Raw Dataset**
- URL: `http://www.cvlibs.net/datasets/kitti/raw_data.php`
- License: Academic use only (Creative Commons Attribution-NonCommercial-ShareAlike 3.0)
- Required subsets:
  - `2011_09_26_drive_0005` - city driving, ~100 frames, ~800MB
  - `2011_09_26_drive_0009` - residential with lane markings, ~100 frames
  - `2011_09_26_calib` - sensor calibration files (required for all scenes)
- What it provides: Velodyne HDL-64E LiDAR (64 beam, 100K pts/frame), GPS/IMU poses, camera images, calibration
- Do NOT download full KITTI Raw (190GB). Download only listed drives.

**nuScenes v1.0-mini**
- URL: `https://www.nuscenes.org/nuscenes`
- License: Free for non-commercial use (registration required)
- Required: v1.0-mini only (~4GB, 10 scenes)
- What it provides: 32-beam LiDAR, HD map annotations (lane dividers, road segments, crosswalks), camera images
- Used for: U-Net training ground truth (map annotation layer), cross-dataset QA validation

### 3.2 Storage Layout

```text
data/
├── raw/                        # original dataset files, never modified
│   ├── kitti/
│   │   ├── 2011_09_26_calib/   # calibration (shared across drives)
│   │   ├── 2011_09_26_drive_0005_sync/
│   │   └── 2011_09_26_drive_0009_sync/
│   └── nuscenes_mini/
│       ├── maps/
│       ├── samples/
│       └── v1.0-mini/
├── processed/                  # pipeline-generated, reproducible from raw
│   ├── kitti_0005/
│   │   ├── accumulated_cloud.parquet
│   │   ├── ground_cloud.parquet
│   │   ├── obstacle_cloud.parquet
│   │   └── bev_intensity/      # per-frame .npy BEV images
│   └── nuscenes_mini/
│       ├── bev_images/         # training data for U-Net
│       └── bev_labels/         # annotation masks
├── models/                     # trained model weights
│   └── bev_segmentation_v1.pt
└── outputs/                    # final artifacts
    ├── features_kitti_0005.geojson
    ├── features_kitti_0009.geojson
    └── qa_report_kitti_0005.json
```

**Git policy:**
- `data/raw/` - NOT committed (too large; download instructions in README)
- `data/processed/` - NOT committed (reproducible; see preprocessing guide)
- `data/models/*.pt` - committed if < 100MB; otherwise Git LFS or download link in README
- `data/outputs/*.geojson` - committed as demo artifacts for README visualization
- `data/outputs/*.json` - committed as benchmark reference

### 3.3 Preprocessing Workflow

Preprocessing converts raw dataset files into formats the pipeline consumes efficiently. It is run once per scene and the outputs cached. Re-running is safe (idempotent by design).

```bash
# Download KITTI scene (example - actual instructions in README)
# Place under data/raw/kitti/ per the layout above

# Run preprocessing (creates data/processed/kitti_0005/)
python scripts/preprocess_kitti.py \
    --scene data/raw/kitti/2011_09_26_drive_0005_sync \
    --calib data/raw/kitti/2011_09_26_calib \
    --output data/processed/kitti_0005 \
    --n_frames 50

# Run U-Net training data preparation
python scripts/prepare_nuscenes_training.py \
    --nuscenes data/raw/nuscenes_mini \
    --output data/processed/nuscenes_mini

# Expected runtimes on M1 MacBook Pro:
# preprocess_kitti: ~3 minutes for 50 frames
# prepare_nuscenes_training: ~8 minutes for 10 scenes
```

**Caching strategy:** Processed outputs use content-addressed filenames (SHA256 of input parameters) stored in a lightweight manifest (`data/processed/.manifest.json`). Preprocessing is skipped if the manifest records a matching entry. This prevents redundant reprocessing while remaining transparent.

---

## 4. System Architecture

### 4.1 Repository Structure

```text
hd-map-pipeline/
│
├── src/
│   ├── geometry/           # coordinate frames, transforms, spatial math
│   │   ├── transforms.py   # SE3, coordinate frame management
│   │   ├── polyline.py     # polyline fitting, simplification, metrics
│   │   └── spatial.py      # KD-tree queries, voxel indexing
│   │
│   ├── filters/            # point cloud processing
│   │   ├── ground_plane.py # RANSAC ground separation
│   │   ├── voxel.py        # Python interface to C++ voxel filter
│   │   └── outlier.py      # statistical and radius outlier removal
│   │
│   ├── ext/                # C++ pybind11 extensions
│   │   ├── voxel_filter.cpp
│   │   ├── CMakeLists.txt
│   │   └── bindings.cpp
│   │
│   ├── pipeline/           # data flow stages
│   │   ├── ingest.py       # dataset loading, frame accumulation
│   │   ├── bev.py          # BEV intensity projection
│   │   ├── extract.py      # geometric feature extraction
│   │   ├── fuse.py         # geometric + ML feature fusion
│   │   └── qa.py           # quality assessment and scoring
│   │
│   ├── ml/                 # machine learning components
│   │   ├── unet.py         # U-Net architecture
│   │   ├── dataset.py      # PyTorch dataset for BEV images
│   │   ├── train.py        # training loop
│   │   └── infer.py        # batch inference wrapper
│   │
│   ├── data/               # dataset-specific loaders
│   │   ├── kitti.py        # KITTI calibration, LiDAR, pose parsing
│   │   └── nuscenes.py     # nuScenes LiDAR, annotation parsing
│   │
│   └── viz/                # Three.js visualization app
│       ├── src/
│       │   ├── main.tsx
│       │   ├── App.tsx
│       │   ├── renderer/
│       │   │   ├── PointCloudRenderer.ts
│       │   │   ├── FeatureRenderer.ts
│       │   │   └── QAAnnotationRenderer.ts
│       │   ├── controls/
│       │   │   ├── CameraController.ts
│       │   │   └── LayerToggle.tsx
│       │   ├── io/
│       │   │   └── GeoJSONLoader.ts
│       │   └── types/
│       │       └── spatial.ts
│       ├── package.json
│       ├── vite.config.ts
│       └── tsconfig.json
│
├── scripts/
│   ├── preprocess_kitti.py
│   ├── prepare_nuscenes_training.py
│   ├── run_pipeline.py         # end-to-end pipeline runner
│   └── benchmark.py            # performance measurement
│
├── tests/
│   ├── geometry/
│   │   ├── test_transforms.py
│   │   └── test_polyline.py
│   ├── filters/
│   │   ├── test_ground_plane.py
│   │   └── test_voxel.py
│   ├── pipeline/
│   │   ├── test_bev.py
│   │   └── test_qa.py
│   └── fixtures/               # small synthetic test data
│       ├── synthetic_ground_plane.npz
│       └── synthetic_lane_markings.npz
│
├── configs/
│   ├── default.yaml            # default pipeline parameters
│   ├── kitti.yaml              # KITTI-specific overrides
│   └── nuscenes.yaml           # nuScenes-specific overrides
│
├── docker/
│   ├── Dockerfile.pipeline
│   ├── Dockerfile.viz
│   └── docker-compose.yml
│
├── docs/
│   ├── coordinate_frames.md    # required reading before contributing
│   ├── dataset_setup.md        # download and preprocessing instructions
│   └── benchmarks.md           # performance results with hardware specs
│
├── data/                       # layout described in Section 3.2
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── setup.py                    # installs src as package + builds C++ ext
└── README.md
```

**Directory intent:**
- `src/geometry/` is the mathematical foundation. It has no imports from other `src/` modules. Everything depends on it.
- `src/filters/` depends only on `src/geometry/`. Never imports pipeline logic.
- `src/pipeline/` is where stages connect. It imports from geometry, filters, data, and ml.
- `src/ml/` is self-contained. Training and inference are separate concerns.
- `src/data/` knows about specific dataset formats. Pipeline stages use its output, never raw file formats directly.
- `src/viz/` is an independent Node.js application. It reads GeoJSON and Parquet files. No Python imports.
- `scripts/` are entry points, not library code. They wire together `src/` modules.
- `tests/fixtures/` contains synthetic data with known properties. Tests never depend on downloaded datasets.

### 4.2 Pipeline Stages and Data Flow

```text
KITTI Raw LiDAR Frames (.bin)
KITTI GPS/IMU Poses (oxts)
KITTI Calibration (calib)
        │
        ▼
┌───────────────────────────────┐
│ Stage 1: Ingest               │
│ src/pipeline/ingest.py        │
│                               │
│ Inputs:  raw .bin files       │
│          oxts pose files      │
│          calib files          │
│                               │
│ Operations:                   │
│  - Parse LiDAR to numpy       │
│  - Parse oxts to SE3 poses    │
│  - Transform sensor->vehicle   │
│  - Accumulate N frames        │
│    into local map segment     │
│                               │
│ Output:  accumulated.parquet  │
│   schema: x,y,z,intensity,    │
│           timestamp,frame_id  │
│   frame: world (ENU)          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Stage 2: Filter               │
│ src/filters/                  │
│                               │
│ Input:  accumulated.parquet   │
│ frame:  world                 │
│                               │
│ Operations:                   │
│  - Voxel grid downsample      │
│    (C++ extension, 5cm voxel) │
│  - Radius outlier removal     │
│  - RANSAC ground separation   │
│                               │
│ Outputs:                      │
│   ground.parquet    (inliers) │
│   obstacle.parquet  (outliers)│
│ frame: world (unchanged)      │
└───────────────┬───────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
┌────────────────┐  ┌──────────────────────────────┐
│ Stage 3A:      │  │ Stage 3B:                    │
│ Geometric      │  │ ML Segmentation              │
│ Extraction     │  │ src/ml/infer.py              │
│ src/pipeline/  │  │                              │
│ extract.py     │  │ Input: ground.parquet        │
│                │  │                              │
│ - BEV project  │  │ Operations:                  │
│ - Intensity    │  │  - BEV intensity image       │
│   threshold    │  │  - U-Net forward pass        │
│ - DBSCAN       │  │  - Mask post-processing      │
│ - Polyline fit │  │  - Back-project to 3D        │
│                │  │                              │
│ Output:        │  │ Output:                      │
│ geom_feat.json │  │ ml_feat.json                 │
└───────┬────────┘  └──────────────┬───────────────┘
        │                          │
        └──────────────┬───────────┘
                       ▼
┌───────────────────────────────┐
│ Stage 4: Fuse + QA            │
│ src/pipeline/fuse.py          │
│ src/pipeline/qa.py            │
│                               │
│ Inputs:  geom_feat.json       │
│          ml_feat.json         │
│          ground_truth (GT)    │
│                               │
│ Operations:                   │
│  - IOU-based feature matching │
│  - Confidence scoring         │
│  - GT comparison              │
│  - QA metric computation      │
│                               │
│ Outputs:                      │
│  features.geojson  (fused)    │
│  qa_report.json               │
└───────────────────────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Stage 5: Visualization        │
│ src/viz/ (Three.js app)       │
│                               │
│ Inputs: features.geojson      │
│         qa_report.json        │
│         ground.parquet        │
│         (served via local API)│
│                               │
│ Renders:                      │
│  - Point cloud (500K+ pts)    │
│  - Feature overlays           │
│  - QA annotations             │
│  - BEV mode toggle            │
└───────────────────────────────┘
```

### 4.3 Configuration Strategy

All pipeline parameters live in `configs/`. No magic numbers in source files.

```yaml
# configs/default.yaml

pipeline:
  n_frames_accumulate: 30       # frames to accumulate per map segment
  world_frame: "enu"            # east-north-up, standard for AV

filters:
  voxel_size: 0.05              # meters - 5cm grid
  outlier_radius: 0.30          # meters for radius outlier removal
  outlier_min_neighbors: 5      # minimum neighbors to keep point
  ransac:
    max_iterations: 150
    distance_threshold: 0.15    # meters
    min_inlier_ratio: 0.35
    seed_z_percentile: 20       # only consider lowest Z% as seed
    seed_xy_radius: 20.0        # meters, max range for seed points

bev:
  resolution: 0.05              # meters per pixel
  extent: 50.0                  # meters from vehicle center
  # image size = 2 * extent / resolution = 2000x2000 px

extraction:
  intensity_percentile: 85      # threshold for road marking candidates
  dbscan_eps: 0.15              # meters, cluster radius
  dbscan_min_samples: 10
  polyline_rdp_epsilon: 0.05    # Ramer-Douglas-Peucker simplification

ml:
  model_path: "data/models/bev_segmentation_v1.pt"
  batch_size: 4
  confidence_threshold: 0.60
  device: "cpu"                 # override to "cuda" if available

qa:
  match_iou_threshold: 0.40
  positional_accuracy_percentiles: [50, 95]
  max_gt_match_distance: 1.0    # meters
```

Dataset-specific configs extend defaults via inheritance and override only what differs.

---

## 5. Component Specifications

### 5.1 Coordinate Frame Management

This is the most critical module in the codebase. Every spatial data bug traces back to coordinate frame confusion. The module defines the canonical representation and provides all transforms used downstream.

**Frame conventions (KITTI-following):**

| Frame | Origin | X | Y | Z | Units |
|---|---|---|---|---|---|
| `lidar` | Velodyne sensor center | forward | left | up | meters |
| `vehicle` | Rear axle center | forward | left | up | meters |
| `world` | GPS origin (ENU) | east | north | up | meters |

All intermediate computations use explicit frame labels. No raw float arrays are passed without a comment or docstring specifying which frame they are in.

```python
# src/geometry/transforms.py

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SE3:
    """
    Rigid body transform representing a coordinate frame change.

    Convention: this transform converts points FROM a source frame
    TO a target frame. Written as T_target_source.

    Example: T_vehicle_lidar converts LiDAR-frame points to vehicle frame.

    Attributes:
        rotation:    (3, 3) float64 rotation matrix (SO(3))
        translation: (3,)   float64 translation vector, in target frame
    """
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        assert self.rotation.shape == (3, 3), f"Expected (3,3), got {self.rotation.shape}"
        assert self.translation.shape == (3,), f"Expected (3,), got {self.translation.shape}"
        # Rotation matrix orthogonality check (loose tolerance for float precision)
        RtR = self.rotation.T @ self.rotation
        assert np.allclose(RtR, np.eye(3), atol=1e-4), "Rotation matrix is not orthogonal"

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply transform to Nx3 point array.

        Args:
            points: (N, 3) float32 or float64, in SOURCE frame

        Returns:
            (N, 3) float64, in TARGET frame
        """
        assert points.ndim == 2 and points.shape[1] == 3
        return (points.astype(np.float64) @ self.rotation.T) + self.translation

    def inverse(self) -> 'SE3':
        """Return T_source_target (reverse direction)."""
        R_inv = self.rotation.T
        t_inv = -(R_inv @ self.translation)
        return SE3(rotation=R_inv, translation=t_inv)

    def compose(self, other: 'SE3') -> 'SE3':
        """
        Return the composed transform T_other_target @ T_self.
        Equivalent to: apply self, then apply other.
        """
        R = other.rotation @ self.rotation
        t = other.rotation @ self.translation + other.translation
        return SE3(rotation=R, translation=t)

    @classmethod
    def identity(cls) -> 'SE3':
        return cls(rotation=np.eye(3), translation=np.zeros(3))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'SE3':
        """Parse 4x4 homogeneous transform matrix."""
        assert T.shape == (4, 4)
        return cls(rotation=T[:3, :3], translation=T[:3, 3])
```

**Round-trip test (required, always passes before merge):**

```python
# tests/geometry/test_transforms.py

def test_se3_round_trip():
    """Applying T then T.inverse() recovers original points within tolerance."""
    rng = np.random.default_rng(42)
    # Random valid rotation via QR decomposition
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    t = rng.standard_normal(3)
    T = SE3(rotation=Q, translation=t)

    points = rng.standard_normal((1000, 3)).astype(np.float32)
    recovered = T.inverse().transform_points(T.transform_points(points))
    np.testing.assert_allclose(points, recovered, atol=1e-5,
        err_msg="SE3 round-trip failed: transform or inverse is incorrect")
```

### 5.2 RANSAC Ground Plane Estimation

Ground plane separation is required before any road feature extraction. The RANSAC implementation uses a seed set (lowest Z percentile within a radius) rather than the full cloud. This is not an optimization - without the seed set, RANSAC finds building walls and vehicle sides rather than the ground plane.

**Known limitations (documented, not hidden):**
- Fails on roads with >5° cross-slope (banked turns). Production systems use terrain models.
- Fails near bridges and overpasses where sky-level and road-level points coexist.
- Per-frame fitting does not enforce temporal consistency. Adjacent frames may produce slightly different plane parameters.

```python
# src/filters/ground_plane.py

def ransac_ground_plane(
    points: np.ndarray,
    cfg: RansacConfig,
) -> GroundPlaneResult:
    """
    Separate ground and non-ground points via RANSAC plane fitting.

    Args:
        points: (N, 3) float32 in VEHICLE frame.
                x=forward, y=left, z=up, origin=rear axle center.
                Input must be in vehicle frame. World-frame input
                will produce incorrect results because vehicle
                height offset is baked into the thresholds.
        cfg:    RansacConfig from configs/

    Returns:
        GroundPlaneResult with:
            ground_mask:  (N,) bool
            obstacle_mask: (N,) bool
            plane:        (4,) [a, b, c, d] unit-normal plane equation
            inlier_ratio: float, diagnostic metric

    Notes:
        Seed filtering pre-selects candidate ground points by:
        1. Taking the lowest cfg.seed_z_percentile percent by Z
        2. Restricting to cfg.seed_xy_radius meters from origin
        This is necessary because RANSAC on the full cloud samples
        from walls and vehicles, rarely finding the ground plane.

        After RANSAC convergence, a plane refinement step re-fits
        using all inliers (not just the 3-point sample), which
        substantially improves normal accuracy.
    """
    assert points.ndim == 2 and points.shape[1] == 3, \
        f"Expected (N,3) in vehicle frame, got {points.shape}"

    # Seed filtering
    xy_dist = np.linalg.norm(points[:, :2], axis=1)
    z_thresh = np.percentile(points[:, 2], cfg.seed_z_percentile)
    seed_mask = (points[:, 2] <= z_thresh) & (xy_dist <= cfg.seed_xy_radius)
    seed_idx = np.where(seed_mask)[0]

    if len(seed_idx) < 3:
        raise InsufficientSeedPointsError(
            f"Only {len(seed_idx)} seed points found. "
            f"Check that input is in vehicle frame and road is visible."
        )

    best_mask = np.zeros(len(points), dtype=bool)
    best_plane = np.array([0., 0., 1., 0.])  # default: flat ground
    best_count = 0
    rng = np.random.default_rng(cfg.random_seed)

    for _ in range(cfg.max_iterations):
        idx = rng.choice(seed_idx, size=3, replace=False)
        sample = points[idx].astype(np.float64)

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue  # degenerate: near-collinear points

        normal /= norm_len
        d = -float(normal @ sample[0])
        dists = np.abs(points.astype(np.float64) @ normal + d)
        mask = dists < cfg.distance_threshold
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask
            best_plane = np.append(normal, d)

    # Refinement: refit using all inliers
    if best_count > 10:
        inliers = points[best_mask].astype(np.float64)
        centroid = inliers.mean(axis=0)
        _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
        refined_normal = Vt[-1]
        refined_d = -float(refined_normal @ centroid)
        refined_dists = np.abs(points.astype(np.float64) @ refined_normal + refined_d)
        best_mask = refined_dists < cfg.distance_threshold
        best_plane = np.append(refined_normal, refined_d)

    return GroundPlaneResult(
        ground_mask=best_mask,
        obstacle_mask=~best_mask,
        plane=best_plane,
        inlier_ratio=best_mask.mean(),
    )
```

### 5.3 BEV Intensity Projection

The Bird's Eye View intensity image is the primary input to the U-Net. It projects ground-plane points onto a 2D grid where each pixel's value is the maximum LiDAR return intensity within that grid cell. High-intensity returns correspond to painted road markings (white/yellow paint has significantly higher reflectance than asphalt at LiDAR wavelengths - typically 905nm).

```python
# src/pipeline/bev.py

def project_to_bev(
    ground_points: np.ndarray,
    cfg: BEVConfig,
) -> BEVImage:
    """
    Project ground point cloud onto a Bird's Eye View intensity image.

    Args:
        ground_points: (N, 4) float32 [x, y, z, intensity]
                       FRAME: world (ENU). Ground points only.
                       Vehicle position assumed at (0, 0) in local coords.
                       Caller must translate to vehicle-centered coords
                       before calling.
        cfg:           BEVConfig from configs/

    Returns:
        BEVImage with:
            image:      (H, W) float32, intensity in [0, 1]
            origin_xy:  (2,) float64, world coords of image (0,0) corner
            resolution: float, meters per pixel

    Notes:
        Normalization uses per-scan max intensity, not a global constant.
        LiDAR intensity varies significantly across sensors and conditions.
        A fixed threshold that works on a sunny day will fail in rain.

        Image H = W = int(2 * cfg.extent / cfg.resolution).
        At 5cm resolution over 50m extent: 2000x2000 pixels.
        This is approximately the resolution used in production BEV pipelines.
    """
    H = W = int(2 * cfg.extent / cfg.resolution)
    bev = np.zeros((H, W), dtype=np.float32)

    # Translate to vehicle-centered coordinates
    # (caller provides world-frame points; vehicle is at world origin)
    x = ground_points[:, 0]
    y = ground_points[:, 1]
    intensity = ground_points[:, 3]

    # Normalize intensity per-scan
    max_intensity = intensity.max()
    if max_intensity > 0:
        intensity = intensity / max_intensity

    # Convert to pixel indices
    px = np.floor((x + cfg.extent) / cfg.resolution).astype(np.int32)
    py = np.floor((y + cfg.extent) / cfg.resolution).astype(np.int32)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, intensity = px[valid], py[valid], intensity[valid]

    # Max pooling via scatter: np.maximum.at handles duplicate indices
    np.maximum.at(bev, (py, px), intensity)

    return BEVImage(
        image=bev,
        origin_xy=np.array([-cfg.extent, -cfg.extent]),
        resolution=cfg.resolution,
    )
```

**Test case for BEV correctness:**

```python
def test_bev_single_point():
    """A single point at known world position maps to the correct pixel."""
    cfg = BEVConfig(resolution=0.10, extent=10.0)  # 200x200 image
    # Place point at (3.0m east, 2.0m north) with max intensity
    point = np.array([[3.0, 2.0, 0.0, 1.0]], dtype=np.float32)
    result = project_to_bev(point, cfg)

    # Expected pixel: px = (3.0 + 10.0) / 0.10 = 130, py = (2.0 + 10.0) / 0.10 = 120
    assert result.image[120, 130] == pytest.approx(1.0, abs=1e-5), \
        "Point did not land in expected pixel"
    # All other pixels should be zero
    assert result.image.sum() == pytest.approx(1.0, abs=1e-5)
```

### 5.4 Geometric Lane Boundary Extraction

Lane boundaries are extracted from the BEV image using intensity thresholding followed by DBSCAN clustering in 3D world space. Clustering happens in 3D (not 2D pixel space) to avoid merging nearby parallel lanes that may appear connected in the BEV image.

```python
# src/pipeline/extract.py (key section)

def extract_lane_boundaries(
    ground_points: np.ndarray,
    cfg: ExtractionConfig,
) -> list[LaneBoundaryFeature]:
    """
    Extract lane boundary polylines from ground point cloud.

    Args:
        ground_points: (N, 4) float32 [x, y, z, intensity]
                       FRAME: world (ENU). Ground plane only.

    Returns:
        List of LaneBoundaryFeature, each containing:
            geometry:     list of (x, y, z) world-frame points
            feature_type: LaneType enum
            confidence:   float in [0, 1]

    Algorithm:
        1. Threshold: keep points above cfg.intensity_percentile
           of the per-scan intensity distribution
        2. DBSCAN: cluster in 3D space (eps in meters)
        3. Filter: discard clusters below minimum point count
        4. Fit: Ramer-Douglas-Peucker polyline per cluster
        5. Classify: lane_line vs curb based on cluster geometry
    """
    # Intensity threshold - percentile-based, not fixed value
    intensity = ground_points[:, 3]
    threshold = np.percentile(intensity, cfg.intensity_percentile)
    marking_candidates = ground_points[intensity >= threshold]

    if len(marking_candidates) < cfg.dbscan_min_samples:
        return []  # sparse scene, no markings detected

    # DBSCAN on XY coordinates only (Z irrelevant for road-plane clustering)
    labels = DBSCAN(
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        n_jobs=-1,
    ).fit_predict(marking_candidates[:, :2])

    features = []
    for label in set(labels):
        if label == -1:
            continue  # noise cluster
        cluster = marking_candidates[labels == label]
        if len(cluster) < 20:
            continue  # too small to be a lane boundary

        polyline = fit_polyline_rdp(
            cluster[:, :3],
            epsilon=cfg.rdp_epsilon,
        )
        feature_type = classify_cluster(cluster)
        confidence = compute_cluster_confidence(cluster)

        features.append(LaneBoundaryFeature(
            geometry=polyline.tolist(),
            feature_type=feature_type,
            confidence=confidence,
            point_count=len(cluster),
        ))

    return features
```

### 5.5 U-Net Architecture

The segmentation model is a lightweight U-Net operating on BEV intensity images. It is intentionally small (~2M parameters) to run on CPU in under 2 seconds. Production systems use much larger multi-modal models; the signal here is that the architecture and training methodology are sound, not that the model is SOTA.

```python
# src/ml/unet.py

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU - standard U-Net building block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BEVSegNet(nn.Module):
    """
    U-Net for BEV road marking segmentation.

    Input:  (B, 1, H, W)  BEV intensity image, float32, [0, 1]
    Output: (B, C, H, W)  per-class logits (pre-softmax)

    Classes (C=4):
        0: background
        1: lane_line  (solid or dashed marking)
        2: crosswalk
        3: arrow / symbol

    Design constraints:
        - Must run on CPU in < 2s for 512x512 input
        - ~2M parameters (verified: count in tests)
        - Uses batch norm for training stability on small BEV dataset
        - No depthwise separable convolutions (keep architecture readable)

    Training notes:
        - Trained on nuScenes-mini BEV images with map annotation labels
        - Evaluated on KITTI (cross-dataset); this is required for generalization claim
        - Class imbalance handled via weighted cross-entropy (background ~90%)
        - Input resolution during training: 512x512 (cropped from 2000x2000)
    """

    def __init__(self, num_classes: int = 4, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBNReLU(1, base_ch)
        self.enc2 = ConvBNReLU(base_ch, base_ch * 2)
        self.enc3 = ConvBNReLU(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBNReLU(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBNReLU(base_ch * 8, base_ch * 16)
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBNReLU(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBNReLU(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBNReLU(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBNReLU(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
```

### 5.6 QA Scoring Module

QA metrics are computed against nuScenes map annotations (the ground truth layer). The metrics mirror what a production map QA system would report.

```python
# src/pipeline/qa.py

@dataclass
class QAReport:
    scene_id: str
    # Completeness: fraction of GT features that were detected
    completeness: float
    # Positional accuracy: distance distribution between matched features
    positional_accuracy_p50: float  # meters
    positional_accuracy_p95: float  # meters
    # False positive rate: fraction of detections with no GT match
    false_positive_rate: float
    # Classification accuracy: correct type on matched features
    classification_accuracy: float
    # Per-class breakdown
    per_class_completeness: dict[str, float]
    # Unmatched GT features (for visualization)
    missed_gt_features: list[str]
    # False positive detection IDs (for visualization)
    false_positive_ids: list[str]


def compute_qa_metrics(
    predicted: list[LaneBoundaryFeature],
    ground_truth: list[LaneBoundaryFeature],
    cfg: QAConfig,
) -> QAReport:
    """
    Compute QA metrics by matching predicted features to ground truth.

    Matching uses Hausdorff distance between polylines with a maximum
    match threshold of cfg.max_gt_match_distance meters. Unmatched
    predictions are false positives. Unmatched GT features are misses.

    This is a simplified version of the IoU-based matching used in
    production. Hausdorff distance is a reasonable proxy for polyline
    similarity that does not require polygon area computation.
    """
    ...
```

### 5.7 3D Visualization

The viewer is a Three.js application served locally. It reads GeoJSON feature files and Parquet point clouds via a minimal Python HTTP server (`scripts/serve_viz.py`). There is no backend API - the server is a file server with CORS headers.

**Rendering constraints:**
- Point cloud: up to 500K points at ≥30fps. Achieved via `THREE.Points` with `BufferGeometry` typed arrays. Never use legacy `THREE.Geometry`.
- Feature lines: GeoJSON polylines rendered as `THREE.Line` objects with `THREE.LineSegments` for QA annotation coloring.
- Color modes: intensity (grayscale), height (jet colormap), classification (per-class palette). Mode switch re-assigns vertex colors in place - no geometry rebuild.
- Memory: each KITTI frame is ~800KB as float32. 50 accumulated frames = ~40MB. This fits in browser memory. Dispose previous frame geometry on scene switch.
- Frame rate: measured in the browser DevTools Performance panel and reported in `docs/benchmarks.md`.

**UI philosophy:** The visualization interface is modeled after internal robotics tooling - dark background, monospace labels, minimal chrome, toggle controls rather than dropdowns. No gradients, no glassmorphism, no dashboard widgets. The viewer exists to inspect spatial data efficiently, not to impress non-engineers.

```typescript
// src/viz/src/renderer/PointCloudRenderer.ts

export class PointCloudRenderer {
  private geometry: THREE.BufferGeometry;
  private material: THREE.PointsMaterial;
  private mesh: THREE.Points;

  constructor(private scene: THREE.Scene) {
    this.geometry = new THREE.BufferGeometry();
    this.material = new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: true,
      sizeAttenuation: true,
    });
    this.mesh = new THREE.Points(this.geometry, this.material);
    scene.add(this.mesh);
  }

  load(positions: Float32Array, intensities: Float32Array): void {
    // Dispose previous buffers to free GPU memory
    this.geometry.dispose();
    this.geometry = new THREE.BufferGeometry();

    this.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(positions, 3)
    );
    this.geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(
        this.intensityToColors(intensities),
        3
      )
    );
    this.geometry.computeBoundingSphere();
    this.mesh.geometry = this.geometry;
  }

  setColorMode(mode: 'intensity' | 'height', positions: Float32Array, intensities: Float32Array): void {
    // Update color buffer in place - no geometry rebuild required
    const colors = mode === 'intensity'
      ? this.intensityToColors(intensities)
      : this.heightToColors(positions);

    const colorAttr = this.geometry.getAttribute('color') as THREE.BufferAttribute;
    colorAttr.array = colors;
    colorAttr.needsUpdate = true;
  }

  private intensityToColors(intensities: Float32Array): Float32Array {
    const colors = new Float32Array(intensities.length * 3);
    for (let i = 0; i < intensities.length; i++) {
      const v = intensities[i];
      colors[i * 3] = v;
      colors[i * 3 + 1] = v;
      colors[i * 3 + 2] = v;
    }
    return colors;
  }

  private heightToColors(positions: Float32Array): Float32Array {
    // Jet colormap: compute Z range, map to blue->green->red
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 2; i < positions.length; i += 3) {
      if (positions[i] < minZ) minZ = positions[i];
      if (positions[i] > maxZ) maxZ = positions[i];
    }
    const range = (maxZ - minZ) || 1;
    const colors = new Float32Array(positions.length);
    for (let i = 0; i < positions.length / 3; i++) {
      const t = (positions[i * 3 + 2] - minZ) / range;
      colors[i * 3]     = Math.max(0, Math.min(1, 1.5 - Math.abs(t - 1.0) * 2));
      colors[i * 3 + 1] = Math.max(0, Math.min(1, 1.5 - Math.abs(t - 0.5) * 2));
      colors[i * 3 + 2] = Math.max(0, Math.min(1, 1.5 - Math.abs(t - 0.0) * 2));
    }
    return colors;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.scene.remove(this.mesh);
  }
}
```

---

## 6. Non-Negotiable Invariants

These are correctness requirements that cannot be relaxed for any reason. Each has a corresponding automated test.

```text
COORDINATE FRAMES
─────────────────
INV-1: Every function accepting point arrays documents the expected
       coordinate frame in its docstring or type annotation.
       "Frame: world (ENU)" is required. "xyz coordinates" is not sufficient.

INV-2: SE3 round-trip test passes for all transforms:
       T.inverse().transform_points(T.transform_points(points)) == points
       within atol=1e-5 meters.

INV-3: Visualization always displays points in ENU world frame.
       Vehicle-frame visualization produces tilted roads on curves.

INV-4: Frame transforms are never hardcoded as float literals.
       All rotation matrices and translations are parsed from calibration
       files or computed from measurement data.

GEOMETRY
────────
INV-5: RANSAC ground plane uses seed-set pre-filtering.
       Full-cloud RANSAC is prohibited. This is not a performance
       optimization; it is a correctness requirement.
       Test: on a synthetic scene with vertical walls, seed-set RANSAC
       finds the ground plane. Full-cloud RANSAC does not.

INV-6: Intensity thresholding uses a per-scan percentile, not a
       global fixed value. Fixed thresholds fail across sensors
       and weather conditions.
       Test: two synthetic scans with different overall intensity
       ranges must produce consistent road marking detection rates.

INV-7: Lane boundary polylines are stored and exported in world-frame
       (ENU) coordinates, not pixel or vehicle-frame coordinates.
       GeoJSON output with pixel coordinates is incorrect.

ML
──
INV-8: U-Net is trained on nuScenes and evaluated on KITTI only.
       Cross-dataset evaluation is non-negotiable. In-distribution
       evaluation on nuScenes would not demonstrate generalization.

INV-9: BEV image normalization is per-image, not global.
       Global normalization constants that were computed on the
       training set must not be used at inference time.

QA
──
INV-10: QA completeness metric is computed against ground truth
        annotations, not against the pipeline's own geometric
        extraction output. Self-evaluation is not QA.

VISUALIZATION
─────────────
INV-11: Point cloud uses THREE.BufferGeometry with Float32Array.
        THREE.Geometry (legacy API) is prohibited.
        Test: renderer instantiates without THREE.Geometry usage
        (verified by static analysis or runtime attribute check).

INV-12: Previous frame geometry is disposed before loading new frame.
        THREE.BufferGeometry.dispose() must be called.
        Memory leak otherwise; browser tab crashes on scene 3.
```

---

## 7. Performance Targets

All targets measured on an Apple M1 Pro (8-core CPU, 16GB RAM). Hardware spec is documented in `docs/benchmarks.md`.

| Stage | Target | Measurement Method |
|---|---|---|
| Ingest + preprocess 30 frames | < 90s | `time scripts/preprocess_kitti.py` |
| Ground plane RANSAC (per frame, 200K pts) | < 400ms | `scripts/benchmark.py --stage=ransac` |
| Voxel downsample (200K -> 20K pts) | < 80ms | `scripts/benchmark.py --stage=voxel` |
| BEV intensity projection | < 120ms | `scripts/benchmark.py --stage=bev` |
| DBSCAN extraction (20K ground pts) | < 300ms | `scripts/benchmark.py --stage=extract` |
| U-Net inference (512×512, CPU) | < 1800ms | `scripts/benchmark.py --stage=unet` |
| Full pipeline (30 frames) | < 8 min | `scripts/benchmark.py --stage=full` |
| Visualization: 500K pts render | ≥ 30fps | Chrome DevTools Performance |
| Visualization: color mode switch | < 50ms | `performance.now()` in browser |

Benchmarks are re-run and committed to `docs/benchmarks.md` for each milestone tag.

---

## 8. Testing Strategy

Tests live in `tests/` and mirror the `src/` structure. The test suite must run without downloaded datasets - all tests use synthetic fixtures or small embedded test data.

**Test tiers:**

| Tier | What | How | Pass criteria |
|---|---|---|---|
| Unit | Individual functions | pytest, synthetic data | All assertions pass |
| Integration | Stage-to-stage data flow | pytest, fixtures | Output schema matches contract |
| End-to-end | Full pipeline on 5 KITTI frames | pytest, requires dataset | No crash, QA metrics logged |
| Performance | Runtime targets above | `scripts/benchmark.py` | All targets met on reference hardware |
| Visual | Three.js renders correctly | Manual + screenshot | FPS ≥ 30, no Z-fighting, correct colors |

**Synthetic test fixtures (`tests/fixtures/`):**

- `synthetic_ground_plane.npz`: 50,000 points on a flat plane + 5,000 noise points above. Known inlier ratio: 0.909. Used for RANSAC tests.
- `synthetic_lane_markings.npz`: two parallel 50m polylines of high-intensity points + low-intensity background. Used for extraction tests.
- `synthetic_se3_pairs.npz`: 100 random SE3 transforms with their inverses. Used for round-trip tests.

---

## 9. Containerization

The pipeline runs in Docker to ensure reproducible execution across machines.

```yaml
# docker/docker-compose.yml

services:
  pipeline:
    build:
      context: ..
      dockerfile: docker/Dockerfile.pipeline
    volumes:
      - ../data:/app/data
      - ../configs:/app/configs
    command: >
      python scripts/run_pipeline.py
        --config configs/kitti.yaml
        --scene data/raw/kitti/2011_09_26_drive_0005_sync
        --output data/outputs

  viz:
    build:
      context: ../src/viz
      dockerfile: ../../docker/Dockerfile.viz
    ports:
      - "5173:5173"
    volumes:
      - ../data/outputs:/app/public/data
    command: npm run dev -- --host
```

---

## 10. Repository and Git Standards

### 10.1 Repository Setup

This repository is public-facing and intended for resume and interview use. Engineers who clone it must be able to understand the architecture and run the pipeline from the README alone, without access to `prd.md` or `claude.md`.

```bash
git init hd-map-pipeline
cd hd-map-pipeline
git remote add origin git@github.com:<username>/hd-map-pipeline.git
```

### 10.2 What to Commit

```
COMMIT:
  src/            all source code
  tests/          all tests including fixtures
  configs/        all configuration files
  scripts/        all pipeline runner scripts
  docker/         all Docker and Compose files
  docs/           all documentation including benchmarks
  data/outputs/   generated GeoJSON and QA JSON artifacts (small, useful for demo)
  .gitignore      see below
  requirements.txt
  requirements-dev.txt
  setup.py
  README.md

DO NOT COMMIT:
  data/raw/       too large; instructions in README
  data/processed/ reproducible from raw; instructions in README
  data/models/    if > 50MB; link in README instead
  __pycache__/    Python bytecode
  *.pyc
  .env            any secrets or credentials
  node_modules/
  dist/           build artifacts
  .DS_Store
```

`prd.md` and `claude.md` are kept locally during development. They are not committed and not in `.gitignore`. They are working documents for the engineer building the project. The repository must stand on its own documentation without them.

### 10.3 Commit Hygiene

```text
Commit message format:
  short description

Examples:
  Add RANSAC ground plane separation with seed filtering
  Rplace Python loop with C++ extension, 40ms->8ms
  fix(transforms): SE3.inverse() was transposing wrong matrix dimension
  test(bev): add round-trip projection test for edge pixels
  docs(benchmarks): add M1 Pro timing results for all pipeline stages

Anti-patterns (do not do these):
  "fix stuff"
  "wip"
  "updates"
  "final version"
  A single commit containing 5 pipeline stages built all at once
```

### 10.4 Branch Strategy

```text
main          - always passes all tests; tagged at milestones
dev           - active development branch
feature/<name> - specific feature branches merged to dev

Milestone tags:
  v0.1  ingest + coordinate frames working, tests passing
  v0.2  ground plane separation + voxel filter complete
  v0.3  geometric feature extraction producing GeoJSON
  v0.4  U-Net training complete, cross-dataset evaluation logged
  v0.5  QA module working with real GT annotations
  v1.0  visualization complete, full pipeline demo-ready
```

### 10.5 README Requirements

The README must contain:
- One-paragraph project description
- One or two screenshots of the 3D viewer with real data
- Quick start: exact commands to download data, preprocess, run pipeline, launch viewer
- Architecture diagram (copy from Section 4.2 above, simplified, USE MERMAID)
- Dataset download instructions with exact URLs and subset names
- Benchmark table (copy from docs/benchmarks.md summary)
- Known limitations section (RANSAC on banked roads, CPU-only U-Net)

The README is what a NVIDIA engineer will read before your interview. It should take 3 minutes to understand what you built.

---

## 11. Design Aesthetic and UX Philosophy

The visualization is modeled after internal AV tooling - not a consumer dashboard, not a data science notebook widget, not a startup product demo.

**Visual principles:**
- Dark background (`#0d1117` or similar)
- Monospace font for all numeric labels and coordinates
- Point cloud colors: grayscale for intensity, jet colormap for height
- Feature overlays: thin lines, class-specific colors, no fills
- QA annotations: false positives in amber (`#f59e0b`), missed GT in red (`#ef4444`)
- No animation except camera movement
- No shadows, no bloom effects, no post-processing
- Control panel: left sidebar, fixed width, plain labels with keyboard shortcuts

**What the viewer does not do:**
- No hover tooltips that obscure the point cloud
- No animated transitions between views
- No loading spinners longer than 100ms
- No marketing copy or explanatory text in the viewport
- No mobile layout (this is an engineering tool, desktop only)

**What the viewer prioritizes:**
- Immediate point cloud rendering on file load
- Layer toggle that is instant (< 50ms)
- Keyboard shortcut for BEV/perspective toggle (`V`)
- Click on feature to show per-feature QA metrics in sidebar
- Stable camera between layer toggles

---

## 12. Definition of Done

### 4-Week MVP (minimum for resume inclusion)

```text
□ KITTI LiDAR frames ingest and transform to ENU world frame
□ SE3 coordinate frame round-trip test passing
□ RANSAC ground plane separation with documented failure modes
□ BEV intensity projection with per-pixel test
□ Basic Three.js viewer showing point cloud with height coloring
□ One KITTI scene processed end-to-end without crash
□ README with dataset setup instructions and one screenshot
```

### 8-Week Full Version (target)

```text
□ Full five-stage pipeline producing features.geojson
□ U-Net trained on nuScenes, evaluated on KITTI, metrics logged
□ QA module with completeness/accuracy/FP metrics vs. GT
□ Three.js viewer: point cloud + feature overlays + QA annotation
□ Layer toggle, color mode switch, BEV/perspective mode
□ Docker Compose: full pipeline in one command
□ All unit and integration tests passing
□ docs/benchmarks.md with timing for all stages
□ README with screenshot, quickstart, architecture overview
□ Milestone tags v0.1 through v1.0 on main branch
```

### Elite Additions (time permitting)

```text
□ Map change detection: diff two pipeline runs of the same road
□ Temporal consistency: enforce lane boundary continuity across frames
□ OpenDRIVE export: standard HD map interchange format
□ Uncertainty visualization: per-feature confidence coloring
□ C++ pybind11 voxel filter with benchmarked speedup vs. Python
```

---

## 13. Resume Presentation

**Primary bullet (SWE/AV targeting):**
```text
Built a five-stage HD map feature extraction pipeline over KITTI LiDAR 
point clouds: SE3 coordinate frame transforms (vehicle/sensor/world), 
RANSAC ground plane separation with seed-set pre-filtering, BEV 
intensity projection for road marking detection, DBSCAN polyline 
extraction, and a U-Net segmentation model trained on nuScenes 
annotations - cross-dataset evaluated on KITTI with completeness, 
positional accuracy, and false positive rate QA metrics
```

**Secondary bullet (visualization):**
```text
Built a Three.js 3D map inspection tool rendering 500K+ point cloud 
frames at ≥30fps via BufferGeometry typed arrays, with GeoJSON feature 
overlays, per-feature QA annotation (missed GT / false positives), 
intensity/height color modes, and BEV/perspective toggle - designed 
to mirror map QA engineer workflows
```
