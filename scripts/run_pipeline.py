#!/usr/bin/env python3
"""Command-line entry point for pipeline stages."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.types import LaneBoundaryFeature, LaneType
from src.data.kitti import load_lidar_frame, parse_calibration, parse_oxts_pose
from src.pipeline.bev import project_to_bev, BEVConfig
from src.pipeline.extract import ExtractionConfig, extract_lane_boundaries
from src.pipeline.ingest import accumulate_kitti_frames, _origin_from_oxts
from src.pipeline.qa import QAConfig, compute_qa_metrics
import pandas as pd
import zlib

from src.filters.ground_plane import ransac_ground_plane, RansacConfig
from src.filters.voxel import voxel_downsample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HD map pipeline stages.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--stage", required=True, choices=("ingest", "full"))
    parser.add_argument("--scene", default=None, help="Override KITTI scene path.")
    parser.add_argument("--calib", default=None, help="Override KITTI calib path.")
    parser.add_argument("--n_frames", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        stage=args.stage,
        output_dir=args.output,
        scene=args.scene,
        calib=args.calib,
        n_frames=args.n_frames,
    )


def run_pipeline(
    *,
    config_path: str | Path,
    stage: str,
    output_dir: str | Path,
    scene: str | Path | None = None,
    calib: str | Path | None = None,
    n_frames: int | None = None,
) -> None:
    """Run one pipeline stage and write file outputs.

    Args:
        config_path: YAML config path.
        stage: ``ingest`` or ``full``.
        output_dir: Directory for stage artifacts. FRAME: full-stage feature
            outputs are world ENU GeoJSON coordinates.
        scene: Optional KITTI scene override for ingest.
        calib: Optional KITTI calibration override for ingest.
        n_frames: Optional frame count override for ingest.
    """
    config = _load_yaml(Path(config_path))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if stage == "ingest":
        dataset = config.get("dataset", {})
        scene_path = Path(scene or dataset["scene"])
        calib_path = Path(calib or dataset["calib"])
        frame_count = n_frames or int(config["pipeline"]["n_frames_accumulate"])
        result = accumulate_kitti_frames(
            scene_dir=scene_path,
            calib_dir=calib_path,
            output_path=output / "accumulated.parquet",
            n_frames=frame_count,
        )
        print(f"Wrote {result.output_path}")
        print(f"Cumulative point counts: {result.per_frame_point_counts}")
        return

    if stage == "full":
        if config.get("dataset"):
            _run_full_kitti(config, output)
        else:
            _run_full_smoke(config, output)
        return

    raise ValueError(f"Unsupported pipeline stage: {stage}")


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

    # Stage 6: QA — no external GT for KITTI scene 0005
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

    _generate_kitti_frames(config, output)


def _run_full_smoke(config: dict[str, Any], output: Path) -> None:
    points = _synthetic_lane_points()
    extraction_cfg = ExtractionConfig(**config["extraction"])
    features = extract_lane_boundaries(points, extraction_cfg)
    ground_truth = [
        LaneBoundaryFeature(
            geometry=[[0.0, -1.5, 0.0], [50.0, -1.5, 0.0]],
            feature_type=LaneType.LANE_LINE,
            confidence=1.0,
            source="synthetic_gt",
        ),
        LaneBoundaryFeature(
            geometry=[[0.0, 1.5, 0.0], [50.0, 1.5, 0.0]],
            feature_type=LaneType.LANE_LINE,
            confidence=1.0,
            source="synthetic_gt",
        ),
    ]
    report = compute_qa_metrics(
        features,
        ground_truth,
        QAConfig(max_gt_match_distance=config["qa"]["max_gt_match_distance"]),
        scene_id="synthetic_full_smoke",
    )
    _write_geojson(output / "features.geojson", features)
    _write_points_bin(output / "points.bin", points)
    (output / "qa_report.json").write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output / 'features.geojson'}")
    print(f"Wrote {output / 'points.bin'}")
    print(f"Wrote {output / 'qa_report.json'}")

    _generate_realistic_frames(output)


def _generate_kitti_frames(config: dict[str, Any], output: Path) -> None:
    """Write per-frame viewer binaries from real KITTI Velodyne data.

    Transforms each raw frame from LiDAR frame → vehicle frame → world ENU,
    then splits into ground (z < 0.5 m) and obstacle layers.

    Output (world ENU, x=east, y=north, z=up):
        frames/frame_{N}_raw.bin       – all returns for frame N
        frames/frame_{N}_ground.bin    – returns with z < 0.5 m
        frames/frame_{N}_obstacles.bin – returns with z >= 0.5 m
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
    lidar_files = sorted((scene_path / "velodyne_points" / "data").glob("*.bin"))[:n_frames]
    oxts_files = sorted((scene_path / "oxts" / "data").glob("*.txt"))[:n_frames]

    t_vehicle_lidar = parse_calibration(calib_path)["T_vehicle_lidar"]
    origin = _origin_from_oxts(oxts_files[0])

    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, (lidar_path, oxts_path) in enumerate(zip(lidar_files, oxts_files)):
        lidar_pts = load_lidar_frame(lidar_path)            # (N,4) LiDAR frame
        vehicle_xyz = t_vehicle_lidar.transform_points(lidar_pts[:, :3])
        t_world_vehicle = parse_oxts_pose(oxts_path, origin=origin)
        world_xyz = t_world_vehicle.transform_points(vehicle_xyz).astype(np.float32)

        # Per-scan intensity normalisation: map sensor raw values → [0, 1]
        intensities = lidar_pts[:, 3].astype(np.float32)
        max_i = intensities.max()
        if max_i > 0:
            intensities = intensities / max_i

        raw = np.column_stack([world_xyz, intensities])

        # Ground split: z < 0.5 m captures road surface, curbs, low markings (~88%)
        gnd_mask = raw[:, 2] < 0.5
        _write_points_bin(frames_dir / f"frame_{frame_idx}_raw.bin", raw)
        _write_points_bin(frames_dir / f"frame_{frame_idx}_ground.bin", raw[gnd_mask])
        _write_points_bin(frames_dir / f"frame_{frame_idx}_obstacles.bin", raw[~gnd_mask])
        print(
            f"[kitti frames] frame {frame_idx}: {len(raw):,} pts "
            f"({gnd_mask.sum():,} ground, {(~gnd_mask).sum():,} obstacles)"
        )

    _generate_bev_images(output, n_frames)


def _generate_bev_images(output: Path, n_frames: int = 5) -> None:
    """Render BEV intensity PNGs from ground-layer point clouds.

    Applies the same turbo colormap used by the 3D viewer so colours are
    consistent across the two views.  Background pixels (no return) are
    rendered as near-black so the road surface stands out.

    Output: data/outputs/bev/frame_{N}_bev.png  (1600×1600, 5 cm/px)
    """
    _STOPS = np.array([
        [0.00, 0.05, 0.05, 0.55],
        [0.40, 0.00, 0.85, 0.90],
        [0.70, 1.00, 0.88, 0.00],
        [1.00, 1.00, 0.08, 0.08],
    ], dtype=np.float32)

    def _turbo(img: np.ndarray) -> np.ndarray:
        flat = img.ravel()
        rgb = np.zeros((len(flat), 3), dtype=np.float32)
        for i in range(1, len(_STOPS)):
            p, n_ = _STOPS[i - 1], _STOPS[i]
            mask = (flat >= p[0]) & (flat <= n_[0])
            if not mask.any():
                continue
            t = (flat[mask] - p[0]) / (n_[0] - p[0])
            for c in range(3):
                rgb[mask, c] = p[c + 1] + t * (n_[c + 1] - p[c + 1])
        rgb[flat == 0] = [0.04, 0.04, 0.12]
        return (rgb.reshape(*img.shape, 3) * 255).clip(0, 255).astype(np.uint8)

    def _write_png(path: Path, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
        def _chunk(tag: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        with path.open("wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
            fh.write(_chunk(b"IHDR", ihdr))
            fh.write(_chunk(b"IDAT", zlib.compress(raw, 6)))
            fh.write(_chunk(b"IEND", b""))

    bev_dir = output / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    cfg = BEVConfig(resolution=0.05, extent=40.0)

    for frame_idx in range(n_frames):
        gnd_path = output / "frames" / f"frame_{frame_idx}_ground.bin"
        with gnd_path.open("rb") as fh:
            data = fh.read()
        n_pts = struct.unpack_from("<I", data)[0]
        xyz = np.frombuffer(data, dtype=np.float32, count=n_pts * 3, offset=4).reshape(n_pts, 3)
        intensities = np.frombuffer(data, dtype=np.float32, count=n_pts, offset=4 + n_pts * 12)
        pts = np.column_stack([xyz, intensities])

        bev = project_to_bev(pts, cfg)
        out_path = bev_dir / f"frame_{frame_idx}_bev.png"
        _write_png(out_path, _turbo(bev.image))
        print(f"[bev]    frame {frame_idx}: {out_path} ({out_path.stat().st_size // 1024}KB)")


def _generate_realistic_frames(output: Path, n_frames: int = 5) -> None:
    """Generate per-frame binary point clouds with realistic Velodyne HDL-64E geometry.

    Scene: flat road (z=0), building walls at |y|=10 m, lane markings at y=±1.5 m
    with high reflectance.  Vehicle advances 10 m per frame along world-ENU x-axis;
    LiDAR is at height z=1.73 m.

    Writes (world ENU, x=east, y=north, z=up):
        frames/frame_{N}_raw.bin       – all scan returns
        frames/frame_{N}_ground.bin    – returns with z < 0.15 m
        frames/frame_{N}_obstacles.bin – returns with z >= 0.15 m
    """
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # HDL-64E: 64 rings from -24.8° to +2.0°, 1800 azimuth steps (0.2° resolution)
    elevations = np.linspace(np.radians(-24.8), np.radians(2.0), 64, dtype=np.float32)
    azimuths = np.linspace(0.0, 2.0 * np.pi, 1800, endpoint=False, dtype=np.float32)

    cos_el = np.cos(elevations)[:, np.newaxis]  # (64, 1)
    sin_el = np.sin(elevations)[:, np.newaxis]  # (64, 1)
    cos_az = np.cos(azimuths)[np.newaxis, :]    # (1, 1800)
    sin_az = np.sin(azimuths)[np.newaxis, :]    # (1, 1800)

    dx = (cos_el * cos_az).astype(np.float32)           # (64, 1800)
    dy = (cos_el * sin_az).astype(np.float32)           # (64, 1800)
    dz = np.broadcast_to(sin_el, (64, 1800)).copy().astype(np.float32)  # (64, 1800)

    lidar_z = 1.73
    far_clip = 60.0

    for frame_idx in range(n_frames):
        origin_x = float(frame_idx) * 10.0

        t = np.full((64, 1800), far_clip, dtype=np.float32)

        # Ground plane z=0: t = -lidar_z / dz (valid when dz < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_gnd = np.where(dz < -1e-6, -lidar_z / dz, far_clip)
        t = np.where((t_gnd > 0.1) & (t_gnd < far_clip), np.minimum(t, t_gnd), t)

        # Wall y=+10 m
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wp = np.where(dy > 1e-6, 10.0 / dy, far_clip)
        t = np.where((t_wp > 0.1) & (t_wp < far_clip), np.minimum(t, t_wp), t)

        # Wall y=-10 m
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wn = np.where(dy < -1e-6, -10.0 / dy, far_clip)
        t = np.where((t_wn > 0.1) & (t_wn < far_clip), np.minimum(t, t_wn), t)

        hit_x = (origin_x + t * dx).astype(np.float32)
        hit_y = (t * dy).astype(np.float32)
        hit_z = (lidar_z + t * dz).astype(np.float32)

        valid = t < (far_clip - 0.5)

        on_ground = valid & (hit_z < 0.3)
        on_lane = on_ground & (
            (np.abs(hit_y - 1.5) < 0.15) | (np.abs(hit_y + 1.5) < 0.15)
        )
        on_road = on_ground & ~on_lane
        on_wall = valid & ~on_ground

        intensity = np.zeros((64, 1800), dtype=np.float32)
        n_lane = int(on_lane.sum())
        n_road = int(on_road.sum())
        n_wall = int(on_wall.sum())
        if n_lane:
            intensity[on_lane] = (0.85 + rng.uniform(0, 0.12, n_lane)).astype(np.float32)
        if n_road:
            intensity[on_road] = (0.15 + rng.uniform(0, 0.18, n_road)).astype(np.float32)
        if n_wall:
            intensity[on_wall] = (0.25 + rng.uniform(0, 0.20, n_wall)).astype(np.float32)

        vf = valid.ravel()
        raw = np.column_stack([
            hit_x.ravel()[vf],
            hit_y.ravel()[vf],
            hit_z.ravel()[vf],
            intensity.ravel()[vf],
        ]).astype(np.float32)

        gnd_mask = raw[:, 2] < 0.15
        _write_points_bin(frames_dir / f"frame_{frame_idx}_raw.bin", raw)
        _write_points_bin(frames_dir / f"frame_{frame_idx}_ground.bin", raw[gnd_mask])
        _write_points_bin(frames_dir / f"frame_{frame_idx}_obstacles.bin", raw[~gnd_mask])
        print(
            f"[frames] frame {frame_idx}: {len(raw):,} pts "
            f"({gnd_mask.sum():,} ground, {(~gnd_mask).sum():,} obstacles)"
        )


def _write_points_bin(path: Path, points: np.ndarray) -> None:
    """Write xyz+intensity point cloud as a packed binary blob.

    Format (little-endian):
        bytes 0-3       uint32  N (point count)
        bytes 4..4+N*12 float32 xyz triples (N*3 values). FRAME: world ENU.
        bytes 4+N*12..  float32 intensities (N values, range [0, 1]).
    """
    arr = np.asarray(points, dtype=np.float32)
    n = len(arr)
    xyz = np.ascontiguousarray(arr[:, :3])
    intensity = np.ascontiguousarray(arr[:, 3])
    with path.open("wb") as fh:
        fh.write(struct.pack("<I", n))
        fh.write(xyz.tobytes())
        fh.write(intensity.tobytes())


def _synthetic_lane_points() -> np.ndarray:
    line_x = np.linspace(0.0, 50.0, 500, dtype=np.float32)
    line_z = np.zeros(500, dtype=np.float32)
    intensity = np.ones(500, dtype=np.float32)
    line1 = np.column_stack(
        [line_x, np.full(500, -1.5, dtype=np.float32), line_z, intensity]
    )
    line2 = np.column_stack(
        [line_x, np.full(500, 1.5, dtype=np.float32), line_z, intensity]
    )
    background = np.random.default_rng(42).uniform(0.0, 0.3, (5_000, 4)).astype(
        np.float32
    )
    background[:, 2] = 0.0
    return np.vstack([line1, line2, background]).astype(np.float32)


def _write_geojson(path: Path, features: list[LaneBoundaryFeature]) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "feature_type": feature.feature_type.value,
                    "confidence": feature.confidence,
                    "source": feature.source,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": feature.geometry,
                },
            }
            for feature in features
        ],
    }
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_path = config.pop("base", None)
    if base_path is None:
        return config

    base = _load_yaml(Path(base_path))
    return _deep_merge(base, config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
