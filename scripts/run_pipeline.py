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
from src.pipeline.extract import ExtractionConfig, extract_lane_boundaries
from src.pipeline.ingest import accumulate_kitti_frames
from src.pipeline.qa import QAConfig, compute_qa_metrics


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
        _run_full_smoke(config, output)
        return

    raise ValueError(f"Unsupported pipeline stage: {stage}")


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
