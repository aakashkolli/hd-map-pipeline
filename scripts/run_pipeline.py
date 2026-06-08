#!/usr/bin/env python3
"""Command-line entry point for pipeline stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.ingest import accumulate_kitti_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HD map pipeline stages.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--stage", required=True, choices=("ingest",))
    parser.add_argument("--scene", default=None, help="Override KITTI scene path.")
    parser.add_argument("--calib", default=None, help="Override KITTI calib path.")
    parser.add_argument("--n_frames", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    dataset = config.get("dataset", {})
    scene = Path(args.scene or dataset["scene"])
    calib = Path(args.calib or dataset["calib"])
    n_frames = args.n_frames or int(config["pipeline"]["n_frames_accumulate"])
    output_dir = Path(args.output)

    if args.stage == "ingest":
        result = accumulate_kitti_frames(
            scene_dir=scene,
            calib_dir=calib,
            output_path=output_dir / "accumulated.parquet",
            n_frames=n_frames,
        )
        print(f"Wrote {result.output_path}")
        print(f"Cumulative point counts: {result.per_frame_point_counts}")


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
