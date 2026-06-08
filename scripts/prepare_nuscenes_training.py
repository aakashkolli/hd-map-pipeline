#!/usr/bin/env python3
"""Prepare BEV image and mask pairs from lightweight nuScenes-mini scenes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import load_config
from src.data.nuscenes import load_map_annotations, load_nuscenes_lidar
from src.pipeline.bev import BEVConfig, project_to_bev


MASK_LANE_VALUE = 1
LIDAR_XYZI_COLUMNS = 4


@dataclass(frozen=True)
class TrainingPrepResult:
    """Generated BEV training-pair paths."""

    image_paths: list[Path]
    mask_paths: list[Path]
    pair_count: int


def prepare_nuscenes_training(
    *,
    nuscenes_root: str | Path,
    output_dir: str | Path,
    bev_config: BEVConfig,
) -> TrainingPrepResult:
    """Generate aligned BEV intensity images and annotation masks.

    Args:
        nuscenes_root: nuScenes-mini root. FRAME: LiDAR and map annotations
            are interpreted in the same local world frame for preparation.
        output_dir: Directory where ``bev_images`` and ``bev_labels`` are
            written.
        bev_config: BEV origin and resolution parameters.

    Returns:
        TrainingPrepResult with written image/mask paths.
            FRAME: image and mask share the same BEV world-frame metadata.
    """
    root = Path(nuscenes_root)
    output = Path(output_dir)
    image_dir = output / "bev_images"
    mask_dir = output / "bev_labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    lidar_files = sorted((root / "samples" / "LIDAR_TOP").glob("*.bin"))
    for lidar_path in lidar_files:
        scene_id = lidar_path.stem
        lidar = load_nuscenes_lidar(lidar_path)
        ground_points = lidar[:, :LIDAR_XYZI_COLUMNS].astype(np.float32)
        bev = project_to_bev(ground_points, bev_config)
        annotations = load_map_annotations(
            root / "maps" / f"{scene_id}_annotations.json"
        )
        mask = _rasterize_annotations(
            annotations=annotations,
            image_shape=bev.image.shape,
            bev_config=bev_config,
        )

        image_path = image_dir / f"{scene_id}.npy"
        mask_path = mask_dir / f"{scene_id}.npy"
        np.save(image_path, bev.image)
        np.save(mask_path, mask)
        image_paths.append(image_path)
        mask_paths.append(mask_path)

    return TrainingPrepResult(
        image_paths=image_paths,
        mask_paths=mask_paths,
        pair_count=len(image_paths),
    )


def _rasterize_annotations(
    *,
    annotations,
    image_shape: tuple[int, int],
    bev_config: BEVConfig,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    if not annotations:
        return mask

    geometries = [
        np.asarray(annotation.geometry, dtype=np.float32)
        for annotation in annotations
        if annotation.geometry
    ]
    if not geometries:
        return mask

    vertices = np.vstack(geometries)
    pixel_x = np.floor((vertices[:, 0] + bev_config.extent) / bev_config.resolution)
    pixel_y = np.floor((vertices[:, 1] + bev_config.extent) / bev_config.resolution)
    pixel_x = pixel_x.astype(np.int64)
    pixel_y = pixel_y.astype(np.int64)
    height, width = image_shape
    valid = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    mask[pixel_y[valid], pixel_x[valid]] = MASK_LANE_VALUE
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = prepare_nuscenes_training(
        nuscenes_root=args.nuscenes,
        output_dir=args.output,
        bev_config=BEVConfig(
            resolution=cfg.bev.resolution,
            extent=cfg.bev.extent,
        ),
    )
    print(f"Generated {result.pair_count} BEV training pairs")


if __name__ == "__main__":
    main()
