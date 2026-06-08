"""Lightweight nuScenes-mini LiDAR and map annotation parsing.

Input schema:
    LiDAR samples are binary float32 files with columns
    ``x, y, z, intensity, ring``. Map annotations are JSON files with a
    ``lane_dividers`` list containing ``id`` and ``geometry`` fields.

Output schema:
    LiDAR arrays are ``(N, 5)`` float32. Lane annotations are
    ``LaneBoundaryFeature`` objects with xyz geometry.

Coordinate frames:
    LiDAR points are sensor-frame coordinates. Map annotation geometries are
    local world-frame coordinates and remain world-frame throughout parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.types import LaneBoundaryFeature, LaneType


LIDAR_COLUMNS = 5
POINT_DIM = 3


@dataclass(frozen=True)
class NuScenesScene:
    """Loaded nuScenes mini-scene data."""

    scene_id: str
    lidar_points: np.ndarray
    annotations: list[LaneBoundaryFeature]


def load_nuscenes_lidar(path: str | Path) -> np.ndarray:
    """Load a nuScenes LiDAR sample.

    Args:
        path: Binary float32 LiDAR file.
            FRAME: nuScenes sensor frame.

    Returns:
        ``(N, 5)`` float32 array ``x, y, z, intensity, ring``.
            FRAME: nuScenes sensor frame.
    """
    lidar_path = Path(path)
    raw = np.fromfile(lidar_path, dtype=np.float32)
    if raw.size % LIDAR_COLUMNS != 0:
        raise ValueError(
            f"nuScenes LiDAR file {lidar_path} has {raw.size} float32 values, "
            f"not divisible by {LIDAR_COLUMNS} columns."
        )
    return raw.reshape((-1, LIDAR_COLUMNS))


def load_map_annotations(path: str | Path) -> list[LaneBoundaryFeature]:
    """Load lane divider annotations from a mini-scene JSON file.

    Args:
        path: JSON file with ``lane_dividers`` geometry records.
            FRAME: annotation geometry is local world frame.

    Returns:
        LaneBoundaryFeature objects. FRAME: world coordinates.
    """
    annotation_path = Path(path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    lane_dividers = payload.get("lane_dividers", [])
    if not isinstance(lane_dividers, list):
        raise ValueError(f"{annotation_path} lane_dividers must be a list.")

    annotations: list[LaneBoundaryFeature] = []
    for record in lane_dividers:
        geometry = _parse_geometry(record, annotation_path)
        annotations.append(
            LaneBoundaryFeature(
                geometry=geometry.astype(np.float32).tolist(),
                feature_type=LaneType.LANE_LINE,
                confidence=1.0,
                point_count=int(geometry.shape[0]),
                source="nuscenes_map",
            )
        )
    return annotations


def load_nuscenes_scene(root: str | Path, *, scene_id: str) -> NuScenesScene:
    """Load one lightweight nuScenes-mini scene.

    Args:
        root: nuScenes-mini root directory.
            FRAME: LiDAR samples are sensor frame; annotations are world.
        scene_id: Scene token/name used for file lookup.

    Returns:
        NuScenesScene with LiDAR sensor-frame points and world-frame
        annotations. FRAME: mixed by field as documented.
    """
    root_path = Path(root)
    lidar = load_nuscenes_lidar(
        root_path / "samples" / "LIDAR_TOP" / f"{scene_id}.bin"
    )
    annotations = load_map_annotations(
        root_path / "maps" / f"{scene_id}_annotations.json"
    )
    return NuScenesScene(
        scene_id=scene_id,
        lidar_points=lidar,
        annotations=annotations,
    )


def annotation_intersects_extent(
    annotation: LaneBoundaryFeature,
    *,
    extent: float,
) -> bool:
    """Return whether an annotation intersects a centered BEV extent.

    Args:
        annotation: LaneBoundaryFeature geometry. FRAME: local world.
        extent: Half-width of the centered BEV region in meters.

    Returns:
        True when any annotation vertex lies inside the extent.
            FRAME: local world.
    """
    geometry = np.asarray(annotation.geometry, dtype=np.float32)
    if geometry.ndim != 2 or geometry.shape[1] != POINT_DIM:
        raise ValueError(
            f"Expected annotation geometry shape (N, {POINT_DIM}), got "
            f"{geometry.shape}."
        )
    inside_x = (geometry[:, 0] >= -extent) & (geometry[:, 0] <= extent)
    inside_y = (geometry[:, 1] >= -extent) & (geometry[:, 1] <= extent)
    return bool(np.any(inside_x & inside_y))


def _parse_geometry(record: Any, path: Path) -> np.ndarray:
    if not isinstance(record, dict):
        raise ValueError(f"Annotation record in {path} must be an object.")
    geometry = np.asarray(record.get("geometry"), dtype=np.float32)
    if geometry.ndim != 2 or geometry.shape[1] != POINT_DIM:
        raise ValueError(
            f"Annotation geometry in {path} must have shape (N, {POINT_DIM}), "
            f"got {geometry.shape}."
        )
    return geometry
