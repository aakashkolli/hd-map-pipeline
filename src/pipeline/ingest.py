"""KITTI frame ingestion and local map accumulation.

Input schema:
    KITTI scene directory with ``velodyne_points/data/*.bin`` LiDAR frames
    and ``oxts/data/*.txt`` pose packets. Calibration directory contains a
    Velodyne calibration file parsed into an SE3 transform.

Output schema:
    Parquet file with columns ``x, y, z, intensity, timestamp, frame_id``.
    ``x, y, z, intensity`` are float32. ``frame_id`` is integer.

Coordinate frames:
    LiDAR input points are transformed from LiDAR frame to vehicle frame
    with calibration, then from vehicle frame to world ENU with OXTS pose.
    The accumulated parquet output is always world ENU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.kitti import (
    ALTITUDE_INDEX,
    LATITUDE_INDEX,
    LONGITUDE_INDEX,
    OxtsOrigin,
    load_lidar_frame,
    parse_calibration,
    parse_oxts_pose,
)


POINT_COLUMNS = ("x", "y", "z")
INTENSITY_COLUMN = "intensity"
TIMESTAMP_COLUMN = "timestamp"
FRAME_ID_COLUMN = "frame_id"
PARQUET_COLUMNS = (
    "x",
    "y",
    "z",
    "intensity",
    "timestamp",
    "frame_id",
)


@dataclass(frozen=True)
class IngestResult:
    """Summary of an accumulated KITTI ingest run."""

    output_path: Path
    per_frame_point_counts: list[int]


def accumulate_kitti_frames(
    *,
    scene_dir: str | Path,
    calib_dir: str | Path,
    output_path: str | Path,
    n_frames: int,
) -> IngestResult:
    """Accumulate KITTI LiDAR frames into a local world-frame map segment.

    Args:
        scene_dir: KITTI raw drive directory containing LiDAR and OXTS data.
            FRAME: LiDAR files contain raw LiDAR-frame points.
        calib_dir: KITTI calibration directory. FRAME: calibration maps
            LiDAR frame to vehicle frame.
        output_path: Destination parquet path.
            FRAME: output points are world ENU.
        n_frames: Number of sorted frames to accumulate.

    Returns:
        IngestResult containing cumulative point counts after each frame.
            FRAME: written parquet is world ENU.
    """
    scene_path = Path(scene_dir)
    output = Path(output_path)
    lidar_files = _frame_files(scene_path / "velodyne_points" / "data", "*.bin")
    oxts_files = _frame_files(scene_path / "oxts" / "data", "*.txt")

    if n_frames > len(lidar_files):
        raise ValueError(
            f"Requested {n_frames} LiDAR frames, but only "
            f"{len(lidar_files)} exist in {scene_path}."
        )
    if n_frames > len(oxts_files):
        raise ValueError(
            f"Requested {n_frames} OXTS poses, but only "
            f"{len(oxts_files)} exist in {scene_path}."
        )

    transforms = parse_calibration(calib_dir)
    t_vehicle_lidar = transforms["T_vehicle_lidar"]
    origin = _origin_from_oxts(oxts_files[0])
    frames: list[pd.DataFrame] = []
    cumulative_counts: list[int] = []
    total_points = 0

    for frame_id, (lidar_path, oxts_path) in enumerate(
        zip(lidar_files[:n_frames], oxts_files[:n_frames], strict=True)
    ):
        lidar_points = load_lidar_frame(lidar_path)
        vehicle_points = t_vehicle_lidar.transform_points(lidar_points[:, :3])
        t_world_vehicle = parse_oxts_pose(oxts_path, origin=origin)
        world_points = t_world_vehicle.transform_points(vehicle_points).astype(
            np.float32
        )

        frame = pd.DataFrame(
            {
                POINT_COLUMNS[0]: world_points[:, 0],
                POINT_COLUMNS[1]: world_points[:, 1],
                POINT_COLUMNS[2]: world_points[:, 2],
                INTENSITY_COLUMN: lidar_points[:, 3].astype(np.float32),
                TIMESTAMP_COLUMN: np.full(
                    lidar_points.shape[0],
                    frame_id,
                    dtype=np.int64,
                ),
                FRAME_ID_COLUMN: np.full(
                    lidar_points.shape[0],
                    frame_id,
                    dtype=np.int64,
                ),
            },
            columns=PARQUET_COLUMNS,
        )
        frames.append(frame)
        total_points += int(lidar_points.shape[0])
        cumulative_counts.append(total_points)

    accumulated = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    accumulated.to_parquet(output, index=False)

    return IngestResult(
        output_path=output,
        per_frame_point_counts=cumulative_counts,
    )


def _frame_files(directory: Path, pattern: str) -> list[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}.")
    return files


def _origin_from_oxts(path: Path) -> OxtsOrigin:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    required_fields = ALTITUDE_INDEX + 1
    if values.size < required_fields:
        raise ValueError(
            f"OXTS packet {path} has {values.size} fields; expected at "
            f"least {required_fields}."
        )

    return OxtsOrigin(
        latitude=float(values[LATITUDE_INDEX]),
        longitude=float(values[LONGITUDE_INDEX]),
        altitude=float(values[ALTITUDE_INDEX]),
    )
