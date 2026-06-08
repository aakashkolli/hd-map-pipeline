"""KITTI raw dataset parsers.

Input contract:
    LiDAR frames are KITTI Velodyne ``.bin`` files with float32 columns
    ``x, y, z, intensity``. OXTS pose files are one text row per frame.
    Calibration files contain rotation and translation records parsed
    from dataset files.

Output contract:
    LiDAR arrays are ``(N, 4)`` float32 without in-place mutation.
    Calibration and OXTS pose parsers return labeled ``SE3`` transforms.

Coordinate frames:
    LiDAR points use the KITTI Velodyne frame: x forward, y left, z up.
    OXTS poses return vehicle-to-world transforms where world is local
    ENU: x east, y north, z up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.geometry.transforms import SE3


LIDAR_COLUMNS = 4
ROTATION_VALUES = 9
TRANSLATION_VALUES = 3
LATITUDE_INDEX = 0
LONGITUDE_INDEX = 1
ALTITUDE_INDEX = 2
ROLL_INDEX = 3
PITCH_INDEX = 4
YAW_INDEX = 5
EARTH_RADIUS_METERS = 6_378_137.0


@dataclass(frozen=True)
class OxtsOrigin:
    """Geodetic origin for local ENU pose conversion."""

    latitude: float
    longitude: float
    altitude: float


def load_lidar_frame(path: str | Path) -> np.ndarray:
    """Read one KITTI Velodyne frame.

    Args:
        path: Path to a KITTI ``.bin`` file. FRAME: output points are
            LiDAR frame, x forward, y left, z up.

    Returns:
        ``(N, 4)`` float32 array ``x, y, z, intensity``.
            FRAME: LiDAR frame.
    """
    frame_path = Path(path)
    raw = np.fromfile(frame_path, dtype=np.float32)
    if raw.size % LIDAR_COLUMNS != 0:
        raise ValueError(
            f"KITTI LiDAR file {frame_path} has {raw.size} float32 values, "
            f"not divisible by {LIDAR_COLUMNS} columns."
        )

    return raw.reshape((-1, LIDAR_COLUMNS))


def parse_calibration_file(
    path: str | Path,
    *,
    source_frame: str,
    target_frame: str,
) -> SE3:
    """Parse a KITTI calibration file into an SE3 transform.

    Args:
        path: Text calibration file containing ``R:`` and ``T:`` records.
            FRAME: parsed transform maps ``source_frame`` to
            ``target_frame``.
        source_frame: Input coordinate frame label.
        target_frame: Output coordinate frame label.

    Returns:
        SE3 transform from ``source_frame`` to ``target_frame``.
    """
    calib_path = Path(path)
    records = _read_key_value_records(calib_path)

    if "R" not in records:
        raise ValueError(f"Calibration file {calib_path} is missing R record.")
    if "T" not in records:
        raise ValueError(f"Calibration file {calib_path} is missing T record.")

    rotation_values = _parse_float_record(records["R"], ROTATION_VALUES, "R")
    translation = _parse_float_record(records["T"], TRANSLATION_VALUES, "T")
    rotation = rotation_values.reshape((TRANSLATION_VALUES, TRANSLATION_VALUES))

    return SE3(
        rotation=rotation,
        translation=translation,
        source_frame=source_frame,
        target_frame=target_frame,
    )


def parse_oxts_pose(path: str | Path, *, origin: OxtsOrigin | None = None) -> SE3:
    """Parse one KITTI OXTS packet into a vehicle-to-world pose.

    Args:
        path: Text OXTS packet with latitude, longitude, altitude, roll,
            pitch, and yaw fields. FRAME: output transform maps vehicle
            frame to world ENU frame.
        origin: Geodetic origin for local ENU coordinates. If omitted,
            the packet position is used as the origin.

    Returns:
        SE3 transform from vehicle frame to world ENU frame.
            FRAME: vehicle to world (ENU).
    """
    values = np.fromstring(Path(path).read_text(encoding="utf-8"), sep=" ")
    required_fields = YAW_INDEX + 1
    if values.size < required_fields:
        raise ValueError(
            f"OXTS packet {path} has {values.size} fields; expected at "
            f"least {required_fields}."
        )

    latitude = float(values[LATITUDE_INDEX])
    longitude = float(values[LONGITUDE_INDEX])
    altitude = float(values[ALTITUDE_INDEX])
    roll = float(values[ROLL_INDEX])
    pitch = float(values[PITCH_INDEX])
    yaw = float(values[YAW_INDEX])

    enu_origin = origin or OxtsOrigin(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )
    translation = _geodetic_to_local_enu(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        origin=enu_origin,
    )
    rotation = _rotation_from_roll_pitch_yaw(roll=roll, pitch=pitch, yaw=yaw)

    return SE3(
        rotation=rotation,
        translation=translation,
        source_frame="vehicle",
        target_frame="world",
    )


def parse_calibration(calib_dir: str | Path) -> dict[str, SE3]:
    """Parse supported KITTI calibration transforms from a directory.

    Args:
        calib_dir: Directory containing KITTI calibration text files.
            FRAME: returned transforms preserve source and target labels.

    Returns:
        Dictionary containing ``T_vehicle_lidar`` when a Velodyne
        calibration file is present.
    """
    directory = Path(calib_dir)
    candidates = (
        directory / "calib_velo_to_vehicle.txt",
        directory / "calib_velo_to_cam.txt",
    )
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No supported Velodyne calibration file found in {directory}."
        )

    return {
        "T_vehicle_lidar": parse_calibration_file(
            existing[0],
            source_frame="lidar",
            target_frame="vehicle",
        )
    }


def _read_key_value_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            records[key.strip()] = value.strip()
    return records


def _parse_float_record(value: str, expected_size: int, key: str) -> np.ndarray:
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.size != expected_size:
        raise ValueError(
            f"Calibration record {key} has {parsed.size} values; expected "
            f"{expected_size}."
        )
    return parsed


def _geodetic_to_local_enu(
    *,
    latitude: float,
    longitude: float,
    altitude: float,
    origin: OxtsOrigin,
) -> np.ndarray:
    latitude_delta = np.deg2rad(latitude - origin.latitude)
    longitude_delta = np.deg2rad(longitude - origin.longitude)
    origin_latitude = np.deg2rad(origin.latitude)

    east = EARTH_RADIUS_METERS * np.cos(origin_latitude) * longitude_delta
    north = EARTH_RADIUS_METERS * latitude_delta
    up = altitude - origin.altitude

    return np.asarray((east, north, up), dtype=np.float64)


def _rotation_from_roll_pitch_yaw(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    rotation = np.empty((TRANSLATION_VALUES, TRANSLATION_VALUES), dtype=np.float64)
    rotation[0, 0] = cos_yaw * cos_pitch
    rotation[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    rotation[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    rotation[1, 0] = sin_yaw * cos_pitch
    rotation[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    rotation[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    rotation[2, 0] = -sin_pitch
    rotation[2, 1] = cos_pitch * sin_roll
    rotation[2, 2] = cos_pitch * cos_roll
    return rotation
