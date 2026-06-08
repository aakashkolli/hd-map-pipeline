"""Bird's-eye-view intensity projection.

Input schema:
    ``ground_points`` is an ``(N, 4)`` float32 array with
    ``x, y, z, intensity`` columns.

Output schema:
    ``BEVImage`` contains a 2D float32 intensity image, the world coordinate
    of the image corner, and the meter-per-pixel resolution.

Coordinate frames:
    Input points are local world ENU ground points. Output pixels are image
    coordinates, while ``origin_xy`` records the world ENU coordinate of
    pixel ``(0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


POINT_COLUMNS = 4
X_COLUMN = 0
Y_COLUMN = 1
INTENSITY_COLUMN = 3


@dataclass(frozen=True)
class BEVConfig:
    """BEV projection configuration."""

    resolution: float
    extent: float


@dataclass(frozen=True)
class BEVImage:
    """Projected BEV intensity image and spatial metadata."""

    image: np.ndarray
    origin_xy: np.ndarray
    resolution: float


def project_to_bev(ground_points: np.ndarray, cfg: BEVConfig) -> BEVImage:
    """Project ground points into a per-scan normalized BEV image.

    Args:
        ground_points: ``(N, 4)`` float32 array ``x, y, z, intensity``.
            FRAME: local world ENU, ground points only.
        cfg: BEV extent and resolution from configuration.

    Returns:
        BEVImage with a float32 image in ``[0, 1]``.
            FRAME: origin_xy is local world ENU.
    """
    point_array = np.asarray(ground_points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != POINT_COLUMNS:
        raise ValueError(
            f"Expected point array shape (N, {POINT_COLUMNS}) in world ENU, "
            f"got {point_array.shape}."
        )
    if cfg.resolution <= np.finfo(np.float32).eps:
        raise ValueError(f"resolution must be positive, got {cfg.resolution}.")
    if cfg.extent <= np.finfo(np.float32).eps:
        raise ValueError(f"extent must be positive, got {cfg.extent}.")

    image_size = int((cfg.extent + cfg.extent) / cfg.resolution)
    image = np.zeros((image_size, image_size), dtype=np.float32)
    origin_xy = np.asarray((-cfg.extent, -cfg.extent), dtype=np.float64)
    if point_array.shape[0] == 0:
        return BEVImage(image=image, origin_xy=origin_xy, resolution=cfg.resolution)

    intensities = point_array[:, INTENSITY_COLUMN].astype(np.float32, copy=True)
    max_intensity = np.max(intensities)
    if max_intensity > np.finfo(np.float32).eps:
        intensities = intensities / max_intensity

    pixel_x = np.floor(
        (point_array[:, X_COLUMN] + cfg.extent) / cfg.resolution
    ).astype(np.int64)
    pixel_y = np.floor(
        (point_array[:, Y_COLUMN] + cfg.extent) / cfg.resolution
    ).astype(np.int64)
    valid = (
        (pixel_x >= 0)
        & (pixel_x < image_size)
        & (pixel_y >= 0)
        & (pixel_y < image_size)
    )
    np.maximum.at(image, (pixel_y[valid], pixel_x[valid]), intensities[valid])

    return BEVImage(image=image, origin_xy=origin_xy, resolution=cfg.resolution)
