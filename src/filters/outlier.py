"""Radius outlier removal for point clouds.

Input schema:
    ``points`` is an ``(N, 4)`` float32 array with ``x, y, z, intensity``.

Output schema:
    ``OutlierRemovalResult`` contains masks over input rows and a filtered
    ``(M, 4)`` float32 point cloud.

Coordinate frames:
    The input frame is unchanged. Neighbor distances are measured in the
    caller's metric frame, and no coordinate transform is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


POINT_COLUMNS = 4
SPATIAL_COLUMNS = 3


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for radius outlier removal."""

    radius: float
    min_neighbors: int


@dataclass(frozen=True)
class OutlierRemovalResult:
    """Radius outlier removal result over the input point rows."""

    filtered_points: np.ndarray
    inlier_mask: np.ndarray
    outlier_mask: np.ndarray
    neighbor_counts: np.ndarray


def remove_radius_outliers(
    points: np.ndarray,
    cfg: OutlierConfig,
) -> OutlierRemovalResult:
    """Remove points with too few neighbors inside a metric radius.

    Args:
        points: ``(N, 4)`` float32 point array ``x, y, z, intensity``.
            FRAME: any metric point-cloud frame.
        cfg: Radius and minimum-neighbor parameters from configuration.

    Returns:
        OutlierRemovalResult with filtered points and masks.
            FRAME: unchanged from input.
    """
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != POINT_COLUMNS:
        raise ValueError(
            f"Expected point array shape (N, {POINT_COLUMNS}), got "
            f"{point_array.shape}."
        )
    if cfg.radius <= np.finfo(np.float32).eps:
        raise ValueError(f"radius must be positive, got {cfg.radius}.")
    if cfg.min_neighbors < 0:
        raise ValueError(
            f"min_neighbors must be non-negative, got {cfg.min_neighbors}."
        )
    if point_array.shape[0] == 0:
        empty_mask = np.empty(0, dtype=bool)
        empty_counts = np.empty(0, dtype=np.int64)
        return OutlierRemovalResult(
            filtered_points=np.empty((0, POINT_COLUMNS), dtype=np.float32),
            inlier_mask=empty_mask,
            outlier_mask=empty_mask.copy(),
            neighbor_counts=empty_counts,
        )

    spatial = point_array[:, :SPATIAL_COLUMNS].astype(np.float64)
    deltas = spatial[:, np.newaxis, :] - spatial[np.newaxis, :, :]
    squared_distances = np.sum(deltas * deltas, axis=2)
    radius_squared = float(cfg.radius) * float(cfg.radius)
    neighbor_counts = np.count_nonzero(
        squared_distances <= radius_squared,
        axis=1,
    )
    inlier_mask = neighbor_counts >= cfg.min_neighbors

    return OutlierRemovalResult(
        filtered_points=point_array[inlier_mask].astype(np.float32, copy=True),
        inlier_mask=inlier_mask,
        outlier_mask=~inlier_mask,
        neighbor_counts=neighbor_counts.astype(np.int64, copy=False),
    )
