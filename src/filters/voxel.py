"""Voxel grid downsampling for point clouds.

Input schema:
    ``points`` is an ``(N, 4)`` float32 array with ``x, y, z, intensity``.

Output schema:
    ``(M, 4)`` float32 array containing one representative point per voxel.

Coordinate frames:
    The input coordinate frame is unchanged. The filter groups by metric
    ``x, y, z`` coordinates and does not apply a spatial transform.
"""

from __future__ import annotations

import numpy as np


POINT_COLUMNS = 4
SPATIAL_COLUMNS = 3

try:
    from src.ext._voxel_filter import voxel_downsample_cpp
except ImportError:
    voxel_downsample_cpp = None


def voxel_downsample(points: np.ndarray, *, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud to one representative point per voxel.

    Args:
        points: ``(N, 4)`` float32 array ``x, y, z, intensity``.
            FRAME: any metric point-cloud frame.
        voxel_size: Voxel edge length in meters from configuration.

    Returns:
        ``(M, 4)`` float32 array. FRAME: unchanged from input.
    """
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != POINT_COLUMNS:
        raise ValueError(
            f"Expected point array shape (N, {POINT_COLUMNS}), got "
            f"{point_array.shape}."
        )
    if voxel_size <= np.finfo(np.float32).eps:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}.")
    if point_array.shape[0] == 0:
        return np.empty((0, POINT_COLUMNS), dtype=np.float32)

    if voxel_downsample_cpp is not None:
        return voxel_downsample_cpp(point_array, float(voxel_size))

    return _voxel_downsample_numpy(point_array, voxel_size=voxel_size)


def _voxel_downsample_numpy(points: np.ndarray, *, voxel_size: float) -> np.ndarray:
    voxel_indices = np.floor(
        points[:, :SPATIAL_COLUMNS].astype(np.float64) / float(voxel_size)
    ).astype(np.int64)
    _, representative_indices = np.unique(
        voxel_indices,
        axis=0,
        return_index=True,
    )
    ordered_indices = np.sort(representative_indices)
    return points[ordered_indices].astype(np.float32, copy=True)

