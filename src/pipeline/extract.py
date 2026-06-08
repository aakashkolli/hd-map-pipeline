"""Geometric lane boundary extraction from ground point clouds.

Input schema:
    ``ground_points`` is an ``(N, 4)`` float32 array with
    ``x, y, z, intensity`` columns.

Output schema:
    A list of ``LaneBoundaryFeature`` objects with 3D polyline geometry.

Coordinate frames:
    Input points are world ENU ground points. Output feature geometries are
    world ENU coordinates, not BEV pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from src.data.types import LaneBoundaryFeature, LaneType


POINT_COLUMNS = 4
SPATIAL_COLUMNS = 3
INTENSITY_COLUMN = 3
DBSCAN_CONNECTIVITY_MIN_SAMPLES = 1


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for geometric lane extraction."""

    intensity_percentile: float
    dbscan_eps: float
    dbscan_min_samples: int
    polyline_rdp_epsilon: float


def extract_lane_boundaries(
    ground_points: np.ndarray,
    cfg: ExtractionConfig,
) -> list[LaneBoundaryFeature]:
    """Extract high-intensity lane boundary polylines.

    Args:
        ground_points: ``(N, 4)`` float32 array ``x, y, z, intensity``.
            FRAME: world ENU ground points.
        cfg: Extraction parameters from configuration.

    Returns:
        Lane boundary features with geometry in world ENU.
            FRAME: world ENU.
    """
    point_array = np.asarray(ground_points, dtype=np.float32)
    if point_array.ndim != 2 or point_array.shape[1] != POINT_COLUMNS:
        raise ValueError(
            f"Expected point array shape (N, {POINT_COLUMNS}) in world ENU, "
            f"got {point_array.shape}."
        )
    if point_array.shape[0] < cfg.dbscan_min_samples:
        return []

    intensities = point_array[:, INTENSITY_COLUMN]
    threshold = np.percentile(intensities, cfg.intensity_percentile)
    candidates = point_array[intensities >= threshold]
    if candidates.shape[0] < cfg.dbscan_min_samples:
        return []

    labels = DBSCAN(
        eps=cfg.dbscan_eps,
        min_samples=DBSCAN_CONNECTIVITY_MIN_SAMPLES,
    ).fit_predict(candidates[:, :SPATIAL_COLUMNS])

    features: list[LaneBoundaryFeature] = []
    for label in np.unique(labels):
        if label < 0:
            continue
        cluster = candidates[labels == label]
        if cluster.shape[0] < cfg.dbscan_min_samples:
            continue
        polyline = _fit_ordered_polyline(cluster[:, :SPATIAL_COLUMNS])
        confidence = _cluster_confidence(cluster[:, INTENSITY_COLUMN])
        features.append(
            LaneBoundaryFeature(
                geometry=polyline.astype(np.float32).tolist(),
                feature_type=LaneType.LANE_LINE,
                confidence=confidence,
                point_count=int(cluster.shape[0]),
            )
        )

    return features


def _fit_ordered_polyline(points_xyz: np.ndarray) -> np.ndarray:
    centroid = np.mean(points_xyz, axis=0)
    centered = points_xyz.astype(np.float64) - centroid
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    direction = right_singular_vectors[0]
    projection = centered @ direction
    ordered = points_xyz[np.argsort(projection)]
    return ordered[[0, -1]]


def _cluster_confidence(intensities: np.ndarray) -> float:
    max_intensity = np.max(intensities)
    if max_intensity <= np.finfo(np.float32).eps:
        return float(max_intensity)
    return float(np.clip(np.mean(intensities) / max_intensity, 0.0, 1.0))
