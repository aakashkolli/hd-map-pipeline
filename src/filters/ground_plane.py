"""Seed-filtered RANSAC ground plane separation.

Input schema:
    ``points`` is an ``(N, 3)`` float32 or float64 point cloud.

Output schema:
    ``GroundPlaneResult`` contains boolean masks over the input point rows,
    a unit-normal plane equation ``a*x + b*y + c*z + d = 0``, and an
    inlier-ratio diagnostic.

Coordinate frames:
    Input points must be in vehicle frame: x forward, y left, z up. Output
    masks index the same vehicle-frame rows and do not transform points.

Known limitations:
    This model fits one dominant plane per frame. It can misclassify banked
    roads, ramps, overpasses, and road crowns where a terrain surface model
    would be more appropriate. It also does not enforce temporal consistency
    between adjacent frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PLANE_SAMPLE_SIZE = 3
POINT_DIM = 3
PLANE_COEFFS = 4


class InsufficientSeedPointsError(ValueError):
    """Raised when seed filtering leaves too few points for plane fitting."""


@dataclass(frozen=True)
class RansacConfig:
    """Configuration for seed-filtered RANSAC plane fitting."""

    max_iterations: int
    distance_threshold: float
    min_inlier_ratio: float
    seed_z_percentile: float
    seed_xy_radius: float
    random_seed: int


@dataclass(frozen=True)
class GroundPlaneResult:
    """Ground separation result over the input point cloud."""

    ground_mask: np.ndarray
    obstacle_mask: np.ndarray
    plane: np.ndarray
    inlier_ratio: float


def ransac_ground_plane(points: np.ndarray, cfg: RansacConfig) -> GroundPlaneResult:
    """Separate ground points from obstacles with seed-filtered RANSAC.

    Args:
        points: ``(N, 3)`` point array. FRAME: vehicle frame, x forward,
            y left, z up.
        cfg: RANSAC parameters from configuration.

    Returns:
        GroundPlaneResult with masks over the input rows. FRAME: vehicle
        frame is unchanged.
    """
    point_array = np.asarray(points)
    if point_array.ndim != 2 or point_array.shape[1] != POINT_DIM:
        raise ValueError(
            f"Expected point array shape (N, {POINT_DIM}) in vehicle frame, "
            f"got {point_array.shape}."
        )
    if point_array.shape[0] < PLANE_SAMPLE_SIZE:
        raise InsufficientSeedPointsError(
            f"Need at least {PLANE_SAMPLE_SIZE} points for a plane, got "
            f"{point_array.shape[0]}."
        )

    points64 = point_array.astype(np.float64)
    xy_distance = np.linalg.norm(points64[:, :2], axis=1)
    z_threshold = np.percentile(points64[:, 2], cfg.seed_z_percentile)
    seed_mask = (points64[:, 2] <= z_threshold) & (
        xy_distance <= cfg.seed_xy_radius
    )
    seed_indices = np.flatnonzero(seed_mask)
    if seed_indices.size < PLANE_SAMPLE_SIZE:
        raise InsufficientSeedPointsError(
            f"Only {seed_indices.size} seed points found in vehicle frame; "
            "check frame convention and seed configuration."
        )

    best_mask = np.zeros(point_array.shape[0], dtype=bool)
    best_plane = np.zeros(PLANE_COEFFS, dtype=np.float64)
    best_count = 0
    rng = np.random.default_rng(cfg.random_seed)

    for _ in range(cfg.max_iterations):
        sample_indices = rng.choice(
            seed_indices,
            size=PLANE_SAMPLE_SIZE,
            replace=False,
        )
        candidate = _plane_from_sample(points64[sample_indices])
        if candidate is None:
            continue

        distances = np.abs(points64 @ candidate[:POINT_DIM] + candidate[POINT_DIM])
        mask = distances < cfg.distance_threshold
        count = int(np.count_nonzero(mask))

        if count > best_count:
            best_count = count
            best_mask = mask
            best_plane = candidate

    if best_count < PLANE_SAMPLE_SIZE:
        raise InsufficientSeedPointsError(
            "RANSAC did not find a non-degenerate plane from seed points."
        )

    refined_plane = _refine_plane(points64[best_mask])
    refined_distances = np.abs(
        points64 @ refined_plane[:POINT_DIM] + refined_plane[POINT_DIM]
    )
    ground_mask = refined_distances < cfg.distance_threshold

    return GroundPlaneResult(
        ground_mask=ground_mask,
        obstacle_mask=~ground_mask,
        plane=refined_plane,
        inlier_ratio=float(np.mean(ground_mask)),
    )


def _plane_from_sample(sample: np.ndarray) -> np.ndarray | None:
    first_vector = sample[1] - sample[0]
    second_vector = sample[2] - sample[0]
    normal = np.cross(first_vector, second_vector)
    normal_length = np.linalg.norm(normal)
    if normal_length <= np.finfo(np.float64).eps:
        return None

    normal = _orient_up(normal / normal_length)
    offset = -float(normal @ sample[0])
    plane = np.empty(PLANE_COEFFS, dtype=np.float64)
    plane[:POINT_DIM] = normal
    plane[POINT_DIM] = offset
    return plane


def _refine_plane(inliers: np.ndarray) -> np.ndarray:
    centroid = np.mean(inliers, axis=0)
    _, _, right_singular_vectors = np.linalg.svd(
        inliers - centroid,
        full_matrices=False,
    )
    normal = _orient_up(right_singular_vectors[-1])
    offset = -float(normal @ centroid)
    plane = np.empty(PLANE_COEFFS, dtype=np.float64)
    plane[:POINT_DIM] = normal
    plane[POINT_DIM] = offset
    return plane


def _orient_up(normal: np.ndarray) -> np.ndarray:
    return np.where(normal[2] < 0.0, -normal, normal)
