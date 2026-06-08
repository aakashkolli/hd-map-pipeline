"""Polyline simplification and distance metrics.

Input schema:
    Polylines are ``(N, 3)`` float32 or float64 arrays of xyz vertices.

Output schema:
    Simplification returns a new ``(M, 3)`` float32 array. Distance metrics
    return scalar meters.

Coordinate frames:
    These functions operate on world ENU polylines and preserve coordinates.
    They do not project points to BEV pixels or other image frames.
"""

from __future__ import annotations

import numpy as np


POINT_DIM = 3
MIN_POLYLINE_POINTS = 2


def simplify_rdp(polyline: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Simplify a world-frame polyline with Ramer-Douglas-Peucker.

    Args:
        polyline: ``(N, 3)`` point array. FRAME: world ENU.
        epsilon: Maximum perpendicular deviation in meters.

    Returns:
        Simplified ``(M, 3)`` polyline. FRAME: world ENU.
    """
    points = _validate_polyline(polyline, name="polyline")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    if points.shape[0] <= MIN_POLYLINE_POINTS:
        return points.astype(np.float32, copy=True)

    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True
    segments = [(0, points.shape[0] - 1)]

    while segments:
        start, end = segments.pop()
        if end - start <= 1:
            continue
        segment_points = points[start + 1 : end]
        distances = _point_to_segment_distances(
            segment_points,
            points[start],
            points[end],
        )
        max_offset = int(np.argmax(distances))
        max_distance = float(distances[max_offset])
        if max_distance > epsilon:
            index = start + 1 + max_offset
            keep[index] = True
            segments.append((start, index))
            segments.append((index, end))

    return points[keep].astype(np.float32, copy=True)


def hausdorff_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compute vertex-sampled Hausdorff distance between world polylines.

    Args:
        first: ``(N, 3)`` point array. FRAME: world ENU.
        second: ``(M, 3)`` point array. FRAME: world ENU.

    Returns:
        Symmetric Hausdorff distance in meters. FRAME: world ENU distances.
    """
    first_points = _validate_polyline(first, name="first")
    second_points = _validate_polyline(second, name="second")
    deltas = first_points[:, np.newaxis, :] - second_points[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    first_to_second = np.max(np.min(distances, axis=1))
    second_to_first = np.max(np.min(distances, axis=0))
    return float(max(first_to_second, second_to_first))


def _validate_polyline(polyline: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(polyline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != POINT_DIM:
        raise ValueError(
            f"Expected {name} shape (N, {POINT_DIM}) in world ENU, got "
            f"{points.shape}."
        )
    if points.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one world-frame point.")
    return points


def _point_to_segment_distances(
    points: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> np.ndarray:
    segment = segment_end - segment_start
    segment_length_squared = float(segment @ segment)
    if segment_length_squared <= np.finfo(np.float64).eps:
        return np.linalg.norm(points - segment_start, axis=1)

    projection = ((points - segment_start) @ segment) / segment_length_squared
    projection = np.clip(projection, 0.0, 1.0)
    closest = segment_start + projection[:, np.newaxis] * segment
    return np.linalg.norm(points - closest, axis=1)
