"""Fusion of geometric and ML map features.

Input schema:
    Geometric and ML predictions are lists of LaneBoundaryFeature objects.

Output schema:
    Fused LaneBoundaryFeature list with source labels preserved or combined.

Coordinate frames:
    All feature geometries are world ENU polylines. Matching and output
    preserve world coordinates and never use BEV pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.types import LaneBoundaryFeature
from src.geometry.polyline import hausdorff_distance


CONFIDENCE_CAP = 1.0


@dataclass(frozen=True)
class FuseConfig:
    """Feature fusion configuration."""

    max_merge_distance: float


def fuse_features(
    geometric_features: list[LaneBoundaryFeature],
    ml_features: list[LaneBoundaryFeature],
    cfg: FuseConfig,
) -> list[LaneBoundaryFeature]:
    """Fuse geometric and ML lane features.

    Args:
        geometric_features: Geometric extractor outputs.
            FRAME: world ENU polylines.
        ml_features: ML segmentation outputs.
            FRAME: world ENU polylines.
        cfg: Maximum Hausdorff distance for merging.

    Returns:
        Fused features. FRAME: world ENU.
    """
    if cfg.max_merge_distance <= np.finfo(np.float32).eps:
        raise ValueError(
            f"max_merge_distance must be positive, got {cfg.max_merge_distance}."
        )

    fused: list[LaneBoundaryFeature] = []
    used_ml: set[int] = set()
    for geometric in geometric_features:
        match_index = _best_match(geometric, ml_features, used_ml, cfg)
        if match_index is None:
            fused.append(geometric)
            continue
        used_ml.add(match_index)
        fused.append(_merge_pair(geometric, ml_features[match_index]))

    fused.extend(
        feature for index, feature in enumerate(ml_features) if index not in used_ml
    )
    return fused


def _best_match(
    feature: LaneBoundaryFeature,
    candidates: list[LaneBoundaryFeature],
    used_indices: set[int],
    cfg: FuseConfig,
) -> int | None:
    best_index = None
    best_distance = float("inf")
    for index, candidate in enumerate(candidates):
        if index in used_indices:
            continue
        distance = hausdorff_distance(
            np.asarray(feature.geometry, dtype=np.float32),
            np.asarray(candidate.geometry, dtype=np.float32),
        )
        if distance < best_distance:
            best_distance = distance
            best_index = index
    if best_index is None or best_distance > cfg.max_merge_distance:
        return None
    return best_index


def _merge_pair(
    geometric: LaneBoundaryFeature,
    ml_feature: LaneBoundaryFeature,
) -> LaneBoundaryFeature:
    confidence = min(
        CONFIDENCE_CAP,
        (geometric.confidence + ml_feature.confidence) / 2.0
        + (CONFIDENCE_CAP - max(geometric.confidence, ml_feature.confidence)) / 2.0,
    )
    return LaneBoundaryFeature(
        geometry=geometric.geometry,
        feature_type=geometric.feature_type,
        confidence=float(confidence),
        point_count=max(geometric.point_count, ml_feature.point_count),
        source=f"{geometric.source}+{ml_feature.source}",
    )
