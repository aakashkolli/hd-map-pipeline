"""Quality assessment metrics for extracted map features.

Input schema:
    Predicted and ground-truth features are lists of LaneBoundaryFeature
    objects with xyz polyline geometry.

Output schema:
    QAReport stores completeness, positional accuracy percentiles,
    false-positive rate, class accuracy, and feature IDs for visualization.

Coordinate frames:
    All feature geometries are world ENU polylines. QA computes distances in
    meters and never compares BEV pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.types import LaneBoundaryFeature
from src.geometry.polyline import hausdorff_distance


@dataclass(frozen=True)
class QAConfig:
    """Configuration for QA feature matching."""

    max_gt_match_distance: float


@dataclass(frozen=True)
class QAReport:
    """Map feature QA metric report."""

    scene_id: str
    completeness: float
    positional_accuracy_p50: float
    positional_accuracy_p95: float
    false_positive_rate: float
    classification_accuracy: float
    per_class_completeness: dict[str, float]
    missed_gt_features: list[str]
    false_positive_ids: list[str]


def compute_qa_metrics(
    predicted: list[LaneBoundaryFeature],
    ground_truth: list[LaneBoundaryFeature],
    cfg: QAConfig,
    *,
    scene_id: str = "synthetic",
) -> QAReport:
    """Compute QA metrics against external ground truth annotations.

    Args:
        predicted: Extracted features. FRAME: world ENU geometry.
        ground_truth: External ground truth features. FRAME: world ENU
            geometry.
        cfg: Matching threshold in meters.
        scene_id: Identifier copied into the report.

    Returns:
        QAReport comparing predictions to ground truth. FRAME: world ENU
        distances and feature IDs.
    """
    if cfg.max_gt_match_distance <= np.finfo(np.float32).eps:
        raise ValueError(
            f"max_gt_match_distance must be positive, got "
            f"{cfg.max_gt_match_distance}."
        )
    if not ground_truth:
        false_positive_rate = 0.0 if not predicted else 1.0
        return QAReport(
            scene_id=scene_id,
            completeness=1.0,
            positional_accuracy_p50=0.0,
            positional_accuracy_p95=0.0,
            false_positive_rate=false_positive_rate,
            classification_accuracy=1.0,
            per_class_completeness={},
            missed_gt_features=[],
            false_positive_ids=_feature_ids(predicted),
        )

    matches: list[tuple[int, int, float]] = []
    used_predictions: set[int] = set()
    for gt_index, gt_feature in enumerate(ground_truth):
        best_pred_index = None
        best_distance = float("inf")
        for pred_index, pred_feature in enumerate(predicted):
            if pred_index in used_predictions:
                continue
            distance = hausdorff_distance(
                np.asarray(pred_feature.geometry, dtype=np.float32),
                np.asarray(gt_feature.geometry, dtype=np.float32),
            )
            if distance < best_distance:
                best_distance = distance
                best_pred_index = pred_index
        if (
            best_pred_index is not None
            and best_distance <= cfg.max_gt_match_distance
        ):
            used_predictions.add(best_pred_index)
            matches.append((best_pred_index, gt_index, best_distance))

    matched_gt = {gt_index for _, gt_index, _ in matches}
    missed_gt = [
        str(index)
        for index in range(len(ground_truth))
        if index not in matched_gt
    ]
    false_positive_ids = [
        str(index)
        for index in range(len(predicted))
        if index not in used_predictions
    ]
    distances = np.asarray([distance for _, _, distance in matches], dtype=np.float64)
    class_matches = [
        predicted[pred_index].feature_type == ground_truth[gt_index].feature_type
        for pred_index, gt_index, _ in matches
    ]

    return QAReport(
        scene_id=scene_id,
        completeness=float(len(matches) / len(ground_truth)),
        positional_accuracy_p50=_percentile_or_zero(distances, percentile=50.0),
        positional_accuracy_p95=_percentile_or_zero(distances, percentile=95.0),
        false_positive_rate=float(len(false_positive_ids) / len(predicted))
        if predicted
        else 0.0,
        classification_accuracy=float(np.mean(class_matches))
        if class_matches
        else 0.0,
        per_class_completeness=_per_class_completeness(ground_truth, matched_gt),
        missed_gt_features=missed_gt,
        false_positive_ids=false_positive_ids,
    )


def _percentile_or_zero(values: np.ndarray, *, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _per_class_completeness(
    ground_truth: list[LaneBoundaryFeature],
    matched_gt: set[int],
) -> dict[str, float]:
    feature_types = {feature.feature_type.value for feature in ground_truth}
    result: dict[str, float] = {}
    for feature_type in feature_types:
        class_indices = [
            index
            for index, feature in enumerate(ground_truth)
            if feature.feature_type.value == feature_type
        ]
        matched_count = sum(index in matched_gt for index in class_indices)
        result[feature_type] = float(matched_count / len(class_indices))
    return result


def _feature_ids(features: list[LaneBoundaryFeature]) -> list[str]:
    return [str(index) for index in range(len(features))]
