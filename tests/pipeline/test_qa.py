import inspect

from src.data.types import LaneBoundaryFeature, LaneType
from src.pipeline.qa import QAConfig, compute_qa_metrics


def _feature(offset: float = 0.0) -> LaneBoundaryFeature:
    return LaneBoundaryFeature(
        geometry=[
            [0.0 + offset, 0.0, 0.0],
            [10.0 + offset, 0.0, 0.0],
            [20.0 + offset, 0.0, 0.0],
        ],
        feature_type=LaneType.LANE_LINE,
        confidence=1.0,
        point_count=3,
    )


def test_qa_perfect_prediction_edge_case():
    cfg = QAConfig(max_gt_match_distance=1.0)

    report = compute_qa_metrics([_feature(offset=0.05)], [_feature()], cfg)

    assert report.completeness == 1.0
    assert report.false_positive_rate == 0.0


def test_qa_all_wrong_edge_case():
    cfg = QAConfig(max_gt_match_distance=1.0)

    report = compute_qa_metrics([_feature(offset=100.0)], [_feature()], cfg)

    assert report.completeness == 0.0
    assert report.false_positive_rate == 1.0


def test_qa_documents_ground_truth_contract():
    docstring = inspect.getdoc(compute_qa_metrics)
    assert docstring is not None, "compute_qa_metrics needs a docstring."
    assert "FRAME:" in docstring, "QA must document feature geometry frames."
    assert "ground truth" in docstring


# ANTI-VIBE GATE - qa.py
#
# 1. COORDINATE FRAME CONTRACT
#    Predicted and ground-truth features arrive as world ENU polylines. QA
#    compares geometry in that frame and does not accept pixel coordinates.
#
# 2. SILENT FAILURE MODE
#    If ground truth comes from the same extractor output, completeness can be
#    perfect while the map is wrong. The API requires an explicit ground_truth
#    argument and tests use separate GT features.
#
# 3. VECTORIZATION STRATEGY
#    Geometry distance delegates to vectorized Hausdorff distance. Matching
#    loops over features, not point arrays.
#
# 4. KNOWN LIMITATIONS
#    Matching is greedy by Hausdorff distance and does not solve global
#    assignment optimally. Dense map QA should use bipartite matching.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, missed GT should highlight red and false positives amber.
#    A correct perfect prediction should show no QA error overlays.
