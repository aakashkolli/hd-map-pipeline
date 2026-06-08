import inspect

from src.data.types import LaneBoundaryFeature, LaneType
from src.pipeline.fuse import FuseConfig, fuse_features


def _feature(offset: float, source: str) -> LaneBoundaryFeature:
    return LaneBoundaryFeature(
        geometry=[[0.0 + offset, 0.0, 0.0], [10.0 + offset, 0.0, 0.0]],
        feature_type=LaneType.LANE_LINE,
        confidence=0.8,
        point_count=2,
        source=source,
    )


def test_identical_predictions_merge_to_single_feature():
    cfg = FuseConfig(max_merge_distance=0.5)

    fused = fuse_features([_feature(0.0, "geometric")], [_feature(0.1, "ml")], cfg)

    assert len(fused) == 1
    assert fused[0].source == "geometric+ml"
    assert fused[0].confidence > 0.8


def test_conflicting_predictions_are_retained_with_sources():
    cfg = FuseConfig(max_merge_distance=0.5)

    fused = fuse_features([_feature(0.0, "geometric")], [_feature(10.0, "ml")], cfg)

    assert len(fused) == 2
    assert {feature.source for feature in fused} == {"geometric", "ml"}


def test_fuse_documents_world_frame_contract():
    docstring = inspect.getdoc(fuse_features)
    assert docstring is not None, "fuse_features needs a docstring."
    assert "FRAME:" in docstring, "Fusion must document feature frames."
    assert "world" in docstring


# ANTI-VIBE GATE - fuse.py
#
# 1. COORDINATE FRAME CONTRACT
#    Geometric and ML features arrive as world ENU polylines and fused outputs
#    remain world ENU. No pixel-space matching or geometry export occurs.
#
# 2. SILENT FAILURE MODE
#    Two nearby but distinct lane markings can be merged if max_merge_distance
#    is too large. Viewer overlays should show whether fused lines collapse
#    adjacent markings.
#
# 3. VECTORIZATION STRATEGY
#    Feature matching delegates geometry distance to vectorized Hausdorff
#    distance. Fusion loops over features, not point arrays.
#
# 4. KNOWN LIMITATIONS
#    Greedy matching does not solve all pair assignments globally. Dense urban
#    markings may need bipartite matching and semantic class constraints.
#
# 5. OBSERVABILITY CHECK
#    Fused features should overlay either geometric or ML predictions without
#    duplicate coincident lines. Conflicts should remain visible as separate
#    source-labeled features.
