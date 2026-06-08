import inspect

import numpy as np

from src.pipeline.extract import ExtractionConfig, extract_lane_boundaries


def test_extract_lane_boundaries_detects_two_parallel_lanes():
    line_x = np.linspace(0.0, 50.0, 500, dtype=np.float32)
    line_z = np.zeros(500, dtype=np.float32)
    line_i = np.ones(500, dtype=np.float32)
    line1 = np.column_stack(
        [line_x, np.full(500, -1.5, dtype=np.float32), line_z, line_i]
    )
    line2 = np.column_stack(
        [line_x, np.full(500, 1.5, dtype=np.float32), line_z, line_i]
    )
    background = np.random.default_rng(42).uniform(
        0.0,
        0.3,
        (5_000, 4),
    ).astype(np.float32)
    background[:, 2] = 0.0
    points = np.vstack([line1, line2, background]).astype(np.float32)
    cfg = ExtractionConfig(
        intensity_percentile=85.0,
        dbscan_eps=0.25,
        dbscan_min_samples=10,
        polyline_rdp_epsilon=0.05,
    )

    features = extract_lane_boundaries(points, cfg)

    assert len(features) == 2, f"Expected 2 lane boundaries, got {len(features)}."
    for feature in features:
        geometry = np.asarray(feature.geometry, dtype=np.float32)
        assert geometry.shape[1] == 3, "Feature geometry must be 3D world coords."
        assert np.max(geometry[:, 0]) > 45.0, (
            "Lane polyline should preserve world-frame x extent, not pixel coords."
        )
        assert np.isclose(abs(float(np.mean(geometry[:, 1]))), 1.5, atol=0.1)


def test_extract_lane_boundaries_sparse_input_returns_empty():
    points = np.array(
        [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    cfg = ExtractionConfig(
        intensity_percentile=85.0,
        dbscan_eps=0.25,
        dbscan_min_samples=10,
        polyline_rdp_epsilon=0.05,
    )

    assert extract_lane_boundaries(points, cfg) == []


def test_extract_documents_world_frame_contract():
    docstring = inspect.getdoc(extract_lane_boundaries)
    assert docstring is not None, "extract_lane_boundaries needs a docstring."
    assert "FRAME:" in docstring, "Extraction must document input frame."
    assert "world" in docstring, "Extraction output must remain world frame."


# ANTI-VIBE GATE - extract.py
#
# 1. COORDINATE FRAME CONTRACT
#    Input ground points arrive in world ENU and extracted feature geometries
#    leave as world ENU xyz coordinates. No BEV pixel coordinates are stored
#    in LaneBoundaryFeature.geometry.
#
# 2. SILENT FAILURE MODE
#    Rain or worn paint can compress intensity contrast so percentile
#    thresholding still selects points but DBSCAN clusters background texture.
#    Cluster count and candidate count should be logged by pipeline callers.
#
# 3. VECTORIZATION STRATEGY
#    Intensity thresholding uses a vectorized percentile and boolean mask.
#    DBSCAN receives the full candidate coordinate array. Cluster slicing is
#    vectorized by label mask; there is no per-point Python loop.
#
# 4. KNOWN LIMITATIONS
#    The extractor assumes painted lane boundaries are high-intensity ground
#    returns and can miss low-reflectance paint or merge very close markings.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, extracted features should overlay lane paint in world
#    coordinates. Pixel-coordinate leakage would place lines near the BEV
#    image origin or far outside the road cloud.
