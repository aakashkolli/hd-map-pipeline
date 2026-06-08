import inspect

import numpy as np

from src.filters.outlier import OutlierConfig, remove_radius_outliers


def test_radius_outlier_removal_preserves_dense_cluster_and_removes_isolated():
    rng = np.random.default_rng(42)
    cluster = rng.normal(0.0, 0.03, (20, 4)).astype(np.float32)
    cluster[:, 3] = np.linspace(0.1, 1.0, 20, dtype=np.float32)
    isolated = np.array(
        [
            [5.0, 5.0, 5.0, 0.5],
            [-5.0, -5.0, 5.0, 0.6],
        ],
        dtype=np.float32,
    )
    points = np.vstack([cluster, isolated]).astype(np.float32)
    cfg = OutlierConfig(radius=0.20, min_neighbors=5)

    result = remove_radius_outliers(points, cfg)

    assert result.filtered_points.shape[0] == cluster.shape[0], (
        f"Expected {cluster.shape[0]} cluster points, got "
        f"{result.filtered_points.shape[0]}."
    )
    assert np.all(result.inlier_mask[: cluster.shape[0]]), (
        "Dense cluster points should be preserved."
    )
    assert not np.any(result.inlier_mask[cluster.shape[0] :]), (
        "Isolated points should be removed."
    )


def test_radius_outlier_removal_empty_input():
    points = np.empty((0, 4), dtype=np.float32)
    cfg = OutlierConfig(radius=0.20, min_neighbors=5)

    result = remove_radius_outliers(points, cfg)

    assert result.filtered_points.shape == (0, 4)
    assert result.filtered_points.dtype == np.float32
    assert result.inlier_mask.shape == (0,)


def test_outlier_filter_documents_frame_contract():
    docstring = inspect.getdoc(remove_radius_outliers)
    assert docstring is not None, "remove_radius_outliers needs a docstring."
    assert "FRAME:" in docstring, "Outlier filter must document frame behavior."
    assert "unchanged" in docstring, "Outlier filter must preserve input frame."


# ANTI-VIBE GATE - outlier.py
#
# 1. COORDINATE FRAME CONTRACT
#    Input points arrive in the caller's metric point-cloud frame, and output
#    points leave in the same unchanged frame. The filter only computes local
#    neighbor density in x, y, z coordinates.
#
# 2. SILENT FAILURE MODE
#    Sparse but valid structures such as poles or lane marking endpoints can
#    be removed if radius or min_neighbors is too aggressive. The output count
#    and removed count should be logged by pipeline callers.
#
# 3. VECTORIZATION STRATEGY
#    Pairwise spatial differences are computed through NumPy broadcasting and
#    reduced into neighbor counts. No per-point Python loop is used.
#
# 4. KNOWN LIMITATIONS
#    The fallback pairwise implementation is intended for synthetic tests and
#    moderate inputs. Production-sized clouds should use a KD-tree backend or
#    tiled neighbor search.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, isolated speckles should disappear while continuous road
#    surfaces, curb lines, and lane markings remain. Missing thin structures
#    indicate over-filtering.
