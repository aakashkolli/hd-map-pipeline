import inspect

import numpy as np
import pytest

from src.pipeline.bev import BEVConfig, project_to_bev


def test_bev_single_point_maps_to_expected_pixel():
    cfg = BEVConfig(resolution=0.10, extent=10.0)
    point = np.array([[3.0, 2.0, 0.0, 1.0]], dtype=np.float32)

    result = project_to_bev(point, cfg)

    assert result.image[120, 130] == pytest.approx(1.0, abs=1e-5), (
        "Point did not land in expected BEV pixel."
    )
    assert result.image.sum() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(result.origin_xy, np.array([-10.0, -10.0]))


def test_bev_normalization_is_per_scan_not_global():
    cfg = BEVConfig(resolution=0.50, extent=5.0)
    low_scale = np.array(
        [[0.0, 0.0, 0.0, 2.0], [1.0, 1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    high_scale = low_scale.copy()
    high_scale[:, 3] *= np.float32(100.0)

    low_result = project_to_bev(low_scale, cfg)
    high_result = project_to_bev(high_scale, cfg)

    np.testing.assert_allclose(
        low_result.image,
        high_result.image,
        atol=1e-6,
        err_msg="BEV images should match under per-scan intensity scaling.",
    )


def test_bev_documents_world_frame_contract():
    docstring = inspect.getdoc(project_to_bev)
    assert docstring is not None, "project_to_bev needs a docstring."
    assert "FRAME:" in docstring, "BEV projection must document input frame."
    assert "world" in docstring, "BEV projection must require world frame."


# ANTI-VIBE GATE - bev.py
#
# 1. COORDINATE FRAME CONTRACT
#    Input ground points arrive in local world ENU and output pixels are an
#    image representation with origin_xy storing the world coordinates of
#    image corner. The docstring documents this explicitly.
#
# 2. SILENT FAILURE MODE
#    If callers pass vehicle-frame points while origin_xy is interpreted as
#    world ENU, the image still renders but exported features will be offset.
#    This should be caught by comparing known world point locations to pixels.
#
# 3. VECTORIZATION STRATEGY
#    Pixel coordinates are computed through vectorized floor operations, valid
#    masks are boolean-indexed, and np.maximum.at performs scatter max pooling.
#    There is no per-point Python loop.
#
# 4. KNOWN LIMITATIONS
#    Multiple heights in the same BEV cell collapse to max intensity. This is
#    acceptable for ground-only lane marking projection but not for overpasses.
#
# 5. OBSERVABILITY CHECK
#    In the viewer or BEV image, high-intensity lane paint should appear in
#    stable world locations. Scaling all intensities should not change visible
#    road-marking structure.
