import inspect

import numpy as np
import pytest

from src.geometry.polyline import hausdorff_distance, simplify_rdp


def test_rdp_simplifies_known_polyline_and_preserves_endpoints():
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [2.0, -0.01, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    simplified = simplify_rdp(polyline, epsilon=0.05)

    assert simplified.shape[0] < polyline.shape[0], (
        "RDP should reduce nearly collinear intermediate points."
    )
    np.testing.assert_allclose(simplified[0], polyline[0])
    np.testing.assert_allclose(simplified[-1], polyline[-1])


def test_hausdorff_distance_identical_polylines_is_zero():
    polyline = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    assert hausdorff_distance(polyline, polyline) == pytest.approx(0.0, abs=1e-8)


def test_polyline_functions_document_frame_contracts():
    for function in (simplify_rdp, hausdorff_distance):
        docstring = inspect.getdoc(function)
        assert docstring is not None, f"{function.__name__} needs a docstring."
        assert "FRAME:" in docstring, (
            f"{function.__name__} must document spatial frame behavior."
        )
        assert "world" in docstring, (
            f"{function.__name__} should document world-frame usage."
        )


# ANTI-VIBE GATE - polyline.py
#
# 1. COORDINATE FRAME CONTRACT
#    Input polylines are world ENU xyz arrays, and outputs remain world ENU.
#    The module computes geometry only; it does not project to BEV pixels.
#
# 2. SILENT FAILURE MODE
#    If point order does not follow the lane direction, RDP can simplify a
#    scrambled polyline into an invalid shape without raising. Extraction must
#    order clusters before calling simplification.
#
# 3. VECTORIZATION STRATEGY
#    Point-to-segment and pairwise point distances are computed as NumPy array
#    operations over full coordinate blocks. There is no per-point Python loop.
#
# 4. KNOWN LIMITATIONS
#    Hausdorff distance compares sampled vertices, not continuous line
#    segments. Dense resampling is needed for centimeter-grade QA metrics.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, simplified lane polylines should follow the same road
#    markings with fewer vertices. Oversimplification cuts corners or removes
#    dashed-line shape changes.
