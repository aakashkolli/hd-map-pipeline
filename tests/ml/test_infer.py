import inspect

import numpy as np
import torch

from src.ml.infer import backproject_mask_to_world, normalize_bev_batch_per_image


def test_segmentation_mask_backprojects_to_world_coordinates():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[12, 13] = 1

    points = backproject_mask_to_world(
        mask,
        origin_xy=np.array([-10.0, -10.0], dtype=np.float64),
        resolution=1.0,
        z_value=0.0,
    )

    assert points.shape == (1, 3)
    np.testing.assert_allclose(
        points[0],
        np.array([3.5, 2.5, 0.0]),
        atol=1e-6,
        err_msg="Mask pixel did not back-project to expected world coordinate.",
    )


def test_bev_batch_normalization_is_per_image():
    batch = torch.tensor(
        [
            [[[0.0, 2.0], [1.0, 0.0]]],
            [[[0.0, 200.0], [100.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_bev_batch_per_image(batch)

    torch.testing.assert_close(normalized[0], normalized[1])
    assert torch.max(normalized[0]) == torch.tensor(1.0)


def test_infer_functions_document_frame_contracts():
    for function in (backproject_mask_to_world, normalize_bev_batch_per_image):
        docstring = inspect.getdoc(function)
        assert docstring is not None, f"{function.__name__} needs a docstring."
        assert "FRAME:" in docstring, (
            f"{function.__name__} must document spatial frame behavior."
        )


# ANTI-VIBE GATE - infer.py
#
# 1. COORDINATE FRAME CONTRACT
#    Masks are BEV pixels with origin/resolution metadata, and back-projected
#    output points are world-frame xyz coordinates at the supplied z value.
#
# 2. SILENT FAILURE MODE
#    If origin_xy or resolution comes from a different BEV image, mask pixels
#    back-project to plausible but wrong world coordinates without crashing.
#    Tests pin one known pixel-to-world mapping.
#
# 3. VECTORIZATION STRATEGY
#    np.argwhere collects all active pixels, coordinate conversion is vectorized
#    over arrays, and batch normalization reduces over tensor dimensions.
#
# 4. KNOWN LIMITATIONS
#    Back-projection returns pixel centers at a constant z value, not a fitted
#    ground surface height. Real inference should sample ground elevations.
#
# 5. OBSERVABILITY CHECK
#    In the viewer, ML feature points should overlay the same lane paint seen
#    in BEV. Systematic offset implies wrong origin/resolution metadata.
