import torch

from src.ml.unet import BEVSegNet


def test_bev_segnet_forward_shape():
    model = BEVSegNet(num_classes=4)
    model.eval()
    x = torch.randn(2, 1, 512, 512)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 4, 512, 512), f"Wrong output shape: {y.shape}."


def test_bev_segnet_parameter_count_under_limit():
    model = BEVSegNet(num_classes=4)
    param_count = sum(parameter.numel() for parameter in model.parameters())

    assert param_count < 3_000_000, f"Model too large: {param_count}."


# ANTI-VIBE GATE - unet.py
#
# 1. COORDINATE FRAME CONTRACT
#    The model consumes BEV raster tensors derived from world-frame ground
#    points. Spatial frame metadata lives outside the tensor, but the tensor
#    shape contract preserves H/W resolution for mask alignment.
#
# 2. SILENT FAILURE MODE
#    If training images and labels are cropped differently, the model can
#    train without shape errors while learning shifted labels. Shape tests only
#    guarantee resolution preservation, not semantic alignment.
#
# 3. VECTORIZATION STRATEGY
#    PyTorch convolutions operate over full image tensors. There is no Python
#    iteration over pixels or point arrays in the model forward pass.
#
# 4. KNOWN LIMITATIONS
#    The architecture is CPU-friendly and intentionally small. It lacks the
#    multi-scale context and sensor fusion used in production mapping models.
#
# 5. OBSERVABILITY CHECK
#    In BEV overlay, predicted lane masks should align with bright lane paint
#    pixels. Systematic offsets indicate preprocessing or crop alignment bugs,
#    not a U-Net shape issue.
