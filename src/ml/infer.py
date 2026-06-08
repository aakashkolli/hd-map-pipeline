"""Inference helpers for BEV segmentation.

Input schema:
    BEV batches are ``(B, 1, H, W)`` float tensors. Segmentation masks are
    ``(H, W)`` integer arrays.

Output schema:
    Batch inference returns class-index masks. Back-projection returns
    ``(N, 3)`` float32 world coordinates.

Coordinate frames:
    BEV tensors are pixel rasters with external world-grid metadata. Mask
    back-projection uses ``origin_xy`` and ``resolution`` to recover world
    coordinates.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


BEV_DIMS = (2, 3)
MASK_POINT_DIM = 3


def normalize_bev_batch_per_image(batch: torch.Tensor) -> torch.Tensor:
    """Normalize each BEV image independently before inference.

    Args:
        batch: ``(B, 1, H, W)`` BEV intensity tensor.
            FRAME: BEV world-grid raster.

    Returns:
        Tensor with each image scaled by its own maximum.
            FRAME: unchanged BEV world-grid raster.
    """
    if batch.ndim != 4:
        raise ValueError(f"Expected batch shape (B, 1, H, W), got {batch.shape}.")
    max_values = torch.amax(batch, dim=BEV_DIMS, keepdim=True)
    safe_max = torch.where(max_values > 0.0, max_values, torch.ones_like(max_values))
    return batch / safe_max


def run_batch_inference(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Run model inference on per-image normalized BEV tensors.

    Args:
        model: Segmentation model.
        batch: ``(B, 1, H, W)`` BEV images. FRAME: BEV world-grid raster.

    Returns:
        ``(B, H, W)`` class-index masks. FRAME: BEV world-grid raster.
    """
    device = next(model.parameters()).device
    normalized = normalize_bev_batch_per_image(batch.to(device=device))
    model.eval()
    with torch.no_grad():
        logits = model(normalized)
    return torch.argmax(logits, dim=1).cpu()


def backproject_mask_to_world(
    mask: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution: float,
    z_value: float,
) -> np.ndarray:
    """Back-project nonzero BEV mask pixels to world coordinates.

    Args:
        mask: ``(H, W)`` integer segmentation mask.
            FRAME: BEV pixel raster.
        origin_xy: ``(2,)`` world coordinate of pixel ``(0, 0)`` corner.
            FRAME: world.
        resolution: Meters per pixel.
        z_value: World-frame z coordinate assigned to back-projected points.

    Returns:
        ``(N, 3)`` float32 xyz points. FRAME: world.
    """
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected mask shape (H, W), got {mask_array.shape}.")
    origin = np.asarray(origin_xy, dtype=np.float64)
    if origin.shape != (2,):
        raise ValueError(f"Expected origin_xy shape (2,), got {origin.shape}.")
    if resolution <= np.finfo(np.float32).eps:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    pixels_yx = np.argwhere(mask_array > 0)
    if pixels_yx.shape[0] == 0:
        return np.empty((0, MASK_POINT_DIM), dtype=np.float32)

    x = origin[0] + (pixels_yx[:, 1].astype(np.float64) + 0.5) * resolution
    y = origin[1] + (pixels_yx[:, 0].astype(np.float64) + 0.5) * resolution
    z = np.full(pixels_yx.shape[0], z_value, dtype=np.float64)
    return np.column_stack([x, y, z]).astype(np.float32)
