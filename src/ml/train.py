"""Training utilities for BEV segmentation.

Input schema:
    Images are ``(B, 1, H, W)`` float tensors. Labels are ``(B, H, W)``
    integer class tensors.

Output schema:
    Training helpers return scalar loss histories and leave model weights
    updated in place.

Coordinate frames:
    Training tensors are BEV rasters generated from world-frame nuScenes map
    annotations. Evaluation tensors are expected to come from KITTI BEV
    rasters for cross-dataset validation. Tensor height and width are
    preserved by the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop hyperparameters."""

    learning_rate: float
    steps: int


def train_on_batch(
    *,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    cfg: TrainingConfig,
) -> list[float]:
    """Overfit one BEV batch with weighted cross-entropy.

    Args:
        model: BEV segmentation model.
        images: ``(B, 1, H, W)`` normalized BEV tensors.
            FRAME: nuScenes world-grid BEV during training; KITTI world-grid
            BEV is reserved for evaluation.
        labels: ``(B, H, W)`` class labels aligned to ``images``.
            FRAME: same BEV world-grid as images.
        class_weights: Per-class loss weights for label imbalance.
        cfg: Optimizer parameters from configuration.

    Returns:
        List of loss values, one per optimization step.
            FRAME: tensor geometry unchanged.
    """
    if cfg.steps <= 0:
        raise ValueError(f"steps must be positive, got {cfg.steps}.")
    if cfg.learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be positive, got {cfg.learning_rate}."
        )
    if images.ndim != 4:
        raise ValueError(f"Expected images shape (B, 1, H, W), got {images.shape}.")
    if labels.ndim != 3:
        raise ValueError(f"Expected labels shape (B, H, W), got {labels.shape}.")

    device = next(model.parameters()).device
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    weights = class_weights.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights)

    losses: list[float] = []
    model.train()
    for _ in range(cfg.steps):
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return losses
