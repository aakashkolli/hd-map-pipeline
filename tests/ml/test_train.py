import inspect

import torch

from src.ml.train import TrainingConfig, train_on_batch
from src.ml.unet import BEVSegNet


def test_training_loss_decreases_over_five_steps():
    torch.manual_seed(42)
    model = BEVSegNet(num_classes=4, base_channels=4)
    x = torch.zeros(4, 1, 64, 64)
    y = torch.zeros(4, 64, 64, dtype=torch.long)
    cfg = TrainingConfig(learning_rate=1e-2, steps=5)
    class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0])

    losses = train_on_batch(
        model=model,
        images=x,
        labels=y,
        class_weights=class_weights,
        cfg=cfg,
    )

    assert losses[-1] < losses[0], (
        f"Expected loss to decrease over 5 steps, got {losses}."
    )


def test_train_module_documents_cross_dataset_contract():
    docstring = inspect.getdoc(train_on_batch)
    assert docstring is not None, "train_on_batch needs a docstring."
    assert "FRAME:" in docstring, "Training tensors must document BEV frame."
    assert "nuScenes" in docstring and "KITTI" in docstring


# ANTI-VIBE GATE - train.py
#
# 1. COORDINATE FRAME CONTRACT
#    Training images are BEV tensors generated from world-frame nuScenes map
#    annotations. Evaluation is reserved for KITTI BEV tensors; train_on_batch
#    preserves tensor H/W alignment and does not alter frame metadata.
#
# 2. SILENT FAILURE MODE
#    If labels are shifted by one pixel, training loss can still decrease
#    while learned masks are spatially wrong. Alignment tests in preparation
#    cover this; the training loop only verifies optimization.
#
# 3. VECTORIZATION STRATEGY
#    PyTorch computes loss and gradients over full tensors. There is no Python
#    iteration over pixels or point arrays.
#
# 4. KNOWN LIMITATIONS
#    The unit test overfits a synthetic batch and does not prove real KITTI
#    generalization. Real cross-dataset IoU belongs in evaluation artifacts.
#
# 5. OBSERVABILITY CHECK
#    Correct training should produce masks that overlay BEV lane paint. A
#    decreasing loss with shifted masks indicates preprocessing mismatch, not
#    an optimizer issue.
