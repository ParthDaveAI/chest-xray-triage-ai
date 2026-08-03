"""
Unit tests for src/explain.py — L9 contracts.

Tests (4):

  U29: Grad-CAM output shape (224,224) and values in [0,1]

  U30: Hooks removed after compute_gradcam — no memory leak

  U31: AssertionError for batch_size > 1

  U32: Works with frozen model weights (requires_grad_(True) fix)

"""

import pytest

import torch

pytestmark = pytest.mark.unit


def test_u29_gradcam_shape_and_range(synthetic_model):
    """U29: Grad-CAM output (224,224) with values in [0,1]."""
    from src.explain import compute_gradcam

    dummy        = torch.randn(1, 3, 224, 224)
    target_layer = synthetic_model.backbone.features[-1]

    heatmap = compute_gradcam(synthetic_model, dummy, target_layer, 1, torch.device("cpu"))

    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_u30_hooks_removed_after_gradcam(synthetic_model):
    """U30: Forward and backward hooks removed after compute_gradcam — no memory leak."""
    from src.explain import compute_gradcam

    target_layer = synthetic_model.backbone.features[-1]

    compute_gradcam(synthetic_model, torch.randn(1, 3, 224, 224), target_layer, 1, torch.device("cpu"))

    assert len(target_layer._forward_hooks)  == 0, "Forward hook not removed"
    assert len(target_layer._backward_hooks) == 0, "Backward hook not removed"


def test_u31_batch_size_assertion(synthetic_model):
    """U31: AssertionError for batch_size > 1."""
    from src.explain import compute_gradcam

    with pytest.raises(AssertionError, match="batch_size=1"):
        compute_gradcam(synthetic_model, torch.randn(2, 3, 224, 224),
                        synthetic_model.backbone.features[-1], 1, torch.device("cpu"))


def test_u32_gradcam_frozen_weights(synthetic_model):
    """U32: compute_gradcam works with all weights frozen (requires_grad_(True) fix).

    If input tensor lacks requires_grad=True and all model weights are frozen,
    score.backward() raises RuntimeError. The fix sets requires_grad_(True) on
    the input tensor before the forward pass.
    """
    from src.explain import compute_gradcam

    # Freeze all weights
    for p in synthetic_model.parameters():
        p.requires_grad = False

    try:
        heatmap = compute_gradcam(
            synthetic_model, torch.randn(1, 3, 224, 224),
            synthetic_model.backbone.features[-1], 1, torch.device("cpu")
        )
        assert heatmap.shape == (224, 224)
    finally:
        # Restore for other tests
        for p in synthetic_model.parameters():
            p.requires_grad = True