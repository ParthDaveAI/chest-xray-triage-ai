"""
Unit tests for src/model.py — L5 contracts.

Tests (5):

  U9:  Forward pass returns (1,2) logits

  U10: freeze_backbone sets backbone params to requires_grad=False

  U11: unfreeze_backbone restores backbone requires_grad=True

  U12: BN layers remain eval after freeze_backbone + model.train() — critical invariant

  U13: Classifier head stays trainable after freeze_backbone

"""

import pytest

import torch

import torch.nn as nn

pytestmark = pytest.mark.unit


def test_u9_forward_shape(synthetic_model):
    """U9: Forward pass on (1,3,224,224) returns (1,2) logits."""
    out = synthetic_model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 2)


def test_u10_freeze_backbone(synthetic_model):
    """U10: freeze_backbone sets all backbone feature params to requires_grad=False."""
    synthetic_model.freeze_backbone()

    params = list(synthetic_model.backbone.features.parameters())
    assert all(not p.requires_grad for p in params)


def test_u11_unfreeze_backbone(synthetic_model):
    """U11: unfreeze_backbone restores backbone requires_grad=True."""
    synthetic_model.freeze_backbone()
    synthetic_model.unfreeze_backbone()

    params = list(synthetic_model.backbone.features.parameters())
    assert any(p.requires_grad for p in params)


def test_u12_bn_stays_eval_after_freeze(synthetic_model):
    """U12: BatchNorm layers remain in eval mode after freeze_backbone + model.train().

    CRITICAL INVARIANT (L5 Decision 10):

    requires_grad=False does NOT prevent BatchNorm from updating running statistics
    when called in train() mode. freeze_backbone must override train() for BN layers.

    If this invariant breaks, BatchNorm will update statistics with training data
    patterns during inference, producing inconsistent predictions.
    """
    synthetic_model.freeze_backbone()
    synthetic_model.train()   # would normally set all BN to train mode

    bn_layers = [m for m in synthetic_model.backbone.features.modules()
                 if isinstance(m, nn.BatchNorm2d)]

    assert len(bn_layers) > 0, "EfficientNet-B0 must have BatchNorm2d layers"

    for bn in bn_layers:
        assert not bn.training, \
            "BatchNorm must remain in eval mode after freeze_backbone() + train(). " \
            "requires_grad=False alone does not prevent BN from updating running stats."


def test_u13_classifier_unfrozen_after_freeze(synthetic_model):
    """U13: Classifier head remains trainable after freeze_backbone."""
    synthetic_model.freeze_backbone()

    params = list(synthetic_model.backbone.classifier.parameters())
    assert any(p.requires_grad for p in params)