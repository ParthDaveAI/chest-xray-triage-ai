"""
Numerical stability and determinism contracts.

Tests (4):

  U37: Softmax output contains no NaN or Inf for standard inputs

  U38: Softmax output is in [0,1] and sums to 1

  U39: Inference is deterministic in eval mode (same input → same output)

  U40: Training/serving transform produces identical tensor values (skew prevention)

"""

import pytest
import torch
from pytest import approx

pytestmark = pytest.mark.unit


def test_u37_no_nan_inf_in_output(synthetic_model):
    """U37: Softmax output contains no NaN or Inf for valid and edge-case inputs."""
    inputs = [
        torch.zeros(1, 3, 224, 224),  # all-zero tensor
        torch.ones(1, 3, 224, 224),  # all-one tensor
        torch.randn(1, 3, 224, 224),  # random Gaussian
        torch.randn(1, 3, 224, 224) * 10,  # large magnitude
    ]

    for tensor in inputs:
        with torch.no_grad():
            logits = synthetic_model(tensor)
            probs = torch.softmax(logits, dim=1)

        assert not torch.isnan(probs).any(), (
            f"NaN in softmax output for input with max={tensor.max():.2f}"
        )
        assert not torch.isinf(probs).any(), "Inf in softmax output"


def test_u38_softmax_sums_to_one(synthetic_model):
    """U38: Softmax probabilities are in [0,1] and sum to 1."""
    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        logits = synthetic_model(dummy)
        probs = torch.softmax(logits, dim=1)

    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0
    assert probs.sum(dim=1).item() == approx(1.0, abs=1e-6)


def test_u39_inference_deterministic(synthetic_model):
    """U39: Repeated inference on the same input produces identical output.

    Non-determinism in eval mode indicates: dropout is active, BatchNorm is
    in train mode, or a non-deterministic CUDA op is in the graph.
    All are bugs — eval mode must be fully deterministic on CPU.
    """
    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out1 = synthetic_model(dummy)
        out2 = synthetic_model(dummy)
        out3 = synthetic_model(dummy)

    assert torch.equal(out1, out2), "Inference not deterministic: run 1 vs run 2 differ"
    assert torch.equal(out2, out3), "Inference not deterministic: run 2 vs run 3 differ"


def test_u40_training_serving_transform_identical(test_config, synthetic_image_pil):
    """U40: dataset.get_inference_transform and serve.get_inference_transform
    are the SAME object and produce identical tensor values.

    This is the training/serving skew prevention invariant from L11.
    If serve.py redefines its own transform, it can silently diverge from
    the evaluation pipeline — test metrics become invalid in production.
    """
    import src.serve as serve_module
    from src.dataset import get_inference_transform as dataset_fn

    # Contract 1: same function object (imported, not redefined)
    assert serve_module.get_inference_transform is dataset_fn, (
        "serve.py must import get_inference_transform from src.dataset, not redefine it. "
        "Redefining creates training/serving skew risk."
    )

    # Contract 2: same input produces identical tensor values
    transform = dataset_fn(test_config)
    tensor_a = transform(synthetic_image_pil)
    tensor_b = transform(synthetic_image_pil)

    assert torch.equal(tensor_a, tensor_b), (
        "Same image through same transform must produce identical tensors. "
        "Non-determinism in preprocessing indicates a bug."
    )
