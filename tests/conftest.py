"""
Shared pytest fixtures for P4 test suite.

SCOPE DECISIONS:

  test_config: session — immutable dict, safe to share

  synthetic_model: FUNCTION — mutable object; session scope causes cascading

    failures when model-mutating tests abort before restoring state

  tmp_image_dir: function — creates real files, must be isolated per test

  image fixtures: function — PIL objects, lightweight to recreate

"""

import io

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session")
def test_config():
    """Minimal config matching training_config.yaml structure. Immutable — session scope safe."""
    return {
        "model": {
            "architecture": "efficientnet_b0",
            "pretrained": False,
            "num_classes": 2,
            "dropout": 0.3,
        },
        "data": {
            "image_size": 224,
            "train_val_test_split": [0.70, 0.15, 0.15],
            "num_workers": 0,
        },
        "training": {
            "batch_size": 4,
            "phase1_epochs": 1,
            "phase2_epochs": 1,
            "phase1_lr": 1e-3,
            "phase2_lr": 1e-4,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "patience": 2,
            "grad_clip": 1.0,
            "amp_enabled": False,
            "class_weights": [1.0, 1.0],
        },
        "evaluation": {
            "recall_threshold": 0.80,
            "precision_threshold": 0.60,
            "auc_threshold": 0.85,
            "fn_cost_weight": 5.0,
            "fp_cost_weight": 1.0,
            "brier_naive_baseline": 0.25,
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation_degrees": 15,
            "brightness": 0.2,
            "contrast": 0.2,
        },
    }


@pytest.fixture(scope="function")
def synthetic_model(test_config):
    """
    ChestXRayClassifier in eval mode, CPU, random weights.

    MUST be function-scoped (not session).

    Model-mutating tests (freeze_backbone, unfreeze_backbone) can fail mid-test,
    leaving the model in corrupted state if session-scoped. Function scope gives
    every test a clean, pristine instance at the cost of ~50ms instantiation.
    """
    from src.model import ChestXRayClassifier

    model = ChestXRayClassifier(test_config)
    model.to(torch.device("cpu")).eval()
    return model


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Temp directory with 30 synthetic PNG images. Cleaned up by pytest."""
    for i in range(30):
        arr = np.random.randint(30, 220, (224, 224, 3), dtype=np.uint8)
        arr[60:100, 60:100] = 200  # ensure variance > MIN_IMAGE_VARIANCE
        Image.fromarray(arr).save(tmp_path / f"img_{i:03d}.png")
    return tmp_path


@pytest.fixture
def synthetic_dataframe(tmp_image_dir):
    """30-row DataFrame with all required columns."""
    n = 30
    return pd.DataFrame(
        {
            "image_path": [str(tmp_image_dir / f"img_{i:03d}.png") for i in range(n)],
            "binary_label": [1 if i % 2 == 0 else 0 for i in range(n)],
            "Patient ID": list(range(n)),
            "Patient Age": [30 + i for i in range(n)],
            "Patient Gender": ["M" if i % 2 == 0 else "F" for i in range(n)],
            "View Position": ["AP" if i % 3 == 0 else "PA" for i in range(n)],
            "Finding Labels": ["Cardiomegaly" if i % 2 == 0 else "No Finding" for i in range(n)],
            "probability": np.random.uniform(0, 1, n),
            "predicted_label": [1 if i % 3 != 0 else 0 for i in range(n)],
            "model_confidence": np.random.uniform(0.5, 1.0, n),
            "conf_level": ["High" if i % 3 == 0 else "Moderate" for i in range(n)],
            "error_type": [
                "TP" if i % 4 == 0 else "TN" if i % 4 == 1 else "FP" if i % 4 == 2 else "FN"
                for i in range(n)
            ],
            "triage_tier": [
                "Tier1"
                if i % 4 == 0
                else "Tier2"
                if i % 4 == 1
                else "Normal"
                if i % 4 == 2
                else "Tier3"
                for i in range(n)
            ],
        }
    )


@pytest.fixture
def synthetic_image_pil():
    """224×224 RGB PIL Image with variance — passes validate_image."""
    arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    arr[50:100, 50:100] = 220
    arr[150:200, 150:200] = 30
    return Image.fromarray(arr)


@pytest.fixture
def synthetic_png_bytes(synthetic_image_pil):
    buf = io.BytesIO()
    synthetic_image_pil.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiny_png_bytes():
    """10×10 PNG — fails MIN_IMAGE_SIZE_PX in validate_image."""
    arr = np.full((10, 10, 3), 128, dtype=np.uint8)
    arr[3:7, 3:7] = 200
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def blank_png_bytes():
    """224×224 uniform PNG — fails MIN_IMAGE_VARIANCE in validate_image."""
    arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
