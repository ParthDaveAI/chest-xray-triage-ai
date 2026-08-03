"""
Unit tests for src/dataset.py — L4 contracts.

Tests (5):

  U4: get_inference_transform returns callable

  U5: ChestXRayDataset __len__ matches DataFrame rows

  U6: ChestXRayDataset __getitem__ returns (float32 tensor (3,224,224), int)

  U7: Train mode: missing image skips (returns next valid item)

  U8: Val mode: missing image raises RuntimeError

"""

import pytest

import torch

pytestmark = pytest.mark.unit


def test_u4_inference_transform_callable(test_config):
    """U4: get_inference_transform returns a callable."""
    from src.dataset import get_inference_transform

    assert callable(get_inference_transform(test_config))


def test_u5_dataset_len(synthetic_dataframe, test_config):
    """U5: Dataset length matches DataFrame rows."""
    from src.dataset import ChestXRayDataset

    ds = ChestXRayDataset(synthetic_dataframe, test_config, mode="val")
    assert len(ds) == len(synthetic_dataframe)


def test_u6_dataset_getitem_shape(synthetic_dataframe, test_config):
    """U6: __getitem__ returns (float32 tensor (3,224,224), binary int label)."""
    from src.dataset import ChestXRayDataset

    ds = ChestXRayDataset(synthetic_dataframe, test_config, mode="val")
    img, label = ds[0]

    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert label in [0, 1]


def test_u7_train_mode_missing_image_skips(synthetic_dataframe, test_config, tmp_path):
    """U7: Train mode skips missing images — returns next valid item."""
    import pandas as pd
    from src.dataset import ChestXRayDataset

    bad_df = synthetic_dataframe.copy()
    bad_df.loc[0, "image_path"] = str(tmp_path / "nonexistent.png")

    ds = ChestXRayDataset(bad_df, test_config, mode="train", missing_file_threshold=0.1)
    img, _ = ds[0]

    assert img.shape == (3, 224, 224)


def test_u8_val_mode_missing_image_raises(synthetic_dataframe, test_config, tmp_path):
    """U8: Val/test mode raises RuntimeError for missing images."""
    import pandas as pd
    from src.dataset import ChestXRayDataset

    bad_df = synthetic_dataframe.copy()
    bad_df.loc[0, "image_path"] = str(tmp_path / "nonexistent.png")

    with pytest.raises(RuntimeError):
        ds = ChestXRayDataset(bad_df, test_config, mode="val", missing_file_threshold=0.01)
        _ = ds[0]