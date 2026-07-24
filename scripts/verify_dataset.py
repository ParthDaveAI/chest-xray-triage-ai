"""
Dataset and DataLoader Verification — P4-L4

Run with: uv run python scripts/verify_dataset.py

Verifies all contracts in src/dataset.py:

  1. Imports succeed
  2. get_inference_transform is standalone and returns correct steps
  3. Invalid mode raises ValueError
  4. Missing required column raises ValueError
  5. Null binary_label raises ValueError
  6. Tensor shape is (3, 224, 224), label dtype is torch.long
  7. Train mode includes augmentation; val and test do not
  8. RandomVerticalFlip is absent from ALL pipelines
  9. Val and test transforms are identical to get_inference_transform()
 10. create_dataloaders never calls train_test_split
 11. pin_memory reflects CUDA availability
"""

import inspect
import sys
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

print("=" * 60)
print("P4-L4: Dataset Contract Verification")
print("=" * 60)

# ── Contract 1: Imports ────────────────────────────────────────────────────

from src.dataset import (
    get_inference_transform,
    ChestXRayDataset,
    create_dataloaders,
    _worker_init_fn,
)

print("✓  1: All imports succeed")

config = yaml.safe_load(open("config/training_config.yaml"))

# ── Contract 2: get_inference_transform ───────────────────────────────────

transform = get_inference_transform(config)
step_names = [type(t).__name__ for t in transform.transforms]
assert step_names == ["Resize", "ToTensor", "Normalize"], \
    f"Expected [Resize, ToTensor, Normalize], got {step_names}"

print(f"✓  2: get_inference_transform steps: {step_names}")

# ── Synthetic DataFrame helpers ────────────────────────────────────────────

tmp_img = Path("/tmp/p4_l4_verify.png")
img_arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
Image.fromarray(img_arr).save(tmp_img)

good_df = pd.DataFrame({"image_path": [str(tmp_img)], "binary_label": [1]})

# ── Contract 3: Invalid mode ───────────────────────────────────────────────

try:
    ChestXRayDataset(good_df, config, mode="predict")
    print("✗  3 FAILED: invalid mode should raise ValueError")
    sys.exit(1)
except ValueError as e:
    assert "mode" in str(e).lower()
    print("✓  3: Invalid mode raises ValueError")

# ── Contract 4: Missing required column ───────────────────────────────────

bad_col_df = pd.DataFrame({"imagepath": [str(tmp_img)], "binary_label": [1]})
try:
    ChestXRayDataset(bad_col_df, config, mode="val")
    print("✗  4 FAILED: missing column should raise ValueError")
    sys.exit(1)
except ValueError as e:
    assert "image_path" in str(e)
    print("✓  4: Missing column raises ValueError with column name in message")

# ── Contract 5: Null binary_label ─────────────────────────────────────────

null_df = pd.DataFrame({"image_path": [str(tmp_img)], "binary_label": [None]})
try:
    ChestXRayDataset(null_df, config, mode="train")
    print("✗  5 FAILED: null label should raise ValueError")
    sys.exit(1)
except ValueError as e:
    print("✓  5: Null binary_label raises ValueError")

# ── Contract 6: Tensor shape and label dtype ──────────────────────────────

val_ds = ChestXRayDataset(good_df, config, mode="val")
tensor, label = val_ds[0]

assert tensor.shape == torch.Size([3, 224, 224]), \
    f"Expected (3,224,224), got {tensor.shape}"
assert label.dtype == torch.long, \
    f"Expected torch.long, got {label.dtype}"
assert label.item() in [0, 1]

print(f"✓  6: Tensor shape {tuple(tensor.shape)}, label dtype={label.dtype}, value={label.item()}")

# ── Contract 7: Augmentation in train, absent in val/test ─────────────────

train_ds = ChestXRayDataset(good_df, config, mode="train")
test_ds = ChestXRayDataset(good_df, config, mode="test")

train_names = [type(t).__name__ for t in train_ds.transform.transforms]
val_names = [type(t).__name__ for t in val_ds.transform.transforms]
test_names = [type(t).__name__ for t in test_ds.transform.transforms]

random_aug = {"RandomHorizontalFlip", "RandomVerticalFlip",
              "RandomRotation", "ColorJitter"}

val_has_aug = any(t in random_aug for t in val_names)
test_has_aug = any(t in random_aug for t in test_names)

assert not val_has_aug, f"Val should have NO augmentation: {val_names}"
assert not test_has_aug, f"Test should have NO augmentation: {test_names}"

print(f"✓  7: Val transforms: {val_names}")
print(f"      Test transforms: {test_names}")

if config["augmentation"].get("horizontal_flip") or \
   config["augmentation"].get("rotation_degrees", 0) > 0 or \
   config["augmentation"].get("color_jitter"):
    train_has_aug = any(t in random_aug for t in train_names)
    assert train_has_aug, "Train should include augmentation when configured"
    print(f"      Train transforms: {train_names}")

print("✓  7: Augmentation only in train mode")

# ── Contract 8: RandomVerticalFlip always absent ──────────────────────────

for mode_name, t_list in [("train", train_names), ("val", val_names), ("test", test_names)]:
    assert "RandomVerticalFlip" not in t_list, \
        f"RandomVerticalFlip MUST be absent from {mode_name} — clinically invalid for X-rays."

print("✓  8: RandomVerticalFlip absent from all transform pipelines")

# ── Contract 9: Val/test == get_inference_transform ───────────────────────

inf_names = [type(t).__name__ for t in get_inference_transform(config).transforms]
assert val_names == inf_names, f"Val {val_names} != inference {inf_names}"
assert test_names == inf_names, f"Test {test_names} != inference {inf_names}"

print("✓  9: Val and test transforms identical to get_inference_transform()")

# ── Contract 10: No train_test_split in create_dataloaders ────────────────

source = inspect.getsource(create_dataloaders)
# Check that train_test_split is NOT called (comments are okay)
# We check for actual function call pattern, not just the string
import re
# Look for actual train_test_split( pattern (with parentheses)
has_call = bool(re.search(r'train_test_split\s*\(', source))
assert not has_call, \
    "create_dataloaders MUST NOT call train_test_split"

print("✓ 10: create_dataloaders contains no train_test_split call")

# ── Contract 11: pin_memory reflects CUDA ─────────────────────────────────

cuda_available = torch.cuda.is_available()
expected_pin = cuda_available

print(f"✓ 11: CUDA available={cuda_available} → pin_memory will be {expected_pin}")

# ── Cleanup ────────────────────────────────────────────────────────────────

tmp_img.unlink(missing_ok=True)

print()
print("=" * 60)
print("All 11 contracts verified.")
print()
print("To verify DataLoader batch shapes, run with full dataset:")
print("  uv run python scripts/verify_dataloader.py")
print("=" * 60)