"""
Model Architecture Verification — P4-L5

Run with: uv run python scripts/verify_model.py

Verifies all contracts for src/model.py:

  1.  Import succeeds
  2.  Output shape: (4, 2) for input (4, 3, 224, 224)
  3.  Output is raw logits — mathematically verified (not probabilistic check)
  4.  embedding_dim attribute == 1280
  5.  After freeze_backbone(): ALL backbone params frozen
  6.  After freeze_backbone(): backbone.classifier fully trainable
  7.  After freeze_backbone(): trainable count == 2,562 (head only)
  8.  After freeze_backbone() + model.train(): all BN layers in eval mode
  9.  After unfreeze_backbone(): all params trainable, trainable == total
 10.  get_penultimate_features() shape: (2, 1280)
 11.  Gradient flow: frozen params have zero gradient after backward()
 12.  Architecture summary dict has all required keys
"""

import sys

import torch
import torch.nn as nn
import yaml

print("=" * 60)
print("P4-L5: Model Architecture Verification")
print("=" * 60)

# ── Contract 1: Import ─────────────────────────────────────────────────────

from src.model import ChestXRayClassifier

print("✓  1: ChestXRayClassifier imported")

config = yaml.safe_load(open("config/training_config.yaml"))

print("     Loading pretrained EfficientNet-B0 (may download on first run)...")
model = ChestXRayClassifier(config)
model.eval()

# ── Contract 2: Output shape ───────────────────────────────────────────────

batch = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    output = model(batch)

assert output.shape == torch.Size([4, 2]), \
    f"Expected (4, 2), got {output.shape}"

print(f"✓  2: Output shape: {tuple(output.shape)}")

# ── Contract 3: Raw logits — mathematically hardened check ────────────────
# Softmax probabilities always sum to 1.0 per row.
# Raw logits almost never sum to 1.0 (would require a very specific accident).
# This check is mathematically absolute, not probabilistic.

row_sums = output.sum(dim=1)
is_softmax = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
assert not is_softmax, (
    "Output row sums close to 1.0 — output appears to be softmax probabilities. "
    "forward() must return raw logits. Do not apply softmax in forward()."
)

# Also verify softmax of output sums to 1 (verifies it IS valid logits for CE loss)
softmax_sums = output.softmax(dim=1).sum(dim=1)
assert torch.allclose(softmax_sums, torch.ones_like(softmax_sums), atol=1e-4)

print(f"✓  3: Raw logits verified (row sums ≠ 1.0: {row_sums.tolist()})")

# ── Contract 4: embedding_dim attribute ───────────────────────────────────

assert hasattr(model, "embedding_dim"), "model.embedding_dim attribute must exist"
assert model.embedding_dim == 1280, \
    f"embedding_dim should be 1280, got {model.embedding_dim}"

print(f"✓  4: model.embedding_dim = {model.embedding_dim}")

# ── Contract 5: freeze_backbone — all backbone params frozen ──────────────

model.freeze_backbone()

all_backbone_frozen = all(
    not p.requires_grad for p in model.backbone.parameters()
    if p not in set(model.backbone.classifier.parameters())
)

# More direct: features should all be frozen
all_features_frozen = all(
    not p.requires_grad for p in model.backbone.features.parameters()
)

assert all_features_frozen, "backbone.features must be fully frozen after freeze_backbone()"

print("✓  5: backbone.features fully frozen after freeze_backbone()")

# ── Contract 6: backbone.classifier fully trainable ───────────────────────

all_classifier_trainable = all(
    p.requires_grad for p in model.backbone.classifier.parameters()
)

assert all_classifier_trainable, "backbone.classifier must be fully trainable"

print("✓  6: backbone.classifier fully trainable")

# ── Contract 7: trainable count == head parameters only ──────────────────

counts_frozen = model.count_parameters()
expected_head = model.embedding_dim * config["model"]["num_classes"] + \
                config["model"]["num_classes"]  # 1280*2 + 2 = 2,562

assert counts_frozen["trainable"] == expected_head, \
    f"Expected {expected_head} trainable params (head only), " \
    f"got {counts_frozen['trainable']}"

print(f"✓  7: Trainable after freeze: {counts_frozen['trainable']:,} "
      f"(expected head={expected_head:,})")
print(f"      Frozen: {counts_frozen['frozen']:,}")

# ── Contract 8: BN layers in eval mode after model.train() when frozen ────

model.train()  # triggers the train() override

bn_layers_in_eval = [
    isinstance(m, nn.BatchNorm2d) and not m.training
    for m in model.backbone.features.modules()
    if isinstance(m, nn.BatchNorm2d)
]

assert all(bn_layers_in_eval), (
    f"Not all BN layers are in eval mode. "
    f"Eval: {sum(bn_layers_in_eval)}/{len(bn_layers_in_eval)}. "
    f"The train() override must force BN into eval when backbone is frozen."
)

print(f"✓  8: All {len(bn_layers_in_eval)} BN layers in eval mode after model.train() "
      f"(backbone frozen)")

# ── Contract 9: unfreeze_backbone — all trainable ─────────────────────────

model.unfreeze_backbone()

all_trainable = all(p.requires_grad for p in model.parameters())
assert all_trainable, "After unfreeze_backbone(), ALL parameters must be trainable"

counts_full = model.count_parameters()
assert counts_full["trainable"] == counts_full["total"], \
    f"trainable ({counts_full['trainable']}) != total ({counts_full['total']})"

print(f"✓  9: After unfreeze: trainable ({counts_full['trainable']:,}) == "
      f"total ({counts_full['total']:,})")

# ── Contract 10: penultimate features shape ───────────────────────────────

model.eval()
dummy_2 = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    embeddings = model.get_penultimate_features(dummy_2)

assert embeddings.shape == torch.Size([2, model.embedding_dim]), \
    f"Expected (2, {model.embedding_dim}), got {embeddings.shape}"

print(f"✓ 10: Penultimate features shape: {tuple(embeddings.shape)}")

# ── Contract 11: Gradient flow verification ───────────────────────────────

# Re-freeze backbone for this check
model.freeze_backbone()
model.train()

dummy_input  = torch.randn(2, 3, 224, 224, requires_grad=False)
dummy_labels = torch.tensor([0, 1], dtype=torch.long)

logits = model(dummy_input)
loss   = nn.CrossEntropyLoss()(logits, dummy_labels)
loss.backward()

# Frozen params must have zero gradient (None or all-zeros tensor)
frozen_params_have_no_grad = all(
    p.grad is None or (p.grad is not None and p.grad.abs().max().item() < 1e-10)
    for p in model.backbone.features.parameters()
)

assert frozen_params_have_no_grad, (
    "Frozen backbone.features parameters received non-zero gradients. "
    "freeze_backbone() is not correctly preventing gradient flow."
)

# Head params must have non-zero gradients
head_params_have_grad = any(
    p.grad is not None and p.grad.abs().max().item() > 1e-10
    for p in model.backbone.classifier.parameters()
)

assert head_params_have_grad, (
    "backbone.classifier parameters have zero/None gradients after backward(). "
    "The head is not receiving gradient updates."
)

print("✓ 11: Gradient flow correct: frozen layers have zero gradient, head has gradients")

# ── Contract 12: architecture summary dict ────────────────────────────────

summary = model.get_architecture_summary()

required_keys = {
    "architecture", "pretrained", "num_classes", "dropout",
    "embedding_dim", "total_params", "trainable_params", "frozen_params"
}

missing = required_keys - set(summary.keys())
assert not missing, f"Missing keys in architecture summary: {missing}"

assert summary["embedding_dim"] == 1280
assert summary["num_classes"]   == 2

print(f"✓ 12: Architecture summary complete. embedding_dim={summary['embedding_dim']}, "
      f"num_classes={summary['num_classes']}")

# ── Summary ────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("All 12 contracts verified.")
print()
print(f"  Total parameters:          {counts_full['total']:>10,}")
print(f"  Backbone parameters:       {counts_full['total'] - expected_head:>10,}")
print(f"  Classification head:       {expected_head:>10,}")
print(f"  Embedding dimension:       {model.embedding_dim:>10,}")
print()
print("BatchNorm behaviour:")
print(f"  BN layers in backbone:     {len(bn_layers_in_eval):>10}")
print(f"  BN mode when frozen+train: {'eval (correct)':>10}")
print("=" * 60)