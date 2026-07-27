"""
EfficientNet-B0 model architecture for P4 Radiology AI.

Key design decisions implemented here:

  1. Production-safe freeze pattern: freeze ALL backbone params, then re-enable classifier.
     Naive pattern (freeze only backbone.features) misses future architecture changes.

  2. BatchNorm override: train() is overridden to keep BN layers in eval() mode
     when the backbone is frozen. requires_grad=False does NOT stop BN running
     statistics from updating. Without this override, frozen conv weights calibrated
     to ImageNet BN stats receive activations normalised by X-ray BN stats — mismatch
     that silently degrades Phase 1 accuracy.

  3. Explicit head initialisation: Xavier uniform weights, zero biases.
     Removes one source of training run variance, improves early convergence.

  4. self.embedding_dim: explicit attribute for P5 contract checking.

  5. Raw logits from forward(): CrossEntropyLoss expects logits.
     Apply torch.softmax in evaluate.py and serve.py when probabilities needed.

  6. Device-agnostic: caller is responsible for model.to(device) and batch.to(device).

See decisions.md Decision 9 (architecture) and Decision 10 (head replacement).
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# Expected embedding dimension for EfficientNet-B0's penultimate layer.
# Used as a constant so P5 can validate the contract without querying the model.
EFFICIENTNET_B0_EMBEDDING_DIM = 1280


class ChestXRayClassifier(nn.Module):
    """
    EfficientNet-B0 adapted for binary chest X-ray classification.

    Two-phase training design:
      Phase 1: backbone frozen (BN in eval mode) → train head only (phase1_lr)
      Phase 2: full network unfrozen → fine-tune everything (phase2_lr = phase1_lr/10)

    BatchNorm behaviour during freezing:
      requires_grad=False stops weight/bias updates but NOT running_mean/running_var.
      The overridden train() method forces BN layers in backbone.features to eval()
      mode when the backbone is frozen. This preserves pretrained ImageNet BN stats.

    Optimizer momentum on Phase 1→2 transition:
      L6 creates a NEW optimizer for Phase 2, discarding Phase 1 momentum buffers.
      This is deliberate: head and backbone should start Phase 2 with equal momentum,
      not asymmetric (head with built-up momentum, backbone with zero).

    Device contract:
      This class is device-agnostic. Caller must call:
        model.to(device)
        images.to(device), labels.to(device)

    Inference input contract:
      forward() accepts float32 tensors normalised to ImageNet statistics,
      shape (batch_size, 3, 224, 224). Raw pixel values [0,255] or uint8
      tensors will produce incorrect outputs without error.

    Args:
        config: loaded training_config.yaml dict.
            Required: model.num_classes, model.dropout, model.pretrained
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        num_classes  = config["model"]["num_classes"]   # 2
        dropout_rate = config["model"]["dropout"]       # 0.3
        pretrained   = config["model"]["pretrained"]    # True

        # ── Load pretrained backbone ───────────────────────────────────────────
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
            logger.warning(
                "pretrained=False: EfficientNet-B0 loaded with random initialisation. "
                "Transfer learning does not apply. Use only for debugging or ablation studies."
            )

        self.backbone = models.efficientnet_b0(weights=weights)

        # Store embedding dimension as explicit attribute.
        # P5 monitoring reads this to validate the contract:
        #   assert model.embedding_dim == 1280
        in_features = self.backbone.classifier[1].in_features
        self.embedding_dim = in_features  # 1280 for EfficientNet-B0

        # ── Replace classification head ────────────────────────────────────────
        # Original: Sequential(Dropout(0.2), Linear(1280, 1000))
        # Ours:     Sequential(Dropout(0.3), Linear(1280, 2))
        #
        # dropout_rate is 0.3 (vs original 0.2) because:
        # - Original model trained on 1.2M diverse ImageNet images
        # - Our head trains on ~78K chest X-rays
        # - Less data → more overfitting risk → higher dropout

        linear_head = nn.Linear(in_features, num_classes)

        # ── Explicit head weight initialisation ───────────────────────────────
        # Xavier uniform: designed for layers with sigmoid-like activations.
        # Produces symmetric initial output distribution, no class bias.
        # Zero bias: initial logit determined by weights only — no systematic bias.
        # Improves reproducibility and early convergence over PyTorch defaults.
        nn.init.xavier_uniform_(linear_head.weight)
        nn.init.zeros_(linear_head.bias)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            linear_head,
        )

        counts = self.count_parameters()
        logger.info(
            "ChestXRayClassifier initialised:\n"
            "  Backbone:      EfficientNet-B0 (pretrained=%s)\n"
            "  Embedding dim: %d\n"
            "  Head:          Dropout(%.1f) + Linear(%d, %d) [Xavier init]\n"
            "  Total params:  %s",
            pretrained, self.embedding_dim,
            dropout_rate, in_features, num_classes,
            f"{counts['total']:,}",
        )

        # Track whether backbone is currently frozen.
        # Used by train() to decide whether to force BN eval mode.
        self._backbone_frozen = False

    # ── BatchNorm Override ─────────────────────────────────────────────────────

    def train(self, mode: bool = True) -> "ChestXRayClassifier":
        """
        Override nn.Module.train() to correctly handle BatchNorm during frozen phase.

        THE PROBLEM:
        requires_grad=False stops weight/bias updates for frozen parameters.
        It does NOT stop BatchNorm2d from updating running_mean and running_var.

        During Phase 1, when model.train() is called by the training loop,
        BN layers inside backbone.features would normally update their running
        statistics from each chest X-ray batch. But the frozen conv weights
        were calibrated to ImageNet BN statistics. Allowing BN stats to drift
        toward X-ray statistics while conv weights remain frozen produces a
        mismatch that silently degrades Phase 1 accuracy.

        THE FIX:
        When the backbone is frozen (self._backbone_frozen=True) and we are
        switching to training mode, force all BatchNorm2d layers inside
        backbone.features back to eval() mode. Their running_mean and
        running_var will not update. The pretrained ImageNet statistics are
        preserved. Correctness is restored.

        This override is transparent: the model reports as training (model.training=True),
        the head and dropout operate in training mode, but backbone BN stats are frozen.
        """
        super().train(mode)

        if mode and self._backbone_frozen:
            for module in self.backbone.features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        return self

    # ── Phase Control ──────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Phase 1: freeze all backbone parameters (features + avgpool).

        Production-safe pattern:
          Step 1: freeze ALL backbone parameters
          Step 2: selectively re-enable backbone.classifier

        This pattern is safer than freezing only backbone.features because
        it handles any future architecture change to backbone.avgpool or other
        components automatically. All backbone components are frozen by default;
        the classifier is the explicit exception.

        BatchNorm note:
          After calling this method, call model.train() to activate the
          train() override, which will force BN layers in backbone.features
          into eval mode. The sequence in L6 train_epoch():
            model.train()  ← triggers override, BN stays in eval
            ... training loop ...

        Trainable after freeze: only backbone.classifier
          = Linear(1280, 2) weights + biases = 1280*2 + 2 = 2,562 parameters
        """
        # Step 1: freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Step 2: selectively re-enable classifier
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        self._backbone_frozen = True

        counts = self.count_parameters()

        # Verify: trainable should equal head parameters only
        expected_head_params = self.embedding_dim * self.config["model"]["num_classes"] + \
                               self.config["model"]["num_classes"]

        if counts["trainable"] != expected_head_params:
            logger.warning(
                "After freeze_backbone(), trainable params = %d but expected %d (head only). "
                "Investigate if backbone has additional trainable components.",
                counts["trainable"], expected_head_params,
            )

        logger.info(
            "Phase 1 — Backbone frozen:\n"
            "  Frozen params:    %s\n"
            "  Trainable params: %s (head only)\n"
            "  BatchNorm: will be kept in eval mode via train() override",
            f"{counts['frozen']:,}", f"{counts['trainable']:,}",
        )

    def unfreeze_backbone(self) -> None:
        """
        Phase 2: unfreeze all parameters for full fine-tuning.

        CRITICAL: Use phase2_lr = phase1_lr / 10 = 0.0001 in L6.
        Large learning rates in Phase 2 produce gradient updates that
        overwrite pretrained backbone weights — catastrophic forgetting.

        MOMENTUM RESET (documented deliberate choice):
        L6 creates a NEW Adam optimizer for Phase 2, discarding momentum
        buffers from Phase 1. This is correct:
          - Head had 10 epochs of accumulated momentum
          - Backbone parameters had zero momentum (frozen)
          - Reusing head's old momentum with a newly unfrozen backbone
            creates asymmetric updates that destabilise early Phase 2
          - New optimizer gives all parameters equal starting momentum (zero)

        BatchNorm note:
        After calling unfreeze_backbone(), the train() override no longer
        forces BN into eval mode. BN running statistics update normally
        during Phase 2 — correct, because backbone weights are now being
        refined and BN stats should adapt alongside them.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        self._backbone_frozen = False

        counts = self.count_parameters()

        logger.info(
            "Phase 2 — Full fine-tuning:\n"
            "  All %s parameters unfrozen.\n"
            "  Use phase2_lr=%.4f (10x lower than phase1_lr) to prevent forgetting.\n"
            "  Create NEW optimizer in L6 to reset momentum buffers.",
            f"{counts['total']:,}",
            self.config["training"]["phase2_lr"],
        )

    # ── Feature Extraction ─────────────────────────────────────────────────────

    def get_penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate layer embedding before the classifier head.

        Used by P5 drift monitoring to detect input distribution shift.
        P5 computes the distribution of these embeddings on the training set
        and monitors for statistical deviation at inference time.

        WHY EMBEDDINGS NOT PROBABILITIES:
        Output probability shift is a lagging indicator — detectable only after
        performance has already degraded. Embedding shift is a leading indicator —
        detects that input data has changed before predictions are affected.

        EfficientNet-B0 architecture path:
          backbone.features(x)  →  (batch, 1280, 7, 7)  conv feature maps
          backbone.avgpool       →  (batch, 1280, 1, 1)  global average pooling
          torch.flatten(_, 1)    →  (batch, 1280)         1D embedding

        Output shape: (batch_size, self.embedding_dim) = (batch_size, 1280)
        P5 contract: assert model.embedding_dim == 1280

        Args:
            x: float32 tensor (batch_size, 3, 224, 224), ImageNet-normalised

        Returns:
            embedding (batch_size, 1280) float32 — not normalised
        """
        features = self.backbone.features(x)      # (B, 1280, 7, 7)
        pooled   = self.backbone.avgpool(features) # (B, 1280, 1, 1)
        return torch.flatten(pooled, 1)            # (B, 1280)

    # ── Forward Pass ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: backbone + classifier head.

        Returns RAW LOGITS — not softmax probabilities.

        WHY RAW LOGITS (not softmax):
        nn.CrossEntropyLoss (used in L6) applies log_softmax internally.
        Passing softmax probabilities creates a double-application:
          log(softmax(logits)) instead of log_softmax(logits)
        These are mathematically different. The model trains without error
        but optimises the wrong objective. This is a silent bug.

        To get Suspicious probability in evaluate.py and serve.py:
          probs = torch.softmax(output, dim=1)[:, 1]

        Input contract:
          - dtype: float32
          - values: ImageNet-normalised (~[-2.1, 2.6])
          - shape: (batch_size, 3, 224, 224)

        Args:
            x: float32 tensor (batch_size, 3, 224, 224), ImageNet-normalised

        Returns:
            logits (batch_size, 2) float32 — can be negative, not in [0,1]
        """
        return self.backbone(x)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """
        Count total, trainable, and frozen parameters.

        Called:
          - After freeze_backbone() to verify correct frozen parameter count
          - After unfreeze_backbone() to verify all params are trainable
          - In train.py (L6) to log to MLflow

        Returns:
            dict: {total, trainable, frozen}
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable

        return {"total": total, "trainable": trainable, "frozen": frozen}

    def get_architecture_summary(self) -> dict:
        """
        Return architecture details dict for MLflow logging in L6.

        Every training run logs this to MLflow params so the architecture
        is part of the experiment's reproducibility record.
        """
        counts = self.count_parameters()

        return {
            "architecture":     self.config["model"]["architecture"],
            "pretrained":       self.config["model"]["pretrained"],
            "num_classes":      self.config["model"]["num_classes"],
            "dropout":          self.config["model"]["dropout"],
            "embedding_dim":    self.embedding_dim,
            "total_params":     counts["total"],
            "trainable_params": counts["trainable"],
            "frozen_params":    counts["frozen"],
        }