# Decision Log — P4 Radiology AI Image Pipeline

## 60 Seconds Academy — AI & ML

This log continues decisions initiated in the L0 design canvas.

The canvas documented the strategic choices. This log documents

implementation-level choices as the project develops.

New entries are added each lecture. All 17 entries are present by L15.

---

## Decision 1: Binary Classification over Multi-Class

**Date:** July 20, 2026

**Decision:** Binary Normal vs Suspicious from NIH ChestX-ray14.

**Context:** Dataset has 14 disease labels plus "No Finding."

**Alternatives:** 14-class multi-label classification.

**Why rejected:** Severe class imbalance (some diseases < 1% prevalence) makes

stable training very difficult. Defending clinical nuances of all 14 conditions

in an interview requires deep domain expertise beyond this project's scope.

**Trade-off:** Loss of disease-specific signal.

**Gain:** Tractable class balance (~54/46), clear clinical framing as a screening

tool, same core transfer learning skills demonstrated.

---

## Decision 2: Online Inference over Batch

**Date:** July 20, 2026

**Decision:** FastAPI real-time endpoint, one prediction per image upload.

**Context:** Radiologists need immediate triage results when images are produced.

**Alternatives:** Nightly or hourly batch processing.

**Why rejected:** Batch predictions arrive hours after the decision window closes.

The clinical workflow is synchronous — radiologist and patient are both present

at study time.

**Trade-off:** Higher per-request infrastructure cost vs batch.

**Gain:** Direct integration with clinical workflow.

---

## Decision 3: DVC Remote — Google Drive for Development, S3 for Production

**Date:** July 20, 2026

**Decision:** Google Drive as DVC remote for portfolio/development use.

**Context:** NIH ChestX-ray14 is ~42GB and cannot be committed to Git.
A remote storage backend is required for dvc push and dvc pull.

**Alternatives:**

- No remote (local only): cannot collaborate or run in CI. Rejected.

- S3 with IAM roles: PHI isolation, audit logging, enterprise-grade.
  Correct production choice. Requires paid AWS account.

**Portfolio decision:** Google Drive — free, supported by dvc-gdrive,
sufficient for a single developer.

**Production transition:** Replace `gdrive://FOLDER_ID` with
`s3://bucket/p4-radiology-ai/` and configure IAM roles with least-privilege
access. Enable S3 server access logging for PHI audit trail.

**Trade-off:** Less security in portfolio. Fully documented production path
demonstrates architectural awareness without requiring AWS spend.

---

## Decision 4: Class Weights over SMOTE for Class Imbalance

**Date:** July 20, 2026

**Decision:** Class-weighted CrossEntropyLoss for 54/46 imbalance.

**Important distinction:** Class weights address *statistical imbalance*
during training. Cost weights (fn_cost_weight=5.0, fp_cost_weight=1.0 in
config) address *clinical asymmetry* during evaluation. These are separate
mechanisms applied at different stages and are not interchangeable.

**Why SMOTE rejected:** Generates synthetic X-ray images by interpolating
pixel values between real images. Interpolated images have no clinical
validity — they do not correspond to any real patient anatomy. SMOTE is
appropriate for tabular data with severe imbalance (95/5). At 54/46 with
medical imagery, it introduces clinically invalid training examples.

**Note:** Class weight tensor must be moved to model device with .to(device)
before passing to nn.CrossEntropyLoss. CPU/GPU mismatch raises a runtime
error. Handled explicitly in train.py (L6).

---

## Decision 5: AP/PA View Position Mitigation Strategy

**Date:** July 23, 2026

**Context:** Training split AP/PA Suspicious rate gap = 9.99 pp (from Kaggle EDA).

Warning threshold: 10pp. Hard failure threshold: 25pp.

**Options:**

A. Do nothing: Rejected if gap > 10pp. Insufficient for production medical system.

B. Exclude View Position as feature: Already done. Does not prevent the model
   from learning pixel-level AP/PA patterns regardless of the column's absence.

C. Stratified sampling by dominant view position (applied if gap > 10pp):
   Each patient is classified as AP-dominant (>50% of images are AP) or
   PA-dominant. The patient-level split then stratifies by this group to
   ensure AP/PA patient proportions approximately match the overall dataset.

   WHY DOMINANT VIEW, NOT INDIVIDUAL IMAGES: The L2 patient-level split
   guarantee requires all of one patient's images to go into one split.
   Individual-image stratification would break this guarantee. Dominant
   view assignment resolves the patient-level stratification paradox.

D. Adversarial debiasing: Technically sound. Out of scope for P4.
   Documented as a future production upgrade path.

**Decision:** No mitigation required — gap (9.99 pp) ≤ 10pp threshold.

**Actual gap:** 9.99 pp

---

## Decision 6: Long-Tail Patient Dominance Mitigation

**Date:** July 23, 2026

**Context:** Top training patient has 108 images (1.463% of training total).

Threshold: 0.5%.

**Options:**

A. Do nothing: Acceptable if threshold not exceeded. The patient-level split
   prevents test contamination. Document and monitor in failure analysis (L8).

B. Cap images per patient: Limit training images per patient to N (e.g., 10).
   Simple, directly addresses dominance. Reduces training set size modestly.

C. Patient-level weighted sampling: Each image's loss weight = 1/patient_image_count.
   Equalises gradient contribution per patient. Preserves full training data.
   More complex implementation — requires custom sampler or loss weighting in L6.

**Decision:** B — Cap images per patient at 10 images. This directly addresses
the dominance issue while preserving patient-level split integrity.

---

## Decision 7: Vertical Flip Disabled — Clinical Correctness

**Date:** July 23, 2026

**Decision:** vertical_flip: false in training_config.yaml — permanent.

**Clinical reasoning:**

Lung anatomy is NOT vertically symmetric:

  - Right hemidiaphragm is elevated by the liver (higher than left)
  - Heart position has vertical clinical significance
  - Air-fluid levels are gravity-dependent — pool at the bottom of cavities
  - Trachea and carina have specific vertical spatial relationships

A vertically flipped chest X-ray is anatomically impossible. No patient
has this presentation. Training on impossible images teaches the model
incorrect spatial anatomy. This is a patient safety consideration.

**Implementation guard:** _build_train_transform() logs a logger.error
if vertical_flip is ever set to True in config — the error is explicit
and immediate rather than silent.

**Rejected:** enabling vertical flip as a standard augmentation.

---

## Decision 8: Config-Driven Augmentation

**Date:** July 23, 2026

**Decision:** All augmentation parameters read from training_config.yaml.

**Rationale:** MLflow logs the full config for every training run. Config-driven
augmentation makes every augmentation experiment automatically trackable,
reproducible, and auditable. Hardcoded values require code changes and code
diffs — config changes create YAML diffs that are cleaner experiment records.

**Rejected:** hardcoded transform values in dataset.py.

---

## Decision 9: Horizontal Flip — Open Clinical Review Required

**Date:** July 23, 2026

**Status:** OPEN — pending clinical advisor review

**Current setting:** horizontal_flip: true

**Context:** Standard ML papers use horizontal flip for chest X-rays to
increase effective dataset size. Patient positioning varies left/right.

**Clinical concern:** Horizontal flip reverses the heart shadow, simulating
situs inversus (dextrocardia) — a rare congenital condition where the heart
points to the right. This is anatomically valid (the condition exists) but
rare (~0.01% of population). Training on many flipped images may confuse
the model's representation of normal cardiac position.

**Action required:** Clinical domain advisor (pathologist) to review this
specific augmentation choice. If advisor confirms it is acceptable for
a screening tool: close decision as "approved". If advisor overrides:
set horizontal_flip: false and document clinical reasoning here.

**Not rejected yet:** horizontal flip is used in this portfolio version
while awaiting clinical confirmation.

---

## Decision 10: EfficientNet-B0 over ResNet-50 over Custom CNN over ViT

**Date:** July 24, 2026

**Custom CNN from scratch — rejected:**

Insufficient training data (112K images) for learning rich visual representations
from random initialisation. Would significantly underfit.

**ViT (Vision Transformer base) — rejected:**

Three reasons: (1) Lower sample efficiency than CNN — ViT lacks CNN's spatial
locality and translational equivariance inductive biases, requiring more pretraining
data for comparable performance when fine-tuning on 112K images. (2) Higher CPU
inference latency (~200ms+) exceeds EfficientNet-B0's ~80ms, tighter SLA headroom.
(3) Grad-CAM explainability (L9) works on convolutional feature maps — ViT's
patch-based attention requires a different, separate explainability implementation.

**ResNet-50 — rejected:**

25.6M parameters vs 5.3M. ~120ms CPU inference vs ~80ms. ~100MB weights vs ~25MB.
Marginal accuracy improvement (76.1% vs 77.1% ImageNet Top-1) does not justify
the inference, memory, and container size penalties.

**EfficientNet-B0 — chosen:**

5.3M parameters. 77.1% ImageNet Top-1. ~80ms CPU inference. ~25MB weights.
Compound scaling (depth × width × resolution simultaneously) achieves better
accuracy-per-parameter than any single-dimension scaling. All three constraints
(SLA, container size, sample efficiency) are met.

Note: An empirical comparison (training both EfficientNet-B0 and ResNet-50 on
this dataset and measuring test recall) would provide stronger evidence than
published benchmarks alone. For a portfolio project with limited compute,
the benchmark-based justification is defensible. Document this limitation.

---

## Decision 11: Replace Final Classifier Layer + BatchNorm Freeze Protocol

**Date:** July 24, 2026

**Replace final layer only:**

Preserve 99%+ of pretrained backbone. Replace only
`Sequential(Dropout(0.2), Linear(1280, 1000))` with
`Sequential(Dropout(0.3), Linear(1280, 2))`.

**Higher dropout rate (0.3 vs 0.2):**

Original model trained on 1.2M diverse images. Our head trains on ~78K chest
X-rays. Less data → more overfitting risk → higher dropout.

**Explicit Xavier uniform head initialisation:**

Removes training run variance from PyTorch's default Kaiming initialisation.
Symmetric initial logit distribution with zero bias improves early convergence
and reproducibility, especially with class imbalance.

**BatchNorm freeze protocol — overriding train():**

requires_grad=False freezes weights and biases but NOT BatchNorm running_mean
and running_var. Without the train() override, BN stats in the frozen backbone
update toward X-ray statistics while conv weights remain calibrated to ImageNet
statistics — a mismatch that silently degrades Phase 1 accuracy.

The overridden train() method forces all BatchNorm2d layers in backbone.features
to eval() mode whenever the backbone is frozen, preserving pretrained BN stats.

**Momentum reset on Phase 1→2 transition:**

L6 creates a NEW Adam optimizer for Phase 2, explicitly discarding Phase 1
momentum buffers. This is a deliberate choice — starting Phase 2 with equal
(zero) momentum for all parameters, rather than asymmetric momentum (head with
10 epochs of history, backbone with none).

**Differential learning rates (not implemented — documented alternative):**

Production fine-tuning sometimes assigns lower learning rates to earlier layers:
  backbone_lr = 1e-5, head_lr = 1e-3

This provides more granular control over backbone refinement rate.
For this project, the two-phase approach (uniform Phase 2 lr) achieves the
same goal with simpler code. Differential LR is the documented upgrade path.