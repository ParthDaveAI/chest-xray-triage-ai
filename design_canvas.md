# ML System Design Canvas — P4 Radiology AI Image Pipeline

## Protocol Phase: -1 | Status: COMPLETE before L1 begins

---

## SECTION 1 — PROBLEM DEFINITION

**Problem:**

Radiologists review thousands of chest X-rays daily under time pressure.

Visual fatigue causes miss rates of 3–5% even among experienced readers.

Suspicious findings go undetected, resulting in delayed diagnoses of pneumonia,

cancer, and other serious conditions.

**Who is affected:**

- Radiologists: workload, cognitive fatigue, liability for missed findings

- Patients: risk of missed findings, delayed diagnosis, delayed treatment

**Cost of the unsolved problem:**

Delayed diagnosis of cancer or pneumonia. Patient harm. Hospital liability.

Every percentage point improvement in recall for a high-volume screening system

translates to a measurable reduction in late-stage disease detection at the

population level.

**"Solved" means:**

Suspicious X-rays are automatically flagged for priority review. The radiologist's

attention is directed to high-risk images first. The false negative rate is reduced

by at least 15% compared to the majority-class baseline (which has a false negative

rate of 100% for the Suspicious class).

**Stakeholders and Domain Expertise:**

| Stakeholder | Role | Interaction with System |

|---|---|---|

| Radiologist | Primary user | Uploads images, acts on predictions, provides ground truth feedback |

| Hospital IT | Infrastructure | Deploys and maintains serving layer |

| Clinical Domain Advisor (Pathologist) | Design validator | Reviews failure analysis (L8), validates Grad-CAM heatmaps (L9), confirms EDA findings are medically sound |

| Regulator (FDA / CE) | Compliance | Requires audit trails, SaMD documentation, justification for design choices |

Domain advisor involvement is documented at Phase -1 so that every downstream

clinical decision is explicitly connected to qualified review.

---

## SECTION 2 — CURRENT STATE

**Today:** Fully manual review. Images processed in first-in-first-out queue.

No prioritisation exists. Equal attention given to clearly normal images and images

with subtle early-stage findings.

**Key limitation:** Scale is growing. The global radiologist shortage is worsening.

More images are produced per radiologist per day than a decade ago.

**Why the current state is insufficient:** FIFO queue management treats all images as

equivalent. Human attentional capacity does not support sustained high-quality review

at scale without systematic prioritisation support.

---

## SECTION 3 — ML SOLUTION STRATEGY

**Is ML justified?**

Yes. Rule-based systems require explicit feature engineering — brightness thresholds,

edge detection, shape descriptors. These cannot generalise to the diversity of

abnormality patterns in real chest X-rays. Deep learning learns spatial feature

hierarchies directly from pixel data. No hand-crafted feature can capture the

morphological complexity of pathological patterns across 112,120 images.

**Problem type:** Binary classification — Normal (0) vs Suspicious (1)

**Training signal:** Supervised. NIH ChestX-ray14 labels derived from radiology

reports via NLP. Label noise acknowledged and documented in Data Card (L2).

**Serving mode:** Online inference via FastAPI endpoint.

Justification: Radiologist uploads image and requires immediate result. Batch

processing returns predictions hours after the decision window closes. The clinical

workflow is synchronous. The serving mode must match.

Rejected serving mode: Batch inference — breaks clinical workflow.

**Pre-Inference Quality Validation Layer:**

Every incoming image passes through validate_image() before reaching the model.

Checks: file format, minimum resolution, pixel contrast (std dev of pixel values),

corruption detection.

If any check fails: HTTP 422 with specific failure reason. Model is never called.

Rationale: training data (NIH ChestX-ray14) is clean and standardised. Real hospital

inputs include corrupted files, low-exposure images, and scanner artifacts. Rejecting

invalid inputs before inference prevents unreliable predictions from appearing as

legitimate outputs.

**System mental model:**

Input → Quality Gate → Model → Decision Engine → Response

(Not: Input → Model → Response)

**Human-in-the-Loop Constraint:**

The model's output is always a recommendation, never a final clinical decision.

Every flagged case is reviewed by a qualified radiologist before any clinical action.

Radiologist judgment always overrides model output. This is not a limitation — it is

the correct architecture for a medical decision support system under SaMD constraints.

**SaMD Boundary:**

| Permitted | Forbidden |

|---|---|

| Flag images for priority radiologist review | Output a specific diagnosis (e.g., "Pneumonia") |

| Return confidence score with advisory note | Recommend or rule out specific treatments |

| Log predictions for audit trail | Override radiologist clinical judgment |

| Escalate low-confidence cases for re-scan | Make autonomous triage decisions without human review |

Disclaimer on every response: "This is not medical advice. Results require qualified clinical review."

---

## SECTION 4 — ALGORITHM LANDSCAPE

| Architecture | Parameters | Pretrained | Assessment | Decision |

|---|---|---|---|---|

| Custom CNN from scratch | ~5–20M | No | Insufficient training data (112K images) for scratch training. Underfitting risk high. | **Rejected** |

| Pretrained ResNet-50 | 25.6M | Yes (ImageNet) | Strong features, widely used in medical imaging. 25.6M params → slower inference, higher VRAM. Accuracy comparable to EfficientNet-B0 with worse efficiency. | **Rejected** |

| Pretrained EfficientNet-B0 | 5.3M | Yes (ImageNet) | Compound scaling (depth × width × resolution simultaneously). 5x fewer params than ResNet-50. Comparable accuracy. Faster inference supports SLA. | **✓ CHOSEN** |

**EfficientNet-B0 justification:**

Compound scaling optimises depth, width, and resolution together rather than scaling

any single dimension. This achieves better accuracy per parameter than ResNet's

single-dimension scaling. At 5.3M parameters, server-side CPU inference is measurably

faster, which is relevant for the serving container where GPU is not assumed.

---

## SECTION 5 — METRICS PRE-SELECTION

**LOCKED. Thresholds defined before any training run. Not changed after seeing results.**

| Metric | Formula | Threshold | Type | Clinical Justification |

|---|---|---|---|---|

| Recall | TP / (TP + FN) | ≥ 0.80 | Primary | Of all truly Suspicious images, ≥ 80% caught. Bounds FN rate at 20%. |

| Precision | TP / (TP + FP) | ≥ 0.60 | Guard-rail | Of all flagged images, ≥ 60% actually suspicious. Prevents flagging everything. |

| AUC-ROC | (threshold-independent) | ≥ 0.85 | Guard-rail | Overall class separation quality. |

| Server-Side Latency p99 | (processing time only, excludes network) | < 500ms | SLA | API gateway + preprocessing + inference. Network RTT excluded — outside system control. |

**Why accuracy is not used:**

With ~54% Normal images, predicting Normal for every image gives ~54% accuracy and

catches zero suspicious cases. Accuracy rewards the majority class and hides the

clinically dangerous failure mode. Not used anywhere in this project.

**Naive Baseline:**

Majority-class classifier: predict Normal for every image.

- Recall for Suspicious class: 0.0 — catches nothing

- Precision for Suspicious class: undefined — never predicts Suspicious

- Expected cost: maximum — every suspicious case is a false negative

All trained model metrics are compared against this baseline.

**Three-Tier Uncertainty System with Action Mapping:**

| Tier | Confidence | API Response | Clinical Action |

|---|---|---|---|

| 1 — High | ≥ 0.80 | Standard prediction | If Suspicious: auto-flag to priority review queue |

| 2 — Moderate | 0.50–0.79 | Prediction + soft advisory | Normal workflow + "Low confidence — radiologist discretion advised" |

| 3 — Low | < 0.50 | Prediction + mandatory review flag | Escalate. Recommend re-scan. "Model confidence insufficient for automated triage — mandatory clinical review required before any action." |

Every tier has exactly one defined action. Ambiguity in a medical system is a safety failure.

---

## SECTION 6 — FAILURE MODE MAPPING

| Failure Mode | Clinical Consequence | System Response |

|---|---|---|

| False Negative | Suspicious finding missed → delayed diagnosis → patient harm | Recall ≥ 0.80 quality gate; Tier 3 escalation for low-confidence predictions |

| False Positive | Radiologist reviews unnecessary image → ~2 min wasted | Precision ≥ 0.60 guard-rail |

| Model unavailable | No automated triage | HTTP 503: "Screening unavailable." Fallback: FIFO queue continues without prioritisation |

| Low-quality image input | Unreliable prediction from corrupted / low-contrast image | validate_image() rejects before inference; HTTP 422 + specific reason |

| Spurious Correlation (AP/PA Hardware Bias) | Model learns portable machine signature not pathology → fails on new equipment | AP/PA view correlation check gated in EDA (L3); stratified sampling mitigation if gap > 10pp |

| High-Confidence Wrong Prediction | Radiologist over-trusts incorrect output | Calibration monitoring: if high-confidence error rate exceeds threshold, trigger calibration review or retraining flag |

| Input Distribution Shift (Data Drift) | Silent performance degradation as equipment or patient population changes | P5: PSI ≥ 0.20 triggers alert + model promotion freeze |

| Concept Drift | Label relationship changes without input distribution change | P5: prediction distribution monitoring + radiologist correction feedback |

**Graceful Degradation Strategy:**

Failure must never be worse than the system not existing.

- Model load failure → HTTP 503, workflow reverts to FIFO baseline

- validate_image() failure → HTTP 422, no prediction returned, no silent garbage output

- Runtime inference error → HTTP 503, logged with trace_id for debugging

---

## SECTION 7 — SYSTEM BOUNDARY

**System mental model:**

Input → Quality Gate → Model → Decision Engine → Human Review → Clinical Action

**Regulatory Traceability:**

Every prediction is traceable to three identifiers:

- model_version: which checkpoint produced this output

- config_hash: which hyperparameters and thresholds were active

- dataset_version (DVC hash): which training data was used

This traceability is the SaMD audit trail.

**Drift Response Logic (implemented in P5):**

| PSI Level | Condition | Action |

|---|---|---|

| No drift | PSI < 0.10 | Normal operation |

| Mild drift | 0.10 ≤ PSI < 0.20 | Log warning, increase monitoring frequency |

| Severe drift | PSI ≥ 0.20 | Alert on-call, freeze model promotion pipeline, flag for retraining |

**Data Drift vs Concept Drift:**

- Data drift (input distribution shift): equipment change, population change. Monitor via embedding drift (P5).

- Concept drift (label relationship shift): e.g., new disease pattern changes clinical interpretation. Monitor via prediction distribution and radiologist correction feedback (P5).

**System Boundary Diagram:**

OFFLINE PATH [NIH ChestX-ray14] → [data_prep.py] → [training_config.yaml] → [dataset.py / DataLoader] → [train.py (Phase 1 + Phase 2, AMP)] → [evaluate.py (threshold on train only)] → [MLflow Registry (bundle)]

ONLINE PATH [Radiologist] → POST /predict/image → [validate_image()] Quality Gate ❌ HTTP 422 if invalid → [get_inference_transform(config)] Preprocessing → [ChestXRayClassifier.forward()] EfficientNet-B0 → [Decision Engine] Three-Tier Assignment → [PredictionResponse Pydantic strict] JSON response → [log_prediction()] PHI-safe: image_hash, tier, model_version, config_hash, dvc_hash → [HUMAN REVIEW] Radiologist — final decision

MONITORING HANDOFF TO P5 POST /predict/image/with-embedding (internal, not in OpenAPI docs) → [P5 Drift Monitoring] KS + PSI on embeddings → [Drift Response] PSI thresholds → alert → freeze → retrain signal

---

## SECTION 8 — COST AND SCALE AWARENESS

| Dimension | Estimate | Notes |

|---|---|---|

| Training (GPU T4, Colab) | ~2–4 hours | Two-phase, 25 epochs, 112K images |

| Server-side inference (CPU) | ~80–150ms | Within 500ms SLA |

| Server-side inference (GPU) | ~15–25ms | Production GPU path |

| Model artifact | ~25MB | EfficientNet-B0 |

| Dataset | ~42GB | DVC-tracked, never in Git |

| Docker container | ~800MB | python:3.11-slim + CPU-only PyTorch |

**Scaling Strategy:**

| Volume | Strategy | Rationale |

|---|---|---|

| < 100 images/hour | CPU inference, single instance | Cost-efficient, within SLA |

| 100–1,000/hour | GPU inference, single instance | Meets SLA with headroom |

| > 1,000/hour | GPU + request batching + autoscaling | Batching amortises GPU transfer cost |

**CPU vs GPU serving decision:**

CPU-only PyTorch is used in the serving container (python:3.11-slim base,

--index-url https://download.pytorch.org/whl/cpu). Keeps container at ~800MB

instead of 3–4GB. EfficientNet-B0 on CPU meets the 500ms SLA for clinical volumes.

GPU serving is the documented upgrade path for high-volume deployments.

**DVC Remote:**

Development: Google Drive (free, sufficient for portfolio).

Production path: S3 bucket with IAM role access controls for PHI isolation and

audit logging (documented in decisions.md, implemented if deployed commercially).