# Problem Framing Document — P4 Radiology AI Image Pipeline

## Protocol Phase: 0 | Status: LOCKED before first training run

---

## 1. System Contract (Not Just a Model Problem)

This document specifies what the *system* does, not just what the model predicts.

**System contract:**

"For each chest X-ray image uploaded by a radiologist, the system shall:

(1) validate the image quality before any inference,

(2) classify the image as Normal or Suspicious using EfficientNet-B0 transfer learning

    trained on NIH ChestX-ray14,

(3) assign the prediction to a confidence tier that determines a specific clinical triage

    action,

(4) return a structured, Pydantic-validated response including prediction, confidence,

    tier, action recommendation, and a traceable audit record,

(5) such that the false negative rate is reduced by at least 15% compared to the

    majority-class baseline, and the server-side processing latency p99 is under 500ms."

**Why this is a system contract, not a model problem statement:**

A model problem statement says "predict whether an X-ray is suspicious."

A system contract specifies the full pipeline: quality gate → model → decision engine →

structured response → compliance log. This project delivers a system, not a model.

---

## 2. Justification for ML over Rule-Based Systems

Rule-based systems require explicit feature engineering — brightness thresholds, edge

detection, shape descriptors. These cannot generalise to the diversity of abnormality

patterns across 112,120 real X-rays from tens of thousands of patients on different

equipment over multiple years.

Deep learning learns spatial feature hierarchies directly from pixel data. No

hand-crafted feature can capture the morphological complexity of pathological patterns

at this scale.

---

## 3. Success Metrics — LOCKED

Thresholds defined before any training run. Not changed after seeing results.

| Metric | Formula | Threshold | Type | Reason |

|---|---|---|---|---|

| Recall | TP / (TP + FN) | ≥ 0.80 | Primary | Bounds FN rate at 20% |

| Precision | TP / (TP + FP) | ≥ 0.60 | Guard-rail | Prevents flooding the priority queue |

| AUC-ROC | (threshold-independent) | ≥ 0.85 | Guard-rail | Overall discrimination quality |

| Server-side latency p99 | (processing only, excludes network) | < 500ms | SLA | Clinical workflow requirement |

**Fairness metric (evaluated in L10):**

Recall gap across gender and age subgroups: < 0.05

---

## 4. Naive Baseline — Two Versions That Both Fail

A useful system must beat both naive baselines from opposite ends.

**Naive Baseline A — Predict Normal always (majority class):**

| Metric | Value | Clinical consequence |

|---|---|---|

| Accuracy | ~54% | Misleading — all Suspicious cases missed |

| Recall (Suspicious) | 0.0 | Every suspicious case is a false negative |

| Precision (Suspicious) | Undefined | Never predicts Suspicious |

| Expected Cost | fn_cost_weight × N_suspicious | Maximum possible cost |

**Naive Baseline B — Predict Suspicious always:**

| Metric | Value | Clinical consequence |

|---|---|---|

| Recall (Suspicious) | 1.0 | Catches everything — but at massive FP cost |

| Precision (Suspicious) | ~0.46 | 54% of all flags are false alarms |

| Expected Cost | fp_cost_weight × N_normal | Priority queue is meaningless — everything is "priority" |

**The model must beat both:** recall > 0.0 (beats Baseline A) and precision > 0.46

(beats Baseline B). The locked thresholds — Recall ≥ 0.80, Precision ≥ 0.60 — are

chosen specifically to clear both baselines with meaningful margin.

**Cost baseline (computed in L7):**

Naive Expected Cost (Baseline A) = 5.0 × total_suspicious_in_test_set

This is the maximum cost the model must beat on the cost-sensitive evaluation metric.

---

## 5. Build vs Buy Decision

The real competitors in clinical AI are not generic vision APIs. They are

FDA-cleared clinical AI vendors.

| Option | Explainability | SaMD Auditability | Vendor Lock-in | Clinical X-Ray Specificity | Decision |

|---|---|---|---|---|---|

| Google Cloud Vision API | None (black box) | Cannot trace to checkpoint | High | General-purpose | **Rejected** |

| Amazon Rekognition Medical | None (black box) | Cannot trace to checkpoint | High | Primarily pathology | **Rejected** |

| Aidoc / Zebra Medical / Nuance (FDA-cleared) | None (proprietary) | Partial, vendor-managed | Very high | Yes, but opaque | **Rejected** |

| Train EfficientNet-B0 (this project) | Full (Grad-CAM) | Complete (model + config + data hash) | None | Full control | **Chosen** |

**The SaMD auditability argument for building:**

Every prediction must be traceable to the exact model version, configuration, and

training data version that produced it. Third-party APIs cannot provide this — the

vendor controls the model, may update it silently, and does not expose training data

versions. A custom-trained model with git commit hash, config hash, and DVC data hash

logged to every prediction provides complete traceability.

**Failure to beat baseline = project STOP condition:**

If the trained model does not beat the naive baseline on every locked metric,

the ML approach is not delivering clinical value and the project does not proceed

to deployment. This is evaluated explicitly in L7.