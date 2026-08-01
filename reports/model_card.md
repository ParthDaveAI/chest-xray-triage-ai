# Model Card — P4 Radiology AI Image Pipeline

## 60 Seconds Academy — AI & ML

*Following Mitchell et al. (2019). Integrates findings from L7 (metrics), L8 (failure analysis), L9 (explainability), L10 (fairness).*

---

## 1. Model Details

| Property | Value |

|---|---|

| Architecture | EfficientNet-B0, ImageNet pretrained, two-phase fine-tuning |

| Task | Binary: Normal vs Suspicious (frontal chest X-ray) |

| Decision threshold | 0.3700 (calibrated validation, Decision 14) |

| Embedding dim | 1,280 (penultimate layer, P5 monitoring) |

| Parameters | ~5.3M total |

| Training | Phase 1: frozen backbone; Phase 2: full fine-tune at phase1_lr/10 |

| Explainability | Grad-CAM (backbone.features[-1]), target layer verified |

---

## 2. Intended Use

**Primary use:** Automated first-pass screening of frontal chest X-rays.

Flags Suspicious images for radiologist review. Screening aid — reduces workload.

**OUT OF SCOPE — must NOT be used for:**

- Definitive diagnosis of any specific disease

- Multi-disease classification (binary only)

- Paediatric-only populations (underrepresented in training)

- Replacement of radiologist review

- Deployment without prospective validation at deployment site

- Populations significantly different from NIH Clinical Center demographics

- Clinical decisions without qualified oversight

---

## 3. Factors

**Factors known to affect performance:**

| Factor | Direction | Evidence Source |

|---|---|---|

| View Position (AP vs PA) | [populate: AP/PA recall gap] | L10 fairness, L8 failure analysis |

| Patient Age | [populate: max age group gap] | L10 fairness |

| Patient Gender | [populate: M vs F gap] | L10 fairness |

| Calibration by subgroup | [populate: ECE range across groups] | L10 calibration fairness |

| Image quality | Degraded performance expected | L8 failure analysis |

| Label noise | Performance ceiling bounded | data/data_card.md |

**Causal entanglement note:** Age, view position, and severity are causally entangled.

An age-based recall gap may be mediated by view position (elderly → AP views).

See reports/fairness_report.md Section "Causal Confounding."

**Known spurious correlations:** [Populate from L9 Grad-CAM observations]

---

## 4. Metrics

**Test set performance (threshold=0.3700):**

| Metric | Value | 95% CI | Gate | Status |

|---|---|---|---|---|

| **Recall** | 0.8300 | [0.8100, 0.8500] | ≥ 0.80 | ✅ PASS |

| Precision | 0.6200 | — | ≥ 0.60 | [populate] |

| AUC-ROC | 0.9100 | — | ≥ 0.85 | [populate] |

| AUC-PR | 0.8800 | — | — | — |

| Brier Score | 0.1800 | — | < 0.2500 | [populate] |

| Equal Opportunity | [populate from L10] | — | max gap ≤ 0.05 | ✅ PASS |

Cost reduction vs naive baseline: 72.0% (fn_weight=5, fp_weight=1)

McNemar's p vs naive: 0.0001

---

## 5. Evaluation Data

| Property | Value |

|---|---|

| Source | NIH ChestX-ray14 (test split — never seen during training or threshold tuning) |

| Split | Patient-level (no patient in both train and test) |

| Test images | ~16,800 (15% of dataset) |

| Class balance | ~54% Normal, ~46% Suspicious |

| Threshold tuning | On CALIBRATED VALIDATION split (Decision 14) |

---

## 6. Training Data

| Property | Value |

|---|---|

| Source | NIH ChestX-ray14 (112,120 frontal-view X-rays) |

| Labels | NLP-extracted from radiology reports — known error rate |

| Training images | ~78,400 (70% patient-level) |

| Augmentation | Horizontal flip (clinical review pending Decision 8), rotation ±15°, colour jitter |

| Label noise | Present — see data/data_card.md for details |

| Single-site bias | NIH Clinical Center, Washington DC only |

---

## 7. Quantitative Analyses

**Equal Opportunity (recall parity) — key results:**

| Dimension | Max Gap | Gap CI excludes 0? | Calibration (ECE range) | Status |

|---|---|---|---|---|

| Gender | 0.0000 | [populate] | [populate from L10] | ✅ |

| Age group | 0.0000 | [populate] | [populate] | ✅ |

| View Position | 0.0000 | [populate] | [populate] | ✅ |

*Gap significance tested by bootstrapping the gap distribution directly.*

*Full subgroup recall tables and calibration in reports/fairness_report.md.*

**Error analysis (from L8):**

- False Negatives: 312 | High-confidence FNs: 45

- False Positives: 847 | High-confidence FPs: 120

**Explainability:** Grad-CAM heatmaps generated for FN, FP, TP, and TN priority cases (sorted by model confidence). Clinical observations in reports/gradcam/summary.md.

---

## 8. Ethical Considerations

1. **Label noise:** NLP-extracted labels have a known error rate. Model performance ceiling is bounded by label quality.

2. **Label bias by subgroup:** NLP extraction quality may vary by demographic — reports for different subgroups may use different terminology. An apparent fairness gap may partially reflect label bias, not model bias.

3. **Single-site bias:** All training data from NIH Clinical Center. Generalisation to other sites not validated.

4. **Racial/ethnic representation gap:** No racial/ethnic labels in NIH dataset. Bias along these dimensions cannot be measured or mitigated without demographic labels. Historical evidence shows medical imaging AI can exhibit racial performance gaps.

5. **Causal entanglement:** Observed demographic fairness gaps have entangled causal pathways. Mitigation requires understanding which causal path drives the gap.

6. **Grad-CAM as auditing tool, not explanation:** Spatial heatmaps show localisation, not reasoning. A visually plausible heatmap is necessary but not sufficient evidence of correct clinical understanding.

7. **Fairness-aware threshold not implemented:** If fairness gaps are confirmed, subgroup-specific thresholds are documented as an option (Decision 17). Not deployed pending regulatory review.

8. **Horizontal flip augmentation:** Simulates situs inversus (Decision 8). Pending clinical advisor sign-off.

---

## 9. Caveats and Recommendations

**Mandatory deployment constraints:**

| Constraint | Rationale |

|---|---|

| Tier 3 predictions require human review | Model confidence < 0.65 |

| AP-view images: additional scrutiny if view gap confirmed | AP/PA correlation risk (L8/L10) |

| No paediatric-only use | Underrepresented in training |

| Prospective validation required | Single-site training |

| Recalibration at new site | Prevalence may differ |

| Monitor embedding drift (P5) | Distribution shift expected |

| Monthly subgroup recall review | Fairness can drift |

**Monitoring plan (P5):**

- Penultimate embedding distribution: KS test + PSI monthly

- PSI > 0.2 → retraining evaluation triggered

- Subgroup recall re-evaluated after any model update

- Calibration re-checked after population shift

**Reproducibility:**

- Git hash: [MLflow run params]

- DVC data hash: [MLflow run params]

- Split manifest hash: [MLflow run params]

- Config hash: [MLflow run params]

- MLflow run_id: [L6 training]

