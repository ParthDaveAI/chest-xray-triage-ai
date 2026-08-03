# Grad-CAM Explainability Summary — P4 Radiology AI

## 60 Seconds Academy — AI & ML

**Gate artifact status:** ❌ NEEDS MORE (2/5 minimum)

**Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)

**Target layer:** model.backbone.features[-1] (last MBConv, EfficientNet-B0)

**Feature map shape:** (batch, 1280, 7, 7) for 224×224 input

---

## FRAMING STATEMENT

Grad-CAM is a spatial localisation audit tool. It shows WHICH REGIONS influenced

the model's class score. It does NOT explain WHY the model reached its decision,

NOR does it prove the model understands the pathology.

A heatmap focused on the correct anatomy is necessary but not sufficient evidence

of clinically sound reasoning. The model may use a shortcut (local texture statistics,

AP equipment signature) that happens to overlap with the pathological region.

Use these heatmaps as supporting evidence — not as conclusive proof.

---

## Heatmaps Generated

| Case Type | Count | Location | Priority | target_class |

|---|---|---|---|---|

| False Negatives (FN) | 2 | fn_cases/ | HIGHEST | 1 (Suspicious) |

| False Positives (FP) | 0 | fp_cases/ | HIGH | 1 (Suspicious) |

| True Positives (TP) | 0 | tp_cases/ | BASELINE | 1 (Suspicious) |

| True Negatives (TN) | 0 | tn_cases/ | REFERENCE | 0 (Normal) |

FN high-confidence cases (model_confidence ≥ 0.80): 2

*These are the most dangerous — model was very certain about a wrong Normal prediction.*

---

## Interpreting FN Heatmaps (target_class=1)

**What these show:** Which regions WOULD HAVE needed to activate MORE to push the

model toward a Suspicious prediction. This is the ABSENT Suspicious signal.

**Different from "what caused the Normal prediction":** target_class=1 on a Normal-

predicted image shows missing positive evidence, not active negative evidence.

To see what drove the Normal prediction, you would need target_class=0.

**Clinical interpretation questions:**

- Does the activated region overlap with clinically relevant lung anatomy?

  (If yes: the model found insufficient signal in the right location)

- Is the heatmap flat/diffuse?

  (Possible gradient saturation — model was highly confident Normal with no

   localised Suspicious signal. Data gap or fundamental model limitation.)

- Does the activated region focus on the image border, clavicles, or labels?

  (Spurious correlation — the model is not attending to pathological anatomy)

---

## Interpreting FP Heatmaps (target_class=1)

**What these show:** Which regions drove the FALSE Suspicious prediction.

**Key question:** Do high-confidence FP heatmaps consistently activate in the same

non-pathological region across different patients?

- Consistent activation on AP equipment / image characteristics → confirms

  AP/PA spurious correlation hypothesis from L8 failure analysis

- Consistent activation on pacemaker leads → model learned device ≠ pathology

- Diffuse activation → model is uncertain (low-confidence FP, expected at low threshold)

---

## Heatmap Layout

Left panel: Original X-ray (100% opacity in non-activated regions)

Right panel: Original X-ray with INFERNO heatmap overlay

  - Heatmap is used as a per-pixel alpha mask (not global blend)

  - Zero-activation regions preserve full diagnostic image quality

  - Activated regions highlighted proportionally to activation magnitude

Colormap: INFERNO — perceptually uniform, no false colour discontinuities.

(JET colormap was rejected: introduces artefactual edges at rainbow transitions

 that can mislead clinical reviewers.)

---

## Clinical Observations

*Complete this section after reviewing with the clinical advisor.*

### FN Cases — Missing Suspicious Signal

**Heatmap pattern:** [populate — diffuse/localised? Anatomy region?]

**Clinical assessment:** [relevant anatomy focus? or spurious region?]

**Connects to L8 hypothesis:** [which root cause hypothesis does this support?]

### FP Cases — Spurious Suspicious Signal

**Heatmap pattern:** [populate — consistent region across cases? AP equipment?]

**Confirms/refutes AP/PA spurious correlation:** [Yes/No]

### TP Cases — Correct Suspicious Predictions (baseline)

**Heatmap pattern:** [populate — do correct predictions focus on appropriate anatomy?]

---

## Limitations

1. **Localisation ≠ reasoning.** Heatmaps reveal spatial sensitivity, not causal

   reasoning. The model may activate correctly while using a shortcut.

2. **Gradient saturation.** Very high-confidence predictions produce near-zero gradients

   — heatmaps appear flat. A flat heatmap for a high-confidence FN indicates the model

   had no localised Suspicious signal (distributed or spurious global features).

3. **7×7 spatial resolution.** Last convolutional layer produces 7×7 feature maps

   upsampled to 224×224. The heatmap has 49 effective resolution bins — coarse-grained.

   For finer attribution, Integrated Gradients is the documented upgrade path.

4. **No sanity baseline.** A proper interpretability validation would compare against

   Grad-CAM from a randomly-initialised model (which should produce noise). Without

   this comparison, there is no guarantee heatmaps reflect learned features rather

   than architectural priors.

5. **Model-specific.** These heatmaps reflect this specific trained model. Any retraining

   requires regenerating heatmaps.

6. **Confirmation bias risk.** Visually plausible heatmaps can reinforce clinical

   overconfidence in model predictions. Reviewers should be aware that the heatmap

   may look anatomically reasonable even when the model's reasoning is spurious.

---

## Gate Artifact Checklist

- [ ] At least 5 heatmaps generated (2 total)

- [ ] FN heatmap pattern documented in Clinical Observations above

- [ ] FP heatmap pattern documented and AP/PA correlation assessed

- [ ] TP baseline comparison completed

- [ ] Limitations acknowledged and included in model card (L10)

- [ ] Clinical advisor has reviewed FN and FP heatmaps

**This document must have all Clinical Observations sections completed

before L10 fairness evaluation and model card begins.**

