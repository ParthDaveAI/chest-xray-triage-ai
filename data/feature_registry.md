# Feature Registry — P4 Radiology AI Image Pipeline

## 60 Seconds Academy — AI & ML

## Populated from Kaggle EDA: `eda_summary.json`

## EDA scope: TRAINING SPLIT ONLY (7,384 images in subset)

---

## Used Features

### Input: Chest X-Ray Image (PNG or JPEG)

| Property | Value |
|---|---|
| Format at inference | PNG or JPEG, any resolution |
| Preprocessing | Resize to 224×224, convert to RGB, normalise to ImageNet stats |
| ImageNet mean | [0.485, 0.456, 0.406] — R, G, B channels |
| ImageNet std | [0.229, 0.224, 0.225] — R, G, B channels |
| Why ImageNet stats? | EfficientNet-B0's pretrained weights calibrated for inputs normalised to these statistics. Un-normalised inputs produce meaningless activations in pretrained layers. Transfer learning fails silently. |
| Source | NIH ChestX-ray14 |
| Known limitation | NLP-extracted labels contain noise. See data/data_card.md. |

---

## Excluded Features

| Feature | In CSV | Excluded Reason |
|---|---|---|
| Patient Age | Yes | Inference contract violation — not in raw PNG. Causal reasoning: age correlates with disease prevalence but does not cause pathological appearance in images. Reserved for fairness evaluation L10 only. |
| Patient Gender | Yes | Inference contract violation. Reserved for fairness evaluation L10 only. |
| View Position | Yes | Inference contract violation. Also carries AP/PA spurious correlation risk (see below). |
| Follow-up Number | Yes | Inference contract violation. Cannot be inferred from a single image. |
| Finding Labels | Yes | Direct target leakage — this IS the label. |

**Inference contract:** POST /predict/image receives only a raw PNG or JPEG file.

Any feature dependent on CSV metadata would cause a silent production failure.

**Causal vs correlational:** Demographics excluded not only for inference-time
unavailability but because they are correlational features — the causal pathway to
suspicious findings runs through visible pathology, not demographics. Using
correlational features trains the model on population statistics, not clinical signals.

---

## AP/PA View Position Correlation Finding (Training Split)

| Metric | Value |
|---|---|
| AP Suspicious rate | 47.68% |
| PA Suspicious rate | 37.68% |
| Gap | 9.99 pp |
| Warning threshold | 10.0 pp |
| Hard failure threshold | 25.0 pp |
| Risk flag raised? | NO (9.99 pp < 10.0 pp threshold) |
| AP-dominant patients in training | N/A (not computed in subset mode) |

**Patient-level dominant view assignment:**

For stratification mitigation (if gap > 10pp): each patient is classified as
AP-dominant if > 50% of their training images are AP, otherwise PA-dominant.
Split stratification operates on this patient-level group assignment.

**Status:** No mitigation required — gap ≤ 10pp (9.99 pp)

See decisions.md Decision 5.

---

## Patient Long-Tail Distribution Finding (Training Split)

| Metric | Value |
|---|---|
| Training patients | 1,864 |
| Training images | 7,384 |
| Top patient image count | 108 |
| Top patient as % of training | 1.463% |
| Threshold | 0.5% |
| Threshold exceeded? | YES (1.463% > 0.5%) |

**Status:** Mitigation should be considered. Top patient contributes 1.463% of training images.

See decisions.md Decision 6.

---

## Pixel Intensity Distribution Finding (Training Split)

| Metric | Value |
|---|---|
| Normal mean intensity | 142.57 |
| Suspicious mean intensity | 135.65 |
| Mean difference | 6.92 |
| Mean risk threshold | 20.0 units |
| Normal mean std dev | N/A (not computed in subset mode) |
| Suspicious mean std dev | N/A (not computed in subset mode) |
| Std dev ratio | N/A |
| Contrast risk threshold | 1.5 |
| Mean risk flag? | NO (6.92 < 20.0) |
| Contrast risk flag? | N/A |

**Mitigation:** Colour jitter augmentation configured (brightness=0.2, contrast=0.2
in training_config.yaml). This trains the model to be invariant to intensity and
contrast variations.

---

## EDA Limitations — What This Analysis Cannot Detect

1. **Label noise:** EDA observes label distributions but cannot verify whether
   individual labels are correct. The AP/PA correlation may partially reflect
   label noise in NLP-extracted labels.

2. **Hidden confounders:** Systematic correlations between scanner models, hospital
   wards, and patient demographics may be invisible to standard EDA checks.

3. **Causal structure:** EDA identifies correlations. The causal chain from
   AP imaging to Suspicious labels runs through clinical patient selection,
   not through imaging physics. EDA cannot distinguish spurious from causal.

4. **Future distribution shift:** EDA describes the training distribution.
   It cannot predict shifts when the model deploys at a different hospital or
   on a different patient population. P5 drift monitoring uses this EDA
   summary as the reference baseline for detecting such shifts.

---

## Forward References

| Downstream Lecture | How EDA Feeds In |
|---|---|
| L6 Training | Class distribution confirms class weight calculation |
| L8 Failure Analysis | AP/PA and intensity findings provide error pattern context |
| L10 Fairness Evaluation | Gender demographics confirm subgroup selection |
| L11 Serving | Inference contract (image-only) enforced in API design |
| P5 Drift Monitoring | eda_summary.json provides training distribution baseline |