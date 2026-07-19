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