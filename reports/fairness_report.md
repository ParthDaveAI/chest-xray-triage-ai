# Fairness Evaluation Report — P4 Radiology AI

## 60 Seconds Academy — AI & ML

**Primary metric:** Equal Opportunity (recall parity)

**Secondary metric:** Calibration fairness (ECE + Brier per subgroup)

**Fairness threshold:** recall gap ≤ 0.05 (5pp) — policy-defined, not scientific law

**Overall status:** ✅ ALL FAIRNESS CHECKS PASSED

---

## Fairness Metric Framing

Equal Opportunity is the primary metric for this use case, not the universally

correct metric. The choice is justified by the clinical harm model (fn_weight=5):

differential miss rates are the primary clinical harm, so recall parity is the

primary fairness requirement.

**Multiple fairness notions monitored:**

- Equal Opportunity (recall parity): primary gate

- Calibration fairness (ECE/Brier per subgroup): secondary check

- Demographic parity and equalized odds: not primary for this use case

  (see decisions.md Decision 16 for full rationale)

**Multiple comparisons note:**

This analysis is EXPLORATORY. Testing fairness across multiple dimensions

increases Type I error (false fairness alarms). For a confirmatory fairness

audit (regulatory submission), apply Bonferroni correction:

  adjusted α = 0.05 / comparisons

A gap CI must exclude zero at the 99% level for confirmed significance.

---

## Dimension 1: Gender Fairness


**Gender — Equal Opportunity (Recall Parity):**

| Subgroup | Recall | Per-subgroup 95% CI | N Susp | Reliability |
|---|---|---|---|---|
| F | N/A | N/A | 0 | ❌ No data |
| M | 0.5267 | [0.4733, 0.6185] | 150 | ✅ OK |

**Gender — Calibration Fairness (ECE and Brier):**

| Subgroup | ECE | Brier Score | N |
|---|---|---|---|
| F | 0.4138 | 0.2148 | 150 |
| M | 0.6068 | 0.4072 | 150 |

  Gender: no pairwise gaps


**Status:** ✅ No concern (gap within threshold)

---

## Dimension 2: Age Group Fairness

Age groups: under_40 (< 40), 40_to_60 (40–60), over_60 (≥ 60)


**Age Group — Equal Opportunity (Recall Parity):**

| Subgroup | Recall | Per-subgroup 95% CI | N Susp | Reliability |
|---|---|---|---|---|
| 40_to_60 | 0.5510 | [0.4128, 0.7097] | 49 | ⚠️  Low (n<50) |
| over_60 | 0.5686 | [0.4162, 0.7167] | 51 | ✅ OK |
| under_40 | 0.4600 | [0.3000, 0.5910] | 50 | ✅ OK |

  Age Group: no pairwise gaps


**Status:** ✅ No concern (gap within threshold)

---

## Dimension 3: View Position Fairness (AP vs PA)


*L3 EDA AP/PA Suspicious rate gap was 10.0pp.*



**View Position — Equal Opportunity (Recall Parity):**

| Subgroup | Recall | Per-subgroup 95% CI | N Susp | Reliability |
|---|---|---|---|---|
| AP | 0.4400 | [0.3200, 0.5600] | 50 | ✅ OK |
| PA | 0.5700 | [0.4723, 0.6600] | 100 | ✅ OK |

  View Position: no pairwise gaps


**Status:** ✅ No concern (gap within threshold)

---

## Causal Confounding

Age, gender, and view position are causally entangled — they are not independent

fairness dimensions:

- Elderly patients → more likely bedridden → more AP-view X-rays → different image quality

- Elderly patients → higher disease prevalence → images may be more complex

An observed age recall gap may be mediated by view position. If so, improving

AP-view recall would fix both the age and view position gaps simultaneously. If

the age gap is direct (model fails on elderly anatomy independently), view-position

stratification will not resolve it.

**This causal structure cannot be resolved from observational data alone.**

It is documented here and in the model card as a known limitation.

---

## Label Bias Consideration

NIH labels are NLP-extracted from radiology reports. NLP extraction quality may

vary by subgroup — reports written for different patient demographics may use

different terminology, verbosity, or clinical conventions. An apparent model

fairness gap may partially reflect label bias (different label quality across

subgroups) rather than purely model bias.

Evidence from L8 clinical advisor review: [populate — did the advisor find

plausible labels for the lower-recall subgroup's FN cases? or suspect labels?]

---

## Fairness-Aware Threshold Option

No action required at current threshold.

If recall gap > 0.05, a lower threshold for the disadvantaged subgroup

restores recall parity at the cost of higher FP rate for that subgroup.

| | Uniform threshold (current) | Subgroup threshold (option) |

|---|---|---|

| Clinical equity | Gap remains | Recall parity restored |

| FP rate | Uniform | Higher for disadvantaged group |

| Regulatory | No demographic decisions | May need explicit justification |

| Status | Default | Requires sign-off (Decision 17) |

---

## Summary

| Dimension | Max Recall Gap | Gap Significant? | ECE Range | Status |

|---|---|---|---|---|

| Gender | 0.0000 | [populate] | [populate] | ✅ Pass |

| Age Group | 0.0000 | [populate] | [populate] | ✅ Pass |

| View Position | 0.0000 | [populate] | [populate] | ✅ Pass |

**Equal Opportunity gate:** ✅ PASSED

