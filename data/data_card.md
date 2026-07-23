# Data Card — NIH Chest-Xray14

## P4 Radiology AI Image Pipeline | 60 Seconds Academy

## Source
**Organisation:** National Institutes of Health, Clinical Center
**URL:** https://nihcc.app.box.com/v/ChestXray-NIHCC
**Citation:** Wang et al., 2017 — "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks"

## Binary Label Mapping
| Original Label | Binary | Count | Percentage |
|---|---|---|---|
| "No Finding" | 0 (Normal) | 60,361 | 53.8% |
| Any finding present | 1 (Suspicious) | 51,759 | 46.2% |

## Patient-Level Split
| Split | Patients | Images | Suspicious % | Deviation |
|---|---|---|---|---|
| Train (70%) | 21,563 | 78,566 | 46.3% | 0.1pp |
| Validation (15%) | 4,620 | 17,062 | 45.8% | 0.3pp |
| Test (15%) | 4,622 | 16,492 | 46.0% | 0.2pp |

## Reproducibility
**Split manifest SHA256:** 607ed9f845f3d1e952e772541028874c59660b02b3a8a6ba961e19f2016c199b

## Known Limitations
- NLP-extracted labels have ~90% accuracy (some label noise)
- Single-site bias (NIH Clinical Center only)
- AP view correlation risk (portable X-rays for sicker patients)
- Temporal leakage possible (patient-level split, not time-based)
