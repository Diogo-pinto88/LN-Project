# Experiment 16 — Metadata-Gated Hybrid

**Date:** 13 October 2026  
**Status:** Completed

## Objective

Combine the reliable label-like keyword cue with the stronger text-based behaviour of the no-prefix ClinicalBERT model on instances for which that cue is unavailable.

## Method

This experiment used the 522 validation instances and saved predictions from Experiment 13. It did not train a new model and did not use the official unlabelled evaluation set.

The hybrid applied the following deterministic decision process:

1. The first comma-separated element of `keywords` was normalised and compared with the twelve known specialty names.
2. If it mapped to a specialty, that specialty became the hybrid prediction.
3. Otherwise, the prediction produced by the no-prefix ClinicalBERT model from Experiment 13 was used.

This separation was motivated by the earlier data analysis. The keyword prefix provides a direct specialty cue in 83.24% of the cleaned training set, whereas the no-prefix ClinicalBERT model is forced to rely on the remaining metadata and clinical text.

The same validation split from the previous experiments was retained:

- Training subset used by the underlying ClinicalBERT model: 2,086 instances
- Validation: 522 instances
- Random seed: 42

The hybrid used the existing Experiment 13 model without retraining or adjusting its parameters after inspecting the validation errors.

## Results

- Accuracy: 0.9387
- Macro-F1: 0.9233
- Weighted F1: 0.9401
- Minimum per-class F1: 0.7368 (`Psychiatry-Psychology`)
- Correct predictions: 490 of 522
- Classes with F1 below 0.25: 0

The decision sources were:

- Keyword rule: 434 instances, 434 correct (100.00%)
- ClinicalBERT fallback: 88 instances, 56 correct (63.64%)

### Per-class results

| Specialty | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Cardiovascular-Pulmonary | 0.903 | 0.933 | 0.918 | 60 |
| Dermatology | 1.000 | 1.000 | 1.000 | 5 |
| Gastroenterology | 0.947 | 0.947 | 0.947 | 38 |
| General Medicine | 0.796 | 0.951 | 0.867 | 41 |
| Neurology | 0.886 | 0.861 | 0.873 | 36 |
| Neurosurgery | 0.882 | 1.000 | 0.938 | 15 |
| Obstetrics-Gynecology | 0.931 | 1.000 | 0.964 | 27 |
| Ophthalmology | 0.933 | 1.000 | 0.966 | 14 |
| Orthopedic | 0.982 | 0.932 | 0.957 | 59 |
| Psychiatry-Psychology | 0.636 | 0.875 | 0.737 | 8 |
| Radiology | 1.000 | 0.886 | 0.940 | 44 |
| Surgery | 1.000 | 0.949 | 0.974 | 175 |

## Comparison with ClinicalBERT

| Configuration | Accuracy | Macro-F1 | Minimum F1 | Correct |
| --- | ---: | ---: | ---: | ---: |
| Complete-keywords ClinicalBERT (Experiment 12) | 0.9253 | 0.9139 | 0.7778 | 483 |
| No-prefix ClinicalBERT (Experiment 13) | 0.5345 | 0.5739 | 0.2222 | 279 |
| Metadata-gated hybrid (Experiment 16) | **0.9387** | **0.9233** | 0.7368 | **490** |

The hybrid produced seven more correct predictions than the complete-keywords ClinicalBERT and increased both Accuracy and Macro-F1. Its minimum F1 was slightly lower, but remained far above the 0.25 target.

Experiment 12 classified 49 of the 88 validation instances without a mapped prefix correctly. The Experiment 13 fallback classified 56 of those same instances correctly. The hybrid therefore combines the perfect mapped-prefix behaviour with the more effective no-prefix model for uncovered examples.

## Interpretation

The hybrid is the strongest validation system tested. It also makes the system's dependence on the metadata explicit instead of requiring ClinicalBERT to learn the direct mapping implicitly. When the label-like cue is absent, the decision is delegated to a model trained without that shortcut.

The high score must not be interpreted solely as evidence of clinical-text understanding. Most validation instances are resolved by an unusually direct input cue. The 63.64% fallback accuracy provides a more informative estimate of behaviour when that cue is unavailable.

This approach directly answers the research question: metadata produces the largest performance gain, while class-aware ClinicalBERT remains useful for the subset that cannot be classified through the specialty-like keyword prefix.

## Generated files

- `results/16_metadata_gated_hybrid_report.txt`
- `results/16_metadata_gated_hybrid_validation_predictions.csv`
