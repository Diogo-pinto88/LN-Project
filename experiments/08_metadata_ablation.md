# Experiment 08 — Metadata Ablation with Logistic Regression

**Date:** 13 October 2026  
**Status:** Completed

## Objective

Measure the contribution of the different input fields and determine how much of the best model's performance depends on the label-like first element of `keywords`.

## Method

The cleaned training set contained 2,608 instances. A single stratified split was created and reused for all four configurations:

- Training: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

Four input configurations were compared:

1. `description` only;
2. `description`, `sample_name`, and `transcription`, without keywords;
3. all four fields after removing the first keyword when it directly named a target specialty;
4. all four fields with the complete keywords.

Every configuration used a newly trained copy of the same model: word-level TF-IDF with unigrams and bigrams, a minimum document frequency of 2, sublinear term frequency, and a maximum of 100,000 features, followed by class-balanced Logistic Regression with `C=1.0`, the `lbfgs` solver, and a maximum of 2,000 iterations.

The official unlabelled evaluation set was not used.

## Results

| Configuration | Accuracy | Macro-F1 | Minimum F1 | Classes below 0.25 | Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Description only | 0.406 | 0.433 | 0.198 (`Neurology`) | 3 | 212 |
| Clinical text without keywords | 0.494 | 0.524 | 0.247 (`Radiology`) | 1 | 258 |
| All fields without keyword prefix | 0.496 | 0.527 | 0.256 (`Radiology`) | 0 | 259 |
| All fields with complete keywords | **0.525** | **0.565** | **0.346** (`Radiology`) | **0** | **274** |

Adding `sample_name` and `transcription` to the description produced 46 additional correct predictions. Adding the remaining keyword content after removing its specialty-like prefix produced only one additional correct prediction. Restoring the complete keyword field produced a further 15 correct predictions.

## Interpretation

The largest improvement came from adding the remaining clinical text: Macro-F1 increased from 0.433 to 0.524. In contrast, the keywords left after removing the specialty-like prefix contributed almost no additional performance, increasing Macro-F1 only from 0.524 to 0.527.

Restoring the complete keywords increased Macro-F1 from 0.527 to 0.565 and the minimum per-class F1 from 0.256 to 0.346. Of the 20 validation predictions that changed, 17 became correct and only two became incorrect. The first keyword therefore provides a meaningful and unusually direct advantage.

However, the model does not depend exclusively on this cue. Without the prefix, it still achieved a Macro-F1 of 0.527, predicted all twelve specialties, and kept every class above an F1 of 0.25. The clinical text contains most of the useful information, while the specialty-like keyword prefix supplies an additional performance boost that should be reported transparently.

The complete-keywords configuration exactly reproduced the predictions of Experiment 07, confirming the reproducibility of the evaluation.

## Generated files

- `results/08_metadata_ablation_summary.csv`
- `results/08_metadata_ablation_report.txt`
- `results/08_metadata_ablation_validation_predictions.csv`
