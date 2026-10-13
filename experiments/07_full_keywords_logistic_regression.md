# Experiment 07 — Multi-Field Logistic Regression with Full Keywords

**Date:** 12 October 2026  
**Status:** Completed

## Objective

Determine whether Logistic Regression provides better and more balanced performance than the Linear SVM when both models receive all input fields and the complete keywords.

## Method

The experiment used the same cleaned dataset and stratified split as the preceding experiments:

- Training: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

All four fields—`description`, `sample_name`, `transcription`, and complete `keywords`—were combined into one marked document. The TF-IDF configuration was kept identical to Experiment 05: word unigrams and bigrams, a minimum document frequency of 2, sublinear term frequency, and a maximum of 100,000 features.

Only the classifier was changed. The Linear SVM was replaced by class-balanced Logistic Regression with `C=1.0`, the `lbfgs` solver, and a maximum of 2,000 iterations. This makes Experiment 07 a controlled classifier comparison with Experiment 05.

The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.525
- Macro-F1: 0.565
- Minimum per-class F1: 0.346 (`Radiology`)
- Correct predictions: 274 of 522

No class obtained an F1 score below 0.25. The highest per-class results were obtained for `Psychiatry-Psychology` (F1 0.800), `Dermatology` (F1 0.750), and `Ophthalmology` (F1 0.706).

## Interpretation

Compared with the full-keywords Linear SVM from Experiment 05, Logistic Regression increased accuracy from 0.521 to 0.525 and Macro-F1 from 0.555 to 0.565. The overall improvement was modest, corresponding to two additional correct predictions.

The most important improvement concerned the weakest classes: the minimum per-class F1 rose from 0.154 to 0.346, and no class remained below 0.25. Logistic Regression therefore produced the strongest and most balanced result obtained up to this point and became the best candidate for the final system.

The model was subsequently used in Experiment 08 to isolate the contribution of each metadata source.

## Generated files

- `results/07_full_keywords_logistic_regression_report.txt`
- `results/07_full_keywords_logistic_regression_validation_predictions.csv`
