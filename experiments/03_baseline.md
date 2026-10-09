# Experiment 03 — Description-Only Baseline

**Date:** 9 October 2026  
**Status:** Completed

## Objective

Implement a weak baseline based on the project specification and measure its performance on an internal validation set.

## Method

The cleaned training set contained 2,608 instances. A stratified split was used:

- Training: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

The model used only the `description` field. Text was lowercased and stemmed using the English Snowball stemmer. TF-IDF features were classified using Multinomial Naive Bayes.

The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.385
- Macro-F1: 0.117
- Minimum per-class F1: 0.000
- Correct predictions: 201 of 522

Nine classes obtained an F1 score below 0.25.

The model predicted `Surgery` for 445 of the 522 validation instances, corresponding to 85.25% of all predictions. However, only 175 validation instances had `Surgery` as their correct label.

## Interpretation

The baseline was strongly affected by class imbalance and collapsed towards the largest class. It achieved 97.7% recall for `Surgery`, but only 38.4% precision because many instances from other specialties were incorrectly classified as `Surgery`.

The model failed to identify several minority classes, including `Dermatology`, `Neurosurgery`, `Obstetrics-Gynecology`, `Ophthalmology`, and `Psychiatry-Psychology`.

These results motivate the use of additional input fields and a classifier that can compensate for class imbalance. The next experiment will combine multiple text fields with a class-balanced linear classifier.

## Generated files

- `results/03_baseline_report.txt`
- `results/03_baseline_validation_predictions.csv`