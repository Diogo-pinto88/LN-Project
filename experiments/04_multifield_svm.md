# Experiment 04 — Multi-Field Linear SVM

**Date:** 10 October 2026  
**Status:** Completed

## Objective

Evaluate whether combining all four input fields with a class-balanced linear classifier improves performance over the description-only baseline.

## Method

The cleaned training set contained 2,608 instances. The same stratified split used in Experiment 03 was maintained:

- Training: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

The `description`, `sample_name`, `transcription`, and `keywords` fields were combined into a single marked document. When the first comma-separated keyword directly named one of the twelve target specialties, that label-like prefix was removed. This prevented the model from relying on an unusually direct indication of the correct class.

The combined text was represented using word-level TF-IDF with unigrams and bigrams, a minimum document frequency of 2, sublinear term frequency, and a maximum of 100,000 features. Classification was performed using a Linear Support Vector Machine with `C=1.0` and balanced class weights.

The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.379
- Macro-F1: 0.418
- Minimum per-class F1: 0.154 (`Neurosurgery`)
- Correct predictions: 198 of 522

Three classes obtained an F1 score below 0.25: `Neurology`, `Neurosurgery`, and `Radiology`.

Unlike the baseline, the model produced predictions for all twelve specialties. Its strongest per-class results included `General Medicine` (F1 0.757), `Psychiatry-Psychology` (F1 0.737), and `Ophthalmology` (F1 0.667).

## Interpretation

Accuracy decreased slightly relative to the baseline, from 0.385 to 0.379, but Macro-F1 increased substantially from 0.117 to 0.418. The model no longer collapsed towards `Surgery` and distributed its predictions across all twelve classes.

This experiment demonstrates why accuracy alone is insufficient for this imbalanced dataset. Combining the available fields with class balancing produced a much fairer classifier, although performance remained weak for `Neurology`, `Neurosurgery`, and `Radiology`.

The next experiment keeps the same split, features, and classifier while restoring the complete `keywords` field to measure the influence of its label-like first element.

## Generated files

- `results/04_multifield_svm_report.txt`
- `results/04_multifield_svm_validation_predictions.csv`
