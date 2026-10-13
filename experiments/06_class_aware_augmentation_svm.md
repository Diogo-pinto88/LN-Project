# Experiment 06 — Class-Aware Augmentation with Linear SVM

**Date:** 10 October 2026  
**Status:** Completed

## Objective

Test whether targeted text augmentation improves the recognition of minority specialties when the label-like keyword prefix is unavailable.

## Method

The cleaned dataset was divided using the same stratified split as the preceding experiments:

- Original training subset: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

The split was performed before augmentation, ensuring that validation records remained untouched and that no augmented version of a validation document entered the training data.

The first keyword was removed whenever it directly named a target specialty. Classes with fewer than 150 training instances were augmented by creating modified copies in which each whitespace-separated word had a 10% probability of deletion. At least one word was always preserved, fields with fewer than five words were left unchanged, and each original example generated at most three augmented copies.

A total of 385 synthetic examples were created, increasing the training subset from 2,086 to 2,471 instances:

- `Dermatology`: 60
- `Neurology`: 5
- `Neurosurgery`: 88
- `Obstetrics-Gynecology`: 43
- `Ophthalmology`: 93
- `Psychiatry-Psychology`: 96

The same multi-field TF-IDF representation and class-balanced Linear SVM from Experiment 04 were used. The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.377
- Macro-F1: 0.416
- Minimum per-class F1: 0.154 (`Neurosurgery`)
- Correct predictions: 197 of 522

Three classes obtained an F1 score below 0.25: `Neurology`, `Neurosurgery`, and `Radiology`.

## Interpretation

The augmentation did not improve the model. Relative to the otherwise comparable Experiment 04, accuracy decreased from 0.379 to 0.377, Macro-F1 decreased from 0.418 to 0.416, and the number of correct predictions decreased from 198 to 197. The minimum per-class F1 also remained unchanged at 0.154.

Random word deletion creates variations of existing documents but introduces no new clinical vocabulary or information. Its contribution may therefore be redundant with TF-IDF, while balanced class weights already compensate for the unequal class frequencies. This particular augmentation strategy should not be selected for the final model, but its negative result is useful evidence when answering the project's research question.

## Generated files

- `results/06_class_aware_augmentation_svm_report.txt`
- `results/06_class_aware_augmentation_svm_validation_predictions.csv`
