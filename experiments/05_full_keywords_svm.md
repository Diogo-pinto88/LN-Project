# Experiment 05 — Multi-Field Linear SVM with Full Keywords

**Date:** 10 October 2026  
**Status:** Completed

## Objective

Measure how much the Linear SVM benefits from keeping the complete `keywords` field, including its specialty-like first element.

## Method

The experiment used the same cleaned dataset and the same stratified split as Experiments 03 and 04:

- Training: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

The `description`, `sample_name`, `transcription`, and complete `keywords` fields were combined into one marked document. In contrast to Experiment 04, the first keyword was not removed.

All other components were kept unchanged: word-level TF-IDF with unigrams and bigrams, a minimum document frequency of 2, sublinear term frequency, a maximum of 100,000 features, and a class-balanced Linear SVM with `C=1.0`. Consequently, the treatment of the keyword prefix was the only intended difference between Experiments 04 and 05.

The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.521
- Macro-F1: 0.555
- Minimum per-class F1: 0.154 (`Neurosurgery`)
- Correct predictions: 272 of 522

Only `Neurosurgery` obtained an F1 score below 0.25. The model achieved particularly strong results for `Dermatology` (F1 0.889), `General Medicine` (F1 0.796), `Psychiatry-Psychology` (F1 0.737), and `Gastroenterology` (F1 0.725).

## Interpretation

Restoring the complete keywords increased accuracy from 0.379 to 0.521 and Macro-F1 from 0.418 to 0.555 relative to Experiment 04. The model produced 74 additional correct predictions on the same validation set.

This large improvement confirms that the first keyword provides a powerful label-like cue. Although the field is part of the provided input, relying on it may lead to an overly optimistic estimate of how well the system understands the clinical text. The result therefore motivates a controlled metadata ablation and the reporting of performance both with and without this prefix.

Despite the overall improvement, `Neurosurgery` remained difficult and retained the same minimum F1 of 0.154.

## Generated files

- `results/05_full_keywords_svm_report.txt`
- `results/05_full_keywords_svm_validation_predictions.csv`
