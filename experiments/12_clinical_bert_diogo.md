# Experiment 12 — ClinicalBERT with Complete Keywords

**Date:** 13 October 2026  
**Status:** Completed

## Objective

Evaluate whether a transformer pretrained on clinical text can outperform the word- and character-based TF-IDF model when all provided input fields are available.

## Method

The experiment used the same cleaned dataset and stratified split as the classical models:

- Training: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

The pretrained `emilyalsentzer/Bio_ClinicalBERT` model was fine-tuned for twelve-class sequence classification. Each input contained `sample_name`, `description`, complete `keywords`, and `transcription`, in that order. Metadata was placed before the transcription so that it remained visible after truncation.

The main training parameters were:

- Maximum sequence length: 128 tokens
- Epochs: 2
- Batch size: 8
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Optimiser: AdamW
- Class weights: balanced
- Model-selection metric: validation Macro-F1

Training used the Mac MPS device. The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.9253
- Macro-F1: 0.9139
- Weighted F1: 0.927
- Minimum per-class F1: 0.7778 (`Psychiatry-Psychology`)
- Correct predictions: 483 of 522

Every specialty obtained an F1 score above 0.77. The strongest results included `Dermatology` (1.000), `Surgery` (0.971), `Ophthalmology` (0.966), and `Orthopedic` (0.947).

A post-hoc analysis showed that 434 validation instances contained a first keyword that directly identified their target specialty. ClinicalBERT followed this prefix and classified all 434 correctly. On the remaining 88 instances, it classified 49 correctly, corresponding to an accuracy of 0.557.

## Interpretation

ClinicalBERT greatly outperformed the classical reference, increasing accuracy from 0.561 to 0.925 and Macro-F1 from 0.590 to 0.914. However, the perfect result on all 434 prefix-covered examples shows that much of this improvement came from the label-like keyword cue.

The experiment is valid because `keywords` is a provided input field, but the headline result cannot be interpreted purely as clinical-text understanding. A controlled ClinicalBERT ablation without the specialty-like prefix was therefore required.

The second epoch was selected as the best checkpoint. Accuracy remained at 0.925, while Macro-F1 increased from 0.907 to 0.914 and validation loss decreased from 0.488 to 0.297.

## Generated files

- `results/12_clinical_bert_diogo_report.txt`
- `results/12_clinical_bert_diogo_training_log.csv`
- `results/12_clinical_bert_diogo_validation_predictions.csv`
