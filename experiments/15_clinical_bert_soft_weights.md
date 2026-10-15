# Experiment 15 — ClinicalBERT with Softened Class Weights

**Date:** 13 October 2026  
**Status:** Completed

## Objective

Test an intermediate class-balancing strategy that protects minority specialties without penalising `Surgery` as strongly as the fully balanced loss.

## Method

The experiment retained the complete setup of Experiment 13, including removal of the specialty-like keyword prefix. The only intended change was the strength of the class weights.

Balanced inverse-frequency weights were first calculated from the training subset. Their square roots were then taken and normalised to have a mean of one. This preserved the ordering of class importance while reducing the difference between majority- and minority-class weights.

All other parameters remained unchanged:

- Training: 2,086 instances
- Validation: 522 instances
- Random seed: 42
- Maximum sequence length: 128 tokens
- Epochs: 2
- Batch size: 8
- Learning rate: `2e-5`
- Weight decay: `0.01`

The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.5517
- Macro-F1: 0.5270
- Weighted F1: 0.5246
- Minimum per-class F1: 0.0000 (`Dermatology`)
- Correct predictions: 288 of 522

The softened weights improved `Surgery` F1 from 0.249 to 0.430 relative to the fully balanced model and retained useful performance for `Neurosurgery` (0.465) and `Ophthalmology` (0.688). `Radiology` also improved from 0.222 to 0.316.

However, the model did not predict any `Dermatology` instances. Its strongest F1 scores were obtained for `General Medicine` (0.742), `Gastroenterology` (0.727), `Psychiatry-Psychology` (0.700), and `Ophthalmology` (0.688).

## Interpretation

Softened weights produced the expected intermediate behaviour. Compared with unweighted training, they preserved the same accuracy while increasing Macro-F1 from 0.427 to 0.527 and restoring predictions for `Neurosurgery` and `Ophthalmology`.

Compared with fully balanced training, they increased accuracy from 0.534 to 0.552 and improved `Surgery` and `Radiology`, but Macro-F1 decreased from 0.574 to 0.527 and `Dermatology` fell from an F1 of 0.750 to zero.

Experiment 13 therefore remains the preferred no-prefix ClinicalBERT configuration when balanced performance is the main objective. Experiment 15 demonstrates the trade-off between overall accuracy, majority-class recall, and minority-class coverage. Further tuning of weight strength was not pursued to avoid overfitting decisions to the small validation set, particularly the five `Dermatology` examples.

## Generated files

- `results/15_clinical_bert_soft_weights_report.txt`
- `results/15_clinical_bert_soft_weights_training_log.csv`
- `results/15_clinical_bert_soft_weights_validation_predictions.csv`
