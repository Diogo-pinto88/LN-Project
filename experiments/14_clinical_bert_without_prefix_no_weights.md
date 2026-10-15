# Experiment 14 — ClinicalBERT without Prefix or Class Weights

**Date:** 13 October 2026  
**Status:** Completed

## Objective

Determine whether removing balanced class weights corrects the extremely low `Surgery` recall observed in the no-prefix ClinicalBERT model.

## Method

This experiment kept the data, split, prefix removal, input construction, ClinicalBERT model, 128-token limit, two epochs, batch size, learning rate, weight decay, optimiser, and random seed from Experiment 13.

The only intended change was the loss function. Standard unweighted cross-entropy replaced cross-entropy with balanced class weights. Consequently, every training instance contributed equally regardless of its specialty.

The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.5517
- Macro-F1: 0.4275
- Weighted F1: 0.5245
- Minimum per-class F1: 0.0000 (`Dermatology`)
- Correct predictions: 288 of 522

`Surgery` recall increased from 0.143 to 0.749 and its F1 increased from 0.249 to 0.636. The model predicted `Surgery` 237 times, compared with 175 true validation instances.

However, the model made no predictions for `Dermatology`, `Neurosurgery`, or `Ophthalmology`, giving all three classes an F1 score of zero.

## Interpretation

Removing class weights increased accuracy by 0.017 and produced nine additional correct predictions relative to Experiment 13. It successfully corrected the low `Surgery` recall but shifted the model towards the majority class and severely damaged minority-class recognition.

Macro-F1 decreased from 0.574 to 0.427, showing that the small accuracy improvement did not represent better balanced performance. Balanced training is therefore necessary for this dataset, although the full inverse-frequency weights used in Experiment 13 are too aggressive for `Surgery`.

This experiment provides a useful negative ablation: unweighted training is not appropriate when the objective includes reliable performance across all twelve specialties.

## Generated files

- `results/14_clinical_bert_without_prefix_no_weights_report.txt`
- `results/14_clinical_bert_without_prefix_no_weights_training_log.csv`
- `results/14_clinical_bert_without_prefix_no_weights_validation_predictions.csv`
