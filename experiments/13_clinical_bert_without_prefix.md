# Experiment 13 — ClinicalBERT without the Label-Like Keyword Prefix

**Date:** 13 October 2026  
**Status:** Completed

## Objective

Measure ClinicalBERT's performance after removing the first keyword whenever it directly names one of the twelve target specialties.

## Method

This was a controlled ablation of Experiment 12. The same cleaned data, stratified split, field order, model, random seed, maximum sequence length, optimiser, learning rate, batch size, number of epochs, and balanced class weights were retained.

The only intended change was the treatment of `keywords`. A specialty-like first element was removed before constructing the model input. This affected:

- 1,737 of the 2,086 training instances;
- 434 of the 522 validation instances.

The official unlabelled evaluation set was not used.

## Results

- Accuracy: 0.5345
- Macro-F1: 0.5739
- Weighted F1: 0.472
- Minimum per-class F1: 0.2222 (`Radiology`)
- Correct predictions: 279 of 522

The model produced predictions for all twelve classes. Its strongest F1 scores included `Dermatology` (0.750), `General Medicine` (0.733), `Psychiatry-Psychology` (0.700), and `Gastroenterology` (0.692).

The weakest classes were `Radiology` (0.222) and `Surgery` (0.249). For `Surgery`, precision was 0.962 but recall was only 0.143: the model predicted this label only 26 times despite 175 validation examples belonging to the class.

## Interpretation

Removing the specialty-like prefix reduced accuracy from 0.925 to 0.534 and Macro-F1 from 0.914 to 0.574. The difference demonstrates that the complete-keywords result depends strongly on the direct metadata cue.

Despite this reduction, ClinicalBERT remained useful. Under the same no-prefix condition, it outperformed the Logistic Regression result from Experiment 08, increasing accuracy from 0.496 to 0.534 and Macro-F1 from 0.527 to 0.574.

On the 88 validation examples that originally lacked a specialty-like prefix, the no-prefix model classified 56 correctly, compared with 49 for the complete-keywords ClinicalBERT. Removing the shortcut during training therefore encouraged the model to learn more from the remaining text.

Balanced class weights protected the minority specialties, but appeared to overcompensate for imbalance by severely reducing `Surgery` recall. This motivated an ablation without class weights.

## Generated files

- `results/13_clinical_bert_without_prefix_report.txt`
- `results/13_clinical_bert_without_prefix_training_log.csv`
- `results/13_clinical_bert_without_prefix_validation_predictions.csv`
