# Experiment 09 — Word and Character TF-IDF

**Date:** 13 October 2026  
**Status:** Completed

## Objective

Determine whether character-level patterns complement word-level TF-IDF and improve the full-keywords Logistic Regression model from Experiment 07.

## Method

The experiment used the same cleaned dataset and stratified split as the preceding experiments:

- Training: 2,086 instances (80%)
- Validation: 522 instances (20%)
- Random seed: 42

All four fields—`description`, `sample_name`, `transcription`, and complete `keywords`—were combined into one marked document. Two representations were compared:

1. word-level TF-IDF with unigrams and bigrams;
2. the same word representation combined with character `char_wb` n-grams of length 3 to 5.

Both vectorisers used a minimum document frequency of 2, sublinear term frequency, and a maximum of 100,000 features. The combined representation therefore contained up to 200,000 features. Classification used Logistic Regression with `C=1.0`, balanced class weights, the `lbfgs` solver, and a maximum of 2,000 iterations.

The official unlabelled evaluation set was not used.

## Results

| Configuration | Features | Accuracy | Macro-F1 | Minimum F1 | Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Word TF-IDF | 100,000 | 0.525 | 0.565 | 0.346 (`Radiology`) | 274 |
| Word + character TF-IDF | 200,000 | **0.561** | **0.590** | 0.286 (`Neurosurgery`) | **293** |

Adding character features changed 41 predictions. Twenty-eight incorrect predictions became correct, while nine correct predictions became incorrect, producing a net gain of 19 correct classifications.

The combined model obtained its strongest F1 scores for `Psychiatry-Psychology` (0.800), `General Medicine` (0.769), and `Dermatology` (0.750). Every class remained above an F1 score of 0.25.

## Interpretation

Character n-grams improved both accuracy and Macro-F1. They capture useful subword information such as medical prefixes, suffixes, spelling variants, and terminology shared across related expressions. This representation is therefore more robust than word features alone.

The minimum per-class F1 decreased from 0.346 to 0.286 because `Neurosurgery` became weaker, but the overall class-balanced performance improved. This experiment produced the strongest classical model and established the main non-transformer reference for the ClinicalBERT experiments.

## Generated files

- `results/09_word_character_tfidf_report.txt`
- `results/09_word_character_tfidf_summary.csv`
- `results/09_word_character_tfidf_validation_predictions.csv`
