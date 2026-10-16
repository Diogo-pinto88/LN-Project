# Medical Specialty Classification

Natural Language project developed by **Group 38**:

- Diogo Pinto (`ist1103976`) — 50%
- Pedro Sanguinetti (`ist1107283`) — 50%

The objective is to assign each clinical document to one of twelve medical
specialties. The repository contains the data inspection, classical machine
learning baselines, data augmentation, ClinicalBERT experiments, final error
analysis, and the code used to generate the official `results.txt` file.

## Final system

The selected system is a **metadata-gated hybrid**:

1. The first comma-separated keyword is normalised and compared with the
   twelve valid specialties.
2. If it directly names a specialty, that deterministic metadata prediction
   is used.
3. Otherwise, the instance is classified by
   `emilyalsentzer/Bio_ClinicalBERT`, fine-tuned without the label-like keyword
   prefix and with balanced class weights.

This design makes the use of the unusually strong metadata cue explicit while
retaining a clinical-text classifier for records where the cue is unavailable.
The official evaluation set is never used for training, validation, or model
selection.

On the fixed 522-instance validation split, the final hybrid obtained:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9387 |
| Macro-F1 | 0.9233 |
| Weighted F1 | 0.9401 |
| Minimum per-class F1 | 0.7368 |

These results should not be interpreted purely as clinical-text understanding:
434 of the 522 validation cases were resolved by the direct keyword cue. The
no-prefix ClinicalBERT fallback classified 56 of the remaining 88 cases
correctly.

## Data

All CSV files use a semicolon (`;`) as the separator.

- `data/train.csv`: 2,616 parsed labelled records supplied for training.
- `data/test_no_labels_clean.csv`: updated clean evaluation set with 398
  unlabelled records; this is the test set used to generate `results.txt`.
- `data/test_no_labels.csv`: earlier 409-record evaluation file, retained only
  for provenance and not used by the final system.

Each retained record may contain four input fields:

- `description`;
- `sample_name`;
- `transcription`;
- `keywords`.

The training label is `medical_specialty`.

The deterministic cleaning procedure removes eight malformed training rows:

- records whose target is not one of the twelve valid specialties;
- records with at least three missing input fields;
- records whose description exceeds 20,000 characters.

Missing input fields in retained records are represented as empty strings. The
cleaned training set contains 2,608 instances. No rows are removed from the
clean evaluation set.

## Repository structure

```text
data/           Training data and unlabelled evaluation data
src/            Shared data loading and deterministic cleaning
experiments/    Experiment scripts and methodological notes
results/        Validation reports, predictions, tables, and figures
scripts/        Final analysis and final-system training scripts
38.pdf          Short paper for Group 38
requirements.txt
results.txt     Official predictions, one label per line and no header
```

The primary entry point is `scripts/train_final_hybrid.py`. The experiment
scripts preserve the development history but are not required merely to read
the already generated `results.txt`.

## Environment setup

The project was tested with Python 3.14.7 on macOS. From the repository root,
create an isolated virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first ClinicalBERT execution requires an internet connection to download
`emilyalsentzer/Bio_ClinicalBERT`. Training can run on CPU but may take several
hours. CUDA or Apple Silicon acceleration is used when available.

## Reproduce the official output

All commands below must be run from the repository root.

### 1. Validate the data without training

```bash
python scripts/train_final_hybrid.py --validate-data-only
```

Expected output summary:

```text
Clean training instances: 2608
Clean test instances: 398
Training specialties: 12
Training keyword prefixes removed: 2171
Test instances covered by keyword rule: 320
Test instances requiring ClinicalBERT: 78
```

### 2. Train the final model and generate the predictions

```bash
python scripts/train_final_hybrid.py
```

The fallback model is trained on all 2,608 cleaned training instances with the
following configuration:

| Parameter | Value |
| --- | --- |
| Pre-trained model | `emilyalsentzer/Bio_ClinicalBERT` |
| Random seed | 42 |
| Maximum sequence length | 128 tokens |
| Epochs | 2 |
| Training batch size | 8 |
| Prediction batch size | 8 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Class weights | Balanced |

One recoverable checkpoint is stored under `.checkpoints/final_hybrid/`. If a
run is interrupted after a checkpoint has been written, resume it with:

```bash
python scripts/train_final_hybrid.py --resume
```

Checkpoints and downloaded model weights are intentionally excluded from the
submission archive.

### 3. Generated files

The final command creates:

- `results.txt`: exactly 398 predictions, one per line, without a header and
  in the same order as `data/test_no_labels_clean.csv`;
- `results/final_test_predictions.csv`: auditable predictions and decision
  source for every evaluation record;
- `results/final_training_report.txt`: final configuration and prediction
  counts;
- `results/final_training_log.csv`: ClinicalBERT training log.

The script stops with an error if the number of rows, prediction order, or
label set is invalid. In the saved final run, 320 predictions were produced by
the keyword rule and 78 by the ClinicalBERT fallback. All outputs belong to the
twelve permitted specialties.

## Experiments

Every validation experiment uses the same stratified 80/20 split and random
seed 42. The official evaluation set is not used in these experiments.

### Data inspection and sparse models

```bash
python experiments/02_keyword_analysis.py
python experiments/03_baseline.py
python experiments/04_multifield_svm.py
python experiments/05_full_keywords_svm.py
python experiments/06_class_aware_augmentation_svm.py
python experiments/07_full_keywords_logistic_regression.py
python experiments/08_metadata_ablation.py
python experiments/09_word_character_tfidf.py
```

Experiment 06 implements the mandatory data-augmentation study. It creates
minority-targeted training copies through controlled random word deletion. The
validation data remain untouched and augmentation is applied only after the
train-validation split.

### ClinicalBERT and the final hybrid

```bash
python experiments/12_clinical_bert_diogo.py
python experiments/13_clinical_bert_without_prefix.py
python experiments/14_clinical_bert_without_prefix_no_weights.py
python experiments/15_clinical_bert_soft_weights.py
python experiments/16_metadata_gated_hybrid.py
```

Experiment 16 consumes the saved validation predictions from Experiment 13
and applies the metadata gate.

### Final analysis

After the experiment outputs are available, regenerate the model comparison,
per-class metrics, confusion matrix, duplicate analysis, and discussion
examples with:

```bash
python scripts/final_analysis.py
```

## Main validation comparison

| Experiment | Model | Accuracy | Macro-F1 | Minimum F1 |
| --- | --- | ---: | ---: | ---: |
| 03 | Description TF-IDF + Multinomial NB | 0.3851 | 0.1173 | 0.0000 |
| 04 | No-prefix multi-field TF-IDF + Linear SVM | 0.3793 | 0.4178 | 0.1538 |
| 06 | Augmented no-prefix TF-IDF + Linear SVM | 0.3774 | 0.4163 | 0.1538 |
| 07 | Full-keyword TF-IDF + Logistic Regression | 0.5249 | 0.5654 | 0.3457 |
| 09 | Word + character TF-IDF + Logistic Regression | 0.5613 | 0.5902 | 0.2857 |
| 12 | Full-keyword ClinicalBERT | 0.9253 | 0.9139 | 0.7778 |
| 13 | No-prefix ClinicalBERT with balanced weights | 0.5345 | 0.5739 | 0.2222 |
| 16 | Metadata-gated hybrid | **0.9387** | **0.9233** | **0.7368** |

The official test labels are unavailable, so no test accuracy or test F1 is
reported.

## Reproducibility notes

- Preprocessing is identical for training and evaluation records.
- The fixed random seed is 42.
- Validation records are never augmented.
- Label-like keyword prefixes are removed from the ClinicalBERT fallback
  inputs in both training and inference.
- The final model is selected using only the labelled validation split.
- The clean evaluation set is used only once the system design is fixed.
- `results.txt` preserves the original order of the 398 clean test records.

## Submission checklist

The Group 38 submission archive must be named `38.zip` and contain:

- `38.pdf`, with the three-page paper body, bibliography, and optional
  one-page appendix;
- the source code and `requirements.txt`;
- `results.txt`, with exactly 398 lines and no header;
- the data files required to reproduce the submitted output.

Do not include `.venv/`, `.checkpoints/`, downloaded model weights,
`__pycache__/`, `.git/`, `.history/`, or operating-system metadata such as
`.DS_Store`.
