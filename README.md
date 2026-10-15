# Medical Specialty Classification

Natural Language project for Group 38:

- Diogo Pinto (`ist1103976`)
- Pedro Sanguinetti (`ist1107283`)

The task is to assign each clinical document to one of twelve medical
specialties. The repository contains the data inspection, classical machine
learning baselines, ClinicalBERT experiments, final error analysis, and the
code used to generate the official `results.txt` file.

## Final system

The selected system is a metadata-gated hybrid:

1. The first comma-separated keyword is normalised and compared with the
   twelve valid specialties.
2. If it names a specialty, that deterministic metadata prediction is used.
3. Otherwise, the instance is classified by
   `emilyalsentzer/Bio_ClinicalBERT`, fine-tuned without the label-like keyword
   prefix and with balanced class weights.

This design makes the system's use of the strong metadata cue explicit while
retaining a clinical-text model for records where the cue is unavailable. The
official evaluation set is never used for training or model selection.

On the fixed 522-instance validation split, the final hybrid obtained:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9387 |
| Macro-F1 | 0.9233 |
| Weighted F1 | 0.9401 |
| Minimum per-class F1 | 0.7368 |

These validation results should not be interpreted purely as clinical-text
understanding: 434 of the 522 validation cases were resolved by the unusually
direct keyword cue. The no-prefix ClinicalBERT fallback classified 56 of the
remaining 88 cases correctly.

## Data

All CSV files use a semicolon (`;`) as the separator.

- `data/train.csv`: 2,616 parsed labelled records.
- `data/test_no_labels_clean.csv`: the updated clean evaluation set containing
  398 unlabelled records.
- `data/test_no_labels.csv`: the earlier evaluation file, retained only for
  provenance and not used by the final system.

The deterministic cleaning procedure removes eight malformed training rows:
rows with an invalid target label, at least three missing input fields, or a
description longer than 20,000 characters. Missing input fields in retained
records are represented as empty strings. This produces 2,608 training
instances. No rows are removed from the clean evaluation set.

## Repository structure

```text
data/               Training and unlabelled evaluation data
src/                Shared data loading, cleaning, and early utilities
experiments/        Experiment scripts and their methodological notes
results/            Validation reports, predictions, tables, and figures
scripts/            Final analysis and final-system training scripts
Project-Template/   LaTeX paper template and bibliography
results.txt         Final predictions in official submission format
```

The primary entry point for the submitted system is
`scripts/train_final_hybrid.py`. The files under `experiments/` preserve the
development history and are not required merely to read the already generated
`results.txt`.

## Environment setup

The project was tested with Python 3.14.7 on macOS. From the repository root,
create an isolated virtual environment and install the locked dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first ClinicalBERT execution requires an internet connection to download
`emilyalsentzer/Bio_ClinicalBERT`. Training can run on CPU; runtime depends on
the available hardware. CUDA or Apple Silicon acceleration is used when it is
available to PyTorch and selected by the Transformers trainer.

## Reproduce the final output

### 1. Validate the data without training

```bash
python scripts/train_final_hybrid.py --validate-data-only
```

The expected summary is:

```text
Clean training instances: 2608
Clean test instances: 398
Training specialties: 12
Training keyword prefixes removed: 2171
Test instances covered by keyword rule: 320
Test instances requiring ClinicalBERT: 78
```

### 2. Train on all cleaned training data and predict the clean test set

Run this command from a clean checkout:

```bash
python scripts/train_final_hybrid.py
```

The final ClinicalBERT configuration is:

| Parameter | Value |
| --- | --- |
| Pre-trained model | `emilyalsentzer/Bio_ClinicalBERT` |
| Random seed | 42 |
| Maximum sequence length | 128 tokens |
| Epochs | 2 |
| Training batch size | 8 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Class weights | Balanced |

One recoverable checkpoint is stored under `.checkpoints/final_hybrid/`. If a
run is interrupted after a checkpoint has been saved, resume it with:

```bash
python scripts/train_final_hybrid.py --resume
```

Checkpoints and downloaded model weights are intentionally excluded from Git
and from the submission archive.

### 3. Generated final files

The final command creates:

- `results.txt`: exactly 398 predictions, one per line, without a header and
  in the same order as `data/test_no_labels_clean.csv`;
- `results/final_test_predictions.csv`: an auditable table containing the
  prediction and decision source for every evaluation row;
- `results/final_training_report.txt`: configuration and prediction counts;
- `results/final_training_log.csv`: the ClinicalBERT training log.

The script stops with an error if the row count, order, or label set is
invalid. For the saved final run, 320 predictions came from the keyword rule
and 78 from the ClinicalBERT fallback. All 398 outputs belong to the twelve
allowed specialties.

## Reproduce the experiments

Classical baselines and ablations can be run independently:

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

ClinicalBERT experiments are substantially slower:

```bash
python experiments/12_clinical_bert_diogo.py
python experiments/13_clinical_bert_without_prefix.py
python experiments/14_clinical_bert_without_prefix_no_weights.py
python experiments/15_clinical_bert_soft_weights.py
```

Experiment 16 consumes the saved validation predictions from Experiment 13
and applies the metadata gate:

```bash
python experiments/16_metadata_gated_hybrid.py
```

After the experiment outputs are available, regenerate the comparison tables,
confusion matrix, duplicate analysis, and discussion examples with:

```bash
python scripts/final_analysis.py
```

Each completed experiment has a corresponding Markdown explanation under
`experiments/` and a report under `results/`.

## Main validation comparison

All values below use the same stratified 80/20 split with random seed 42.

| Experiment | Model | Accuracy | Macro-F1 | Minimum F1 |
| --- | --- | ---: | ---: | ---: |
| 03 | Description TF-IDF + Multinomial NB | 0.3851 | 0.1173 | 0.0000 |
| 06 | Augmented no-prefix TF-IDF + Linear SVM | 0.3774 | 0.4163 | 0.1538 |
| 07 | Full-keyword TF-IDF + Logistic Regression | 0.5249 | 0.5654 | 0.3457 |
| 09 | Word + character TF-IDF + Logistic Regression | 0.5613 | 0.5902 | 0.2857 |
| 12 | Full-keyword ClinicalBERT | 0.9253 | 0.9139 | 0.7778 |
| 13 | No-prefix ClinicalBERT with balanced weights | 0.5345 | 0.5739 | 0.2222 |
| 16 | Metadata-gated hybrid | **0.9387** | **0.9233** | **0.7368** |

The official test labels are not available, so no test accuracy or test F1 is
reported.

## Submission checklist

The final Group 38 ZIP archive must include:

- `38.pdf`, using the supplied paper structure and respecting the three-page
  limit;
- the project source code and `requirements.txt`;
- `results.txt` with exactly 398 lines and no header.

Do not include `.venv/`, `.checkpoints/`, downloaded pre-trained weights,
Python caches, or Git metadata. Before creating the ZIP, reproduce the project
from a clean copy and confirm that `results.txt` remains aligned with the clean
398-row evaluation file.
