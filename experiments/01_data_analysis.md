# Experiment 01 — Initial Data Analysis

**Date:** 9 October 2026  
**Status:** Completed

## Objective

Inspect the provided training and evaluation datasets before applying preprocessing or training classification models.

## Dataset dimensions

The datasets were loaded as semicolon-separated CSV files.

- Raw training set: 2,616 rows and 5 columns
- Evaluation set: 409 rows and 4 columns
- Expected number of target classes: 12

The training columns are:

- `medical_specialty`
- `description`
- `sample_name`
- `transcription`
- `keywords`

The evaluation set contains the same input fields but does not contain `medical_specialty`.

## Class distribution

The raw counts of the twelve expected specialties were:

| Specialty | Instances |
|---|---:|
| Surgery | 876 |
| Cardiovascular-Pulmonary | 299 |
| Orthopedic | 292 |
| Radiology | 217 |
| General Medicine | 207 |
| Gastroenterology | 192 |
| Neurology | 181 |
| Obstetrics-Gynecology | 135 |
| Neurosurgery | 78 |
| Ophthalmology | 71 |
| Psychiatry-Psychology | 40 |
| Dermatology | 25 |

The dataset is strongly imbalanced. Before removing malformed instances, the largest class, Surgery, contains approximately 35 times more examples than the smallest class, Dermatology.

## Invalid labels

Although the task defines 12 specialties, 15 different values were found in the `medical_specialty` column.

Three rows contained fragments of clinical text instead of valid specialty labels. These rows were identified using the predefined list of twelve expected labels.

## Concatenated records

Four additional rows had apparently valid labels but contained at least three missing input fields and abnormally long descriptions.

The description lengths ranged from approximately 29,000 to 33,000 characters. Manual inspection showed that the descriptions contained multiple concatenated records and fragments such as `Orthopedic` or `Surgery;`.

The affected row indices were:

- 931 — Neurosurgery
- 1058 — Obstetrics-Gynecology
- 2051 — Surgery
- 2280 — Surgery

## Cleaning decision

Seven malformed training rows were excluded:

- three rows with invalid labels;
- four rows containing concatenated records.

The records were excluded only from the in-memory training dataframe. The original CSV file was not modified.

After this operation, the cleaned training set contains:

- 2,609 rows;
- 5 columns;
- exactly 12 target labels.

Missing input values were represented as empty strings so that they could be processed by text-vectorisation methods.

No evaluation-set rows were removed because the order and number of predictions must remain unchanged.

## Missing values before cleaning

### Training set

| Field | Missing | Percentage |
|---|---:|---:|
| medical_specialty | 0 | 0.00% |
| description | 3 | 0.11% |
| sample_name | 7 | 0.27% |
| transcription | 33 | 1.26% |
| keywords | 405 | 15.48% |

### Evaluation set

| Field | Missing | Percentage |
|---|---:|---:|
| description | 1 | 0.24% |
| sample_name | 9 | 2.20% |
| transcription | 11 | 2.69% |
| keywords | 83 | 20.29% |

The keyword field has the highest proportion of missing values in both datasets. Therefore, the final system cannot rely exclusively on this field.

## Main observations

1. The dataset is strongly imbalanced, particularly between Surgery and Dermatology.
2. Seven training records are clearly malformed.
3. Keywords are missing in a substantial proportion of both datasets.
4. Evaluation instances cannot be removed, even when fields are missing, because every input row requires a prediction.

## Next steps

- Analyse the relationship between the keyword field and the target label.
- Measure exact and near-duplicate clinical notes.
- Define a training and validation split.
- Implement the description-only TF-IDF and Multinomial Naive Bayes baseline.