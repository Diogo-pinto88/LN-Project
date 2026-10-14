# Experiment 01 — Initial Data Analysis

**Date:** 9 October 2026  
**Status:** Completed

## Objective

Inspect the provided training and evaluation datasets before applying preprocessing or training classification models.

## Dataset dimensions

The datasets were loaded as semicolon-separated CSV files.

- Raw training set: 2,616 rows and 5 columns
- Final clean evaluation set: 398 rows and 4 columns
- Expected number of target classes: 12

The training columns are:

- `medical_specialty`
- `description`
- `sample_name`
- `transcription`
- `keywords`

The final clean evaluation set contains the same input fields but does not contain `medical_specialty`. It replaces the earlier 409-row evaluation file and is the only evaluation file to be used when generating the final predictions.

## Class distribution

After removing the eight malformed rows, the class counts were:

| Specialty | Instances |
|---|---:|
| Surgery | 874 |
| Cardiovascular-Pulmonary | 299 |
| Orthopedic | 292 |
| Radiology | 217 |
| General Medicine | 206 |
| Gastroenterology | 192 |
| Neurology | 181 |
| Obstetrics-Gynecology | 134 |
| Neurosurgery | 77 |
| Ophthalmology | 71 |
| Psychiatry-Psychology | 40 |
| Dermatology | 25 |

The cleaned dataset is strongly imbalanced. The largest class, Surgery, contains approximately 35 times more examples than the smallest class, Dermatology.

## Invalid labels

Although the task defines 12 specialties, 15 different values were found in the `medical_specialty` column.

Three rows contained fragments of clinical text instead of valid specialty labels. These rows were identified using the predefined list of twelve expected labels.

## Concatenated records

Five additional rows had valid labels but contained abnormally long descriptions, ranging from approximately 28,700 to 32,800 characters. Four of these rows also contained at least three missing input fields. Manual inspection showed that the descriptions contained multiple concatenated records and fragments such as `Orthopedic` or `Surgery;`.

The affected row indices were:

- 710 — General Medicine
- 931 — Neurosurgery
- 1058 — Obstetrics-Gynecology
- 2051 — Surgery
- 2280 — Surgery

## Cleaning decision

Eight malformed training rows were excluded:

- three rows with invalid labels;
- five rows containing abnormally long, concatenated records.

The records were excluded only from the in-memory training dataframe. The original CSV file was not modified.

After this operation, the cleaned training set contains:

- 2,608 rows;
- 5 columns;
- exactly 12 target labels.

No rows were removed from the final clean evaluation set because the order and number of predictions must remain unchanged.

## Missing values after row removal

The following counts were measured after removing malformed training rows and before replacing missing inputs with empty strings.

### Training set

| Field | Missing | Percentage |
|---|---:|---:|
| medical_specialty | 0 | 0.00% |
| description | 0 | 0.00% |
| sample_name | 0 | 0.00% |
| transcription | 26 | 1.00% |
| keywords | 398 | 15.26% |

### Final clean evaluation set

| Field | Missing | Percentage |
|---|---:|---:|
| description | 0 | 0.00% |
| sample_name | 0 | 0.00% |
| transcription | 2 | 0.50% |
| keywords | 72 | 18.09% |

The keyword field has the highest proportion of missing values in both datasets. Therefore, the final system cannot rely exclusively on this field.

For modelling, these missing input values were represented as empty strings so that every record could be processed without changing the number or order of instances.

## Label-like keyword prefix

In 2,171 of the 2,608 cleaned training instances (83.24%), the first comma-separated keyword directly maps to one of the twelve target specialties. For every covered training instance, this mapped specialty agrees with the target label. The prefix is therefore an unusually strong label-like cue and must be analysed separately from the remaining clinical text.

The same kind of prefix occurs in 320 of the 398 final evaluation instances (80.40%). Since this set is unlabelled, no evaluation-set performance was calculated during development.

## Conflicting exact duplicates

The cleaned training set contains 79 groups of instances whose four input fields are exactly identical. These groups contain 165 rows in total, and every group is associated with more than one target specialty. This indicates annotation ambiguity: an identical document may have different correct labels depending on the row.

## Main observations

1. The dataset is strongly imbalanced, particularly between Surgery and Dermatology.
2. Eight training records are clearly malformed.
3. Keywords are missing in a substantial proportion of both datasets.
4. The first keyword directly identifies the target specialty in 83.24% of the cleaned training instances.
5. Exact duplicate inputs can have conflicting target labels.
6. Evaluation instances cannot be removed, even when fields are missing, because every input row requires a prediction.

## Next steps

- Keep the final clean evaluation set completely separate from model development.
- Report performance both with and without the label-like keyword prefix.
- Use per-class metrics to account for class imbalance.
- Inspect misclassifications for annotation conflicts and overlapping specialties.
