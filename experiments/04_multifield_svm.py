import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from data import (
    EXPECTED_LABELS,
    clean_data,
    load_data,
)


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

LABEL_LIKE_KEYWORDS = {
    "cardiovascular / pulmonary",
    "dermatology",
    "gastroenterology",
    "general medicine",
    "neurology",
    "neurosurgery",
    "obstetrics / gynecology",
    "ophthalmology",
    "orthopedic",
    "psychiatry / psychology",
    "radiology",
    "surgery",
}


def remove_label_like_keyword_prefix(keywords):
    """
    Remove the first keyword only when it corresponds directly
    to one of the twelve medical specialties.
    """
    parts = keywords.split(",", maxsplit=1)
    first_keyword = parts[0].strip().lower()

    if first_keyword in LABEL_LIKE_KEYWORDS:
        if len(parts) == 2:
            return parts[1].strip()

        return ""

    return keywords


def combine_fields(dataframe):
    """
    Combine the four input fields into one marked document.
    """
    keywords_without_prefix = dataframe["keywords"].apply(
        remove_label_like_keyword_prefix
    )

    return (
        "DESCRIPTION "
        + dataframe["description"]
        + " SAMPLE_NAME "
        + dataframe["sample_name"]
        + " TRANSCRIPTION "
        + dataframe["transcription"]
        + " KEYWORDS "
        + keywords_without_prefix
    )


# Load and clean data
raw_train, raw_test = load_data()
train, _ = clean_data(raw_train, raw_test)


# Combine the input fields
X = combine_fields(train)
y = train["medical_specialty"]


# Use exactly the same split as the baseline
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# Word-level TF-IDF followed by a class-balanced Linear SVM
model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=100_000,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


print("=== TRAINING MULTI-FIELD SVM ===")
print("Training instances:", len(X_train))
print("Validation instances:", len(X_validation))

model.fit(X_train, y_train)

predictions = model.predict(X_validation)


# Metrics
accuracy = accuracy_score(
    y_validation,
    predictions,
)

macro_f1 = f1_score(
    y_validation,
    predictions,
    average="macro",
    zero_division=0,
)

report_text = classification_report(
    y_validation,
    predictions,
    labels=EXPECTED_LABELS,
    zero_division=0,
    digits=3,
)

report_dictionary = classification_report(
    y_validation,
    predictions,
    labels=EXPECTED_LABELS,
    zero_division=0,
    output_dict=True,
)

per_class_f1 = {
    label: report_dictionary[label]["f1-score"]
    for label in EXPECTED_LABELS
}

minimum_f1_label = min(
    per_class_f1,
    key=per_class_f1.get,
)

minimum_f1 = per_class_f1[minimum_f1_label]

classes_below_threshold = {
    label: score
    for label, score in per_class_f1.items()
    if score < 0.25
}


print("\n=== MULTI-FIELD SVM RESULTS ===")
print(f"Accuracy: {accuracy:.3f}")
print(f"Macro-F1: {macro_f1:.3f}")

print(
    "Minimum per-class F1:",
    f"{minimum_f1_label} = {minimum_f1:.3f}",
)

print("\nClasses with F1 below 0.25:")

if classes_below_threshold:
    for label, score in classes_below_threshold.items():
        print(f"- {label}: {score:.3f}")
else:
    print("None")

print("\n=== CLASSIFICATION REPORT ===")
print(report_text)


# Prediction distribution
prediction_counts = pd.Series(
    predictions,
    name="predicted_label",
).value_counts()

print("\n=== PREDICTION COUNTS ===")
print(prediction_counts)


# Save report
RESULTS_DIR.mkdir(exist_ok=True)

report_path = RESULTS_DIR / "04_multifield_svm_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Multi-field Linear SVM\n")
    file.write("======================\n\n")

    file.write("Keyword prefix removed: yes\n")
    file.write(f"Random state: {RANDOM_STATE}\n")
    file.write(f"Validation size: {VALIDATION_SIZE}\n")
    file.write(f"Training instances: {len(X_train)}\n")
    file.write(
        f"Validation instances: {len(X_validation)}\n\n"
    )

    file.write(f"Accuracy: {accuracy:.3f}\n")
    file.write(f"Macro-F1: {macro_f1:.3f}\n")

    file.write(
        "Minimum per-class F1: "
        f"{minimum_f1_label} = {minimum_f1:.3f}\n\n"
    )

    file.write(report_text)

    file.write("\nPrediction counts:\n")
    file.write(prediction_counts.to_string())


# Save validation predictions
validation_results = train.loc[
    X_validation.index,
    [
        "description",
        "sample_name",
        "keywords",
    ],
].copy()

validation_results["transcription_excerpt"] = (
    train.loc[X_validation.index, "transcription"]
    .str.slice(0, 500)
)

validation_results["correct_label"] = y_validation
validation_results["predicted_label"] = predictions

validation_results["is_correct"] = (
    validation_results["correct_label"]
    == validation_results["predicted_label"]
)

validation_results.index.name = "source_index"

predictions_path = (
    RESULTS_DIR
    / "04_multifield_svm_validation_predictions.csv"
)

validation_results.to_csv(
    predictions_path,
    index=True,
)


print("\nFiles created:")
print("-", report_path)
print("-", predictions_path)
