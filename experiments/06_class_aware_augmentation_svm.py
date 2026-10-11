import random
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
    FEATURE_COLUMNS,
    clean_data,
    load_data,
)


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

WORD_DELETION_PROBABILITY = 0.10
AUGMENTATION_TARGET = 150
MAX_AUGMENTED_COPIES_PER_EXAMPLE = 3

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
    Remove the first keyword when it directly names a target class.
    """
    parts = keywords.split(",", maxsplit=1)
    first_keyword = parts[0].strip().lower()

    if first_keyword in LABEL_LIKE_KEYWORDS:
        if len(parts) == 2:
            return parts[1].strip()

        return ""

    return keywords


def randomly_delete_words(text, random_generator):
    """
    Produce a slightly different version of a text by deleting
    approximately 10% of its whitespace-separated words.

    Very short fields are left unchanged, and at least one word is
    always preserved.
    """
    words = text.split()

    if len(words) < 5:
        return text

    remaining_words = [
        word
        for word in words
        if random_generator.random()
        > WORD_DELETION_PROBABILITY
    ]

    if not remaining_words:
        remaining_words = [random_generator.choice(words)]

    return " ".join(remaining_words)


def create_class_aware_augmentation(training_data):
    """
    Create augmented copies only for classes with fewer than 150
    original training examples.

    No original example generates more than three augmented copies.
    """
    random_generator = random.Random(RANDOM_STATE)
    augmented_rows = []
    augmentation_counts = {}

    for label in EXPECTED_LABELS:
        class_data = training_data.loc[
            training_data["medical_specialty"] == label
        ]

        original_count = len(class_data)
        desired_examples = max(
            0,
            AUGMENTATION_TARGET - original_count,
        )
        maximum_examples = (
            original_count
            * MAX_AUGMENTED_COPIES_PER_EXAMPLE
        )
        number_to_create = min(
            desired_examples,
            maximum_examples,
        )

        source_indices = list(class_data.index)
        candidate_indices = (
            source_indices
            * MAX_AUGMENTED_COPIES_PER_EXAMPLE
        )
        random_generator.shuffle(candidate_indices)

        for source_index in candidate_indices[:number_to_create]:
            augmented_row = training_data.loc[
                source_index
            ].copy()

            for column in FEATURE_COLUMNS:
                augmented_row[column] = randomly_delete_words(
                    augmented_row[column],
                    random_generator,
                )

            augmented_rows.append(augmented_row)

        augmentation_counts[label] = number_to_create

    if not augmented_rows:
        return training_data.copy(), augmentation_counts

    augmented_data = pd.DataFrame(augmented_rows)

    final_training_data = pd.concat(
        [training_data, augmented_data],
        ignore_index=True,
    )

    return final_training_data, augmentation_counts


def combine_fields(dataframe):
    """
    Combine the four input fields into one marked document.
    """
    return (
        "DESCRIPTION "
        + dataframe["description"]
        + " SAMPLE_NAME "
        + dataframe["sample_name"]
        + " TRANSCRIPTION "
        + dataframe["transcription"]
        + " KEYWORDS "
        + dataframe["keywords"]
    )


# Load and clean the data
raw_train, raw_test = load_data()
train, _ = clean_data(raw_train, raw_test)

# Remove the specialty-like keyword before splitting or augmenting
train["keywords"] = train["keywords"].apply(
    remove_label_like_keyword_prefix
)


# Split before augmentation so validation remains completely untouched
training_indices, validation_indices = train_test_split(
    train.index,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train["medical_specialty"],
)

original_training_data = train.loc[
    training_indices
].copy()

validation_data = train.loc[
    validation_indices
].copy()


# Add transformed copies only to the training subset
augmented_training_data, augmentation_counts = (
    create_class_aware_augmentation(
        original_training_data
    )
)

X_train = combine_fields(augmented_training_data)
y_train = augmented_training_data["medical_specialty"]

X_validation = combine_fields(validation_data)
y_validation = validation_data["medical_specialty"]


# Use the same TF-IDF and SVM configuration as Experiment 04
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


print("=== CLASS-AWARE AUGMENTATION ===")
print("Original training instances:", len(original_training_data))
print(
    "Augmented training instances:",
    len(augmented_training_data),
)
print("Validation instances:", len(validation_data))

print("\nAugmented examples created per class:")
for label, count in augmentation_counts.items():
    if count > 0:
        print(f"- {label}: {count}")


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


print("\n=== AUGMENTED SVM RESULTS ===")
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

report_path = (
    RESULTS_DIR
    / "06_class_aware_augmentation_svm_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Class-aware augmented multi-field Linear SVM\n")
    file.write("============================================\n\n")

    file.write("Keyword prefix removed: yes\n")
    file.write("Augmentation: random word deletion\n")
    file.write(
        "Word deletion probability: "
        f"{WORD_DELETION_PROBABILITY}\n"
    )
    file.write(
        f"Augmentation target per class: {AUGMENTATION_TARGET}\n"
    )
    file.write(
        "Maximum augmented copies per example: "
        f"{MAX_AUGMENTED_COPIES_PER_EXAMPLE}\n"
    )
    file.write(f"Random state: {RANDOM_STATE}\n")
    file.write(f"Validation size: {VALIDATION_SIZE}\n")
    file.write(
        "Original training instances: "
        f"{len(original_training_data)}\n"
    )
    file.write(
        "Augmented training instances: "
        f"{len(augmented_training_data)}\n"
    )
    file.write(
        f"Validation instances: {len(validation_data)}\n\n"
    )

    file.write("Augmented examples created per class:\n")
    for label, count in augmentation_counts.items():
        if count > 0:
            file.write(f"- {label}: {count}\n")

    file.write(f"\nAccuracy: {accuracy:.3f}\n")
    file.write(f"Macro-F1: {macro_f1:.3f}\n")

    file.write(
        "Minimum per-class F1: "
        f"{minimum_f1_label} = {minimum_f1:.3f}\n\n"
    )

    file.write(report_text)

    file.write("\nPrediction counts:\n")
    file.write(prediction_counts.to_string())


# Save validation predictions
validation_results = validation_data.loc[
    :, 
    [
        "description",
        "sample_name",
        "keywords",
    ],
].copy()

validation_results["transcription_excerpt"] = (
    validation_data["transcription"].str.slice(0, 500)
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
    / "06_class_aware_augmentation_svm_validation_predictions.csv"
)

validation_results.to_csv(
    predictions_path,
    index=True,
)


print("\nFiles created:")
print("-", report_path)
print("-", predictions_path)
