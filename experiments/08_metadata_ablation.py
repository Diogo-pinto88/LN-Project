import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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


CONFIGURATIONS = [
    (
        "description_only",
        "Description only",
    ),
    (
        "clinical_text_no_keywords",
        "Description + sample name + transcription",
    ),
    (
        "all_fields_without_prefix",
        "All fields without label-like keyword prefix",
    ),
    (
        "all_fields_full_keywords",
        "All fields with complete keywords",
    ),
]


def remove_label_like_keyword_prefix(keywords):
    """
    Remove the first keyword when it directly names
    one of the twelve target specialties.
    """
    parts = keywords.split(",", maxsplit=1)
    first_keyword = parts[0].strip().lower()

    if first_keyword in LABEL_LIKE_KEYWORDS:
        if len(parts) == 2:
            return parts[1].strip()

        return ""

    return keywords


def build_text(dataframe, configuration):
    """
    Construct the input text according to the selected
    metadata configuration.
    """
    if configuration == "description_only":
        return (
            "DESCRIPTION "
            + dataframe["description"]
        )

    if configuration == "clinical_text_no_keywords":
        return (
            "DESCRIPTION "
            + dataframe["description"]
            + " SAMPLE_NAME "
            + dataframe["sample_name"]
            + " TRANSCRIPTION "
            + dataframe["transcription"]
        )

    if configuration == "all_fields_without_prefix":
        keywords_without_prefix = dataframe[
            "keywords"
        ].apply(
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

    if configuration == "all_fields_full_keywords":
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

    raise ValueError(
        f"Unknown configuration: {configuration}"
    )


def create_model():
    """
    Create exactly the same TF-IDF and Logistic Regression
    pipeline for every metadata configuration.
    """
    return Pipeline(
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
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# Load and clean the training dataset
raw_train, raw_test = load_data()
train, _ = clean_data(raw_train, raw_test)


# Split the indices only once so every configuration uses
# exactly the same training and validation examples
training_indices, validation_indices = train_test_split(
    train.index,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train["medical_specialty"],
)

training_data = train.loc[
    training_indices
].copy()

validation_data = train.loc[
    validation_indices
].copy()

y_train = training_data["medical_specialty"]
y_validation = validation_data["medical_specialty"]


print("=== METADATA ABLATION ===")
print("Training instances:", len(training_data))
print("Validation instances:", len(validation_data))


summary_rows = []
detailed_reports = []

validation_results = validation_data[
    [
        "description",
        "sample_name",
        "keywords",
    ]
].copy()

validation_results["correct_label"] = y_validation


for configuration, configuration_name in CONFIGURATIONS:
    print(
        "\n========================================"
    )
    print("Configuration:", configuration_name)
    print(
        "========================================"
    )

    X_train = build_text(
        training_data,
        configuration,
    )

    X_validation = build_text(
        validation_data,
        configuration,
    )

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

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

    minimum_f1 = per_class_f1[
        minimum_f1_label
    ]

    number_below_threshold = sum(
        score < 0.25
        for score in per_class_f1.values()
    )

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")

    print(
        "Minimum per-class F1:",
        f"{minimum_f1_label} = {minimum_f1:.3f}",
    )

    print(
        "Classes with F1 below 0.25:",
        number_below_threshold,
    )

    print("\nClassification report:")
    print(report_text)

    summary_rows.append(
        {
            "configuration": configuration,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "minimum_f1_label": minimum_f1_label,
            "minimum_f1": minimum_f1,
            "classes_below_0.25": (
                number_below_threshold
            ),
        }
    )

    detailed_reports.append(
        (
            configuration_name,
            report_text,
        )
    )

    validation_results[
        f"prediction_{configuration}"
    ] = predictions


# Create summary table
summary = pd.DataFrame(summary_rows)

print("\n=== SUMMARY ===")
print(
    summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)


# Save results
RESULTS_DIR.mkdir(exist_ok=True)

summary_path = (
    RESULTS_DIR
    / "08_metadata_ablation_summary.csv"
)

report_path = (
    RESULTS_DIR
    / "08_metadata_ablation_report.txt"
)

predictions_path = (
    RESULTS_DIR
    / "08_metadata_ablation_validation_predictions.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)


with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:
    file.write("Metadata ablation with Logistic Regression\n")
    file.write("==========================================\n\n")

    file.write(f"Random state: {RANDOM_STATE}\n")
    file.write(f"Validation size: {VALIDATION_SIZE}\n")
    file.write(
        f"Training instances: {len(training_data)}\n"
    )
    file.write(
        f"Validation instances: {len(validation_data)}\n"
    )

    file.write("\nModel configuration:\n")
    file.write("- TF-IDF word n-grams: (1, 2)\n")
    file.write("- Minimum document frequency: 2\n")
    file.write("- Maximum features: 100000\n")
    file.write("- Logistic Regression C: 1.0\n")
    file.write("- Class weight: balanced\n")
    file.write("- Solver: lbfgs\n")
    file.write("- Maximum iterations: 2000\n")

    file.write("\nSummary:\n")
    file.write(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    file.write("\n\n")

    for configuration_name, report_text in detailed_reports:
        file.write(
            "========================================\n"
        )
        file.write(
            f"{configuration_name}\n"
        )
        file.write(
            "========================================\n\n"
        )
        file.write(report_text)
        file.write("\n\n")


validation_results.index.name = "source_index"

validation_results.to_csv(
    predictions_path,
    index=True,
)


print("\nFiles created:")
print("-", summary_path)
print("-", report_path)
print("-", predictions_path)