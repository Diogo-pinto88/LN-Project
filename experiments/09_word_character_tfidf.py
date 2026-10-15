import sys
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from data import EXPECTED_LABELS, clean_data, load_data


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


CONFIGURATIONS = [
    (
        "word_only",
        "Word TF-IDF",
    ),
    (
        "word_and_character",
        "Word and character TF-IDF",
    ),
]


def combine_fields(dataframe):
    """
    Combine all four input fields into one marked document.

    The complete keywords field is maintained so that the
    word-only configuration reproduces Experiment 07.
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


def create_word_vectorizer():
    """
    Create the same word-level TF-IDF representation used
    in Experiments 07 and 08.
    """
    return TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
    )


def create_character_vectorizer():
    """
    Represent fragments of words using character n-grams.

    char_wb creates character fragments inside word
    boundaries, reducing fragments that cross unrelated words.
    """
    return TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
    )


def create_classifier():
    """
    Create the same Logistic Regression classifier used
    in Experiments 07 and 08.
    """
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2_000,
        random_state=RANDOM_STATE,
    )


def create_model(configuration):
    """
    Create either the original word-only model or the new
    model combining word and character features.
    """
    if configuration == "word_only":
        features = create_word_vectorizer()

    elif configuration == "word_and_character":
        features = FeatureUnion(
            [
                (
                    "word_tfidf",
                    create_word_vectorizer(),
                ),
                (
                    "character_tfidf",
                    create_character_vectorizer(),
                ),
            ]
        )

    else:
        raise ValueError(
            f"Unknown configuration: {configuration}"
        )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "classifier",
                create_classifier(),
            ),
        ]
    )


def count_features(model, configuration):
    """
    Count the number of TF-IDF features created by a fitted
    model.
    """
    feature_extractor = model.named_steps["features"]

    if configuration == "word_only":
        return len(feature_extractor.vocabulary_)

    return sum(
        len(transformer.vocabulary_)
        for _, transformer
        in feature_extractor.transformer_list
    )


# Load and clean the data
raw_train, raw_test = load_data()
train, _ = clean_data(raw_train, raw_test)


# Split indices once so both configurations receive
# exactly the same training and validation documents
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


X_train = combine_fields(training_data)
X_validation = combine_fields(validation_data)

y_train = training_data["medical_specialty"]
y_validation = validation_data["medical_specialty"]


print("=== WORD AND CHARACTER TF-IDF EXPERIMENT ===")
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

    model = create_model(configuration)

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

    correct_predictions = (
        predictions == y_validation
    ).sum()

    number_of_features = count_features(
        model,
        configuration,
    )

    print(f"Features: {number_of_features}")
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

    print(
        "Correct predictions:",
        f"{correct_predictions} of {len(y_validation)}",
    )

    print("\nClassification report:")
    print(report_text)

    summary_rows.append(
        {
            "configuration": configuration,
            "number_of_features": number_of_features,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "minimum_f1_label": minimum_f1_label,
            "minimum_f1": minimum_f1,
            "classes_below_0.25": (
                number_below_threshold
            ),
            "correct_predictions": (
                correct_predictions
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

    validation_results[
        f"is_correct_{configuration}"
    ] = (
        predictions == y_validation
    )


# Compare the predictions made by both configurations
word_predictions = validation_results[
    "prediction_word_only"
]

hybrid_predictions = validation_results[
    "prediction_word_and_character"
]

word_correct = validation_results[
    "is_correct_word_only"
]

hybrid_correct = validation_results[
    "is_correct_word_and_character"
]


predictions_changed = (
    word_predictions != hybrid_predictions
).sum()

wrong_to_correct = (
    (~word_correct) & hybrid_correct
).sum()

correct_to_wrong = (
    word_correct & (~hybrid_correct)
).sum()


summary = pd.DataFrame(summary_rows)


print("\n=== SUMMARY ===")

print(
    summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\n=== PREDICTION COMPARISON ===")
print("Predictions changed:", predictions_changed)
print("Wrong to correct:", wrong_to_correct)
print("Correct to wrong:", correct_to_wrong)


# Create output directory
RESULTS_DIR.mkdir(exist_ok=True)


summary_path = (
    RESULTS_DIR
    / "09_word_character_tfidf_summary.csv"
)

report_path = (
    RESULTS_DIR
    / "09_word_character_tfidf_report.txt"
)

predictions_path = (
    RESULTS_DIR
    / "09_word_character_tfidf_validation_predictions.csv"
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
    file.write("Word and character TF-IDF experiment\n")
    file.write("===================================\n\n")

    file.write("Complete keywords used: yes\n")
    file.write(f"Random state: {RANDOM_STATE}\n")
    file.write(f"Validation size: {VALIDATION_SIZE}\n")

    file.write(
        f"Training instances: {len(training_data)}\n"
    )

    file.write(
        f"Validation instances: "
        f"{len(validation_data)}\n\n"
    )

    file.write("Word TF-IDF parameters:\n")
    file.write("- Analyzer: word\n")
    file.write("- N-grams: (1, 2)\n")
    file.write("- Minimum document frequency: 2\n")
    file.write("- Maximum features: 100000\n")
    file.write("- Sublinear term frequency: yes\n\n")

    file.write("Character TF-IDF parameters:\n")
    file.write("- Analyzer: char_wb\n")
    file.write("- N-grams: (3, 5)\n")
    file.write("- Minimum document frequency: 2\n")
    file.write("- Maximum features: 100000\n")
    file.write("- Sublinear term frequency: yes\n\n")

    file.write("Logistic Regression parameters:\n")
    file.write("- C: 1.0\n")
    file.write("- Class weight: balanced\n")
    file.write("- Solver: lbfgs\n")
    file.write("- Maximum iterations: 2000\n\n")

    file.write("Summary:\n")

    file.write(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    file.write("\n\nPrediction comparison:\n")
    file.write(
        f"- Predictions changed: {predictions_changed}\n"
    )
    file.write(
        f"- Wrong to correct: {wrong_to_correct}\n"
    )
    file.write(
        f"- Correct to wrong: {correct_to_wrong}\n"
    )

    for configuration_name, report_text in detailed_reports:
        file.write("\n\n")
        file.write("=" * 50)
        file.write("\n")
        file.write(configuration_name)
        file.write("\n")
        file.write("=" * 50)
        file.write("\n\n")
        file.write(report_text)


validation_results.index.name = "source_index"

validation_results.to_csv(
    predictions_path,
    index=True,
)


print("\nFiles created:")
print("-", summary_path)
print("-", report_path)
print("-", predictions_path)