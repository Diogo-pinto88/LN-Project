import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from data import EXPECTED_LABELS


INPUT_PREDICTIONS = (
    RESULTS_DIR
    / "13_clinical_bert_without_prefix_validation_predictions.csv"
)

OUTPUT_REPORT = RESULTS_DIR / "16_metadata_gated_hybrid_report.txt"
OUTPUT_PREDICTIONS = (
    RESULTS_DIR
    / "16_metadata_gated_hybrid_validation_predictions.csv"
)


KEYWORD_TO_SPECIALTY = {
    "cardiovascular / pulmonary": "Cardiovascular-Pulmonary",
    "dermatology": "Dermatology",
    "gastroenterology": "Gastroenterology",
    "general medicine": "General Medicine",
    "neurology": "Neurology",
    "neurosurgery": "Neurosurgery",
    "obstetrics / gynecology": "Obstetrics-Gynecology",
    "ophthalmology": "Ophthalmology",
    "orthopedic": "Orthopedic",
    "psychiatry / psychology": "Psychiatry-Psychology",
    "radiology": "Radiology",
    "surgery": "Surgery",
}


REQUIRED_COLUMNS = [
    "source_index",
    "correct_label",
    "predicted_label",
    "confidence",
    "description",
    "sample_name",
    "original_keywords",
    "keywords_without_prefix",
    "transcription_excerpt",
]


def map_keyword_prefix(keywords):
    """Map the first keyword to a specialty when it names a target label."""

    first_keyword = (
        keywords
        .fillna("")
        .str.split(",")
        .str[0]
        .str.strip()
        .str.lower()
    )

    return first_keyword.map(KEYWORD_TO_SPECIALTY)


def validate_input(predictions):
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in Experiment 13 predictions: "
            + ", ".join(missing_columns)
        )

    if predictions["source_index"].duplicated().any():
        raise ValueError("Experiment 13 contains duplicate source indices.")

    unexpected_correct_labels = set(
        predictions["correct_label"].dropna()
    ) - set(EXPECTED_LABELS)

    unexpected_predictions = set(
        predictions["predicted_label"].dropna()
    ) - set(EXPECTED_LABELS)

    if unexpected_correct_labels:
        raise ValueError(
            "Unexpected correct labels: "
            + ", ".join(sorted(unexpected_correct_labels))
        )

    if unexpected_predictions:
        raise ValueError(
            "Unexpected ClinicalBERT predictions: "
            + ", ".join(sorted(unexpected_predictions))
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(INPUT_PREDICTIONS)
    validate_input(predictions)

    keyword_prediction = map_keyword_prefix(
        predictions["original_keywords"]
    )
    keyword_covered = keyword_prediction.notna()

    hybrid_prediction = predictions["predicted_label"].copy()
    hybrid_prediction.loc[keyword_covered] = keyword_prediction.loc[
        keyword_covered
    ]

    correct_labels = predictions["correct_label"]
    is_correct = hybrid_prediction.eq(correct_labels)

    accuracy = accuracy_score(correct_labels, hybrid_prediction)
    macro_f1 = f1_score(
        correct_labels,
        hybrid_prediction,
        labels=EXPECTED_LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        correct_labels,
        hybrid_prediction,
        labels=EXPECTED_LABELS,
        average="weighted",
        zero_division=0,
    )
    per_class_f1 = f1_score(
        correct_labels,
        hybrid_prediction,
        labels=EXPECTED_LABELS,
        average=None,
        zero_division=0,
    )

    minimum_f1_position = int(per_class_f1.argmin())
    minimum_f1_label = EXPECTED_LABELS[minimum_f1_position]
    minimum_f1 = float(per_class_f1[minimum_f1_position])

    classification_report_text = classification_report(
        correct_labels,
        hybrid_prediction,
        labels=EXPECTED_LABELS,
        target_names=EXPECTED_LABELS,
        digits=3,
        zero_division=0,
    )

    keyword_examples = int(keyword_covered.sum())
    fallback_examples = int((~keyword_covered).sum())
    keyword_correct = int(is_correct[keyword_covered].sum())
    fallback_correct = int(is_correct[~keyword_covered].sum())
    total_correct = int(is_correct.sum())

    keyword_accuracy = keyword_correct / keyword_examples
    fallback_accuracy = fallback_correct / fallback_examples

    changed_predictions = int(
        hybrid_prediction.ne(predictions["predicted_label"]).sum()
    )

    decision_source = pd.Series(
        "clinicalbert_fallback",
        index=predictions.index,
    )
    decision_source.loc[keyword_covered] = "keyword_rule"

    output = pd.DataFrame(
        {
            "source_index": predictions["source_index"],
            "correct_label": correct_labels,
            "predicted_label": hybrid_prediction,
            "is_correct": is_correct,
            "decision_source": decision_source,
            "keyword_rule_prediction": keyword_prediction,
            "clinicalbert_prediction": predictions["predicted_label"],
            "clinicalbert_confidence": predictions["confidence"],
            "description": predictions["description"],
            "sample_name": predictions["sample_name"],
            "original_keywords": predictions["original_keywords"],
            "keywords_without_prefix": predictions[
                "keywords_without_prefix"
            ],
            "transcription_excerpt": predictions[
                "transcription_excerpt"
            ],
        }
    )

    output.to_csv(OUTPUT_PREDICTIONS, index=False)

    report_lines = [
        "=== METADATA-GATED HYBRID ===",
        "",
        "Input model: Experiment 13 ClinicalBERT without prefix",
        (
            "Decision rule: use the mapped first keyword when available; "
            "otherwise use the no-prefix ClinicalBERT prediction"
        ),
        "Official evaluation set used: no",
        "",
        "=== VALIDATION RESULTS ===",
        f"Validation instances: {len(predictions)}",
        f"Correct predictions: {total_correct}/{len(predictions)}",
        f"Accuracy: {accuracy:.4f}",
        f"Macro-F1: {macro_f1:.4f}",
        f"Weighted F1: {weighted_f1:.4f}",
        (
            f"Minimum per-class F1: {minimum_f1_label} = "
            f"{minimum_f1:.4f}"
        ),
        "",
        "=== DECISION SOURCES ===",
        (
            f"Keyword rule: {keyword_examples} examples, "
            f"{keyword_correct} correct, accuracy {keyword_accuracy:.4f}"
        ),
        (
            f"ClinicalBERT fallback: {fallback_examples} examples, "
            f"{fallback_correct} correct, accuracy {fallback_accuracy:.4f}"
        ),
        (
            "Predictions changed relative to no-prefix ClinicalBERT: "
            f"{changed_predictions}"
        ),
        "",
        "=== CLASSIFICATION REPORT ===",
        classification_report_text,
    ]

    OUTPUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=== METADATA-GATED HYBRID ===")
    print("Validation instances:", len(predictions))
    print("Correct predictions:", total_correct)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(
        "Minimum per-class F1:",
        f"{minimum_f1_label} = {minimum_f1:.4f}",
    )

    print("\n=== DECISION SOURCES ===")
    print(
        f"Keyword rule: {keyword_correct}/{keyword_examples} correct "
        f"({keyword_accuracy:.2%})"
    )
    print(
        f"ClinicalBERT fallback: {fallback_correct}/{fallback_examples} "
        f"correct ({fallback_accuracy:.2%})"
    )
    print(
        "Predictions changed relative to no-prefix ClinicalBERT:",
        changed_predictions,
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report_text)

    print("Files created:")
    print("-", OUTPUT_REPORT)
    print("-", OUTPUT_PREDICTIONS)


if __name__ == "__main__":
    main()
