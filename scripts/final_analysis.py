import os
import sys
import tempfile
from pathlib import Path

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "ln_project_plot_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from data import EXPECTED_LABELS, FEATURE_COLUMNS


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

HYBRID_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "16_metadata_gated_hybrid_validation_predictions.csv"
)


MODEL_CONFIGURATIONS = [
    {
        "experiment": "03",
        "model": "Description TF-IDF + Multinomial NB",
        "file": "03_baseline_validation_predictions.csv",
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "04",
        "model": "Multi-field TF-IDF + Linear SVM, no prefix",
        "file": "04_multifield_svm_validation_predictions.csv",
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "05",
        "model": "Multi-field TF-IDF + Linear SVM, full keywords",
        "file": "05_full_keywords_svm_validation_predictions.csv",
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "06",
        "model": "Augmented TF-IDF + Linear SVM, no prefix",
        "file": "06_class_aware_augmentation_svm_validation_predictions.csv",
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "07",
        "model": "Word TF-IDF + Logistic Regression, full keywords",
        "file": (
            "07_full_keywords_logistic_regression_"
            "validation_predictions.csv"
        ),
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "08a",
        "model": "Logistic Regression, description only",
        "file": "08_metadata_ablation_validation_predictions.csv",
        "prediction_column": "prediction_description_only",
    },
    {
        "experiment": "08b",
        "model": "Logistic Regression, clinical text without keywords",
        "file": "08_metadata_ablation_validation_predictions.csv",
        "prediction_column": "prediction_clinical_text_no_keywords",
    },
    {
        "experiment": "08c",
        "model": "Logistic Regression, all fields without prefix",
        "file": "08_metadata_ablation_validation_predictions.csv",
        "prediction_column": "prediction_all_fields_without_prefix",
    },
    {
        "experiment": "09",
        "model": "Word + character TF-IDF + Logistic Regression",
        "file": "09_word_character_tfidf_validation_predictions.csv",
        "prediction_column": "prediction_word_and_character",
    },
    {
        "experiment": "12",
        "model": "ClinicalBERT, full keywords, balanced weights",
        "file": "12_clinical_bert_diogo_validation_predictions.csv",
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "13",
        "model": "ClinicalBERT, no prefix, balanced weights",
        "file": (
            "13_clinical_bert_without_prefix_"
            "validation_predictions.csv"
        ),
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "14",
        "model": "ClinicalBERT, no prefix, no class weights",
        "file": (
            "14_clinical_bert_without_prefix_no_weights_"
            "validation_predictions.csv"
        ),
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "15",
        "model": "ClinicalBERT, no prefix, softened weights",
        "file": (
            "15_clinical_bert_soft_weights_"
            "validation_predictions.csv"
        ),
        "prediction_column": "predicted_label",
    },
    {
        "experiment": "16",
        "model": "Metadata-gated hybrid",
        "file": "16_metadata_gated_hybrid_validation_predictions.csv",
        "prediction_column": "predicted_label",
    },
]


SHORT_LABELS = [
    "CV-Pulm",
    "Derm",
    "Gastro",
    "Gen Med",
    "Neurol",
    "Neurosurg",
    "OB-GYN",
    "Ophth",
    "Ortho",
    "Psych",
    "Radiol",
    "Surgery",
]


DISCUSSION_EXAMPLES = {
    1747: (
        "This exact input also appears in the training subset with "
        "Neurosurgery and Orthopedic labels. The prediction is therefore "
        "supported by contradictory annotations, while the procedure itself "
        "can reasonably involve several surgical specialties."
    ),
    1887: (
        "The document describes a coronary procedure and an exact copy in "
        "the training subset is labelled Cardiovascular-Pulmonary. The error "
        "reflects ambiguity between the organ-system specialty and the "
        "procedural Surgery label."
    ),
    507: (
        "The short description reports persistent abdominal pain without a "
        "specific gastrointestinal diagnosis or procedure. General Medicine "
        "is plausible from the available text, showing the difficulty of "
        "assigning broad presentations to a single specialty."
    ),
}


def read_predictions(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    return pd.read_csv(path)


def validate_labels(correct_labels, predicted_labels, model_name):
    unexpected_correct = set(correct_labels.dropna()) - set(EXPECTED_LABELS)
    unexpected_predicted = set(predicted_labels.dropna()) - set(
        EXPECTED_LABELS
    )

    if unexpected_correct:
        raise ValueError(
            f"Unexpected correct labels for {model_name}: "
            f"{sorted(unexpected_correct)}"
        )

    if unexpected_predicted:
        raise ValueError(
            f"Unexpected predictions for {model_name}: "
            f"{sorted(unexpected_predicted)}"
        )


def calculate_metrics(correct_labels, predicted_labels):
    per_class_f1 = f1_score(
        correct_labels,
        predicted_labels,
        labels=EXPECTED_LABELS,
        average=None,
        zero_division=0,
    )
    minimum_position = int(per_class_f1.argmin())

    return {
        "validation_instances": len(correct_labels),
        "correct_predictions": int(
            predicted_labels.eq(correct_labels).sum()
        ),
        "accuracy": accuracy_score(correct_labels, predicted_labels),
        "macro_f1": f1_score(
            correct_labels,
            predicted_labels,
            labels=EXPECTED_LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            correct_labels,
            predicted_labels,
            labels=EXPECTED_LABELS,
            average="weighted",
            zero_division=0,
        ),
        "minimum_f1": float(per_class_f1[minimum_position]),
        "minimum_f1_label": EXPECTED_LABELS[minimum_position],
        "classes_below_0_25": int((per_class_f1 < 0.25).sum()),
    }


def create_model_comparison():
    rows = []

    for configuration in MODEL_CONFIGURATIONS:
        predictions = read_predictions(configuration["file"])
        correct_labels = predictions["correct_label"]
        predicted_labels = predictions[
            configuration["prediction_column"]
        ]

        validate_labels(
            correct_labels,
            predicted_labels,
            configuration["model"],
        )

        metrics = calculate_metrics(correct_labels, predicted_labels)
        rows.append(
            {
                "experiment": configuration["experiment"],
                "model": configuration["model"],
                **metrics,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        index=False,
    )

    return comparison


def create_per_class_results(hybrid):
    correct_labels = hybrid["correct_label"]
    predicted_labels = hybrid["predicted_label"]

    precision, recall, f1, support = precision_recall_fscore_support(
        correct_labels,
        predicted_labels,
        labels=EXPECTED_LABELS,
        zero_division=0,
    )

    per_class = pd.DataFrame(
        {
            "specialty": EXPECTED_LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    per_class.to_csv(
        RESULTS_DIR / "final_per_class_metrics.csv",
        index=False,
    )

    return per_class


def create_confusion_matrix(hybrid):
    matrix = confusion_matrix(
        hybrid["correct_label"],
        hybrid["predicted_label"],
        labels=EXPECTED_LABELS,
    )

    matrix_frame = pd.DataFrame(
        matrix,
        index=EXPECTED_LABELS,
        columns=EXPECTED_LABELS,
    )
    matrix_frame.index.name = "correct_label"
    matrix_frame.columns.name = "predicted_label"
    matrix_frame.to_csv(RESULTS_DIR / "final_confusion_matrix.csv")

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    normalized_frame = pd.DataFrame(
        normalized,
        index=EXPECTED_LABELS,
        columns=EXPECTED_LABELS,
    )
    normalized_frame.index.name = "correct_label"
    normalized_frame.columns.name = "predicted_label"
    normalized_frame.to_csv(
        RESULTS_DIR / "final_confusion_matrix_normalized.csv"
    )

    annotations = np.empty_like(matrix, dtype=object)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = matrix[row, column]
            percentage = normalized[row, column]
            annotations[row, column] = (
                f"{count}\n{percentage:.0%}" if count else ""
            )

    sns.set_theme(style="white", context="paper", font_scale=0.90)
    figure, axis = plt.subplots(figsize=(11.5, 9.0))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        linecolor="#d9d9d9",
        square=True,
        xticklabels=SHORT_LABELS,
        yticklabels=SHORT_LABELS,
        cbar_kws={"label": "Number of instances", "shrink": 0.80},
        ax=axis,
    )
    axis.set_title("Metadata-gated hybrid confusion matrix", pad=14)
    axis.set_xlabel("Predicted specialty")
    axis.set_ylabel("Correct specialty")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "final_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        RESULTS_DIR / "final_confusion_matrix.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    return matrix, normalized


def load_clean_training_data():
    train = pd.read_csv(DATA_DIR / "train.csv", sep=";")

    invalid_label = ~train["medical_specialty"].isin(EXPECTED_LABELS)
    too_many_missing = train[FEATURE_COLUMNS].isna().sum(axis=1) >= 3
    too_long = train["description"].fillna("").str.len() > 20_000

    clean_train = train.loc[
        ~(invalid_label | too_many_missing | too_long)
    ].copy()
    clean_train[FEATURE_COLUMNS] = clean_train[FEATURE_COLUMNS].fillna("")

    return clean_train


def make_input_key(row):
    return tuple(row[column] for column in FEATURE_COLUMNS)


def create_duplicate_analysis(clean_train, hybrid):
    grouped = clean_train.groupby(FEATURE_COLUMNS, dropna=False)

    duplicate_rows = []

    for _, group in grouped:
        if len(group) <= 1:
            continue

        labels = sorted(group["medical_specialty"].unique())

        if len(labels) <= 1:
            continue

        first_row = group.iloc[0]
        duplicate_rows.append(
            {
                "row_count": len(group),
                "number_of_labels": len(labels),
                "labels": " | ".join(labels),
                "source_indices": " | ".join(
                    str(index) for index in sorted(group.index)
                ),
                "sample_name": first_row["sample_name"],
                "description": first_row["description"],
            }
        )

    duplicate_groups = pd.DataFrame(duplicate_rows).sort_values(
        ["row_count", "sample_name"],
        ascending=[False, True],
    )
    duplicate_groups.to_csv(
        RESULTS_DIR / "final_conflicting_duplicate_groups.csv",
        index=False,
    )

    training_indices, validation_indices = train_test_split(
        clean_train.index,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean_train["medical_specialty"],
    )
    training_subset = clean_train.loc[training_indices]
    validation_subset = clean_train.loc[validation_indices]

    training_by_input = {}

    for source_index, row in training_subset.iterrows():
        key = make_input_key(row)
        training_by_input.setdefault(key, []).append(
            (source_index, row["medical_specialty"])
        )

    hybrid_by_index = hybrid.set_index("source_index")
    validation_duplicate_rows = []

    for source_index, row in validation_subset.iterrows():
        key = make_input_key(row)
        matching_training_rows = training_by_input.get(key, [])

        if not matching_training_rows:
            continue

        training_labels = sorted(
            {label for _, label in matching_training_rows}
        )
        training_source_indices = sorted(
            index for index, _ in matching_training_rows
        )
        hybrid_row = hybrid_by_index.loc[source_index]

        validation_duplicate_rows.append(
            {
                "source_index": source_index,
                "correct_label": row["medical_specialty"],
                "predicted_label": hybrid_row["predicted_label"],
                "is_correct": bool(hybrid_row["is_correct"]),
                "training_labels_for_same_input": " | ".join(
                    training_labels
                ),
                "training_indices_for_same_input": " | ".join(
                    str(index) for index in training_source_indices
                ),
                "sample_name": row["sample_name"],
                "description": row["description"],
            }
        )

    validation_duplicates = pd.DataFrame(validation_duplicate_rows)
    validation_duplicates.to_csv(
        RESULTS_DIR / "final_validation_conflicting_duplicates.csv",
        index=False,
    )

    conflict_indices = set(validation_duplicates["source_index"])
    error_analysis = hybrid.loc[
        ~hybrid["is_correct"].astype(bool)
    ].copy()
    error_analysis["has_exact_conflicting_training_duplicate"] = (
        error_analysis["source_index"].isin(conflict_indices)
    )

    training_labels_by_validation_index = {
        row["source_index"]: row["training_labels_for_same_input"]
        for row in validation_duplicate_rows
    }
    error_analysis["training_labels_for_same_input"] = (
        error_analysis["source_index"]
        .map(training_labels_by_validation_index)
        .fillna("")
    )
    error_analysis["error_context"] = np.where(
        error_analysis["has_exact_conflicting_training_duplicate"],
        "Exact input appears in training with a different label",
        "No exact input duplicate in the training subset",
    )
    error_analysis.to_csv(
        RESULTS_DIR / "final_error_analysis.csv",
        index=False,
    )

    total_conflicting_groups = len(duplicate_groups)
    total_rows_in_conflicting_groups = int(
        duplicate_groups["row_count"].sum()
    )
    validation_conflicts = len(validation_duplicates)
    total_errors = len(error_analysis)
    errors_with_conflict = int(
        error_analysis[
            "has_exact_conflicting_training_duplicate"
        ].sum()
    )

    conflict_accuracy = validation_duplicates["is_correct"].mean()
    non_conflict = hybrid.loc[
        ~hybrid["source_index"].isin(conflict_indices)
    ]
    non_conflict_accuracy = non_conflict["is_correct"].astype(bool).mean()

    duplicate_summary = {
        "conflicting_groups_full_clean_train": total_conflicting_groups,
        "rows_in_conflicting_groups": total_rows_in_conflicting_groups,
        "validation_inputs_duplicated_in_training": validation_conflicts,
        "hybrid_errors": total_errors,
        "errors_with_conflicting_training_duplicate": errors_with_conflict,
        "conflicting_duplicate_validation_accuracy": conflict_accuracy,
        "remaining_validation_accuracy": non_conflict_accuracy,
    }

    return error_analysis, duplicate_summary


def calculate_subset_accuracy(correct, predicted, mask):
    number = int(mask.sum())

    if number == 0:
        return number, np.nan, 0

    number_correct = int(predicted[mask].eq(correct[mask]).sum())
    return number, number_correct / number, number_correct


def create_prefix_analysis():
    hybrid = read_predictions(
        "16_metadata_gated_hybrid_validation_predictions.csv"
    ).set_index("source_index")
    covered = hybrid["decision_source"].eq("keyword_rule")

    experiment_08 = read_predictions(
        "08_metadata_ablation_validation_predictions.csv"
    ).set_index("source_index")
    experiment_12 = read_predictions(
        "12_clinical_bert_diogo_validation_predictions.csv"
    ).set_index("source_index")
    experiment_13 = read_predictions(
        "13_clinical_bert_without_prefix_validation_predictions.csv"
    ).set_index("source_index")

    configurations = [
        (
            "08c",
            "Logistic Regression without keyword prefix",
            experiment_08,
            "prediction_all_fields_without_prefix",
        ),
        (
            "08d",
            "Logistic Regression with complete keywords",
            experiment_08,
            "prediction_all_fields_full_keywords",
        ),
        (
            "12",
            "ClinicalBERT with complete keywords",
            experiment_12,
            "predicted_label",
        ),
        (
            "13",
            "ClinicalBERT without keyword prefix",
            experiment_13,
            "predicted_label",
        ),
        (
            "16",
            "Metadata-gated hybrid",
            hybrid,
            "predicted_label",
        ),
    ]

    rows = []

    for experiment, name, predictions, prediction_column in configurations:
        predictions = predictions.loc[hybrid.index]
        correct = predictions["correct_label"]
        predicted = predictions[prediction_column]

        covered_count, covered_accuracy, covered_correct = (
            calculate_subset_accuracy(correct, predicted, covered)
        )
        uncovered_count, uncovered_accuracy, uncovered_correct = (
            calculate_subset_accuracy(correct, predicted, ~covered)
        )
        metrics = calculate_metrics(correct, predicted)

        rows.append(
            {
                "experiment": experiment,
                "configuration": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "minimum_f1": metrics["minimum_f1"],
                "covered_examples": covered_count,
                "covered_correct": covered_correct,
                "covered_accuracy": covered_accuracy,
                "uncovered_examples": uncovered_count,
                "uncovered_correct": uncovered_correct,
                "uncovered_accuracy": uncovered_accuracy,
            }
        )

    prefix_comparison = pd.DataFrame(rows)
    prefix_comparison.to_csv(
        RESULTS_DIR / "final_prefix_comparison.csv",
        index=False,
    )

    return prefix_comparison


def truncate_text(text, maximum_characters=420):
    text = " ".join(str(text).split())

    if len(text) <= maximum_characters:
        return text

    return text[: maximum_characters - 3].rstrip() + "..."


def create_discussion_examples(error_analysis):
    errors_by_index = error_analysis.set_index("source_index")
    lines = [
        "# Concrete Misclassification Examples",
        "",
        (
            "These examples come from the internal validation split. "
            "The official unlabelled evaluation set was not used."
        ),
        "",
    ]

    for number, (source_index, interpretation) in enumerate(
        DISCUSSION_EXAMPLES.items(),
        start=1,
    ):
        if source_index not in errors_by_index.index:
            raise ValueError(
                f"Discussion example {source_index} is not a hybrid error."
            )

        row = errors_by_index.loc[source_index]
        excerpt = truncate_text(row["description"])

        lines.extend(
            [
                f"## Example {number}: {str(row['sample_name']).strip()}",
                "",
                f"- Source index: {source_index}",
                f"- Correct label: `{row['correct_label']}`",
                f"- Predicted label: `{row['predicted_label']}`",
                f"- Decision source: `{row['decision_source']}`",
                (
                    "- Exact conflicting duplicate in training: "
                    f"{bool(row['has_exact_conflicting_training_duplicate'])}"
                ),
                f"- Text excerpt: \"{excerpt}\"",
                "",
                f"**Interpretation.** {interpretation}",
                "",
            ]
        )

    output_path = RESULTS_DIR / "final_discussion_examples.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


def create_text_report(
    comparison,
    per_class,
    prefix_comparison,
    duplicate_summary,
    hybrid,
):
    best_row = comparison.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]

    hybrid_metrics = calculate_metrics(
        hybrid["correct_label"],
        hybrid["predicted_label"],
    )

    lines = [
        "=== FINAL VALIDATION ANALYSIS ===",
        "",
        "Official evaluation set used: no",
        "Validation split: stratified 80/20",
        f"Random seed: {RANDOM_STATE}",
        f"Validation instances: {len(hybrid)}",
        "",
        "=== BEST SYSTEM ===",
        f"Experiment: {best_row['experiment']}",
        f"Model: {best_row['model']}",
        f"Accuracy: {best_row['accuracy']:.4f}",
        f"Macro-F1: {best_row['macro_f1']:.4f}",
        f"Minimum per-class F1: {best_row['minimum_f1']:.4f}",
        "",
        "=== HYBRID RESULTS ===",
        (
            f"Correct predictions: "
            f"{hybrid_metrics['correct_predictions']}/{len(hybrid)}"
        ),
        f"Accuracy: {hybrid_metrics['accuracy']:.4f}",
        f"Macro-F1: {hybrid_metrics['macro_f1']:.4f}",
        f"Weighted F1: {hybrid_metrics['weighted_f1']:.4f}",
        (
            "Minimum F1: "
            f"{hybrid_metrics['minimum_f1_label']} = "
            f"{hybrid_metrics['minimum_f1']:.4f}"
        ),
        "",
        "=== DUPLICATE-LABEL ANALYSIS ===",
        (
            "Conflicting exact-input groups in cleaned training data: "
            f"{duplicate_summary['conflicting_groups_full_clean_train']}"
        ),
        (
            "Rows in these groups: "
            f"{duplicate_summary['rows_in_conflicting_groups']}"
        ),
        (
            "Validation inputs with an exact training duplicate: "
            f"{duplicate_summary['validation_inputs_duplicated_in_training']}"
        ),
        (
            "Hybrid errors associated with a conflicting duplicate: "
            f"{duplicate_summary['errors_with_conflicting_training_duplicate']}"
            f"/{duplicate_summary['hybrid_errors']}"
        ),
        (
            "Accuracy on conflicting duplicate validation instances: "
            f"{duplicate_summary['conflicting_duplicate_validation_accuracy']:.4f}"
        ),
        (
            "Accuracy on the remaining validation instances: "
            f"{duplicate_summary['remaining_validation_accuracy']:.4f}"
        ),
        "",
        "=== PER-CLASS F1 ===",
    ]

    for row in per_class.itertuples(index=False):
        lines.append(f"{row.specialty}: {row.f1:.4f}")

    lines.extend(
        [
            "",
            "=== PREFIX COMPARISON ===",
        ]
    )

    for row in prefix_comparison.itertuples(index=False):
        lines.append(
            f"{row.configuration}: overall accuracy {row.accuracy:.4f}; "
            f"covered {row.covered_correct}/{row.covered_examples} "
            f"({row.covered_accuracy:.4f}); uncovered "
            f"{row.uncovered_correct}/{row.uncovered_examples} "
            f"({row.uncovered_accuracy:.4f})"
        )

    lines.extend(
        [
            "",
            "=== CLASSIFICATION REPORT FOR THE HYBRID ===",
            classification_report(
                hybrid["correct_label"],
                hybrid["predicted_label"],
                labels=EXPECTED_LABELS,
                target_names=EXPECTED_LABELS,
                digits=3,
                zero_division=0,
            ),
        ]
    )

    (RESULTS_DIR / "final_analysis_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def verify_outputs(comparison, per_class, hybrid, error_analysis):
    if len(comparison) != len(MODEL_CONFIGURATIONS):
        raise AssertionError("The model comparison is incomplete.")

    if len(per_class) != len(EXPECTED_LABELS):
        raise AssertionError("The per-class table is incomplete.")

    if len(hybrid) != 522:
        raise AssertionError("Unexpected validation-set size.")

    if int(hybrid["is_correct"].astype(bool).sum()) != 490:
        raise AssertionError("Unexpected number of correct hybrid predictions.")

    if len(error_analysis) != 32:
        raise AssertionError("Unexpected number of hybrid errors.")

    required_outputs = [
        "final_model_comparison.csv",
        "final_per_class_metrics.csv",
        "final_confusion_matrix.csv",
        "final_confusion_matrix_normalized.csv",
        "final_confusion_matrix.png",
        "final_confusion_matrix.pdf",
        "final_conflicting_duplicate_groups.csv",
        "final_validation_conflicting_duplicates.csv",
        "final_error_analysis.csv",
        "final_prefix_comparison.csv",
        "final_discussion_examples.md",
        "final_analysis_report.txt",
    ]

    missing_outputs = [
        filename
        for filename in required_outputs
        if not (RESULTS_DIR / filename).exists()
    ]

    if missing_outputs:
        raise AssertionError(
            "Missing final-analysis outputs: "
            + ", ".join(missing_outputs)
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = create_model_comparison()
    hybrid = pd.read_csv(HYBRID_PREDICTIONS_FILE)
    per_class = create_per_class_results(hybrid)
    create_confusion_matrix(hybrid)

    clean_train = load_clean_training_data()
    error_analysis, duplicate_summary = create_duplicate_analysis(
        clean_train,
        hybrid,
    )
    prefix_comparison = create_prefix_analysis()
    create_discussion_examples(error_analysis)
    create_text_report(
        comparison,
        per_class,
        prefix_comparison,
        duplicate_summary,
        hybrid,
    )
    verify_outputs(comparison, per_class, hybrid, error_analysis)

    best = comparison.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]

    print("=== FINAL ANALYSIS COMPLETE ===")
    print("Models compared:", len(comparison))
    print("Best model:", best["model"])
    print(f"Best accuracy: {best['accuracy']:.4f}")
    print(f"Best Macro-F1: {best['macro_f1']:.4f}")
    print("Hybrid validation errors:", len(error_analysis))
    print(
        "Errors with conflicting exact duplicate:",
        duplicate_summary[
            "errors_with_conflicting_training_duplicate"
        ],
    )
    print("All final-analysis outputs were verified.")


if __name__ == "__main__":
    main()
