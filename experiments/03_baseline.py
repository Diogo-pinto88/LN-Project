import re
import sys
from pathlib import Path

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from data import EXPECTED_LABELS, clean_data, load_data


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

stemmer = SnowballStemmer("english")
token_pattern = re.compile(r"\b[a-zA-Z]+\b")


def tokenize_and_stem(text):
    """
    Transform a text into lowercase word stems.

    Example:
    "patients treated surgically"
    becomes approximately:
    ["patient", "treat", "surgic"]
    """
    tokens = token_pattern.findall(text.lower())

    return [
        stemmer.stem(token)
        for token in tokens
    ]


# Carregar e limpar o treino
raw_train, raw_test = load_data()
train, _ = clean_data(raw_train, raw_test)


# A baseline utiliza apenas a coluna description
X = train["description"]
y = train["medical_specialty"]


# Separar 80% para treino e 20% para validação
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# Pipeline da baseline oficial:
# description -> lowercase/stemming -> TF-IDF -> Naive Bayes
baseline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                token_pattern=None,
                lowercase=False,
            ),
        ),
        (
            "classifier",
            MultinomialNB(),
        ),
    ]
)


print("=== TRAINING BASELINE ===")

print("Training instances:", len(X_train))
print("Validation instances:", len(X_validation))

baseline.fit(X_train, y_train)


# Fazer previsões no conjunto de validação
predictions = baseline.predict(X_validation)


# Calcular métricas
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


print("\n=== BASELINE RESULTS ===")

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


# Guardar o relatório
RESULTS_DIR.mkdir(exist_ok=True)

report_path = RESULTS_DIR / "03_baseline_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Description-only baseline\n")
    file.write("=========================\n\n")

    file.write(f"Random state: {RANDOM_STATE}\n")
    file.write(f"Validation size: {VALIDATION_SIZE}\n")
    file.write(f"Training instances: {len(X_train)}\n")
    file.write(f"Validation instances: {len(X_validation)}\n\n")

    file.write(f"Accuracy: {accuracy:.3f}\n")
    file.write(f"Macro-F1: {macro_f1:.3f}\n")

    file.write(
        "Minimum per-class F1: "
        f"{minimum_f1_label} = {minimum_f1:.3f}\n\n"
    )

    file.write(report_text)


# Guardar as previsões para posterior análise de erros
validation_results = train.loc[
    X_validation.index,
    [
        "description",
        "sample_name",
    ],
].copy()

validation_results["correct_label"] = y_validation
validation_results["predicted_label"] = predictions

validation_results["is_correct"] = (
    validation_results["correct_label"]
    == validation_results["predicted_label"]
)

predictions_path = (
    RESULTS_DIR
    / "03_baseline_validation_predictions.csv"
)

validation_results.to_csv(
    predictions_path,
    index=True,
)


print("\nFiles created:")
print("-", report_path)
print("-", predictions_path)