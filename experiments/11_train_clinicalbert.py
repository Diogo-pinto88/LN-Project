import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

sys.path.insert(0, str(SRC_DIR))

from data import (  # noqa: E402
    DATA_DIR,
    EXPECTED_LABELS,
    FEATURE_COLUMNS,
    clean_data,
    load_data,
)


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
MAX_LENGTH = 256
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5


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

LABEL2ID = {label: idx for idx, label in enumerate(EXPECTED_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def remove_label_like_keyword_prefix(keywords: str) -> str:
    """Remove the first keyword when it is just a specialty name."""
    if not isinstance(keywords, str):
        return ""

    parts = keywords.split(",", maxsplit=1)
    first_keyword = parts[0].strip().lower()

    if first_keyword in LABEL_LIKE_KEYWORDS:
        if len(parts) == 2:
            return parts[1].strip()
        return ""

    return keywords.strip()


def build_text(dataframe: pd.DataFrame) -> pd.Series:
    """Build one input text per example using all non-label fields."""
    keywords_without_prefix = dataframe["keywords"].apply(
        remove_label_like_keyword_prefix
    )

    return (
        "DESCRIPTION "
        + dataframe["description"].astype(str)
        + " SAMPLE_NAME "
        + dataframe["sample_name"].astype(str)
        + " TRANSCRIPTION "
        + dataframe["transcription"].astype(str)
        + " KEYWORDS "
        + keywords_without_prefix.astype(str)
    )


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


print("=== LOADING DATA ===")
raw_train, raw_test = load_data()
train, _ = clean_data(raw_train, raw_test)

if (DATA_DIR / "train_clean.csv").is_file():
    train = pd.read_csv(DATA_DIR / "train_clean.csv", sep=";")

train = train.copy()
train[FEATURE_COLUMNS] = train[FEATURE_COLUMNS].fillna("")
train["medical_specialty"] = train["medical_specialty"].astype(str)

train["text"] = build_text(train)
train["labels"] = train["medical_specialty"].map(LABEL2ID)

if train["labels"].isna().any():
    missing = train.loc[train["labels"].isna(), "medical_specialty"].value_counts()
    raise ValueError(
        "Found labels outside the expected specialties: "
        f"{missing.to_dict()}"
    )

train["labels"] = train["labels"].astype(int)

train_df, validation_df = train_test_split(
    train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train["labels"],
)

print("Training instances:", len(train_df))
print("Validation instances:", len(validation_df))
print("Label distribution in training set:")
print(train_df["medical_specialty"].value_counts().reindex(EXPECTED_LABELS, fill_value=0))

train_dataset = Dataset.from_pandas(
    train_df[["text", "labels"]],
    preserve_index=False,
)
validation_dataset = Dataset.from_pandas(
    validation_df[["text", "labels"]],
    preserve_index=False,
)

print("=== LOADING MODEL ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EXPECTED_LABELS),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


train_dataset = train_dataset.map(tokenize_batch, batched=True)
validation_dataset = validation_dataset.map(tokenize_batch, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
validation_dataset = validation_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
validation_dataset.set_format("torch")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(list(range(len(EXPECTED_LABELS)))),
    y=train_df["labels"].to_numpy(),
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

training_args = TrainingArguments(
    output_dir=str(MODELS_DIR / "clinicalbert"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
    seed=RANDOM_STATE,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


trainer = WeightedTrainer(
    class_weights=class_weights_tensor,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("=== TRAINING CLINICALBERT ===")
trainer.train()

print("=== VALIDATION ===")
validation_predictions = trainer.predict(validation_dataset)
validation_logits = validation_predictions.predictions
validation_pred_ids = np.argmax(validation_logits, axis=-1)
validation_true_ids = validation_predictions.label_ids

validation_pred_labels = [ID2LABEL[idx] for idx in validation_pred_ids]
validation_true_labels = [ID2LABEL[idx] for idx in validation_true_ids]

accuracy = accuracy_score(validation_true_labels, validation_pred_labels)
macro_f1 = f1_score(
    validation_true_labels,
    validation_pred_labels,
    average="macro",
    zero_division=0,
)
report_text = classification_report(
    validation_true_labels,
    validation_pred_labels,
    labels=EXPECTED_LABELS,
    zero_division=0,
    digits=3,
)

prediction_counts = pd.Series(validation_pred_labels, name="predicted_label").value_counts()

print(f"Accuracy: {accuracy:.3f}")
print(f"Macro-F1: {macro_f1:.3f}")
print("\n=== CLASSIFICATION REPORT ===")
print(report_text)
print("\n=== PREDICTION COUNTS ===")
print(prediction_counts)

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

model_dir = MODELS_DIR / "clinicalbert"
trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))

report_path = RESULTS_DIR / "09_clinicalbert_report.txt"
with open(report_path, "w", encoding="utf-8") as file:
    file.write("ClinicalBERT fine-tuning\n")
    file.write("========================\n\n")
    file.write(f"Model: {MODEL_NAME}\n")
    file.write(f"Random state: {RANDOM_STATE}\n")
    file.write(f"Validation size: {VALIDATION_SIZE}\n")
    file.write(f"Max length: {MAX_LENGTH}\n")
    file.write(f"Epochs: {NUM_EPOCHS}\n")
    file.write(f"Train batch size: {TRAIN_BATCH_SIZE}\n")
    file.write(f"Eval batch size: {EVAL_BATCH_SIZE}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Training instances: {len(train_df)}\n")
    file.write(f"Validation instances: {len(validation_df)}\n")
    file.write("Class weights:\n")
    for label, weight in zip(EXPECTED_LABELS, class_weights):
        file.write(f"- {label}: {weight:.4f}\n")
    file.write(f"\nAccuracy: {accuracy:.3f}\n")
    file.write(f"Macro-F1: {macro_f1:.3f}\n\n")
    file.write(report_text)
    file.write("\nPrediction counts:\n")
    file.write(prediction_counts.to_string())

validation_results = validation_df[
    ["medical_specialty", "description", "sample_name", "keywords"]
].copy()
validation_results["predicted_label"] = validation_pred_labels
validation_results["is_correct"] = (
    validation_results["medical_specialty"] == validation_results["predicted_label"]
)
validation_results.index.name = "source_index"

predictions_path = RESULTS_DIR / "09_clinicalbert_validation_predictions.csv"
validation_results.to_csv(predictions_path, sep=";", index=True)

print(f"\nModel saved to: {model_dir}")
print(f"Report saved to: {report_path}")
print(f"Validation predictions saved to: {predictions_path}")
