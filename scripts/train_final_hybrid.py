import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"
CHECKPOINT_DIR = PROJECT_DIR / ".checkpoints" / "final_hybrid"

RESULTS_TXT_PATH = PROJECT_DIR / "results.txt"
PREDICTIONS_PATH = RESULTS_DIR / "final_test_predictions.csv"
REPORT_PATH = RESULTS_DIR / "final_training_report.txt"
TRAINING_LOG_PATH = RESULTS_DIR / "final_training_log.csv"

sys.path.insert(0, str(SRC_DIR))

from data import EXPECTED_LABELS, clean_data, load_data


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

RANDOM_STATE = 42
MAX_LENGTH = 128
NUMBER_OF_EPOCHS = 2
TRAIN_BATCH_SIZE = 8
PREDICTION_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

EXPECTED_TRAINING_INSTANCES = 2_608
EXPECTED_TEST_INSTANCES = 398
EXPECTED_TEST_KEYWORD_RULE_INSTANCES = 320


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

LABEL_LIKE_KEYWORDS = set(KEYWORD_TO_SPECIALTY)


# ============================================================
# COMMAND-LINE OPTIONS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train the final no-prefix ClinicalBERT model on all cleaned "
            "training instances and generate hybrid predictions."
        )
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help=(
            "Validate the train and clean test files without loading or "
            "training ClinicalBERT."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest final-model checkpoint.",
    )

    return parser.parse_args()


# ============================================================
# KEYWORD PROCESSING
# ============================================================

def first_keyword(keywords):
    return str(keywords).split(",", maxsplit=1)[0].strip().lower()


def map_keyword_prefix(keywords):
    return KEYWORD_TO_SPECIALTY.get(first_keyword(keywords))


def has_label_like_keyword_prefix(keywords):
    return first_keyword(keywords) in LABEL_LIKE_KEYWORDS


def remove_label_like_keyword_prefix(keywords):
    """Remove the first keyword only when it names a target specialty."""

    parts = str(keywords).split(",", maxsplit=1)

    if parts[0].strip().lower() in LABEL_LIKE_KEYWORDS:
        if len(parts) == 2:
            return parts[1].strip()

        return ""

    return str(keywords).strip()


# ============================================================
# TEXT AND DATASET CONSTRUCTION
# ============================================================

def build_text(row):
    sample_name = str(row["sample_name"]).strip()
    description = str(row["description"]).strip()
    keywords = remove_label_like_keyword_prefix(row["keywords"])
    transcription = str(row["transcription"]).strip()

    return (
        f"SAMPLE NAME: {sample_name} "
        f"DESCRIPTION: {description} "
        f"KEYWORDS: {keywords} "
        f"TRANSCRIPTION: {transcription}"
    )


class ClinicalDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(
                self.labels[index],
                dtype=torch.long,
            )

        return item


# ============================================================
# CLASS-WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_function(outputs.logits, labels)

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# DATA VALIDATION
# ============================================================

def load_and_validate_data():
    raw_train, raw_test = load_data(
        test_filename="test_no_labels_clean.csv"
    )
    train, test = clean_data(raw_train, raw_test)

    if len(train) != EXPECTED_TRAINING_INSTANCES:
        raise ValueError(
            "Unexpected clean training size: "
            f"{len(train)} instead of {EXPECTED_TRAINING_INSTANCES}."
        )

    if len(test) != EXPECTED_TEST_INSTANCES:
        raise ValueError(
            "Unexpected clean test size: "
            f"{len(test)} instead of {EXPECTED_TEST_INSTANCES}."
        )

    unexpected_labels = set(train["medical_specialty"]) - set(
        EXPECTED_LABELS
    )

    if unexpected_labels:
        raise ValueError(
            "Unexpected training labels: "
            + ", ".join(sorted(unexpected_labels))
        )

    if train["medical_specialty"].nunique() != len(EXPECTED_LABELS):
        raise ValueError("The clean training data does not contain 12 classes.")

    test_rule_predictions = test["keywords"].apply(map_keyword_prefix)
    number_covered = int(test_rule_predictions.notna().sum())

    if number_covered != EXPECTED_TEST_KEYWORD_RULE_INSTANCES:
        raise ValueError(
            "Unexpected keyword-rule coverage in the clean test: "
            f"{number_covered} instead of "
            f"{EXPECTED_TEST_KEYWORD_RULE_INSTANCES}."
        )

    return train, test, test_rule_predictions


def print_data_summary(train, test, test_rule_predictions):
    training_prefixes = int(
        train["keywords"].apply(has_label_like_keyword_prefix).sum()
    )
    test_prefixes = int(test_rule_predictions.notna().sum())

    print("\n=== FINAL DATA VALIDATION ===")
    print("Clean training instances:", len(train))
    print("Clean test instances:", len(test))
    print("Training specialties:", train["medical_specialty"].nunique())
    print("Training keyword prefixes removed:", training_prefixes)
    print("Test instances covered by keyword rule:", test_prefixes)
    print("Test instances requiring ClinicalBERT:", len(test) - test_prefixes)


# ============================================================
# FINAL OUTPUT VALIDATION
# ============================================================

def validate_final_predictions(predicted_labels, expected_rows):
    if len(predicted_labels) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} predictions, "
            f"but obtained {len(predicted_labels)}."
        )

    if any(not isinstance(label, str) or not label for label in predicted_labels):
        raise ValueError("The final predictions contain an empty label.")

    unexpected_labels = set(predicted_labels) - set(EXPECTED_LABELS)

    if unexpected_labels:
        raise ValueError(
            "The final predictions contain unexpected labels: "
            + ", ".join(sorted(unexpected_labels))
        )


def write_and_verify_results_txt(predicted_labels):
    RESULTS_TXT_PATH.write_text(
        "\n".join(predicted_labels) + "\n",
        encoding="utf-8",
    )

    saved_lines = RESULTS_TXT_PATH.read_text(
        encoding="utf-8"
    ).splitlines()

    validate_final_predictions(
        saved_lines,
        EXPECTED_TEST_INSTANCES,
    )

    if saved_lines != predicted_labels:
        raise ValueError(
            "The saved results.txt does not preserve prediction order."
        )


# ============================================================
# FINAL TRAINING AND PREDICTION
# ============================================================

def main():
    arguments = parse_arguments()
    set_seed(RANDOM_STATE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test, test_rule_predictions = load_and_validate_data()
    print_data_summary(train, test, test_rule_predictions)

    if arguments.validate_data_only:
        print("\nData validation completed. No model was loaded or trained.")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR))

    if arguments.resume:
        if latest_checkpoint is None:
            raise FileNotFoundError(
                "No final-model checkpoint is available to resume."
            )

        resume_checkpoint = latest_checkpoint
        print("\nResuming from checkpoint:", resume_checkpoint)
    else:
        if latest_checkpoint is not None:
            raise FileExistsError(
                "A final-model checkpoint already exists. Use --resume to "
                "continue it, or move the existing final_hybrid checkpoint "
                "directory before starting a new run."
            )

        resume_checkpoint = None

    label_to_id = {
        label: index
        for index, label in enumerate(EXPECTED_LABELS)
    }
    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    training_texts = train.apply(build_text, axis=1).tolist()
    test_texts = test.apply(build_text, axis=1).tolist()
    training_labels = (
        train["medical_specialty"].map(label_to_id).tolist()
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(EXPECTED_LABELS)),
        y=np.array(training_labels),
    )
    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    print("\n=== FINAL CONFIGURATION ===")
    print("Model:", MODEL_NAME)
    print("Maximum tokens:", MAX_LENGTH)
    print("Epochs:", NUMBER_OF_EPOCHS)
    print("Training batch size:", TRAIN_BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Class weights: balanced")

    print("\nLoading ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    training_dataset = ClinicalDataset(
        training_texts,
        tokenizer,
        labels=training_labels,
    )
    test_dataset = ClinicalDataset(
        test_texts,
        tokenizer,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EXPECTED_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    training_arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUMBER_OF_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PREDICTION_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=20,
        logging_first_step=True,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        class_weights=class_weights,
    )

    print("Device used:", trainer.args.device)
    print("\n=== FINAL TRAINING ON ALL 2,608 INSTANCES ===")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    training_log = pd.DataFrame(trainer.state.log_history)
    training_log.to_csv(TRAINING_LOG_PATH, index=False)

    print("\n=== PREDICTING THE 398 CLEAN TEST INSTANCES ===")
    prediction_output = trainer.predict(test_dataset)
    logits = prediction_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    clinicalbert_ids = np.argmax(logits, axis=-1)
    clinicalbert_predictions = [
        id_to_label[int(label_id)]
        for label_id in clinicalbert_ids
    ]
    clinicalbert_confidence = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy().max(axis=1)

    final_predictions = []
    decision_sources = []

    for keyword_prediction, clinicalbert_prediction in zip(
        test_rule_predictions,
        clinicalbert_predictions,
    ):
        if pd.notna(keyword_prediction):
            final_predictions.append(keyword_prediction)
            decision_sources.append("keyword_rule")
        else:
            final_predictions.append(clinicalbert_prediction)
            decision_sources.append("clinicalbert_fallback")

    validate_final_predictions(final_predictions, len(test))
    write_and_verify_results_txt(final_predictions)

    prediction_table = pd.DataFrame(
        {
            "test_row": np.arange(len(test)),
            "predicted_label": final_predictions,
            "decision_source": decision_sources,
            "keyword_rule_prediction": test_rule_predictions,
            "clinicalbert_prediction": clinicalbert_predictions,
            "clinicalbert_confidence": clinicalbert_confidence,
            "description": test["description"],
            "sample_name": test["sample_name"],
            "original_keywords": test["keywords"],
            "keywords_without_prefix": test["keywords"].apply(
                remove_label_like_keyword_prefix
            ),
        }
    )
    prediction_table.to_csv(PREDICTIONS_PATH, index=False)

    prediction_counts = pd.Series(final_predictions).value_counts()
    keyword_rule_count = decision_sources.count("keyword_rule")
    fallback_count = decision_sources.count("clinicalbert_fallback")

    report_lines = [
        "=== FINAL HYBRID TRAINING AND PREDICTION ===",
        "",
        f"Model: {MODEL_NAME}",
        "ClinicalBERT keyword prefix removed: yes",
        "ClinicalBERT class weights: balanced",
        f"Random seed: {RANDOM_STATE}",
        f"Maximum tokens: {MAX_LENGTH}",
        f"Epochs: {NUMBER_OF_EPOCHS}",
        f"Training batch size: {TRAIN_BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Training instances: {len(train)}",
        f"Test instances: {len(test)}",
        f"Keyword-rule predictions: {keyword_rule_count}",
        f"ClinicalBERT fallback predictions: {fallback_count}",
        f"results.txt lines: {len(final_predictions)}",
        "Official test labels used: no",
        "",
        "=== FINAL PREDICTION COUNTS ===",
        prediction_counts.to_string(),
    ]
    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n=== FINAL OUTPUT VERIFIED ===")
    print("results.txt lines:", len(final_predictions))
    print("Keyword-rule predictions:", keyword_rule_count)
    print("ClinicalBERT fallback predictions:", fallback_count)
    print("Unexpected labels: 0")
    print("Header in results.txt: no")
    print("Prediction order preserved: yes")
    print("\nFiles created:")
    print("-", RESULTS_TXT_PATH)
    print("-", PREDICTIONS_PATH)
    print("-", REPORT_PATH)
    print("-", TRAINING_LOG_PATH)


if __name__ == "__main__":
    main()
