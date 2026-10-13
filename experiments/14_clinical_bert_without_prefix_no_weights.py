import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"
CHECKPOINT_DIR = (
    PROJECT_DIR
    / ".checkpoints"
    / "14_clinical_bert_without_prefix_no_weights"
)

sys.path.insert(0, str(SRC_DIR))

from data import clean_data, load_data


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

MAX_LENGTH = 128
NUMBER_OF_EPOCHS = 2

TRAIN_BATCH_SIZE = 8
EVALUATION_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


EXPECTED_LABELS = [
    "Cardiovascular-Pulmonary",
    "Dermatology",
    "Gastroenterology",
    "General Medicine",
    "Neurology",
    "Neurosurgery",
    "Obstetrics-Gynecology",
    "Ophthalmology",
    "Orthopedic",
    "Psychiatry-Psychology",
    "Radiology",
    "Surgery",
]


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


# ============================================================
# REMOÇÃO DO PREFIXO DAS KEYWORDS
# ============================================================

def remove_label_like_keyword_prefix(keywords):
    """
    Remove o primeiro elemento das keywords quando este
    corresponde diretamente a uma das especialidades.
    """

    parts = str(keywords).split(",", maxsplit=1)
    first_keyword = parts[0].strip().lower()

    if first_keyword in LABEL_LIKE_KEYWORDS:
        if len(parts) == 2:
            return parts[1].strip()

        return ""

    return str(keywords).strip()


def has_label_like_keyword_prefix(keywords):
    """
    Verifica se o primeiro elemento das keywords identifica
    diretamente uma das doze especialidades.
    """

    first_keyword = (
        str(keywords)
        .split(",", maxsplit=1)[0]
        .strip()
        .lower()
    )

    return first_keyword in LABEL_LIKE_KEYWORDS


# ============================================================
# DATASET PARA O PYTORCH
# ============================================================

class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return item


# ============================================================
# CONSTRUÇÃO DO TEXTO
# ============================================================

def build_text(row):
    """
    Constrói o texto sem o primeiro elemento das keywords
    quando esse elemento identifica diretamente a classe.
    """

    sample_name = str(row["sample_name"]).strip()
    description = str(row["description"]).strip()

    keywords = remove_label_like_keyword_prefix(
        row["keywords"]
    )

    transcription = str(row["transcription"]).strip()

    return (
        f"SAMPLE NAME: {sample_name} "
        f"DESCRIPTION: {description} "
        f"KEYWORDS: {keywords} "
        f"TRANSCRIPTION: {transcription}"
    )


# ============================================================
# MÉTRICAS
# ============================================================

def compute_metrics(evaluation_prediction):
    logits = evaluation_prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    true_labels = evaluation_prediction.label_ids
    predicted_labels = np.argmax(logits, axis=-1)

    per_class_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=list(range(len(EXPECTED_LABELS))),
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "minimum_f1": float(per_class_f1.min()),
    }


# ============================================================
# EXPERIÊNCIA
# ============================================================

def main():
    set_seed(RANDOM_STATE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Carregar e limpar os dados
    # --------------------------------------------------------

    raw_train, raw_test = load_data()
    train, _ = clean_data(raw_train, raw_test)

    print("\n=== DATA ===")
    print("Clean training instances:", len(train))

    label_to_id = {
        label: index
        for index, label in enumerate(EXPECTED_LABELS)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    # --------------------------------------------------------
    # Divisão treino/validação
    # --------------------------------------------------------

    training_indices, validation_indices = train_test_split(
        train.index,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train["medical_specialty"],
    )

    training_data = train.loc[training_indices].copy()
    validation_data = train.loc[validation_indices].copy()

    print("Training instances:", len(training_data))
    print("Validation instances:", len(validation_data))

    training_prefixes_removed = (
        training_data["keywords"]
        .apply(has_label_like_keyword_prefix)
        .sum()
    )

    validation_prefixes_removed = (
        validation_data["keywords"]
        .apply(has_label_like_keyword_prefix)
        .sum()
    )

    print("\n=== KEYWORD PREFIX REMOVAL ===")
    print(
        "Training prefixes removed:",
        training_prefixes_removed,
    )
    print(
        "Validation prefixes removed:",
        validation_prefixes_removed,
    )

    print("\n=== CONFIGURATION ===")
    print("Maximum tokens:", MAX_LENGTH)
    print("Epochs:", NUMBER_OF_EPOCHS)
    print("Training batch size:", TRAIN_BATCH_SIZE)
    print("Class weights: no")

    # --------------------------------------------------------
    # Construir os textos
    # --------------------------------------------------------

    training_texts = (
        training_data
        .apply(build_text, axis=1)
        .tolist()
    )

    validation_texts = (
        validation_data
        .apply(build_text, axis=1)
        .tolist()
    )

    training_labels = (
        training_data["medical_specialty"]
        .map(label_to_id)
        .tolist()
    )

    validation_labels = (
        validation_data["medical_specialty"]
        .map(label_to_id)
        .tolist()
    )

    # --------------------------------------------------------
    # Tokenizador e datasets
    # --------------------------------------------------------

    print("\nLoading ClinicalBERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    training_dataset = ClinicalDataset(
        training_texts,
        training_labels,
        tokenizer,
    )

    validation_dataset = ClinicalDataset(
        validation_texts,
        validation_labels,
        tokenizer,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EXPECTED_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # --------------------------------------------------------
    # Configuração do treino
    # --------------------------------------------------------

    training_arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),

        num_train_epochs=NUMBER_OF_EPOCHS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVALUATION_BATCH_SIZE,
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=20,
        logging_first_step=True,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,

        optim="adamw_torch",

        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,

        report_to="none",

        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Trainer normal, sem pesos de classe
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1
            )
        ],
    )

    print("\nDevice used:", trainer.args.device)
    print(
        "\n=== TRAINING CLINICALBERT WITHOUT PREFIX "
        "AND WITHOUT CLASS WEIGHTS ==="
    )

    trainer.train()

    # --------------------------------------------------------
    # Previsões na validação
    # --------------------------------------------------------

    prediction_output = trainer.predict(
        validation_dataset
    )

    logits = prediction_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=-1)

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()

    confidence_scores = probabilities.max(axis=1)

    predicted_labels = [
        id_to_label[int(label_id)]
        for label_id in predicted_ids
    ]

    correct_labels = (
        validation_data["medical_specialty"]
        .tolist()
    )

    # --------------------------------------------------------
    # Calcular resultados
    # --------------------------------------------------------

    accuracy = accuracy_score(
        correct_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        correct_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    report_text = classification_report(
        correct_labels,
        predicted_labels,
        labels=EXPECTED_LABELS,
        target_names=EXPECTED_LABELS,
        digits=3,
        zero_division=0,
    )

    report_dictionary = classification_report(
        correct_labels,
        predicted_labels,
        labels=EXPECTED_LABELS,
        target_names=EXPECTED_LABELS,
        output_dict=True,
        zero_division=0,
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

    correct_predictions = sum(
        predicted == correct
        for predicted, correct in zip(
            predicted_labels,
            correct_labels,
        )
    )

    summary = (
        "=== CLINICALBERT WITHOUT PREFIX "
        "AND WITHOUT CLASS WEIGHTS ===\n"
        f"Model: {MODEL_NAME}\n"
        "Keyword prefix removed: yes\n"
        "Class weights: no\n"
        f"Maximum tokens: {MAX_LENGTH}\n"
        f"Epochs: {NUMBER_OF_EPOCHS}\n"
        f"Training instances: {len(training_data)}\n"
        f"Validation instances: {len(validation_data)}\n"
        f"Training prefixes removed: "
        f"{training_prefixes_removed}\n"
        f"Validation prefixes removed: "
        f"{validation_prefixes_removed}\n"
        f"Correct predictions: "
        f"{correct_predictions}/{len(validation_data)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro-F1: {macro_f1:.4f}\n"
        f"Minimum per-class F1: "
        f"{minimum_f1_label} = {minimum_f1:.4f}\n"
        f"Best checkpoint: "
        f"{trainer.state.best_model_checkpoint}\n"
        "\n=== CLASSIFICATION REPORT ===\n"
        f"{report_text}"
    )

    print("\n" + summary)

    # --------------------------------------------------------
    # Guardar relatório
    # --------------------------------------------------------

    report_path = (
        RESULTS_DIR
        / (
            "14_clinical_bert_without_prefix_"
            "no_weights_report.txt"
        )
    )

    report_path.write_text(
        summary,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Guardar previsões
    # --------------------------------------------------------

    validation_results = pd.DataFrame({
        "source_index": validation_data.index,
        "correct_label": correct_labels,
        "predicted_label": predicted_labels,
        "is_correct": [
            predicted == correct
            for predicted, correct in zip(
                predicted_labels,
                correct_labels,
            )
        ],
        "confidence": confidence_scores,
        "description": validation_data[
            "description"
        ].tolist(),
        "sample_name": validation_data[
            "sample_name"
        ].tolist(),
        "original_keywords": validation_data[
            "keywords"
        ].tolist(),
        "keywords_without_prefix": validation_data[
            "keywords"
        ].apply(
            remove_label_like_keyword_prefix
        ).tolist(),
        "transcription_excerpt": validation_data[
            "transcription"
        ].str[:700].tolist(),
    })

    predictions_path = (
        RESULTS_DIR
        / (
            "14_clinical_bert_without_prefix_"
            "no_weights_validation_predictions.csv"
        )
    )

    validation_results.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Guardar evolução do treino
    # --------------------------------------------------------

    training_log = pd.DataFrame(
        trainer.state.log_history
    )

    training_log_path = (
        RESULTS_DIR
        / (
            "14_clinical_bert_without_prefix_"
            "no_weights_training_log.csv"
        )
    )

    training_log.to_csv(
        training_log_path,
        index=False,
    )

    print("\nFiles created:")
    print(f"- {report_path}")
    print(f"- {predictions_path}")
    print(f"- {training_log_path}")


if __name__ == "__main__":
    main()