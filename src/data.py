from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

FEATURE_COLUMNS = [
    "description",
    "sample_name",
    "transcription",
    "keywords",
]

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


def load_data():
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        sep=";",
    )

    test = pd.read_csv(
        DATA_DIR / "test_no_labels.csv",
        sep=";",
        header=None,
        names=FEATURE_COLUMNS,
    )

    return train, test


def clean_data(train, test):
    clean_train = train.copy()
    clean_test = test.copy()

    # Linhas cuja etiqueta não é uma das 12 especialidades
    invalid_label_mask = ~clean_train["medical_specialty"].isin(
        EXPECTED_LABELS
    )

    # Linhas com pelo menos três campos vazios
    too_many_missing_mask = (
        clean_train[FEATURE_COLUMNS].isna().sum(axis=1) >= 3
    )

    # Descrições anormalmente longas
    description_length = (
        clean_train["description"]
        .fillna("")
        .str.len()
    )

    abnormally_long_description_mask = description_length > 20_000

    rows_to_remove = (
        invalid_label_mask
        | too_many_missing_mask
        | abnormally_long_description_mask
    )

    print("Linhas removidas do treino:", rows_to_remove.sum())

    clean_train = clean_train.loc[~rows_to_remove].copy()

    # Os modelos de texto precisam de strings, não de valores NaN
    clean_train[FEATURE_COLUMNS] = (
        clean_train[FEATURE_COLUMNS].fillna("")
    )

    # Não removemos nenhuma linha do teste
    clean_test[FEATURE_COLUMNS] = (
        clean_test[FEATURE_COLUMNS].fillna("")
    )

    return clean_train, clean_test


if __name__ == "__main__":
    raw_train, raw_test = load_data()
    train, test = clean_data(raw_train, raw_test)

    print("Treino original:", raw_train.shape)
    print("Treino limpo:", train.shape)
    print("Teste:", test.shape)

    print("Especialidades:", train["medical_specialty"].nunique())
    print("Valores em falta no treino:", train.isna().sum().sum())
    print("Valores em falta no teste:", test.isna().sum().sum())