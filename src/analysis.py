from data import FEATURE_COLUMNS, load_data


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


train, test = load_data()


print("=== TAMANHO DOS DADOS ===")
print("Treino:", train.shape)
print("Teste:", test.shape)


print("\n=== PRIMEIROS CASOS DO TREINO ===")
print(
    train[
        ["medical_specialty", "description", "sample_name"]
    ].head().to_string(index=False)
)


print("\n=== NÚMERO DE ETIQUETAS ENCONTRADAS ===")
print(train["medical_specialty"].nunique())


print("\n=== EXEMPLOS POR ESPECIALIDADE ===")
class_counts = (
    train["medical_specialty"]
    .value_counts()
    .reindex(EXPECTED_LABELS)
)

print(class_counts)


print("\n=== ETIQUETAS INESPERADAS ===")
unexpected_mask = ~train["medical_specialty"].isin(EXPECTED_LABELS)
unexpected_labels = train.loc[
    unexpected_mask,
    "medical_specialty",
]

print("Número de linhas com etiquetas inesperadas:", len(unexpected_labels))

for label in unexpected_labels:
    print("-", repr(str(label)[:100]))


print("\n=== VALORES EM FALTA NO TREINO ===")
print(train.isna().sum())


print("\n=== VALORES EM FALTA NO TESTE ===")
print(test.isna().sum())


print("\n=== PERCENTAGEM EM FALTA NO TREINO ===")
print((train.isna().mean() * 100).round(2))


print("\n=== PERCENTAGEM EM FALTA NO TESTE ===")
print((test.isna().mean() * 100).round(2))

print("\n=== LINHAS COM PELO MENOS 3 CAMPOS EM FALTA ===")

valid_train = train[
    train["medical_specialty"].isin(EXPECTED_LABELS)
].copy()

missing_per_row = valid_train[FEATURE_COLUMNS].isna().sum(axis=1)

suspicious_rows = valid_train[missing_per_row >= 3]

print("Número de linhas suspeitas:", len(suspicious_rows))

for index, row in suspicious_rows.iterrows():
    description = str(row["description"])

    print("\nÍndice:", index)
    print("Especialidade:", row["medical_specialty"])
    print("Tamanho da description:", len(description))
    print("Início da description:", repr(description[:150]))