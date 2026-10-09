import sys
from pathlib import Path


# Permite importar funções da pasta src
PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


from data import clean_data, load_data


# Relação entre o primeiro elemento das keywords
# e as 12 especialidades válidas.
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


# Carregar os ficheiros
raw_train, raw_test = load_data()

# Aplicar a limpeza que já definimos
train, _ = clean_data(raw_train, raw_test)


# Obter apenas o primeiro elemento das keywords.
#
# Exemplo:
# "surgery, biopsy, tumour"
# transforma-se em:
# "surgery"
first_keyword = (
    train["keywords"]
    .str.split(",")
    .str[0]
    .str.strip()
    .str.lower()
)


# Tentar transformar o primeiro elemento numa especialidade.
#
# Quando o primeiro elemento não corresponde a uma especialidade,
# o resultado será NaN.
keyword_prediction = first_keyword.map(KEYWORD_TO_SPECIALTY)


# Identificar os casos em que foi possível obter uma especialidade.
covered_mask = keyword_prediction.notna()

number_of_examples = len(train)
number_covered = covered_mask.sum()
coverage = number_covered / number_of_examples


# Comparar a especialidade obtida através das keywords
# com a resposta verdadeira.
correct_mask = (
    keyword_prediction[covered_mask]
    == train.loc[covered_mask, "medical_specialty"]
)

number_correct = correct_mask.sum()
accuracy_when_covered = correct_mask.mean()


print("=== KEYWORD ANALYSIS ===")

print("Training examples:", number_of_examples)
print("Examples covered:", number_covered)
print("Correct predictions:", number_correct)

print(f"Coverage: {coverage:.2%}")
print(f"Accuracy when covered: {accuracy_when_covered:.2%}")


print("\n=== FIRST KEYWORD COUNTS ===")

print(
    first_keyword
    .value_counts()
    .head(20)
)


print("\n=== DISAGREEMENTS ===")

disagreement_mask = (
    covered_mask
    & (
        keyword_prediction
        != train["medical_specialty"]
    )
)

disagreements = train.loc[
    disagreement_mask,
    [
        "medical_specialty",
        "description",
        "keywords",
    ],
].copy()

disagreements["keyword_prediction"] = keyword_prediction[
    disagreement_mask
]

if disagreements.empty:
    print("No disagreements found.")
else:
    print(
        disagreements[
            [
                "medical_specialty",
                "keyword_prediction",
                "description",
                "keywords",
            ]
        ].to_string(index=False)
    )