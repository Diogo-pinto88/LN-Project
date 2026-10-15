#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"

expected_file="${1:-$project_dir/data/expected_classification.txt}"
obtained_file="${2:-$project_dir/data/results.txt}"

if [[ ! -f "$expected_file" ]]; then
	echo "Error: expected file not found: $expected_file" >&2
	exit 1
fi

if [[ ! -f "$obtained_file" ]]; then
	# Fallback for the misspelled filename mentioned in the request.
	if [[ -f "$project_dir/data/optained_classification.txt" ]]; then
		obtained_file="$project_dir/data/optained_classification.txt"
	else
		echo "Error: obtained file not found: $obtained_file" >&2
		exit 1
	fi
fi

python3 - "$expected_file" "$obtained_file" <<'PY'
import sys
from collections import defaultdict
from pathlib import Path

expected_path = Path(sys.argv[1])
obtained_path = Path(sys.argv[2])

specialties = [
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

with expected_path.open("r", encoding="utf-8") as expected_file:
    expected_labels = [line.strip() for line in expected_file if line.strip()]

with obtained_path.open("r", encoding="utf-8") as obtained_file:
    obtained_labels = [line.strip() for line in obtained_file if line.strip()]

max_len = max(len(expected_labels), len(obtained_labels))
metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

all_labels = set(specialties)
all_labels.update(expected_labels)
all_labels.update(obtained_labels)

for index in range(max_len):
    expected = expected_labels[index] if index < len(expected_labels) else None
    obtained = obtained_labels[index] if index < len(obtained_labels) else None

    if expected is not None:
        all_labels.add(expected)
    if obtained is not None:
        all_labels.add(obtained)

    if expected == obtained and expected is not None:
        metrics[expected]["tp"] += 1
    else:
        if expected is not None:
            metrics[expected]["fn"] += 1
        if obtained is not None:
            metrics[obtained]["fp"] += 1


def score(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

print(f"Expected file: {expected_path}")
print(f"Obtained file:  {obtained_path}")
print()
print(f"{'Specialty':<28} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
print("-" * 86)

for label in specialties:
    tp = metrics[label]["tp"]
    fp = metrics[label]["fp"]
    fn = metrics[label]["fn"]
    precision, recall, f1 = score(tp, fp, fn)
    print(f"{label:<28} {precision*100:9.2f}% {recall*100:9.2f}% {f1*100:9.2f}% {tp:8d} {fp:8d} {fn:8d}")

extra_labels = sorted(all_labels - set(specialties))
if extra_labels:
    print()
    print("Labels not in the configured specialty list:")
    for label in extra_labels:
        tp = metrics[label]["tp"]
        fp = metrics[label]["fp"]
        fn = metrics[label]["fn"]
        precision, recall, f1 = score(tp, fp, fn)
        print(f"{label:<28} {precision*100:9.2f}% {recall*100:9.2f}% {f1*100:9.2f}% {tp:8d} {fp:8d} {fn:8d}")
PY
