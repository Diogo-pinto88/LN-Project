#!/usr/bin/env python3

"""Classify medical specialties using a keyword heuristic and ClinicalBERT.

The script reads a semicolon-separated CSV file row by row, ignores the first
column when present, and predicts the medical specialty from the first keyword
entry in the keywords column.

If the first keyword does not match one of the known specialties, the script
falls back to a fine-tuned ClinicalBERT model stored in models/clinicalbert.

Usage:
	python3 src/classificator.py input.csv [output.txt]

If no output file is provided, the predictions are written to stdout.
Each prediction is written on its own line.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "clinicalbert"

SPECIALTIES = [
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


def normalize(text: str) -> str:
	return re.sub(r"[^a-z0-9]+", "", text.casefold())


SPECIALTY_BY_NORMALIZED_KEY = {normalize(specialty): specialty for specialty in SPECIALTIES}


def parse_row(row: list[str]) -> tuple[str, str, str, str]:
	"""Return the four non-label columns from a CSV row.

	The expected format is either:
	- label, description, sample_name, transcription, keywords
	- description, sample_name, transcription, keywords
	"""

	if len(row) >= 5:
		description, sample_name, transcription, keywords = row[1:5]
	elif len(row) == 4:
		description, sample_name, transcription, keywords = row
	else:
		padded = list(row[1:])
		while len(padded) < 4:
			padded.append("")
		description, sample_name, transcription, keywords = padded[:4]

	return (
		description.strip(),
		sample_name.strip(),
		transcription.strip(),
		keywords.strip(),
	)


def first_keyword_label(keywords: str) -> str:
	first_keyword = keywords.split(",", 1)[0].strip().strip('"').strip("'")
	return SPECIALTY_BY_NORMALIZED_KEY.get(normalize(first_keyword), "Unknown")


def build_text(description: str, sample_name: str, transcription: str, keywords: str) -> str:
	return (
		"DESCRIPTION "
		+ description
		+ " SAMPLE_NAME "
		+ sample_name
		+ " TRANSCRIPTION "
		+ transcription
		+ " KEYWORDS "
		+ keywords
	)


def load_clinicalbert_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
	if not MODEL_DIR.exists():
		raise FileNotFoundError(
			f"ClinicalBERT model not found at {MODEL_DIR}. Run experiments/09_clinicalbert.py first."
		)

	tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
	model.eval()
	return tokenizer, model


def predict_with_clinicalbert(texts: list[str], batch_size: int = 8) -> list[str]:
	if not texts:
		return []

	tokenizer, model = load_clinicalbert_model()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	predictions: list[str] = []

	with torch.no_grad():
		for start in range(0, len(texts), batch_size):
			batch_texts = texts[start : start + batch_size]
			encoded = tokenizer(
				batch_texts,
				truncation=True,
				padding=True,
				max_length=256,
				return_tensors="pt",
			)
			encoded = {key: value.to(device) for key, value in encoded.items()}
			logits = model(**encoded).logits
			batch_ids = logits.argmax(dim=-1).tolist()

			for label_id in batch_ids:
				label_name = model.config.id2label.get(int(label_id))
				if label_name is None:
					label_name = SPECIALTIES[int(label_id)]
				predictions.append(label_name)

	return predictions


def classify_csv(input_path: Path, output_path: Path | None = None) -> list[str]:
	predictions: list[str | None] = []
	unknown_texts: list[str] = []
	unknown_positions: list[int] = []

	with input_path.open("r", encoding="utf-8", newline="") as csv_file:
		reader = csv.reader(csv_file, delimiter=";")

		for row_number, row in enumerate(reader):
			if not row:
				continue

			if row_number == 0 and row[0].strip().lower() == "medical_specialty":
				continue

			description, sample_name, transcription, keywords = parse_row(row)
			predicted_label = first_keyword_label(keywords)

			if predicted_label == "Unknown":
				unknown_positions.append(len(predictions))
				unknown_texts.append(build_text(description, sample_name, transcription, keywords))
				predictions.append(None)
			else:
				predictions.append(predicted_label)

	if unknown_texts:
		fallback_predictions = predict_with_clinicalbert(unknown_texts)

		for position, fallback_prediction in zip(unknown_positions, fallback_predictions):
			predictions[position] = fallback_prediction

	final_predictions = [prediction or "Unknown" for prediction in predictions]

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8", newline="") as output_file:
			for prediction in final_predictions:
				output_file.write(prediction + "\n")
	else:
		for prediction in final_predictions:
			print(prediction)

	return final_predictions


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Classify each row by the first keyword in the keywords column, "
			"falling back to ClinicalBERT when needed."
		)
	)
	parser.add_argument("input_csv", type=Path, help="Input CSV file")
	parser.add_argument(
		"output_file",
		nargs="?",
		type=Path,
		help="Optional output file. If omitted, predictions are printed to stdout.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	if not args.input_csv.is_file():
		print(f"Error: file not found: {args.input_csv}", file=sys.stderr)
		return 1

	classify_csv(args.input_csv, args.output_file)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
