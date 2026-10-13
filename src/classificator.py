#!/usr/bin/env python3

"""Classify medical specialties using the first keyword in each row.

The script reads a semicolon-separated CSV file row by row, ignores the first
column when present, and predicts the medical specialty from the first keyword
entry in the last column.

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


SPECIALTY_BY_NORMALIZED_KEY = {
	normalize(specialty): specialty for specialty in SPECIALTIES
}


def predict_specialty(keywords: str) -> str:
	first_keyword = keywords.split(",", 1)[0].strip().strip('"').strip("'")
	normalized = normalize(first_keyword)

	if normalized in SPECIALTY_BY_NORMALIZED_KEY:
		return SPECIALTY_BY_NORMALIZED_KEY[normalized]

	return "Unknown"


def classify_csv(input_path: Path, output_path: Path | None = None) -> list[str]:
	predictions: list[str] = []

	with input_path.open("r", encoding="utf-8", newline="") as csv_file:
		reader = csv.reader(csv_file, delimiter=";")

		for row_number, row in enumerate(reader):
			if not row:
				continue

			if row_number == 0 and row[0].strip().lower() == "medical_specialty":
				continue

			keywords = row[-1] if row else ""
			predictions.append(predict_specialty(keywords))

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8", newline="") as output_file:
			for prediction in predictions:
				output_file.write(prediction + "\n")
	else:
		for prediction in predictions:
			print(prediction)

	return predictions


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Classify each row by the first keyword in the keywords column."
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
