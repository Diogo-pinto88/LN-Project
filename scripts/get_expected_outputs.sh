#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
	echo "Usage: $0 input.csv [output.txt]" >&2
	exit 1
fi

input_csv="$1"

if [[ ! -f "$input_csv" ]]; then
	echo "Error: file not found: $input_csv" >&2
	exit 1
fi

if [[ $# -eq 2 ]]; then
	output_file="$2"
else
	output_dir="$(dirname "$input_csv")"
	output_file="$output_dir/expected_classification.txt"
fi

python3 - "$input_csv" "$output_file" <<'PY'
import csv
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, newline="", encoding="utf-8") as csv_file, open(
		output_path, "w", encoding="utf-8", newline=""
) as out_file:
		reader = csv.reader(csv_file, delimiter=";", quotechar='"')
		next(reader, None)
		for row in reader:
				if row and row[0].strip():
						out_file.write(row[0].strip() + "\n")
PY

echo "Wrote first column to $output_file"
