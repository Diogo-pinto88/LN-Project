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

read -r correct incorrect total < <(
	awk '
		BEGIN {
			correct = 0
			incorrect = 0
			total = 0
		}
		{
			expected = $0
			if ((getline obtained < obtained_file) > 0) {
				if (expected == obtained) {
					correct++
				} else {
					incorrect++
				}
				total++
			} else {
				incorrect++
				total++
			}
		}
		END {
			while ((getline obtained < obtained_file) > 0) {
				incorrect++
				total++
			}
			printf "%d %d %d\n", correct, incorrect, total
		}
	' obtained_file="$obtained_file" "$expected_file"
)

if [[ "$total" -eq 0 ]]; then
	precision="0.00"
else
	precision="$(awk -v c="$correct" -v t="$total" 'BEGIN { printf "%.2f", (c / t) * 100 }')"
fi

echo "Expected file: $expected_file"
echo "Obtained file:  $obtained_file"
echo "Correct lines:  $correct"
echo "Incorrect lines: $incorrect"
echo "Accuracy:      ${precision}%"
