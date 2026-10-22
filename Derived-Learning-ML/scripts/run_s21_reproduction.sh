#!/usr/bin/env sh
set -eu

# Usage: sh scripts/run_s21_reproduction.sh /absolute/or/relative/output-directory
# Runs the frozen S2.1 protocol into a new directory; never overwrite a saved run.
if [ "$#" -ne 1 ]; then
  echo "usage: sh scripts/run_s21_reproduction.sh OUTPUT_DIRECTORY" >&2
  exit 2
fi

output_dir=$1
if [ -e "$output_dir" ]; then
  echo "refusing to overwrite existing output: $output_dir" >&2
  exit 2
fi

export PYTHONPATH=src
export PYTHONHASHSEED=0
python3 -m derived_learning run --config configs/s21_structural_misspecification_intercept.json --output-dir "$output_dir" --project-root .
python3 -m derived_learning evaluate --run-dir "$output_dir" --output "$output_dir/evaluation_verification.json"
python3 -m derived_learning verify --run-dir "$output_dir" --output "$output_dir/manifest_verification.json"
python3 -m derived_learning report --run-dir "$output_dir" --output "$output_dir/REPORT.md"
python3 -m derived_learning analyze --run-dir "$output_dir" --output-dir "$output_dir/analysis"
