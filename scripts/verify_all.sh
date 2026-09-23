#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python; fi
export PYTHONHASHSEED=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
OUT="${1:-runs/acceptance-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$OUT" ]]; then echo "Output already exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
"$PY" -m pip check
"$PY" -m gesture doctor
"$PY" -m ruff check .
"$PY" -m pytest -q --junitxml="$OUT/junit.xml"
for task in reach path pick-place; do
  "$PY" -m gesture demo --task "$task" --seconds 30 --headless --output "$OUT/$task" --assert-success
  "$PY" -m gesture replay --episode "$OUT/$task" --headless
  "$PY" -m gesture verify-data --episode "$OUT/$task"
  "$PY" -m gesture export --episode "$OUT/$task" --output "$OUT/$task.npz"
  "$PY" -m gesture evaluate --episode "$OUT/$task" --output "$OUT/${task}-metrics"
done
"$PY" -m gesture reprocess --episode "$OUT/reach" --output "$OUT/reach-no-filter" --filter none
"$PY" -m gesture verify-data --episode "$OUT/reach-no-filter"
"$PY" -m pip freeze --exclude-editable > "$OUT/environment.txt"
printf '\nAcceptance completed; evidence: %s\n' "$OUT"
