#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing project interpreter: ${PYTHON}" >&2
  exit 1
fi
if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic is required to compile paper/main.pdf" >&2
  exit 1
fi

"${PYTHON}" -S "${ROOT}/scripts/verify_development_v2.py" \
  --artifact-root "${ROOT}/artifacts/kaggle-development-v2-r1"
"${PYTHON}" -S "${ROOT}/scripts/generate_paper_assets.py"
(
  cd "${ROOT}/paper"
  SOURCE_DATE_EPOCH=1792576800 FORCE_SOURCE_DATE=1 tectonic main.tex
)
shasum -a 256 "${ROOT}/paper/main.pdf"
