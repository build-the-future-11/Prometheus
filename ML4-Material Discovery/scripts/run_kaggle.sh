#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/run_kaggle.sh RUN_ID" >&2
  exit 2
fi

run_id="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

export PYTHONPATH="$root/src:$root${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ml4-materials-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ml4-materials-xdg}"
export MPL_IGNORE_SYSTEM_FONTS=1

python -m pytest -q
python -m ml4materials prepare
python -m ml4materials preflight \
  --config configs/study.json \
  --output "artifacts/preflight-${run_id}.json"
python -m ml4materials run --run-id "$run_id"
python -m ml4materials verify --run-id "$run_id"
python scripts/reproduce_submission.py --run-id "$run_id"
python -m ml4materials registry --output "artifacts/run-registry-${run_id}.json"

archive="artifacts/${run_id}.tar.gz"
tar -czf "$archive" \
  "runs/${run_id}" \
  "artifacts/preflight-${run_id}.json" \
  "artifacts/run-registry-${run_id}.json"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$archive" > "${archive}.sha256"
else
  shasum -a 256 "$archive" > "${archive}.sha256"
fi

echo "HOSTED_RUN_COMPLETE=$run_id"
echo "HOSTED_ARCHIVE=$archive"
echo "HOSTED_ARCHIVE_SHA256=$(cut -d ' ' -f 1 "${archive}.sha256")"
