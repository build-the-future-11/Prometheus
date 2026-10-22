#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release="$root/releases/s3-r0"
mkdir -p "$release"
bundle="$release/dlml-s3-source.bundle"

# macOS writes AppleDouble sidecars when archiving files from some external
# volumes unless COPYFILE_DISABLE is set.  Those files are not source and make
# the release identity depend on Finder metadata, so fail closed on them.
export COPYFILE_DISABLE=1
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='._*' \
  -C "$root" -czf "$bundle" \
  src tests build_backend pyproject.toml README.md configs \
  PROTOCOL.md PROTOCOL_S2.md PROTOCOL_S21.md PROTOCOL_S3_REAL_DATA.md \
  scripts/run_s3_gpu_study.py scripts/verify_s3_hosted_archive.py

python3 - "$bundle" <<'PY'
from pathlib import Path, PurePosixPath
import sys
import tarfile

bundle = Path(sys.argv[1])
with tarfile.open(bundle, "r:gz") as archive:
    members = archive.getmembers()

bad_paths = []
for member in members:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        bad_paths.append(member.name)
    if any(part == ".DS_Store" or part.startswith("._") for part in path.parts):
        bad_paths.append(member.name)
    if member.issym() or member.islnk() or member.isdev():
        bad_paths.append(member.name)

if bad_paths:
    raise SystemExit(f"unsafe or metadata-only archive members: {sorted(set(bad_paths))}")
PY

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$bundle" > "$release/dlml-s3-source.sha256"
else
  shasum -a 256 "$bundle" > "$release/dlml-s3-source.sha256"
fi
echo "S3_SOURCE_BUNDLE=$bundle"
echo "S3_SOURCE_SHA256=$(cut -d ' ' -f 1 "$release/dlml-s3-source.sha256")"
