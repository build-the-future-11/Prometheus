"""Verify and import a completed v3 archive without overwriting existing runs."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("checksum", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--artifacts-destination", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--notebook", default="https://www.kaggle.com/code/ryangomez010/ml4-material-discovery-robust-aggregation-v3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    with args.archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    expected = args.checksum.read_text().split()[0]
    if h.hexdigest() != expected:
        raise ValueError("downloaded archive hash mismatch")
    stage = args.stage or root / "kaggle/downloads/robust-v3/verified-import"
    artifact_destination = args.artifacts_destination or root / "artifacts/hosted-v3"
    receipt_path = args.receipt or root / "artifacts/hosted-import-v3.json"
    if artifact_destination.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite retained import artifacts or receipt")
    stage.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as archive:
        archive.extractall(stage, filter="data")
    run_dirs = list((stage / "runs").iterdir())
    if len(run_dirs) != 1:
        raise ValueError("expected exactly one recorded run")
    run = run_dirs[0]
    destination = root / "runs" / run.name
    if destination.exists():
        raise FileExistsError("refusing to overwrite a retained run")
    shutil.copytree(run, destination)
    shutil.copytree(stage / "artifacts", artifact_destination)
    receipt = {"archive_sha256": h.hexdigest(), "archive_bytes": args.archive.stat().st_size,
               "run_id": run.name, "notebook": args.notebook,
               "status": "DOWNLOADED_HASH_VERIFIED_NOT_YET_NUMERICALLY_VERIFIED"}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
