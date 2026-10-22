"""Build an explicitly enumerated source/data transport, excluding credentials."""
from pathlib import Path
import hashlib
import json
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    refuse_existing_freeze(root)
    stage = root / "kaggle/staging-v3"
    stage.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ["src", "tests", "scripts", "configs", "PROTOCOLS", "data/raw", "data/manifests"]:
        files.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.extend(root / name for name in ["pyproject.toml", "uv.lock", "README.md", "LICENSE", "PROTOCOL.md", "RESEARCH_STATE.json"])
    files = sorted(files)
    identities = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_identity = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    freeze = {"status": "FROZEN_FOR_FUTURE_EXECUTION", "evaluation_class": "DEVELOPMENT_PREVIOUSLY_INSPECTED_LABELS", "source_sha256": source_identity, "files": identities}
    with (root / "artifacts/freeze-v3.json").open("x") as stream:
        stream.write(json.dumps(freeze, indent=2) + "\n")
    bundle = stage / "robust-source.bundle"
    with tarfile.open(bundle, "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(root / "artifacts/freeze-v3.json", arcname="artifacts/freeze-v3.json")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (stage / "robust-source.sha256").write_text(digest + "  robust-source.bundle\n")
    metadata = json.loads((root / "kaggle/staging/dataset-metadata.json").read_text())
    metadata["description"] = "Private frozen v3 source/data transport. Historical Kaggle dataset versions preserve v1/v2 source. CPU-only bounded development experiment; no prospective discovery claim."
    (stage / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"files": len(files), "source_sha256": source_identity, "bundle_sha256": digest, "bytes": bundle.stat().st_size}))


def refuse_existing_freeze(root):
    if (root / "artifacts/freeze-v3.json").exists() or (root / "kaggle/staging-v3/robust-source.bundle").exists():
        raise FileExistsError("v3 freeze/transport already exists; reuse the retained bundle for reproduction, or create a separately versioned successor package")


if __name__ == "__main__":
    main()
