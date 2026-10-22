"""Stage a private Kaggle script that replays the retained frozen v3 source."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import re


FROZEN_BUNDLE_SHA256 = "b54be749fc45aaa81625a5fea82e61c83f968db9e9798ee5e49416e6bd0787fb"
SOURCE_RUN_ID = "kaggle-robust-aggregation-v3-reproduction-r1"
RUN_ID_PATTERN = re.compile(r"kaggle-robust-aggregation-v3-replay-r[1-9][0-9]*")
KERNEL_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(run_id: str, kernel_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run ID must be kaggle-robust-aggregation-v3-replay-rN, with N >= 1")
    if not KERNEL_ID_PATTERN.fullmatch(kernel_id):
        raise ValueError("kernel ID must be a lowercase Kaggle owner/slug pair")


def stage(root: Path, run_id: str, kernel_id: str, output: Path) -> dict:
    validate(run_id, kernel_id)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing staging directory: {output}")

    bundle = root / "releases/20261021/robust-source.bundle"
    checksum = root / "releases/20261021/robust-source.sha256"
    if not bundle.is_file() or not checksum.is_file():
        raise FileNotFoundError("retained frozen v3 source transport is missing")
    if checksum.read_text().split()[0] != FROZEN_BUNDLE_SHA256 or digest(bundle) != FROZEN_BUNDLE_SHA256:
        raise ValueError("retained frozen v3 source transport hash mismatch")

    template = (root / "kaggle/reproduction/run_reproduction.py").read_text()
    old_run = f'RUN_ID = "{SOURCE_RUN_ID}"'
    old_work = 'WORK = Path("/kaggle/working/ml4-material-discovery-v3-reproduction")'
    if template.count(old_run) != 1 or template.count(old_work) != 1:
        raise ValueError("reproduction launcher template identity is not recognized")
    launcher = template.replace(old_run, f'RUN_ID = "{run_id}"').replace(
        old_work, f'WORK = Path("/kaggle/working/{run_id}")'
    )

    output.mkdir(parents=True)
    launcher_path = output / "run_replay.py"
    launcher_path.write_text(launcher)
    metadata = {
        "id": kernel_id,
        "title": f"ML4 Material Discovery v3 replay {run_id.rsplit('-', 1)[-1]}",
        "code_file": launcher_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": ["ryangomez010/ml4-material-discovery-source"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    metadata_path = output / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "status": "STAGED_FOR_EXACT_FROZEN_REPLAY",
        "run_id": run_id,
        "kernel_id": kernel_id,
        "frozen_bundle_sha256": FROZEN_BUNDLE_SHA256,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "launcher_sha256": digest(launcher_path),
        "metadata_sha256": digest(metadata_path),
        "submission_command": f"kaggle kernels push -p {output}",
        "scope": "Exact replay of retained v3 source only; this does not authorize a changed scientific configuration.",
    }
    (output / "stage-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kernel-id", default="ryangomez010/ml4-material-discovery-v3-replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(stage(root, args.run_id, args.kernel_id, args.output), indent=2))


if __name__ == "__main__":
    main()
