import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "NEUROGENESIS/scripts/assemble_submission_repaired.py"
spec = importlib.util.spec_from_file_location("assemble_repaired", FILE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def package(root):
    result = root / "reproducibility/results"
    result.mkdir(parents=True)
    (result / "trace.json").write_text('{"fixture":true}\n')
    (result / "manifest.json").write_text('{"nested":"must be hashed"}\n')
    data = (result / "trace.json").read_bytes()
    (root / "manifest.json").write_text(json.dumps({"schema_version": "neurogenesis.submission-package.v1", "files": {"reproducibility/results/trace.json": hashlib.sha256(data).hexdigest()}}))
    return data


def test_preserves_evidence_and_fixes_documented_path(tmp_path):
    before = package(tmp_path)
    result = module.repair_new_package(tmp_path)
    target = tmp_path / "reproducibility" / module.BUNDLE
    assert (target / "trace.json").read_bytes() == before
    assert (target / "manifest.json").relative_to(tmp_path).as_posix() in result["files"]
    for path, expected in result["files"].items():
        assert hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == expected
    assert result["schema_version"] == "neurogenesis.submission-package.v1.1"


def test_repair_cannot_run_twice(tmp_path):
    package(tmp_path); module.repair_new_package(tmp_path)
    with pytest.raises(ValueError): module.repair_new_package(tmp_path)


def test_tampered_input_rejected_before_move(tmp_path):
    package(tmp_path)
    (tmp_path / "reproducibility/results/trace.json").write_text('changed')
    with pytest.raises(ValueError, match="identity mismatch"): module.repair_new_package(tmp_path)
    assert (tmp_path / "reproducibility/results/trace.json").exists()


def test_unsafe_manifest_path_rejected(tmp_path):
    package(tmp_path)
    p = tmp_path / "manifest.json"; data = json.loads(p.read_text())
    data["files"]["../outside"] = "a"*64; p.write_text(json.dumps(data))
    with pytest.raises(ValueError): module.repair_new_package(tmp_path)
