import json

import pytest

from sparking.evidence import create_manifest, verify_manifest


def test_manifest_detects_substitution_and_extra_file(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"value": 1}))
    create_manifest(tmp_path)
    assert verify_manifest(tmp_path)["valid"]
    (tmp_path / "result.json").write_text(json.dumps({"value": 2}))
    with pytest.raises(ValueError, match="changed"):
        verify_manifest(tmp_path)


def test_manifest_detects_unlisted_file(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    create_manifest(tmp_path)
    (tmp_path / "surprise.txt").write_text("not declared")
    with pytest.raises(ValueError, match="extra"):
        verify_manifest(tmp_path)

