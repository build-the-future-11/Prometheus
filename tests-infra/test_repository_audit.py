import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_repository", ROOT / "scripts/audit_repository.py")
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def complete():
    return ({p: {"mode": "040000", "kind": "tree", "sha": "a"*40} for p in audit.PROJECTS},
            {p + "/src/model.py": {} for p in audit.PROJECTS})


def test_complete_inventory_passes_without_requiring_worktree_files():
    root, files = complete(); assert audit.inspect(root, files, {})["passed"]


def test_gitlink_without_gitmodules_fails():
    root, files = complete(); root["FIN-ML"].update(mode="160000", kind="commit")
    result = audit.inspect(root, files, {})
    assert {x["code"] for x in result["findings"]} == {"GITLINK_SOURCE_UNAVAILABLE", "MISSING_GITMODULES"}


def test_gitmodules_does_not_pretend_linked_source_is_present():
    root, files = complete(); root["FIN-ML"].update(mode="160000", kind="commit"); root[".gitmodules"] = {}
    assert [x["code"] for x in audit.inspect(root, files, {})["findings"]] == ["GITLINK_SOURCE_UNAVAILABLE"]


def test_bytecode_is_not_source():
    root, files = complete(); del files["FIN-ML/src/model.py"]; files["FIN-ML/__pycache__/model.pyc"] = {}
    assert not audit.inspect(root, files, {})["passed"]


def test_legacy_workflow_is_detected():
    root, files = complete()
    assert not audit.inspect(root, files, {".github/workflows/ci.yml": "atlas init-run"})["passed"]


def test_nul_tree_parser_preserves_space_and_tab_paths():
    entry = audit.parse_tree("100644 blob " + "a"*40 + "\tML4-Material Discovery/a\tb.py\0")
    assert "ML4-Material Discovery/a\tb.py" in entry
