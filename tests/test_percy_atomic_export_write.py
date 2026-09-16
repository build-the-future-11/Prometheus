import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atlas.percy import write_percy_export


class PercyAtomicExportWriteTests(unittest.TestCase):
    def test_write_replaces_destination_with_complete_json_and_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "exports" / "percy-tasks.json"
            output.parent.mkdir(parents=True)
            output.write_text("old\n", encoding="utf-8")
            if os.name == "posix":
                output.chmod(0o640)

            payload = {"z": 3, "a": {"value": 1}}
            returned = write_percy_export(payload, output)

            self.assertEqual(returned, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertTrue(output.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)

    def test_replace_failure_preserves_existing_export_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "percy-tasks.json"
            output.write_text("original-export\n", encoding="utf-8")

            with patch("atlas.percy.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_percy_export({"new": "payload"}, output)

            self.assertEqual(output.read_text(encoding="utf-8"), "original-export\n")
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

    def test_write_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "exports" / "percy-tasks.json"

            write_percy_export({"ok": True}, output)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})


if __name__ == "__main__":
    unittest.main()
