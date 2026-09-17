import json
import tempfile
import unittest
from pathlib import Path

from atlas.percy import load_json_object


class PercyDuplicateJsonKeyTests(unittest.TestCase):
    def _write_json(self, root: Path, text: str) -> Path:
        path = root / "payload.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_rejects_duplicate_top_level_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(
                Path(tmp),
                '{"atlas_run_id":"first","atlas_run_id":"second"}',
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate JSON object key is not permitted: atlas_run_id",
            ):
                load_json_object(path)

    def test_rejects_duplicate_nested_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(
                Path(tmp),
                '{"status_contract":{"success_status":"FAILED","success_status":"SUCCEEDED"}}',
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate JSON object key is not permitted: success_status",
            ):
                load_json_object(path)

    def test_valid_object_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "atlas_run_id": "run-1",
                "status_contract": {"success_status": "SUCCEEDED"},
            }
            path = self._write_json(Path(tmp), json.dumps(payload))

            self.assertEqual(load_json_object(path), payload)


if __name__ == "__main__":
    unittest.main()
