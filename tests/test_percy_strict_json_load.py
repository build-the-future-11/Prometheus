import tempfile
import unittest
from pathlib import Path

from atlas.percy import load_json_object


class PercyStrictJsonLoadTests(unittest.TestCase):
    def test_load_rejects_non_standard_json_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for token in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(token=token):
                    path = root / f"payload-{token.replace('-', 'neg-')}.json"
                    path.write_text(f'{{"value": {token}}}\n', encoding="utf-8")

                    with self.assertRaisesRegex(
                        ValueError,
                        f"non-standard JSON constant is not permitted: {token}",
                    ):
                        load_json_object(path)

    def test_load_accepts_standards_compliant_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            path.write_text('{"enabled": true, "count": 3}\n', encoding="utf-8")

            self.assertEqual(load_json_object(path), {"enabled": True, "count": 3})

    def test_load_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            path.write_text('[1, 2, 3]\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                load_json_object(path)


if __name__ == "__main__":
    unittest.main()
