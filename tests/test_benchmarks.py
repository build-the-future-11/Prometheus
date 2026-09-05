import json
import tempfile
import unittest
from pathlib import Path

from atlas.benchmarks import (
    build_benchmark_manifest,
    load_cases,
    load_predictions,
    score_exact_match,
    validate_benchmark,
)
from atlas.cli import main


class BenchmarkTests(unittest.TestCase):
    def _write_cases(self, root: Path) -> Path:
        path = root / "cases.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"id": "p1", "category": "physics", "answer": "42"}),
                json.dumps({"id": "m1", "category": "math", "answer": "x = 2"}),
                json.dumps({"id": "m2", "category": "math", "answer": "7"}),
            ]) + "\n",
            encoding="utf-8",
        )
        return path

    def test_scores_exact_match_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = self._write_cases(root)
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                "\n".join([
                    json.dumps({"id": "p1", "prediction": " 42 "}),
                    json.dumps({"id": "m1", "prediction": "X = 2"}),
                    json.dumps({"id": "extra", "prediction": "ignored"}),
                ]) + "\n",
                encoding="utf-8",
            )

            report = score_exact_match(load_cases(cases_path), load_predictions(predictions_path))
            self.assertEqual(report.total, 3)
            self.assertEqual(report.correct, 2)
            self.assertEqual(report.missing, 1)
            self.assertAlmostEqual(report.accuracy, 2 / 3)
            self.assertEqual(report.by_category["math"]["missing"], 1)
            self.assertEqual(report.unexpected_prediction_ids, ["extra"])

    def test_manifest_binds_case_count_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = self._write_cases(root)
            manifest = build_benchmark_manifest(
                cases_path,
                name="tiny",
                version="1.0.0",
                license_name="CC0-1.0",
                task_description="Tiny deterministic benchmark.",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
            report = validate_benchmark(cases_path, manifest_path)
            self.assertTrue(report["valid"])
            self.assertEqual(report["case_count"], 3)
            self.assertEqual(report["manifest"]["metric"], "exact_match")
            self.assertEqual(len(report["dataset_sha256"]), 64)

    def test_manifest_rejects_changed_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = self._write_cases(root)
            manifest = build_benchmark_manifest(
                cases_path,
                name="tiny",
                version="1.0.0",
                license_name="CC0-1.0",
                task_description="Tiny deterministic benchmark.",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
            cases_path.write_text(
                cases_path.read_text(encoding="utf-8")
                + json.dumps({"id": "new", "category": "math", "answer": "9"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "case_count|dataset_sha256"):
                validate_benchmark(cases_path, manifest_path)

    def test_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps({"id": "dup", "category": "a", "answer": "1"}) + "\n" +
                json.dumps({"id": "dup", "category": "b", "answer": "2"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate benchmark case id"):
                load_cases(path)

    def test_rejects_duplicate_prediction_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                json.dumps({"id": "dup", "prediction": "1"}) + "\n" +
                json.dumps({"id": "dup", "prediction": "2"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate prediction id"):
                load_predictions(path)

    def test_rejects_malformed_and_empty_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "malformed.jsonl"
            malformed.write_text('{"id":\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_cases(malformed)

            empty = root / "empty.jsonl"
            empty.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "benchmark contains no cases"):
                load_cases(empty)

    def test_cli_validate_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = self._write_cases(root)
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                json.dumps({"id": "p1", "prediction": "42"}) + "\n",
                encoding="utf-8",
            )
            output = root / "score.json"
            self.assertEqual(main(["benchmark", "validate", "--cases", str(cases_path)]), 0)
            self.assertEqual(
                main([
                    "benchmark",
                    "score",
                    "--cases",
                    str(cases_path),
                    "--predictions",
                    str(predictions_path),
                    "--output",
                    str(output),
                ]),
                0,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["total"], 3)
            self.assertEqual(report["correct"], 1)
            self.assertEqual(report["missing"], 2)


if __name__ == "__main__":
    unittest.main()
