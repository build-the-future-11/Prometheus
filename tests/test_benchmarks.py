import json
import tempfile
import unittest
from pathlib import Path

from atlas.benchmarks import load_cases, load_predictions, score_exact_match


class BenchmarkTests(unittest.TestCase):
    def test_scores_exact_match_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases_path = root / "cases.jsonl"
            predictions_path = root / "predictions.jsonl"
            cases_path.write_text(
                "\n".join([
                    json.dumps({"id": "p1", "category": "physics", "answer": "42"}),
                    json.dumps({"id": "m1", "category": "math", "answer": "x = 2"}),
                    json.dumps({"id": "m2", "category": "math", "answer": "7"}),
                ]) + "\n",
                encoding="utf-8",
            )
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


if __name__ == "__main__":
    unittest.main()
