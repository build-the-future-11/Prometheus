import math
import unittest

from atlas.models import Criterion


class CriterionValidationTests(unittest.TestCase):
    def test_rejects_non_finite_thresholds(self):
        for threshold in (math.nan, math.inf, -math.inf):
            with self.subTest(threshold=threshold):
                criterion = Criterion(
                    metric="primary_metric",
                    operator=">=",
                    threshold=threshold,
                    rationale="Frozen primary success threshold.",
                )
                with self.assertRaisesRegex(ValueError, "threshold must be finite"):
                    criterion.validate()

    def test_accepts_finite_threshold(self):
        Criterion(
            metric="primary_metric",
            operator=">=",
            threshold=0.0,
            rationale="Frozen primary success threshold.",
        ).validate()


if __name__ == "__main__":
    unittest.main()
