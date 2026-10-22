import unittest

from derived_learning.tasks import TaskFamily, generate_task


class TaskGenerationTests(unittest.TestCase):
    def setUp(self):
        self.family = TaskFamily(
            name="test",
            dimension=6,
            group_size=2,
            train_examples=12,
            evaluation_examples=10,
            feature_correlation=0.2,
            coefficient_scale=1.0,
            noise_std=0.0,
        )

    def test_deterministic_generation(self):
        first = generate_task("meta_train", self.family, 7)
        second = generate_task("meta_train", self.family, 7)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_split_and_streams_are_independent(self):
        meta = generate_task("meta_train", self.family, 7)
        final = generate_task("final", self.family, 7)
        self.assertNotEqual(meta.support.features, final.support.features)
        self.assertNotEqual(meta.support.features, meta.evaluation.features)

    def test_hidden_coefficients_are_not_serialized(self):
        task = generate_task("final", self.family, 4)
        serialized = task.to_dict()
        self.assertNotIn("coefficients", serialized)
        self.assertFalse(serialized["provenance"]["coefficient_labels_exposed_to_learner"])
        self.assertFalse(serialized["provenance"]["evaluation_targets_exposed_during_fit"])

    def test_groups_partition_features(self):
        task = generate_task("final", self.family, 4)
        self.assertEqual([index for group in task.groups for index in group], list(range(6)))

    def test_structural_extensions_are_deterministic_and_do_not_expose_truth(self):
        shifted = TaskFamily(
            name="shifted",
            dimension=8,
            group_size=2,
            train_examples=12,
            evaluation_examples=16,
            feature_correlation=0.4,
            coefficient_scale=1.0,
            noise_std=0.1,
            interaction_strength=0.7,
            nonlinear_kind="quadratic",
            supplied_partition="interleaved",
        )
        first = generate_task("final", shifted, 301)
        second = generate_task("final", shifted, 301)
        self.assertEqual(first, second)
        self.assertEqual(sorted(index for group in first.groups for index in group), list(range(8)))
        self.assertNotEqual(first.groups, [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertFalse(first.provenance["true_partition_exposed_to_learner"])
        self.assertNotIn("coefficients", str(first.to_dict()))


if __name__ == "__main__":
    unittest.main()

