import math
import unittest

from derived_learning.linalg import dot, mse, ridge_solution, ridge_solution_with_intercept, solve_linear


class LinearAlgebraTests(unittest.TestCase):
    def test_shape_checks(self):
        with self.assertRaises(ValueError):
            dot([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            mse([], [])
        with self.assertRaises(ValueError):
            solve_linear([[1.0, 2.0]], [1.0])

    def test_partial_pivot_solve(self):
        solution = solve_linear([[0.0, 2.0], [1.0, 3.0]], [4.0, 7.0])
        self.assertAlmostEqual(solution[0], 1.0)
        self.assertAlmostEqual(solution[1], 2.0)

    def test_ridge_solution_satisfies_independent_normal_equation(self):
        rows = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        targets = [1.0, 3.0, 5.0, 7.0]
        regularization = 0.25
        weights = ridge_solution(rows, targets, regularization)
        for index in range(2):
            residual = math.fsum(
                row[index] * (dot(row, weights) - target)
                for row, target in zip(rows, targets, strict=True)
            ) + regularization * weights[index]
            self.assertAlmostEqual(residual, 0.0, places=10)

    def test_ridge_stabilizes_rank_deficient_input(self):
        weights = ridge_solution([[1.0, 1.0], [2.0, 2.0]], [1.0, 2.0], 0.1)
        self.assertTrue(all(math.isfinite(value) for value in weights))

    def test_ridge_with_intercept_recovers_an_affine_relation(self):
        rows = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        targets = [3.0, 5.0, 2.0, 4.0]
        weights, intercept = ridge_solution_with_intercept(rows, targets, 1e-9)
        self.assertAlmostEqual(weights[0], 2.0, places=6)
        self.assertAlmostEqual(weights[1], -1.0, places=6)
        self.assertAlmostEqual(intercept, 3.0, places=6)


if __name__ == "__main__":
    unittest.main()

