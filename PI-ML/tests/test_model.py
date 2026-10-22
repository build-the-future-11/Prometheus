import unittest

from piml.model import EnsembleOperator, RidgeOperator, feature_map, jacobi_correct, pcg_solve, solve_dense
from piml.pde import energy_norm, generate_case, generate_split, norm, residual


class ModelTests(unittest.TestCase):
    def test_quadratic_feature_dimension(self) -> None:
        self.assertEqual(len(feature_map((1.0, 2.0, 3.0))), 10)

    def test_dense_solver_pivots(self) -> None:
        result = solve_dense([[0.0, 2.0], [1.0, 1.0]], [4.0, 3.0])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 2.0)

    def test_ridge_respects_boundaries_and_grid_contract(self) -> None:
        train = generate_split("train", 100, 32, 17)
        model = RidgeOperator.fit(train, 1e-4)
        prediction = model.predict(generate_case("held", "test", 999, 17))
        self.assertEqual(prediction[0], 0.0)
        self.assertEqual(prediction[-1], 0.0)
        with self.assertRaises(ValueError):
            model.predict(generate_case("wrong", "test", 1000, 19))

    def test_ensemble_is_deterministic(self) -> None:
        train = generate_split("train", 200, 24, 13)
        case = generate_case("held", "test", 9999, 13)
        first = EnsembleOperator.fit(train, 1e-4, 3, 77).predict(case)
        second = EnsembleOperator.fit(train, 1e-4, 3, 77).predict(case)
        self.assertEqual(first, second)

    def test_pcg_converges_from_zero_and_learned_initialization(self) -> None:
        train = generate_split("train", 500, 48, 33)
        case = generate_case("held", "test", 800, 33)
        prediction = RidgeOperator.fit(train, 1e-4).predict(case)
        zero = pcg_solve(case, rtol=1e-10, max_iterations=31)
        warm = pcg_solve(case, prediction, rtol=1e-10, max_iterations=31)
        self.assertTrue(zero.converged)
        self.assertTrue(warm.converged)
        self.assertLess(zero.final_relative_residual, 1e-10)
        self.assertLess(warm.final_relative_residual, 1e-10)
        error = tuple(value - truth for value, truth in zip(warm.solution, case.solution))
        self.assertLess(energy_norm(case, error), 1e-8)

    def test_jacobi_reduces_residual_on_valid_case(self) -> None:
        case = generate_case("held", "test", 801, 33)
        initial = (0.0,) * 33
        corrected = jacobi_correct(case, initial, 4, 0.8)
        initial_norm = norm(residual(case.diffusivity, case.forcing, initial))
        corrected_norm = norm(residual(case.diffusivity, case.forcing, corrected))
        self.assertLess(corrected_norm, initial_norm)


if __name__ == "__main__":
    unittest.main()

