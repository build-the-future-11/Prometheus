import math
import unittest

from piml.pde import (
    apply_operator,
    generate_case,
    norm,
    operator_matrix,
    residual,
    tridiagonal_solve,
)


class PDETests(unittest.TestCase):
    def test_tridiagonal_known_system(self) -> None:
        result = tridiagonal_solve([-1.0, -1.0], [2.0, 2.0, 2.0], [-1.0, -1.0], [1.0, 0.0, 1.0])
        self.assertEqual(len(result), 3)
        for value in result:
            self.assertAlmostEqual(value, 1.0, places=12)

    def test_generated_solution_satisfies_discrete_equation(self) -> None:
        case = generate_case("test", "unit", 17, 33)
        relative = norm(residual(case.diffusivity, case.forcing, case.solution)) / norm(case.forcing[1:-1])
        self.assertLess(relative, 1e-12)
        self.assertEqual(case.solution[0], 0.0)
        self.assertEqual(case.solution[-1], 0.0)

    def test_operator_is_symmetric_positive_definite(self) -> None:
        case = generate_case("test", "unit", 21, 17, coefficient_scale=0.9)
        lower, diagonal, upper = operator_matrix(case.diffusivity)
        self.assertEqual(lower, upper)
        vector = tuple([0.0] + [math.sin(i) for i in range(1, 16)] + [0.0])
        action = apply_operator(case.diffusivity, vector)
        quadratic = sum(x * y for x, y in zip(vector[1:-1], action))
        self.assertGreater(quadratic, 0.0)
        self.assertTrue(all(value > 0.0 for value in diagonal))

    def test_invalid_dimensions_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            tridiagonal_solve([], [], [], [])
        with self.assertRaises(ValueError):
            apply_operator((1.0, 1.0, 1.0), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

