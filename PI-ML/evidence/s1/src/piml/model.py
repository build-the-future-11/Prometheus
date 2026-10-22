"""Data-only operator and residual-certified iterative solver controls."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter_ns

from .pde import Case, apply_operator, norm, operator_matrix, residual


def feature_map(features: tuple[float, ...]) -> list[float]:
    """Quadratic feature map with an intercept and unique cross-products."""
    result = [1.0, *features]
    for i, left in enumerate(features):
        result.extend(left * right for right in features[i:])
    return result


def solve_dense(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Pivoted Gauss-Jordan solve for the small ridge normal equations."""
    n = len(matrix)
    if n == 0 or len(rhs) != n or any(len(row) != n for row in matrix):
        raise ValueError("invalid dense system dimensions")
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ArithmeticError("singular dense system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * base
                    for value, base in zip(augmented[row], augmented[column])
                ]
    return [row[-1] for row in augmented]


@dataclass
class RidgeOperator:
    weights: list[list[float]]
    ridge: float
    grid_points: int

    @classmethod
    def fit(cls, cases: list[Case], ridge: float) -> "RidgeOperator":
        if not cases:
            raise ValueError("at least one training case is required")
        grid_points = cases[0].grid_points
        if any(case.grid_points != grid_points for case in cases):
            raise ValueError("all cases must use one grid")
        design = [feature_map(case.features) for case in cases]
        width = len(design[0])
        gram = [[0.0] * width for _ in range(width)]
        for row in design:
            for i in range(width):
                for j in range(i, width):
                    gram[i][j] += row[i] * row[j]
        for i in range(width):
            for j in range(i):
                gram[i][j] = gram[j][i]
            gram[i][i] += ridge if i else ridge * 1e-6
        weights: list[list[float]] = []
        for output in range(1, grid_points - 1):
            target = [case.solution[output] for case in cases]
            cross = [sum(row[i] * y for row, y in zip(design, target)) for i in range(width)]
            weights.append(solve_dense(gram, cross))
        return cls(weights=weights, ridge=ridge, grid_points=grid_points)

    def predict(self, case: Case) -> tuple[float, ...]:
        if case.grid_points != self.grid_points:
            raise ValueError("model and case grid sizes differ")
        features = feature_map(case.features)
        interior = [sum(w * x for w, x in zip(row, features)) for row in self.weights]
        return (0.0, *interior, 0.0)


@dataclass
class EnsembleOperator:
    members: list[RidgeOperator]

    @classmethod
    def fit(
        cls, cases: list[Case], ridge: float, members: int, seed: int
    ) -> "EnsembleOperator":
        if members < 1:
            raise ValueError("members must be positive")
        rng = Random(seed)
        fitted = []
        for _ in range(members):
            sample = [cases[rng.randrange(len(cases))] for _ in cases]
            fitted.append(RidgeOperator.fit(sample, ridge))
        return cls(fitted)

    def predict(self, case: Case) -> tuple[tuple[float, ...], tuple[float, ...]]:
        predictions = [member.predict(case) for member in self.members]
        mean = tuple(sum(values) / len(values) for values in zip(*predictions))
        stddev = tuple(
            (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5
            for average, values in zip(mean, zip(*predictions))
        )
        return mean, stddev


@dataclass(frozen=True)
class SolveResult:
    solution: tuple[float, ...]
    iterations: int
    operator_applications: int
    converged: bool
    initial_relative_residual: float
    final_relative_residual: float
    elapsed_ns: int
    residual_history: tuple[float, ...]
    breakdown: str | None = None


def pcg_solve(
    case: Case,
    initial: tuple[float, ...] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iterations: int | None = None,
) -> SolveResult:
    """Diagonal-preconditioned conjugate gradients with complete diagnostics."""
    n = case.grid_points - 2
    if rtol <= 0.0 or atol < 0.0:
        raise ValueError("invalid tolerances")
    if initial is None:
        initial = (0.0,) * case.grid_points
    if len(initial) != case.grid_points:
        raise ValueError("initial vector size differs from the case")
    max_iterations = max_iterations or n
    x = list(initial[1:-1])
    rhs = list(case.forcing[1:-1])
    _, diagonal, _ = operator_matrix(case.diffusivity)
    rhs_norm = max(norm(rhs), 1e-30)
    started = perf_counter_ns()
    applied = apply_operator(case.diffusivity, (0.0, *x, 0.0))
    operator_applications = 1
    r = [b - ax for b, ax in zip(rhs, applied)]
    initial_relative = norm(r) / rhs_norm
    history = [initial_relative]
    threshold = max(atol, rtol * rhs_norm)
    if norm(r) <= threshold:
        return SolveResult((0.0, *x, 0.0), 0, operator_applications, True,
                           initial_relative, initial_relative, perf_counter_ns() - started,
                           tuple(history))
    z = [value / diagonal[i] for i, value in enumerate(r)]
    p = z[:]
    rz = sum(left * right for left, right in zip(r, z))
    breakdown: str | None = None
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        ap = list(apply_operator(case.diffusivity, (0.0, *p, 0.0)))
        operator_applications += 1
        denominator = sum(left * right for left, right in zip(p, ap))
        if denominator <= 0.0 or not (denominator < float("inf")):
            breakdown = "non-positive-or-non-finite-curvature"
            break
        alpha = rz / denominator
        x = [value + alpha * direction for value, direction in zip(x, p)]
        r = [value - alpha * action for value, action in zip(r, ap)]
        relative = norm(r) / rhs_norm
        history.append(relative)
        if norm(r) <= threshold:
            return SolveResult((0.0, *x, 0.0), iteration, operator_applications, True,
                               initial_relative, relative, perf_counter_ns() - started,
                               tuple(history))
        z = [value / diagonal[i] for i, value in enumerate(r)]
        next_rz = sum(left * right for left, right in zip(r, z))
        if rz <= 0.0:
            breakdown = "non-positive-preconditioned-residual"
            break
        beta = next_rz / rz
        p = [preconditioned + beta * direction for preconditioned, direction in zip(z, p)]
        rz = next_rz
    final_relative = norm(r) / rhs_norm
    return SolveResult((0.0, *x, 0.0), iteration, operator_applications, False,
                       initial_relative, final_relative, perf_counter_ns() - started,
                       tuple(history), breakdown or "iteration-limit")


def jacobi_correct(
    case: Case, initial: tuple[float, ...], steps: int, relaxation: float
) -> tuple[float, ...]:
    if steps < 0 or not 0.0 < relaxation < 1.0:
        raise ValueError("invalid Jacobi parameters")
    current = list(initial)
    _, diagonal, _ = operator_matrix(case.diffusivity)
    for _ in range(steps):
        defect = residual(case.diffusivity, case.forcing, tuple(current))
        for i, value in enumerate(defect, start=1):
            current[i] += relaxation * value / diagonal[i - 1]
    current[0] = current[-1] = 0.0
    return tuple(current)

