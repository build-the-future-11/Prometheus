"""Finite-volume discretization of a one-dimensional elliptic PDE.

We solve ``-(a(x) u'(x))' = f(x)`` on ``[0, 1]`` with homogeneous
Dirichlet boundaries.  The discretized interior operator is symmetric positive
definite whenever the sampled diffusivity is positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sin, sqrt
from random import Random
from typing import Iterable


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    generator_seed: int
    coefficient_scale: float
    source_scale: float
    coefficients: tuple[float, ...]
    source_coefficients: tuple[float, ...]
    grid_points: int
    diffusivity: tuple[float, ...]
    forcing: tuple[float, ...]
    solution: tuple[float, ...]

    @property
    def features(self) -> tuple[float, ...]:
        return self.coefficients + self.source_coefficients


def _field(coefficients: Iterable[float], x: float, exponential: bool) -> float:
    value = sum(c * sin((i + 1) * pi * x) for i, c in enumerate(coefficients))
    return exp(value) if exponential else value


def tridiagonal_solve(
    lower: list[float], diagonal: list[float], upper: list[float], rhs: list[float]
) -> list[float]:
    """Solve a real tridiagonal system using the Thomas algorithm."""
    n = len(diagonal)
    if n == 0 or len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("invalid tridiagonal system dimensions")
    d, b, c = diagonal[:], rhs[:], upper[:]
    for row in range(1, n):
        if abs(d[row - 1]) < 1e-14:
            raise ArithmeticError("near-zero tridiagonal pivot")
        factor = lower[row - 1] / d[row - 1]
        d[row] -= factor * c[row - 1]
        b[row] -= factor * b[row - 1]
    if abs(d[-1]) < 1e-14:
        raise ArithmeticError("near-zero tridiagonal pivot")
    result = [0.0] * n
    result[-1] = b[-1] / d[-1]
    for row in range(n - 2, -1, -1):
        result[row] = (b[row] - c[row] * result[row + 1]) / d[row]
    return result


def operator_matrix(
    diffusivity: tuple[float, ...],
) -> tuple[list[float], list[float], list[float]]:
    """Return lower, diagonal, upper vectors for the interior SPD operator."""
    if len(diffusivity) < 3 or any(a <= 0.0 for a in diffusivity):
        raise ValueError("diffusivity must contain at least three positive values")
    scale = float((len(diffusivity) - 1) ** 2)
    faces = [0.5 * (left + right) * scale for left, right in zip(diffusivity, diffusivity[1:])]
    diagonal = [faces[i] + faces[i + 1] for i in range(len(faces) - 1)]
    off_diagonal = [-faces[i] for i in range(1, len(faces) - 1)]
    return off_diagonal[:], diagonal, off_diagonal[:]


def apply_operator(diffusivity: tuple[float, ...], solution: tuple[float, ...]) -> tuple[float, ...]:
    if len(diffusivity) != len(solution):
        raise ValueError("diffusivity and solution sizes differ")
    lower, diagonal, upper = operator_matrix(diffusivity)
    interior = solution[1:-1]
    values: list[float] = []
    for i, center in enumerate(interior):
        value = diagonal[i] * center
        if i:
            value += lower[i - 1] * interior[i - 1]
        if i + 1 < len(interior):
            value += upper[i] * interior[i + 1]
        values.append(value)
    return tuple(values)


def solve_case(diffusivity: tuple[float, ...], forcing: tuple[float, ...]) -> tuple[float, ...]:
    if len(diffusivity) != len(forcing):
        raise ValueError("diffusivity and forcing sizes differ")
    lower, diagonal, upper = operator_matrix(diffusivity)
    interior = tridiagonal_solve(lower, diagonal, upper, list(forcing[1:-1]))
    return (0.0, *interior, 0.0)


def residual(
    diffusivity: tuple[float, ...], forcing: tuple[float, ...], solution: tuple[float, ...]
) -> tuple[float, ...]:
    applied = apply_operator(diffusivity, solution)
    return tuple(rhs - lhs for rhs, lhs in zip(forcing[1:-1], applied))


def norm(values: Iterable[float]) -> float:
    return sqrt(sum(value * value for value in values))


def energy_norm(case: Case, error: tuple[float, ...]) -> float:
    applied = apply_operator(case.diffusivity, error)
    quadratic = sum(x * y for x, y in zip(error[1:-1], applied))
    return sqrt(max(0.0, quadratic))


def generate_case(
    case_id: str,
    split: str,
    seed: int,
    grid_points: int,
    coefficient_scale: float = 0.45,
    source_scale: float = 1.0,
    modes: int = 4,
) -> Case:
    if grid_points < 5:
        raise ValueError("grid_points must be at least five")
    rng = Random(seed)
    coefficients = tuple(rng.uniform(-coefficient_scale, coefficient_scale) for _ in range(modes))
    source_coefficients = tuple(rng.uniform(-source_scale, source_scale) for _ in range(modes))
    xs = [i / (grid_points - 1) for i in range(grid_points)]
    diffusivity = tuple(_field(coefficients, x, True) for x in xs)
    forcing = tuple(_field(source_coefficients, x, False) for x in xs)
    solution = solve_case(diffusivity, forcing)
    return Case(
        case_id=case_id,
        split=split,
        generator_seed=seed,
        coefficient_scale=coefficient_scale,
        source_scale=source_scale,
        coefficients=coefficients,
        source_coefficients=source_coefficients,
        grid_points=grid_points,
        diffusivity=diffusivity,
        forcing=forcing,
        solution=solution,
    )


def generate_split(
    prefix: str,
    seed: int,
    count: int,
    grid_points: int,
    coefficient_scale: float = 0.45,
    source_scale: float = 1.0,
    modes: int = 4,
) -> list[Case]:
    """Generate independent cases; split seeds occupy disjoint integer ranges."""
    return [
        generate_case(
            f"{prefix}-{index:05d}", prefix, seed + index, grid_points,
            coefficient_scale, source_scale, modes,
        )
        for index in range(count)
    ]

