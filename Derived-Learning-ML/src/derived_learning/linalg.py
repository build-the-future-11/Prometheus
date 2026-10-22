"""Small, dependency-free linear algebra helpers used by the experiment."""

from __future__ import annotations

import math
from collections.abc import Sequence


Vector = list[float]
Matrix = list[Vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"dot shape mismatch: {len(left)} != {len(right)}")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def matvec(rows: Sequence[Sequence[float]], weights: Sequence[float]) -> Vector:
    return [dot(row, weights) for row in rows]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(values) / len(values)


def mse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if len(predictions) != len(targets) or not targets:
        raise ValueError("mse requires equal, non-empty vectors")
    return math.fsum((prediction - target) ** 2 for prediction, target in zip(predictions, targets, strict=True)) / len(targets)


def variance(values: Sequence[float]) -> float:
    center = mean(values)
    return math.fsum((value - center) ** 2 for value in values) / len(values)


def normalized_mse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    denominator = max(variance(targets), 1e-12)
    return mse(predictions, targets) / denominator


def solve_linear(matrix: Sequence[Sequence[float]], vector: Sequence[float], *, pivot_tolerance: float = 1e-12) -> Vector:
    """Solve a dense square system using partial-pivot Gaussian elimination."""

    size = len(matrix)
    if size == 0 or len(vector) != size or any(len(row) != size for row in matrix):
        raise ValueError("solve_linear requires a non-empty square matrix and matching vector")
    augmented = [[float(value) for value in row] + [float(vector[index])] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= pivot_tolerance:
            raise ValueError("singular or numerically unstable linear system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]
    result = [augmented[row][size] for row in range(size)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("linear solve produced a non-finite result")
    return result


def ridge_solution(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    regularization: float,
) -> Vector:
    if not rows or len(rows) != len(targets):
        raise ValueError("ridge_solution requires equal, non-empty rows and targets")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("ridge_solution rows must have a stable non-zero width")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for row, target in zip(rows, targets, strict=True):
        if not math.isfinite(target) or not all(math.isfinite(value) for value in row):
            raise ValueError("ridge_solution rejects non-finite inputs")
        for left in range(width):
            rhs[left] += row[left] * target
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    for index in range(width):
        gram[index][index] += regularization
    return solve_linear(gram, rhs)


def ridge_solution_with_intercept(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    regularization: float,
) -> tuple[Vector, float]:
    """Solve full-feature ridge with an unpenalized affine intercept."""

    if not rows or len(rows) != len(targets):
        raise ValueError("ridge_solution_with_intercept requires equal, non-empty rows and targets")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("ridge_solution_with_intercept rows must have stable non-zero width")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    augmented_rows = [list(row) + [1.0] for row in rows]
    gram = [[0.0 for _ in range(width + 1)] for _ in range(width + 1)]
    rhs = [0.0 for _ in range(width + 1)]
    for row, target in zip(augmented_rows, targets, strict=True):
        if not math.isfinite(target) or not all(math.isfinite(value) for value in row):
            raise ValueError("ridge_solution_with_intercept rejects non-finite inputs")
        for left in range(width + 1):
            rhs[left] += row[left] * target
            for right in range(width + 1):
                gram[left][right] += row[left] * row[right]
    for index in range(width):
        gram[index][index] += regularization
    solution = solve_linear(gram, rhs)
    return solution[:-1], solution[-1]

