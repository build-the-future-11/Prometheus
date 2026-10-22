"""Dependency-free paired uncertainty calculations."""

from __future__ import annotations

from math import sqrt
from random import Random


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires observations")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires observations")
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid quantile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(ordered):
        return ordered[-1]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def bootstrap_mean_ci(
    values: list[float], seed: int, repeats: int = 5000, confidence: float = 0.95
) -> tuple[float, float, float]:
    if not values or repeats < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap request")
    rng = Random(seed)
    estimates = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(repeats)]
    tail = (1.0 - confidence) / 2.0
    return mean(values), quantile(estimates, tail), quantile(estimates, 1.0 - tail)

