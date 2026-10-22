"""Registered paired uncertainty calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    probability_improvement: float
    repetitions: int
    block_length: int
    seed: int
    resampling_scheme: str


def paired_block_bootstrap(
    differences: Mapping[str, Sequence[float]],
    *,
    block_length: int,
    repetitions: int,
    seed: int,
    synchronized: bool = False,
) -> BootstrapResult:
    """Equal-series-weight moving-block bootstrap of paired loss differences.

    ``synchronized=True`` reuses block starts across aligned series, preserving
    cross-series dependence from shared time shocks. Independent per-series blocks
    remain available only for non-aligned diagnostic uses.
    """
    if block_length <= 0 or repetitions <= 0 or not differences:
        raise ValueError("positive block length/repetitions and nonempty data are required")
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in differences.items()}
    if any(len(values) < block_length for values in arrays.values()):
        raise ValueError("every series must contain at least one complete block")
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("differences must be finite")
    lengths = {len(values) for values in arrays.values()}
    if synchronized and len(lengths) != 1:
        raise ValueError("synchronized bootstrap requires equal-length aligned series")
    estimate = float(np.mean([float(values.mean()) for values in arrays.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        series_means: list[float] = []
        shared_starts = None
        if synchronized:
            needed = next(iter(lengths))
            shared_starts = rng.integers(
                0,
                needed - block_length + 1,
                size=(needed + block_length - 1) // block_length,
            )
        for values in arrays.values():
            needed = len(values)
            starts = shared_starts
            if starts is None:
                starts = rng.integers(
                    0,
                    len(values) - block_length + 1,
                    size=(needed + block_length - 1) // block_length,
                )
            sample = np.concatenate([values[start : start + block_length] for start in starts])[:needed]
            series_means.append(float(sample.mean()))
        samples[repetition] = float(np.mean(series_means))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return BootstrapResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        probability_improvement=float(np.mean(samples < 0.0)),
        repetitions=repetitions,
        block_length=block_length,
        seed=seed,
        resampling_scheme=(
            "synchronized-moving-block" if synchronized else "independent-series-moving-block"
        ),
    )

