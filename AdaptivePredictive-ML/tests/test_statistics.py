from __future__ import annotations

import pytest

from adaptive_predictive.statistics import paired_block_bootstrap


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    result = paired_block_bootstrap(
        {"a": [-0.2] * 40, "b": [-0.1] * 40},
        block_length=7,
        repetitions=500,
        seed=17,
    )
    assert result.estimate == pytest.approx(-0.15)
    assert result.ci_upper < 0
    assert result.probability_improvement == 1.0


def test_paired_bootstrap_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="complete block"):
        paired_block_bootstrap({"a": [1.0, 2.0]}, block_length=7, repetitions=10, seed=1)


def test_synchronized_bootstrap_preserves_cross_series_dependence() -> None:
    left = [-1.0, -1.0, 1.0, 1.0] * 10
    result = paired_block_bootstrap(
        {"a": left, "b": [-value for value in left]},
        block_length=2,
        repetitions=200,
        seed=23,
        synchronized=True,
    )
    assert result.estimate == pytest.approx(0.0)
    assert result.ci_lower == pytest.approx(0.0)
    assert result.ci_upper == pytest.approx(0.0)
    assert result.resampling_scheme == "synchronized-moving-block"


def test_synchronized_bootstrap_requires_aligned_series() -> None:
    with pytest.raises(ValueError, match="equal-length aligned"):
        paired_block_bootstrap(
            {"a": [0.0] * 10, "b": [0.0] * 11},
            block_length=2,
            repetitions=10,
            seed=1,
            synchronized=True,
        )

