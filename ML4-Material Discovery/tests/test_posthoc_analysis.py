import numpy as np
import pytest

from scripts.analyze_completed_run import _bootstrap_means, _hypergeometric_tail


def test_hypergeometric_tail_matches_hand_calculation() -> None:
    probability = _hypergeometric_tail(
        population=10, successes=2, draws=2, observed=1
    )
    assert probability == pytest.approx(17 / 45)


def test_bootstrap_means_are_seeded_and_bounded() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    first = _bootstrap_means(values, repetitions=20, seed=7, chunk_size=4)
    second = _bootstrap_means(values, repetitions=20, seed=7, chunk_size=9)
    assert np.array_equal(first, second)
    assert np.all((first >= 1.0) & (first <= 3.0))
