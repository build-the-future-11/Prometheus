"""Engineering equivalence tests; no external or protected outcomes are read."""
import pickle
from dataclasses import replace

import numpy as np
import pytest

from adaptive_predictive.bounded import BoundedDriftDirectedFixedShare, FloatWindow
from adaptive_predictive.model import DDFSConfig, DriftDirectedFixedShare

MODES = ("uniform", "hedge", "adahedge", "fixed_share", "adaptive_share", "directed_fixed_share", "ddfs")


def equal_state(a, b):
    for name in ("weights", "expert_fast_loss", "expert_slow_loss", "cumulative_expert_losses"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    for name in ("fast_loss", "slow_loss", "mixability_gap", "_awaiting_update"):
        assert getattr(a, name) == getattr(b, name)
    for x, y in zip(a.experts, b.experts):
        assert np.array_equal(x.theta, y.theta)
        assert np.array_equal(x.covariance, y.covariance)
        assert x.reset_count == y.reset_count


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("season", (1, 7, 24, 168))
@pytest.mark.parametrize("window", (1, 8, 256))
def test_all_modes_are_exact_after_eviction(mode, season, window):
    config = replace(DDFSConfig(), interval_window=window)
    a = DriftDirectedFixedShare(season=season, mode=mode, config=config)
    b = BoundedDriftDirectedFixedShare(season=season, mode=mode, config=config)
    rng = np.random.default_rng(20261022)
    stream = rng.normal(size=620) + np.sin(np.arange(620) / 5)
    stream[300:] += 8
    for value in stream:
        assert a.ready() == b.ready()
        if a.ready():
            assert a.predict() == b.predict()
            assert np.array_equal(a._last_features, b._last_features)
            a.update(value); b.update(value)
            equal_state(a, b)
        else:
            a.observe_initial(value); b.observe_initial(value)
    assert list(b.history) == a.history[-b.history_capacity:]
    assert len(b.history) <= b.history_capacity
    assert len(b.ensemble_errors) <= max(8, window)
    assert len(b.normalized_residuals) <= window


@pytest.mark.parametrize("capacity", (1, 2, 17))
def test_ring_index_and_slice_semantics(capacity):
    ring = FloatWindow(capacity); reference = []
    for i in range(80):
        ring.append(i); reference = (reference + [float(i)])[-capacity:]
        assert list(ring) == reference
        for key in (slice(None), slice(None, None, -1), slice(-12, None, 2), slice(1, 9), slice(10, 2, -2)):
            assert ring[key] == reference[key]
        for j in range(-len(reference), len(reference)):
            assert ring[j] == reference[j]
    with pytest.raises(IndexError): _ = ring[len(ring)]
    with pytest.raises(IndexError): _ = ring[-len(ring)-1]


@pytest.mark.parametrize("window", (1, 8, 256))
def test_overlong_initialization_constant_stream_and_pickle_resume(window):
    cfg = replace(DDFSConfig(), interval_window=window, lags=(1, 12, 73))
    a = DriftDirectedFixedShare(season=7, config=cfg)
    b = BoundedDriftDirectedFixedShare(season=7, config=cfg)
    for i in range(300):
        a.observe_initial(float(i % 4)); b.observe_initial(float(i % 4))
    for i in range(80):
        assert a.predict() == b.predict()
        if i == 20: b = pickle.loads(pickle.dumps(b))
        a.update(5.0); b.update(5.0); equal_state(a, b)


@pytest.mark.parametrize("window", (0, -1, True, 1.5))
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError):
        BoundedDriftDirectedFixedShare(season=7, config=replace(DDFSConfig(), interval_window=window))


@pytest.mark.parametrize("season", (0, -1, True, 1.5))
def test_invalid_season_rejected(season):
    with pytest.raises(ValueError): BoundedDriftDirectedFixedShare(season=season)


def test_failed_calls_do_not_advance_phase():
    m = BoundedDriftDirectedFixedShare(season=7)
    with pytest.raises(RuntimeError): m.update(1.0)
    with pytest.raises(ValueError): m.observe_initial(float("nan"))
    assert m._phase == 0
    for i in range(m.minimum_history): m.observe_initial(float(i))
    m.predict(); phase = m._phase
    with pytest.raises(RuntimeError): m.observe_initial(2.0)
    with pytest.raises(RuntimeError): m.predict()
    with pytest.raises(ValueError): m.update(float("inf"))
    assert m._phase == phase
    m.update(2.0)
    assert m._phase == (phase + 1) % 7
