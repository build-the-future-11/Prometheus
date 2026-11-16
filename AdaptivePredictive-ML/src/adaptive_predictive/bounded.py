"""Opt-in bounded-state execution; the frozen forecasting implementation is unchanged.

Only suffixes consumed by model.py are retained. Seasonal phase is maintained
separately, because len(history) ceases to represent elapsed time after eviction.
No claim of improved predictive accuracy is made.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from operator import index
from typing import overload

import numpy as np

from .model import DDFSConfig, DriftDirectedFixedShare, Mode


class FloatWindow(Sequence[float]):
    """Fixed-capacity ring supporting chronological indexing and list slices."""

    __slots__ = ("capacity", "_values", "_start", "_size")

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._values = [0.0] * capacity
        self._start = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, key: int) -> float: ...

    @overload
    def __getitem__(self, key: slice) -> list[float]: ...

    def __getitem__(self, key: int | slice) -> float | list[float]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(self._size))]
        position = index(key)
        if position < 0:
            position += self._size
        if position < 0 or position >= self._size:
            raise IndexError("window index out of range")
        return self._values[(self._start + position) % self.capacity]

    def __iter__(self) -> Iterator[float]:
        for i in range(self._size):
            yield self._values[(self._start + i) % self.capacity]

    def append(self, value: float) -> None:
        value = float(value)
        if self._size < self.capacity:
            self._values[(self._start + self._size) % self.capacity] = value
            self._size += 1
        else:
            self._values[self._start] = value
            self._start = (self._start + 1) % self.capacity


class BoundedDriftDirectedFixedShare(DriftDirectedFixedShare):
    """Same forecast/update recurrence, with bounded observation/error buffers.

    Use observe_initial/predict/update; direct mutation of internal buffers is not
    part of the API. Existing entry points still use the preserved legacy class.
    Select this class explicitly in a newly recorded execution configuration.
    """

    def __init__(self, *, season: int, config: DDFSConfig | None = None,
                 mode: Mode = "ddfs") -> None:
        config = config or DDFSConfig()
        if isinstance(season, bool) or not isinstance(season, int) or season < 1:
            raise ValueError("season must be a positive integer")
        if not config.lags or any(isinstance(lag, bool) or not isinstance(lag, int)
                                  for lag in config.lags):
            raise ValueError("lags must be nonempty integer offsets")
        window = config.interval_window
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("interval_window must be a positive integer")
        super().__init__(season=season, config=config, mode=mode)
        self.history_capacity = max(self.minimum_history, max(2 * season, 32) + 1)
        self.history = FloatWindow(self.history_capacity)
        # The legacy interval fallback counts eight past errors even when w < 8.
        self.ensemble_errors = FloatWindow(max(8, window))
        # Legacy normalized_residuals are diagnostics; no prediction reads them.
        self.normalized_residuals = FloatWindow(window)
        self._phase = 0

    def observe_initial(self, value: float) -> None:
        super().observe_initial(value)
        self._phase = (self._phase + 1) % self.season

    def update(self, outcome: float) -> None:
        super().update(outcome)
        self._phase = (self._phase + 1) % self.season

    def _features(self) -> np.ndarray:
        features = super()._features()
        angle = 2.0 * np.pi * self._phase / self.season
        features[-2] = float(np.sin(angle))
        features[-1] = float(np.cos(angle))
        return features

    @property
    def retained_sample_capacity(self) -> int:
        return self.history_capacity + max(8, self.config.interval_window) + self.config.interval_window
