"""Composition-only features available before any candidate measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np

from .chemistry import ATOMIC_NUMBER, ELEMENTS, element_fractions


GLOBAL_FEATURES = (
    "n_elements",
    "fraction_entropy",
    "fraction_min",
    "fraction_max",
    "fraction_l2",
    "atomic_number_mean",
    "atomic_number_std",
    "atomic_number_min",
    "atomic_number_max",
    "atomic_number_range",
)


@dataclass
class CompositionFeaturizer:
    """Fixed-schema featurizer with train-only element-frequency state."""

    train_element_counts: dict[str, int] = field(default_factory=dict)
    fitted: bool = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"fraction_{element}" for element in ELEMENTS) + GLOBAL_FEATURES

    def fit(self, formulas: list[str] | tuple[str, ...]) -> "CompositionFeaturizer":
        counts = {element: 0 for element in ELEMENTS}
        for formula in formulas:
            for element in element_fractions(formula):
                counts[element] += 1
        self.train_element_counts = counts
        self.fitted = True
        return self

    def transform(self, formulas: list[str] | tuple[str, ...]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CompositionFeaturizer must be fitted on training data")
        matrix = np.zeros((len(formulas), len(self.feature_names)), dtype=np.float32)
        for row_index, formula in enumerate(formulas):
            fractions = element_fractions(formula)
            for element, fraction in fractions.items():
                matrix[row_index, ATOMIC_NUMBER[element] - 1] = fraction
            values = np.asarray(list(fractions.values()), dtype=np.float64)
            atomic_numbers = np.asarray(
                [ATOMIC_NUMBER[element] for element in fractions], dtype=np.float64
            )
            weights = values / values.sum()
            mean_atomic_number = float(np.sum(weights * atomic_numbers))
            global_values = (
                float(len(fractions)),
                float(-sum(value * log(value) for value in values)),
                float(values.min()),
                float(values.max()),
                float(np.sqrt(np.sum(values**2))),
                mean_atomic_number,
                float(np.sqrt(np.sum(weights * (atomic_numbers - mean_atomic_number) ** 2))),
                float(atomic_numbers.min()),
                float(atomic_numbers.max()),
                float(atomic_numbers.max() - atomic_numbers.min()),
            )
            matrix[row_index, len(ELEMENTS) :] = global_values
        return matrix

    def unseen_elements(self, formula: str) -> tuple[str, ...]:
        if not self.fitted:
            raise RuntimeError("CompositionFeaturizer must be fitted on training data")
        return tuple(
            sorted(
                (
                    element
                    for element in element_fractions(formula)
                    if self.train_element_counts.get(element, 0) == 0
                ),
                key=ATOMIC_NUMBER.__getitem__,
            )
        )

    def rare_element_fraction(self, formula: str, rare_threshold: int = 20) -> float:
        fractions = element_fractions(formula)
        return float(
            sum(
                fraction
                for element, fraction in fractions.items()
                if self.train_element_counts.get(element, 0) < rare_threshold
            )
        )

