"""Calibration partitions, normalized split conformal intervals, and ranking probability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from math import erf, sqrt
from typing import Iterable

import numpy as np


def calibration_partition(
    identifiers: Iterable[str],
    fraction: float,
    seed: int,
    group_by_identifier: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split without labels using stable identifier hashes."""

    if not 0.0 < fraction < 0.5:
        raise ValueError("calibration fraction must lie strictly between 0 and 0.5")
    identifier_list = list(identifiers)
    group_by_identifier = group_by_identifier or {
        identifier: identifier for identifier in identifier_list
    }
    grouped: dict[str, list[str]] = {}
    for identifier in identifier_list:
        grouped.setdefault(group_by_identifier[identifier], []).append(identifier)
    scored_groups = sorted(
        (
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
            group,
        )
        for group in grouped
    )
    target_calibration_size = max(1, int(round(len(identifier_list) * fraction)))
    calibration_groups: set[str] = set()
    calibration_size = 0
    for _, group in scored_groups:
        if calibration_size >= target_calibration_size:
            break
        calibration_groups.add(group)
        calibration_size += len(grouped[group])
    calibration = [
        identifier
        for identifier in identifier_list
        if group_by_identifier[identifier] in calibration_groups
    ]
    proper_train = [
        identifier
        for identifier in identifier_list
        if group_by_identifier[identifier] not in calibration_groups
    ]
    if set(proper_train) & set(calibration):
        raise AssertionError("calibration split is not disjoint")
    return proper_train, calibration


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    if values.ndim != 1 or not len(values):
        raise ValueError("quantile values must be a non-empty vector")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between zero and one")
    rank = min(len(values) - 1, max(0, int(np.ceil(probability * len(values))) - 1))
    return float(np.partition(values, rank)[rank])


@dataclass(frozen=True)
class NormalizedConformalCalibrator:
    sigma_floor: float
    sigma_scale: float
    normalized_scores: tuple[float, ...]
    calibration_size: int

    @classmethod
    def fit(
        cls,
        targets: np.ndarray,
        predictions: np.ndarray,
        raw_sigma: np.ndarray,
    ) -> "NormalizedConformalCalibrator":
        if not (len(targets) == len(predictions) == len(raw_sigma)):
            raise ValueError("calibration arrays must have equal length")
        positive = raw_sigma[raw_sigma > 0]
        sigma_floor = float(np.quantile(positive, 0.05)) if len(positive) else 0.05
        sigma_floor = max(sigma_floor, 1e-6)
        floored = np.maximum(raw_sigma, sigma_floor)
        ratios = np.abs(targets - predictions) / floored
        robust_scale = float(np.median(ratios) / 0.6744897501960817)
        sigma_scale = max(robust_scale, 1e-6)
        normalized = np.abs(targets - predictions) / (floored * sigma_scale)
        return cls(
            sigma_floor=sigma_floor,
            sigma_scale=sigma_scale,
            normalized_scores=tuple(float(value) for value in normalized),
            calibration_size=len(targets),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NormalizedConformalCalibrator":
        return cls(
            sigma_floor=float(payload["sigma_floor"]),
            sigma_scale=float(payload["sigma_scale"]),
            normalized_scores=tuple(float(x) for x in payload["normalized_scores"]),  # type: ignore[index]
            calibration_size=int(payload["calibration_size"]),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def calibrated_sigma(self, raw_sigma: np.ndarray) -> np.ndarray:
        return np.maximum(raw_sigma, self.sigma_floor) * self.sigma_scale

    def interval(
        self, predictions: np.ndarray, raw_sigma: np.ndarray, coverage: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        scores = np.asarray(self.normalized_scores, dtype=float)
        finite_sample_probability = min(
            1.0 - np.finfo(float).eps,
            np.ceil((len(scores) + 1) * coverage) / len(scores),
        )
        quantile = _higher_quantile(scores, finite_sample_probability)
        half_width = quantile * self.calibrated_sigma(raw_sigma)
        return predictions - half_width, predictions + half_width, quantile


def normal_target_probability(
    predictions: np.ndarray,
    sigma: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Gaussian working score used only to order fixed-pool candidates."""

    safe_sigma = np.maximum(sigma, 1e-9)
    vector_erf = np.vectorize(erf, otypes=[float])
    z_upper = (upper - predictions) / (safe_sigma * sqrt(2.0))
    z_lower = (lower - predictions) / (safe_sigma * sqrt(2.0))
    probabilities = 0.5 * (vector_erf(z_upper) - vector_erf(z_lower))
    return np.clip(probabilities, 0.0, 1.0)

