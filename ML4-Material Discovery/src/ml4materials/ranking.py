"""Label-blind fixed-pool policies and post hoc ranking evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .uncertainty import normal_target_probability


def policy_orders(
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    target_window: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Construct deterministic policies without accepting evaluation labels."""

    lower, upper = target_window
    center = (lower + upper) / 2.0
    probability = normal_target_probability(predictions, uncertainty, lower, upper)
    return {
        "greedy_predicted_property": np.argsort(
            np.abs(predictions - center), kind="mergesort"
        ),
        "uncertainty": np.argsort(-uncertainty, kind="mergesort"),
        "proposed_calibrated_probability": np.argsort(-probability, kind="mergesort"),
    }


def evaluate_ranking(
    targets: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    target_window: tuple[float, float],
    budgets: list[int],
    random_replicates: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    lower, upper = target_window
    hits = (targets >= lower) & (targets <= upper)
    total_hits = int(hits.sum())
    prevalence = float(hits.mean())
    rows: list[dict[str, Any]] = []
    orders = policy_orders(predictions, uncertainty, target_window)
    probability = normal_target_probability(predictions, uncertainty, lower, upper)

    for policy, order in orders.items():
        for budget in budgets:
            chosen = order[:budget]
            hit_count = int(hits[chosen].sum())
            rows.append(
                {
                    "policy": policy,
                    "budget": int(budget),
                    "hits": float(hit_count),
                    "hits_std": 0.0,
                    "precision": float(hit_count / budget),
                    "precision_std": 0.0,
                    "recall": float(hit_count / total_hits) if total_hits else 0.0,
                    "enrichment_vs_pool": float((hit_count / budget) / prevalence)
                    if prevalence
                    else 0.0,
                    "mean_absolute_target_distance_ev": float(
                        np.mean(np.abs(targets[chosen] - (lower + upper) / 2.0))
                    ),
                }
            )

    rng = np.random.default_rng(20261019)
    random_orders = [rng.permutation(len(targets)) for _ in range(random_replicates)]
    for budget in budgets:
        counts = np.asarray([hits[order[:budget]].sum() for order in random_orders], dtype=float)
        distances = np.asarray(
            [
                np.mean(np.abs(targets[order[:budget]] - (lower + upper) / 2.0))
                for order in random_orders
            ],
            dtype=float,
        )
        mean_hits = float(counts.mean())
        rows.append(
            {
                "policy": "random",
                "budget": int(budget),
                "hits": mean_hits,
                "hits_std": float(counts.std(ddof=1)),
                "precision": float(mean_hits / budget),
                "precision_std": float(counts.std(ddof=1) / budget),
                "recall": float(mean_hits / total_hits) if total_hits else 0.0,
                "enrichment_vs_pool": float((mean_hits / budget) / prevalence)
                if prevalence
                else 0.0,
                "mean_absolute_target_distance_ev": float(distances.mean()),
            }
        )
    return rows, orders["proposed_calibrated_probability"], probability

