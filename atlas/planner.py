from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from .models import Criterion, Experiment, RunManifest


def _slug(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return normalized[:48] or "research-run"


def _protocol_digest(manifest: RunManifest) -> str:
    """Bind new plan identities to the full protocol, never creation-time metadata."""
    protocol = manifest.to_dict()
    protocol.pop("run_id")
    protocol.pop("metadata")
    encoded = json.dumps(
        {"identity_scheme": "atlas.protocol-sha256.v1", "protocol": protocol},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    question: str,
    *,
    treatment: str = "proposed method",
    control: str = "strong baseline",
    metric: str = "primary_score",
    minimum_improvement: float = 0.02,
    seeds: tuple[int, ...] = (17, 29, 43, 71, 101),
    perturbations: tuple[str, ...] = ("in_distribution", "distribution_shift", "stress_test"),
) -> RunManifest:
    question = " ".join(question.split())
    if len(question) < 10:
        raise ValueError("research question must contain at least 10 characters")
    if treatment.strip() == control.strip():
        raise ValueError("treatment and control must differ")
    if minimum_improvement <= 0:
        raise ValueError("minimum_improvement must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not perturbations:
        raise ValueError("at least one perturbation is required")

    hypothesis = (
        f"{treatment} will outperform {control} on {metric} by at least "
        f"{minimum_improvement:.4f} under the preregistered evaluation matrix."
    )
    null_hypothesis = (
        f"{treatment} will not outperform {control} on {metric} by at least "
        f"{minimum_improvement:.4f} under the preregistered evaluation matrix."
    )

    experiments: list[Experiment] = []
    for perturbation in perturbations:
        for seed in seeds:
            name = f"{_slug(perturbation)}-seed-{seed}"
            experiments.append(
                Experiment(
                    name=name,
                    treatment=treatment,
                    control=control,
                    seed=seed,
                    perturbation=perturbation,
                    required_artifacts=(
                        "config.json",
                        "metrics.json",
                        "stdout.log",
                        "environment.txt",
                    ),
                )
            )

    manifest = RunManifest(
        run_id="pending-protocol-identity",
        question=question,
        hypothesis=hypothesis,
        null_hypothesis=null_hypothesis,
        primary_criterion=Criterion(
            metric=f"mean_{metric}_improvement",
            operator=">=",
            threshold=minimum_improvement,
            rationale="The minimum effect size is fixed before execution to prevent post-hoc threshold selection.",
        ),
        experiments=experiments,
        stopping_rule=(
            "Run every preregistered seed and perturbation exactly once. Do not stop early for positive or negative interim results. "
            "A failed or crashed run must remain in the evidence record and may be rerun only with a documented reason."
        ),
        evidence_requirements=[
            "Record the exact source commit SHA for every run.",
            "Record dataset identity/version and any preprocessing configuration.",
            "Persist per-seed treatment and control metrics, not only aggregates.",
            "Report all preregistered perturbations, including negative results.",
            "Separate exploratory follow-up experiments from the preregistered confirmatory matrix.",
        ],
        metadata={
            "schema_version": "atlas.run.v0.1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed_count": len(seeds),
            "perturbation_count": len(perturbations),
            "experiment_count": len(experiments),
        },
    )
    manifest.validate()
    digest = _protocol_digest(manifest)
    manifest.run_id = f"{_slug(question)}-{digest[:16]}"
    manifest.metadata["run_id_scheme"] = "atlas.protocol-sha256.v1"
    manifest.metadata["protocol_sha256"] = digest
    return manifest
