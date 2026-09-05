from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    answer: str


@dataclass(frozen=True)
class Prediction:
    case_id: str
    prediction: str


@dataclass(frozen=True)
class BenchmarkManifest:
    name: str
    version: str
    license: str
    task_description: str
    metric: str
    dataset_sha256: str
    case_count: int

    def validate(self) -> None:
        for field_name in ("name", "version", "license", "task_description", "metric"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"benchmark manifest {field_name} must be non-empty")
        if self.metric != "exact_match":
            raise ValueError("benchmark manifest metric must be 'exact_match' for Benchmark Foundry v0")
        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_sha256):
            raise ValueError("benchmark manifest dataset_sha256 must be a lowercase SHA-256 hex digest")
        if isinstance(self.case_count, bool) or not isinstance(self.case_count, int) or self.case_count < 1:
            raise ValueError("benchmark manifest case_count must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "task_description": self.task_description,
            "metric": self.metric,
            "dataset_sha256": self.dataset_sha256,
            "case_count": self.case_count,
        }


@dataclass
class ScoreReport:
    total: int
    correct: int
    missing: int
    accuracy: float
    by_category: dict[str, dict[str, float | int]]
    unexpected_prediction_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "missing": self.missing,
            "accuracy": self.accuracy,
            "by_category": self.by_category,
            "unexpected_prediction_ids": self.unexpected_prediction_ids,
        }


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read {source}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {source}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} of {source} must contain a JSON object")
            yield payload


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"could not hash {source}: {exc}") from exc
    return digest.hexdigest()


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for index, payload in enumerate(_read_jsonl(path), start=1):
        try:
            case_id = str(payload["id"]).strip()
            category = str(payload["category"]).strip()
            answer = str(payload["answer"]).strip()
        except KeyError as exc:
            raise ValueError(f"benchmark case {index} missing field {exc.args[0]!r}") from exc
        if not case_id or not category or not answer:
            raise ValueError(f"benchmark case {index} contains an empty required field")
        if case_id in seen:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        seen.add(case_id)
        cases.append(BenchmarkCase(case_id=case_id, category=category, answer=answer))
    if not cases:
        raise ValueError("benchmark contains no cases")
    return cases


def load_predictions(path: str | Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    seen: set[str] = set()
    for index, payload in enumerate(_read_jsonl(path), start=1):
        try:
            case_id = str(payload["id"]).strip()
            prediction = str(payload["prediction"]).strip()
        except KeyError as exc:
            raise ValueError(f"prediction {index} missing field {exc.args[0]!r}") from exc
        if not case_id:
            raise ValueError(f"prediction {index} has an empty id")
        if case_id in seen:
            raise ValueError(f"duplicate prediction id: {case_id}")
        seen.add(case_id)
        predictions.append(Prediction(case_id=case_id, prediction=prediction))
    return predictions


def build_benchmark_manifest(
    cases_path: str | Path,
    *,
    name: str,
    version: str,
    license_name: str,
    task_description: str,
) -> BenchmarkManifest:
    cases = load_cases(cases_path)
    manifest = BenchmarkManifest(
        name=name.strip(),
        version=version.strip(),
        license=license_name.strip(),
        task_description=task_description.strip(),
        metric="exact_match",
        dataset_sha256=_sha256_file(cases_path),
        case_count=len(cases),
    )
    manifest.validate()
    return manifest


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid benchmark manifest {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark manifest must contain a JSON object")
    required = {
        "name",
        "version",
        "license",
        "task_description",
        "metric",
        "dataset_sha256",
        "case_count",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"benchmark manifest missing fields: {', '.join(missing)}")
    manifest = BenchmarkManifest(
        name=payload["name"],
        version=payload["version"],
        license=payload["license"],
        task_description=payload["task_description"],
        metric=payload["metric"],
        dataset_sha256=payload["dataset_sha256"],
        case_count=payload["case_count"],
    )
    manifest.validate()
    return manifest


def validate_benchmark(cases_path: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    cases = load_cases(cases_path)
    result: dict[str, Any] = {
        "valid": True,
        "case_count": len(cases),
        "dataset_sha256": _sha256_file(cases_path),
        "categories": sorted({case.category for case in cases}),
    }
    if manifest_path is not None:
        manifest = load_benchmark_manifest(manifest_path)
        if manifest.case_count != len(cases):
            raise ValueError(
                f"benchmark manifest case_count={manifest.case_count} does not match dataset count={len(cases)}"
            )
        if manifest.dataset_sha256 != result["dataset_sha256"]:
            raise ValueError("benchmark manifest dataset_sha256 does not match the cases file")
        result["manifest"] = manifest.to_dict()
    return result


def normalize_answer(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def score_exact_match(cases: list[BenchmarkCase], predictions: list[Prediction]) -> ScoreReport:
    prediction_map = {item.case_id: item.prediction for item in predictions}
    case_ids = {case.case_id for case in cases}
    unexpected = sorted(set(prediction_map) - case_ids)

    correct = 0
    missing = 0
    category_totals: dict[str, int] = defaultdict(int)
    category_correct: dict[str, int] = defaultdict(int)
    category_missing: dict[str, int] = defaultdict(int)

    for case in cases:
        category_totals[case.category] += 1
        if case.case_id not in prediction_map:
            missing += 1
            category_missing[case.category] += 1
            continue
        if normalize_answer(prediction_map[case.case_id]) == normalize_answer(case.answer):
            correct += 1
            category_correct[case.category] += 1

    by_category: dict[str, dict[str, float | int]] = {}
    for category in sorted(category_totals):
        total = category_totals[category]
        category_hits = category_correct[category]
        by_category[category] = {
            "total": total,
            "correct": category_hits,
            "missing": category_missing[category],
            "accuracy": category_hits / total if total else 0.0,
        }

    total = len(cases)
    return ScoreReport(
        total=total,
        correct=correct,
        missing=missing,
        accuracy=correct / total if total else 0.0,
        by_category=by_category,
        unexpected_prediction_ids=unexpected,
    )
