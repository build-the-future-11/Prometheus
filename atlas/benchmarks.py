from __future__ import annotations

import json
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
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} of {path} must contain a JSON object")
            yield payload


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
