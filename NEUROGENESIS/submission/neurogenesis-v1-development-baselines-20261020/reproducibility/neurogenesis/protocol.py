"""Protocol loading, hashing, and fail-closed run admission."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, ResourceBudget, ScopeStatus


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ContractError(f"non-standard JSON constant rejected: {value}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key rejected: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=object_pairs,
    )


@dataclass(frozen=True)
class Protocol:
    raw: Mapping[str, Any]
    status: ScopeStatus
    approved_mechanism: str | None
    run_budget: ResourceBudget

    @classmethod
    def load(cls, path: Path) -> "Protocol":
        raw = strict_json_load(path)
        if not isinstance(raw, dict):
            raise ContractError("protocol must be a JSON object")
        required = {
            "schema_version",
            "project",
            "status",
            "approved_mechanism",
            "identity_decision",
            "adaptation_problem",
            "information_timing",
            "primary_endpoint",
            "secondary_endpoints",
            "run_budget",
            "development_matrix_budget",
            "candidate_falsifiers",
            "protected_evaluation",
        }
        if set(raw) != required:
            raise ContractError(
                f"protocol keys mismatch: missing={sorted(required - set(raw))}, "
                f"extra={sorted(set(raw) - required)}"
            )
        if raw["schema_version"] != "neurogenesis.protocol.v1":
            raise ContractError("unsupported protocol schema")
        try:
            status = ScopeStatus(raw["status"])
        except ValueError as error:
            raise ContractError(f"unsupported scope status: {raw['status']!r}") from error
        approved = raw["approved_mechanism"]
        if status is ScopeStatus.UNRESOLVED and approved is not None:
            raise ContractError("unresolved scope cannot name an approved mechanism")
        expected_approved = {
            ScopeStatus.LEARNED_RULE: "learned_rule",
            ScopeStatus.MEMORY_WEIGHT: "memory_weight",
        }
        if status in expected_approved and approved != expected_approved[status]:
            raise ContractError("approved mechanism does not match scope status")
        if raw["primary_endpoint"] != "prequential_mean_squared_error":
            raise ContractError("primary endpoint is frozen to external prequential MSE")
        protected = raw["protected_evaluation"]
        if protected.get("admitted") is not False or protected.get("run_count") != 0:
            raise ContractError("protected evaluation must remain unadmitted with zero runs")
        return cls(
            raw=raw,
            status=status,
            approved_mechanism=approved,
            run_budget=ResourceBudget.from_mapping(raw["run_budget"]),
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.raw)

    def admit(self, split: str, policy_kind: str) -> None:
        if split == "evaluation":
            raise PermissionError("protected evaluation is not admitted by protocol v1")
        if split != "development":
            raise PermissionError(f"only the development matrix is admitted, received {split!r}")
        admitted_baselines = {
            "no_adaptation",
            "standard_sgd",
            "mechanism_off_standard_adam",
        }
        if policy_kind not in admitted_baselines:
            raise PermissionError(
                "candidate mechanism execution is denied until a local owner decision resolves scope"
            )

