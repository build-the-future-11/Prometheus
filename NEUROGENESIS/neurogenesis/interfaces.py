"""Isolated candidate interfaces; neither scientific hypothesis is implemented."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .contracts import ResourceBudget, ScopeStatus


class CandidateNotApproved(PermissionError):
    """Raised when candidate execution is attempted before scope approval."""


class LearnedRuleCandidate(ABC):
    """Interface reserved for a future learned-loss or learned-update decision."""

    candidate_scope = ScopeStatus.LEARNED_RULE

    @abstractmethod
    def meta_train(
        self,
        task_ids: Sequence[str],
        budget: ResourceBudget,
    ) -> Mapping[str, Any]:
        """Fit only on registered meta-training tasks and return retained state."""

    @abstractmethod
    def update(self, features: Sequence[float], target: float) -> None:
        """Apply one admitted inner update after the target is available."""


class MemoryWeightCandidate(ABC):
    """Interface reserved for a future bounded-memory/weight-update decision."""

    candidate_scope = ScopeStatus.MEMORY_WEIGHT

    @abstractmethod
    def predict(self, features: Sequence[float]) -> float:
        """Predict before receiving the current target."""

    @abstractmethod
    def observe_available_label(
        self,
        features: Sequence[float],
        target: float,
        available_at: int,
    ) -> None:
        """Write or update only once the label availability time is reached."""


def require_candidate_approval(scope: ScopeStatus, candidate_scope: ScopeStatus) -> None:
    if scope is ScopeStatus.UNRESOLVED:
        raise CandidateNotApproved("candidate execution denied while scientific scope is unresolved")
    if scope is not candidate_scope:
        raise CandidateNotApproved(
            f"candidate scope {candidate_scope.value} conflicts with approved scope {scope.value}"
        )

