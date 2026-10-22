"""Interchangeable conventional, derived-target, and ablation update rules."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .linalg import mse, ridge_solution, ridge_solution_with_intercept
from .model import GroupedLinearModel


@dataclass
class ResourceCounter:
    arithmetic_work_units: int = 0
    support_examples_read: int = 0
    target_generator_training_steps: int = 0
    teacher_calls: int = 0
    probes: int = 0
    preprocessing_work_units: int = 0
    full_global_gradient_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class OptimizerState:
    step: int
    first_moment: list[float]
    second_moment: list[float]
    intercept_first_moment: float = 0.0
    intercept_second_moment: float = 0.0

    @classmethod
    def zeros(cls, dimension: int) -> "OptimizerState":
        return cls(step=0, first_moment=[0.0] * dimension, second_moment=[0.0] * dimension)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OptimizerState":
        state = cls(
            step=int(payload["step"]),
            first_moment=[float(value) for value in payload["first_moment"]],
            second_moment=[float(value) for value in payload["second_moment"]],
            intercept_first_moment=float(payload.get("intercept_first_moment", 0.0)),
            intercept_second_moment=float(payload.get("intercept_second_moment", 0.0)),
        )
        if state.step < 0 or len(state.first_moment) != len(state.second_moment):
            raise ValueError("invalid optimizer state")
        if not all(math.isfinite(value) for value in state.first_moment + state.second_moment + [state.intercept_first_moment, state.intercept_second_moment]):
            raise ValueError("optimizer state contains non-finite values")
        return state


class UpdateRule(Protocol):
    name: str

    def step(
        self,
        model: GroupedLinearModel,
        rows: list[list[float]],
        targets: list[float],
        resources: ResourceCounter,
    ) -> None: ...


def validate_batch(model: GroupedLinearModel, rows: list[list[float]], targets: list[float]) -> None:
    if not rows or len(rows) != len(targets):
        raise ValueError("training batch must have equal, non-empty features and targets")
    if any(len(row) != model.dimension for row in rows):
        raise ValueError("training batch width does not match model")
    if not all(math.isfinite(value) for row in rows for value in row) or not all(math.isfinite(value) for value in targets):
        raise ValueError("training batch contains non-finite values")


def mse_gradient(model: GroupedLinearModel, rows: list[list[float]], targets: list[float]) -> list[float]:
    """Gradient of mean squared error; kept standalone for finite-difference tests."""

    validate_batch(model, rows, targets)
    predictions = model.predict(rows)
    count = len(rows)
    return [
        2.0 * math.fsum((prediction - target) * row[index] for row, prediction, target in zip(rows, predictions, targets, strict=True)) / count
        for index in range(model.dimension)
    ]


def _intercept_gradient(model: GroupedLinearModel, rows: list[list[float]], targets: list[float]) -> float:
    if not model.fit_intercept:
        return 0.0
    predictions = model.predict(rows)
    return 2.0 * math.fsum(prediction - target for prediction, target in zip(predictions, targets, strict=True)) / len(rows)


class AdamRule:
    name = "adam"

    def __init__(self, dimension: int, learning_rate: float, *, fit_intercept: bool = False, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.state = OptimizerState.zeros(dimension)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "derived-learning.adam-state.v1",
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "fit_intercept": self.fit_intercept,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "AdamRule":
        if payload.get("schema_version") != "derived-learning.adam-state.v1":
            raise ValueError("unsupported Adam state schema")
        state = OptimizerState.from_dict(payload["state"])
        rule = cls(
            len(state.first_moment),
            learning_rate=float(payload["learning_rate"]),
            beta1=float(payload["beta1"]),
            beta2=float(payload["beta2"]),
            epsilon=float(payload["epsilon"]),
            fit_intercept=bool(payload.get("fit_intercept", False)),
        )
        rule.state = state
        return rule

    def step(self, model: GroupedLinearModel, rows: list[list[float]], targets: list[float], resources: ResourceCounter) -> None:
        gradient = mse_gradient(model, rows, targets)
        intercept_gradient = _intercept_gradient(model, rows, targets)
        self.state.step += 1
        step = self.state.step
        for index, value in enumerate(gradient):
            self.state.first_moment[index] = self.beta1 * self.state.first_moment[index] + (1.0 - self.beta1) * value
            self.state.second_moment[index] = self.beta2 * self.state.second_moment[index] + (1.0 - self.beta2) * value * value
            corrected_first = self.state.first_moment[index] / (1.0 - self.beta1**step)
            corrected_second = self.state.second_moment[index] / (1.0 - self.beta2**step)
            model.weights[index] -= self.learning_rate * corrected_first / (math.sqrt(corrected_second) + self.epsilon)
        if model.fit_intercept:
            self.state.intercept_first_moment = self.beta1 * self.state.intercept_first_moment + (1.0 - self.beta1) * intercept_gradient
            self.state.intercept_second_moment = self.beta2 * self.state.intercept_second_moment + (1.0 - self.beta2) * intercept_gradient * intercept_gradient
            corrected_first = self.state.intercept_first_moment / (1.0 - self.beta1**step)
            corrected_second = self.state.intercept_second_moment / (1.0 - self.beta2**step)
            model.set_intercept(model.intercept - self.learning_rate * corrected_first / (math.sqrt(corrected_second) + self.epsilon))
        model.validate()
        resources.support_examples_read += len(rows)
        resources.full_global_gradient_calls += 1
        resources.arithmetic_work_units += 4 * len(rows) * model.dimension + 12 * model.dimension


class AnalyticResidualRule:
    """Gauss-Seidel block updates from analytic residual targets.

    For block g, the admissible target is y minus the current prediction of all
    other blocks. The block is solved by ridge regression. No global gradient is
    formed and no coefficient/component labels are available to this rule.
    """

    name = "analytic_residual"

    def __init__(self, regularization: float):
        if regularization <= 0.0:
            raise ValueError("regularization must be positive")
        self.regularization = regularization

    def step(self, model: GroupedLinearModel, rows: list[list[float]], targets: list[float], resources: ResourceCounter) -> None:
        validate_batch(model, rows, targets)
        if model.fit_intercept:
            model.set_intercept(math.fsum(target - prediction for target, prediction in zip(targets, model.predict(rows), strict=True)) / len(rows) + model.intercept)
        total_prediction = model.predict(rows)
        resources.arithmetic_work_units += 2 * len(rows) * model.dimension
        for group in model.groups:
            old_group_prediction = model.group_prediction(rows, group)
            local_targets = [
                target - (total - old)
                for target, total, old in zip(targets, total_prediction, old_group_prediction, strict=True)
            ]
            local_rows = [[row[index] for index in group] for row in rows]
            new_weights = ridge_solution(local_rows, local_targets, self.regularization)
            model.set_group_weights(group, new_weights)
            new_group_prediction = [
                math.fsum(row[index] * model.weights[index] for index in group)
                for row in rows
            ]
            total_prediction = [
                total - old + new
                for total, old, new in zip(total_prediction, old_group_prediction, new_group_prediction, strict=True)
            ]
            width = len(group)
            resources.arithmetic_work_units += len(rows) * (2 * width * width + 6 * width + 4) + 2 * width**3
            resources.support_examples_read += len(rows)


class ClassicalBackfittingRule(AnalyticResidualRule):
    """Explicit scientific-name alias for the analytic residual update.

    Keeping this as a separate condition prevents reports from presenting the
    exact partial-residual update as if it were distinct from classical cyclic
    ridge backfitting.  The implementation is intentionally inherited without
    modification so a numerical-equivalence test can enforce the boundary.
    """

    name = "classical_backfitting"


class GlobalRidgeRule:
    """Full-feature ridge solution used as the attainable linear-model ceiling."""

    name = "global_ridge"

    def __init__(self, regularization: float, *, fit_intercept: bool = False):
        if regularization <= 0.0 or not math.isfinite(regularization):
            raise ValueError("regularization must be finite and positive")
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        self.converged = False

    def step(self, model: GroupedLinearModel, rows: list[list[float]], targets: list[float], resources: ResourceCounter) -> None:
        validate_batch(model, rows, targets)
        if self.fit_intercept != model.fit_intercept:
            raise ValueError("global ridge intercept setting does not match model")
        if self.fit_intercept:
            weights, intercept = ridge_solution_with_intercept(rows, targets, self.regularization)
            model.set_intercept(intercept)
            solve_width = model.dimension + 1
        else:
            weights = ridge_solution(rows, targets, self.regularization)
            solve_width = model.dimension
        model.weights = list(weights)
        model.validate()
        resources.support_examples_read += len(rows)
        resources.arithmetic_work_units += (
            len(rows) * (2 * solve_width * solve_width + 3 * solve_width)
            + 2 * solve_width**3
        )
        self.converged = True


class BlockJacobiResidualRule:
    """Simultaneous residual-target updates using a frozen sweep prediction.

    Unlike :class:`AnalyticResidualRule`, every local target is computed from
    the same pre-sweep model.  This isolates target derivation from the
    Gauss--Seidel update order used by classical backfitting.
    """

    name = "block_jacobi_residual"

    def __init__(self, regularization: float):
        if regularization <= 0.0:
            raise ValueError("regularization must be positive")
        self.regularization = regularization

    def step(self, model: GroupedLinearModel, rows: list[list[float]], targets: list[float], resources: ResourceCounter) -> None:
        validate_batch(model, rows, targets)
        total_prediction = model.predict(rows)
        proposed: list[tuple[list[int], list[float]]] = []
        resources.arithmetic_work_units += 2 * len(rows) * model.dimension
        for group in model.groups:
            old_group_prediction = model.group_prediction(rows, group)
            local_targets = [
                target - (total - old)
                for target, total, old in zip(targets, total_prediction, old_group_prediction, strict=True)
            ]
            local_rows = [[row[index] for index in group] for row in rows]
            proposed.append((group, ridge_solution(local_rows, local_targets, self.regularization)))
            width = len(group)
            resources.arithmetic_work_units += len(rows) * (2 * width * width + 6 * width + 4) + 2 * width**3
            resources.support_examples_read += len(rows)
        for group, values in proposed:
            model.set_group_weights(group, values)
        if model.fit_intercept:
            model.set_intercept(math.fsum(target - prediction for target, prediction in zip(targets, model.predict(rows), strict=True)) / len(rows) + model.intercept)


class FixedNormalizedLMSRule:
    """A simple, untuned fixed rule using normalized full-batch residual updates."""

    name = "fixed_normalized_lms"

    def __init__(self, step_size: float = 0.25):
        self.step_size = step_size

    def step(self, model: GroupedLinearModel, rows: list[list[float]], targets: list[float], resources: ResourceCounter) -> None:
        validate_batch(model, rows, targets)
        predictions = model.predict(rows)
        residuals = [target - prediction for target, prediction in zip(targets, predictions, strict=True)]
        for index in range(model.dimension):
            numerator = math.fsum(residual * row[index] for residual, row in zip(residuals, rows, strict=True))
            denominator = math.fsum(row[index] ** 2 for row in rows) + 1e-8
            model.weights[index] += self.step_size * numerator / denominator
        if model.fit_intercept:
            model.set_intercept(model.intercept + self.step_size * math.fsum(residuals) / len(rows))
        model.validate()
        resources.support_examples_read += len(rows)
        resources.arithmetic_work_units += 5 * len(rows) * model.dimension + 4 * model.dimension


class BlockGradientAblationRule:
    """Same block schedule as the proposal, but ordinary MSE gradients only."""

    name = "block_gradient_no_derivation"

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, model: GroupedLinearModel, rows: list[list[float]], targets: list[float], resources: ResourceCounter) -> None:
        validate_batch(model, rows, targets)
        for group in model.groups:
            predictions = model.predict(rows)
            count = len(rows)
            updates = [
                2.0 * math.fsum((prediction - target) * row[index] for row, prediction, target in zip(rows, predictions, targets, strict=True)) / count
                for index in group
            ]
            model.set_group_weights(group, [model.weights[index] - self.learning_rate * update for index, update in zip(group, updates, strict=True)])
            resources.support_examples_read += len(rows)
            resources.arithmetic_work_units += 2 * len(rows) * model.dimension + 2 * len(rows) * len(group) + 3 * len(group)
        if model.fit_intercept:
            residuals = [target - prediction for target, prediction in zip(targets, model.predict(rows), strict=True)]
            model.set_intercept(model.intercept + self.learning_rate * 2.0 * math.fsum(residuals) / len(rows))


def training_loss(model: GroupedLinearModel, rows: list[list[float]], targets: list[float]) -> float:
    return mse(model.predict(rows), targets)

