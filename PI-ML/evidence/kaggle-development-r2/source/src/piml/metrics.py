"""Per-case accuracy, physics, calibration, and solver metrics."""

from __future__ import annotations

from .pde import Case, energy_norm, norm, residual


def case_metrics(
    case: Case,
    prediction: tuple[float, ...],
    stddev: tuple[float, ...] | None = None,
    calibration: float | None = None,
) -> dict[str, float]:
    if len(prediction) != case.grid_points:
        raise ValueError("prediction size differs from case")
    error = tuple(value - truth for value, truth in zip(prediction, case.solution))
    truth_norm = max(norm(case.solution), 1e-30)
    rhs_norm = max(norm(case.forcing[1:-1]), 1e-30)
    physics_residual = residual(case.diffusivity, case.forcing, prediction)
    result = {
        "relative_l2": norm(error) / truth_norm,
        "relative_residual_l2": norm(physics_residual) / rhs_norm,
        "energy_error": energy_norm(case, error),
        "boundary_max": max(abs(prediction[0]), abs(prediction[-1])),
    }
    if stddev is not None and calibration is not None:
        covered = [
            abs(value - truth) <= calibration * max(scale, 1e-12)
            for value, truth, scale in zip(prediction[1:-1], case.solution[1:-1], stddev[1:-1])
        ]
        result["pointwise_coverage90"] = sum(covered) / len(covered)
        result["mean_interval_width90"] = sum(
            2.0 * calibration * max(scale, 1e-12) for scale in stddev[1:-1]
        ) / len(covered)
    return result

