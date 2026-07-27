"""Small bounded-step BFGS optimizer for nested Boozer solves."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult


Vector = NDArray[np.float64]
Objective = Callable[[Vector], tuple[float, Vector, bool]]
Callback = Callable[[Vector], None]


def rejected_trial_value_and_gradient(
    trial_x: Vector,
    accepted_x: Vector,
    accepted_value: float,
) -> tuple[float, Vector]:
    """Return a smooth quadratic rejection model around the accepted point."""
    displacement = np.asarray(trial_x) - np.asarray(accepted_x)
    scale = max(abs(float(accepted_value)), 1.0)
    value = float(
        float(accepted_value) + scale * np.dot(displacement, displacement)
    )
    gradient = 2.0 * scale * displacement
    return value, gradient


def minimize_bounded_bfgs(
    objective: Objective,
    x0: Vector,
    *,
    initial_value: float,
    initial_gradient: Vector,
    callback: Callback,
    maxiter: int,
    gtol: float,
    max_step_norm: float,
) -> OptimizeResult:
    """Run BFGS with an L2 step cap and Armijo-only backtracking."""
    if not np.isfinite(max_step_norm) or max_step_norm <= 0.0:
        raise ValueError("max_step_norm must be finite and positive")

    x = np.asarray(x0, dtype=np.float64).copy()
    value = float(initial_value)
    gradient = np.asarray(initial_gradient, dtype=np.float64).copy()
    inverse_hessian = np.eye(x.size, dtype=np.float64)
    evaluations = 0
    step_bound = max_step_norm

    for iteration in range(maxiter + 1):
        if np.linalg.norm(gradient, ord=np.inf) <= gtol:
            return OptimizeResult(
                x=x, fun=value, jac=gradient, hess_inv=inverse_hessian,
                nfev=evaluations, njev=evaluations, nit=iteration,
                status=0, success=True,
                message="Optimization terminated successfully.",
            )
        if iteration == maxiter:
            break

        direction = -(inverse_hessian @ gradient)
        slope = float(np.dot(gradient, direction))
        if slope >= 0.0:
            inverse_hessian = np.eye(x.size, dtype=np.float64)
            direction = -gradient
            slope = -float(np.dot(gradient, gradient))

        direction_norm = float(np.linalg.norm(direction))
        step_length = min(1.0, step_bound / direction_norm)
        accepted = False

        for _ in range(20):
            candidate_x = x + step_length * direction
            candidate_value, candidate_gradient, candidate_valid = objective(candidate_x)
            evaluations += 1
            if candidate_valid and candidate_value <= (
                value + 1.0e-4 * step_length * slope
            ):
                accepted = True
                break
            step_length *= 0.5

        if not accepted:
            objective(x)
            return OptimizeResult(
                x=x, fun=value, jac=gradient, hess_inv=inverse_hessian,
                nfev=evaluations + 1, njev=evaluations + 1,
                nit=iteration, status=2, success=False,
                message="Bounded Armijo line search failed.",
            )

        candidate_gradient = np.asarray(candidate_gradient, dtype=np.float64).copy()
        step = candidate_x - x
        gradient_change = candidate_gradient - gradient
        curvature = float(np.dot(step, gradient_change))
        curvature_scale = float(
            np.linalg.norm(step) * np.linalg.norm(gradient_change)
        )
        if curvature > 1.0e-10 * curvature_scale:
            rho = 1.0 / curvature
            identity = np.eye(x.size, dtype=np.float64)
            left = identity - rho * np.outer(step, gradient_change)
            inverse_hessian = (
                left @ inverse_hessian @ left.T + rho * np.outer(step, step)
            )
            inverse_hessian = 0.5 * (inverse_hessian + inverse_hessian.T)

        x = candidate_x
        value = float(candidate_value)
        gradient = candidate_gradient
        step_bound = min(max_step_norm, 2.0 * float(np.linalg.norm(step)))
        callback(x)

    return OptimizeResult(
        x=x, fun=value, jac=gradient, hess_inv=inverse_hessian,
        nfev=evaluations, njev=evaluations, nit=maxiter,
        status=1, success=False,
        message="Maximum number of accepted iterations exceeded.",
    )
