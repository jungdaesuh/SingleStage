import unittest

import numpy as np

from bounded_bfgs import minimize_bounded_bfgs, rejected_trial_value_and_gradient


class BoundedBFGSTests(unittest.TestCase):
    def test_steps_never_exceed_the_configured_radius(self):
        evaluated = []
        accepted = [np.array([2.0, -1.0])]

        def objective(x):
            evaluated.append((accepted[0].copy(), x.copy()))
            return 0.5 * float(np.dot(x, x)), x.copy(), True

        def callback(x):
            accepted[0] = x.copy()

        x0 = np.array([2.0, -1.0])
        result = minimize_bounded_bfgs(
            objective,
            x0,
            initial_value=2.5,
            initial_gradient=x0,
            callback=callback,
            maxiter=40,
            gtol=1.0e-8,
            max_step_norm=0.2,
        )

        self.assertTrue(result.success, result.message)
        for anchor, point in evaluated:
            self.assertLessEqual(np.linalg.norm(point - anchor), 0.2 + 1.0e-14)

    def test_backtracks_from_a_rejected_trial(self):
        rejected = 0

        def objective(x):
            nonlocal rejected
            if x[0] < 0.15:
                rejected += 1
                return 10.0 + float((x[0] - 0.15) ** 2), 2.0 * (x - 0.15), False
            displacement = x - 0.2
            return float(np.dot(displacement, displacement)), 2.0 * displacement, True

        result = minimize_bounded_bfgs(
            objective,
            np.array([1.0]),
            initial_value=0.64,
            initial_gradient=np.array([1.6]),
            callback=lambda _x: None,
            maxiter=30,
            gtol=1.0e-8,
            max_step_norm=0.9,
        )

        self.assertGreater(rejected, 0)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, np.array([0.2]), atol=1.0e-8)

    def test_roundoff_cannot_accept_an_invalid_trial(self):
        callbacks = []

        def rejected_objective(x):
            value, gradient = rejected_trial_value_and_gradient(
                x, np.zeros(1), 1.0e-3,
            )
            return value, gradient, False

        result = minimize_bounded_bfgs(
            rejected_objective,
            np.zeros(1),
            initial_value=1.0e-3,
            initial_gradient=np.array([1.0001e-4]),
            callback=lambda x: callbacks.append(x.copy()),
            maxiter=1,
            gtol=1.0e-4,
            max_step_norm=1.0e-12,
        )

        self.assertFalse(result.success)
        self.assertEqual(callbacks, [])


if __name__ == "__main__":
    unittest.main()
