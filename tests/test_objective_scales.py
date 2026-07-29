import unittest

from simsopt.field import Current, CurrentPenalty

from stage2_contracts import (
    BOOZER_RESIDUAL_THRESHOLD,
    CURRENT_SCALE_AMPERES,
    boozer_residual_guard,
    current_limit_penalty,
    iota_penalty_curvature,
)


class ObjectiveScaleTests(unittest.TestCase):
    def test_boozer_residual_below_the_threshold_costs_nothing(self):
        residual = CurrentPenalty(
            [Current(0.9 * BOOZER_RESIDUAL_THRESHOLD)],
            p=2,
        )
        guard = boozer_residual_guard(residual)
        self.assertEqual(float(guard.J()), 0.0)

    def test_boozer_residual_one_threshold_above_contributes_one_half(self):
        residual = CurrentPenalty(
            [Current(2.0 * BOOZER_RESIDUAL_THRESHOLD)],
            p=2,
        )
        guard = boozer_residual_guard(residual)
        self.assertAlmostEqual(float(guard.J()), 0.5, places=12)

    def test_iota_error_of_one_scale_contributes_one_half(self):
        # QuadraticPenalty is 0.5*diff^2; at diff = IOTA_SCALE = |target| the
        # scaled contribution must be exactly 0.5, independent of the target.
        for target in (0.3, 0.05, -0.3):
            curvature = iota_penalty_curvature(target)
            scale = abs(target)
            self.assertAlmostEqual(curvature * 0.5 * scale**2, 0.5, places=15)

    def test_zero_target_falls_back_to_unit_scale(self):
        self.assertEqual(iota_penalty_curvature(0.0), 1.0)

    def test_non_finite_target_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            iota_penalty_curvature(float("nan"))

    def test_current_below_the_limit_costs_nothing(self):
        # 150 kA is a limit, not a goal: the one-sided term must be exactly
        # zero below it, so the optimizer feels no pull to shrink admissible
        # currents.
        penalty = current_limit_penalty([Current(0.9 * CURRENT_SCALE_AMPERES)])
        self.assertEqual(float(penalty.J()), 0.0)

    def test_current_one_scale_above_the_limit_contributes_one_half(self):
        penalty = current_limit_penalty([Current(2.0 * CURRENT_SCALE_AMPERES)])
        self.assertAlmostEqual(float(penalty.J()), 0.5, places=12)

    def test_current_scale_is_the_150ka_policy(self):
        self.assertEqual(CURRENT_SCALE_AMPERES, 150_000.0)

    def test_custom_current_limit_moves_the_threshold(self):
        # The --current-limit-amperes CLI knob: the same 0/0.5 contract must
        # hold at whatever limit the caller declares, not just the default.
        limit = 100_000.0
        below = current_limit_penalty([Current(0.9 * limit)], limit_amperes=limit)
        self.assertEqual(float(below.J()), 0.0)
        above = current_limit_penalty([Current(2.0 * limit)], limit_amperes=limit)
        self.assertAlmostEqual(float(above.J()), 0.5, places=12)

    def test_invalid_current_limit_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            current_limit_penalty([Current(1.0)], limit_amperes=0.0)


if __name__ == "__main__":
    unittest.main()
