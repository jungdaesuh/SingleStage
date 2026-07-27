"""Regression guards for the minimal driver fixes.

Every test here pins a defect that was measured on this code, so each one names
the failure it prevents. The driver takes minutes per run and needs a Stage-II
input, so the structural properties are checked against the source and the
cheap behaviour is checked by invoking the CLI.
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).parents[1]
DRIVER = ROOT / "single_stage_dipoles.py"
BOOZER = ROOT / "boozer_functions.py"


def driver_source():
    return DRIVER.read_text(encoding="utf-8")


class PublicationGateTests(unittest.TestCase):
    def test_gates_run_before_any_final_artifact_is_written(self):
        # Measured defect: the gates sat AFTER the save block, so a run that
        # failed them had already written coils_opt.vtu, bs_opt.json,
        # surf_opt.json, surf_opt.vts and vacuum_vessel.vts -- while the error
        # said nothing was published.
        source = driver_source()
        gates = source.index("# ---- Publication gates: fail closed ----")
        for writer in (
            'coils_to_vtk(coils, filename=os.path.join(out_dir, "coils_opt")',
            'VV.to_vtk(os.path.join(out_dir, "vacuum_vessel"))',
            'boozer_surface.surface.to_vtk(os.path.join(out_dir, "surf_opt")',
            'plot_relBfinal_norm_modB(',
        ):
            self.assertIn(writer, source)
            self.assertGreater(
                source.index(writer), gates,
                msg=f"final artifact write must come after the gates: {writer}",
            )

    def test_every_gated_quantity_is_computed_before_the_gates(self):
        source = driver_source()
        gates = source.index("# ---- Publication gates: fail closed ----")
        for computation in (
            "final_iota = float(iota.J())",
            "final_cp = float(Jcurrent.J())",
        ):
            self.assertIn(computation, source)
            self.assertLess(source.index(computation), gates, msg=computation)

    def test_all_three_gates_are_present_and_raise(self):
        source = driver_source()
        block = source.split("# ---- Publication gates: fail closed ----", 1)[1]
        block = block.split("# SAVE FINAL OUTPUTS", 1)[0]
        for gate in ("optimizer convergence", "iota", "current p-norm"):
            self.assertIn(f'"{gate}"', block)
        self.assertIn("raise RuntimeError(", block)

    def test_gate_message_does_not_claim_nothing_exists_on_disk(self):
        # The resume callback checkpoints on every accepted step, so a snapshot
        # always exists. Claiming otherwise is what made the original message
        # false.
        # The message is wrapped across f-string lines, so match its fragments.
        source = driver_source()
        self.assertNotIn("Nothing was published", source)
        self.assertIn("No final ", source)
        self.assertIn("artifact was written; the resume checkpoint under", source)
        self.assertIn("stays marked optimization_success=None", source)


class ObjectiveBookkeepingTests(unittest.TestCase):
    def test_iota_contribution_is_not_scaled_by_the_boozer_penalty_weight(self):
        # Measured defect: Jiota was moved out of the PENALTY_WEIGHT bracket but
        # the history still recorded PENALTY_WEIGHT * Jiota, over-reporting the
        # iota contribution by 100x (1.6571e-08 logged as 1.6571e-06).
        source = driver_source()
        self.assertIn('"J_iota_contrib": float(ji_g),', source)
        self.assertNotIn('"J_iota_contrib": float(PENALTY_WEIGHT * ji_g)', source)
        # The two terms that ARE still inside the bracket must keep their factor.
        self.assertIn('"J_boozer_contrib": float(PENALTY_WEIGHT * jb_g)', source)
        self.assertIn('"J_current_contrib": float(PENALTY_WEIGHT * jc_g)', source)

    def test_objective_is_assembled_with_iota_outside_the_bracket(self):
        source = driver_source()
        self.assertIn(
            "JF = JnonQSRatio + PENALTY_WEIGHT * (JBoozerGuard + JCurrentGuard) + Jiota",
            source,
        )
        self.assertIn("Jiota = IOTA_PENALTY_CURVATURE * QuadraticPenalty(", source)

    def test_printed_objective_formula_matches_the_assembled_objective(self):
        # The banner still advertised the old bracketing, so the log described an
        # objective the code no longer minimised.
        source = driver_source()
        self.assertNotIn(
            "nonQS + {PENALTY_WEIGHT} * (Boozer_guard + iota_penalty + current_guard)",
            source,
        )
        self.assertIn("Boozer_guard + current_guard", source)

    def test_curvature_is_derived_once_and_recorded(self):
        source = driver_source()
        self.assertEqual(source.count("IOTA_WEIGHT / IOTA_SCALE**2"), 1)
        for key in (
            '"iota_penalty_weight": IOTA_WEIGHT',
            '"iota_scale": IOTA_SCALE',
            '"iota_penalty_curvature": IOTA_PENALTY_CURVATURE',
        ):
            self.assertIn(key, source)


class OuterTrialSolveTests(unittest.TestCase):
    def test_outer_trials_are_newton_only(self):
        # Measured defect: fun() called run_code(), whose BoozerLS path is "BFGS
        # followed by Newton" (simsopt BoozerSurface.run_code docstring). Every
        # outer trial therefore restarted a ~300-iteration BFGS from an already
        # converged surface that Newton then fixed in one step -- 10 solves and
        # 2976 inner iterations for 7 accepted outer iterations at order 6.
        source = driver_source()
        objective = source.split("    def fun(x):", 1)[1].split("    def callback(", 1)[0]
        # Match the call, not the bare word: "run_code" legitimately appears in
        # the comment explaining why it is not called here.
        self.assertNotIn("boozer_surface.run_code(", objective)
        self.assertIn(
            "boozer_surface.minimize_boozer_penalty_constraints_newton(", objective
        )
        # The Newton entry point returns self.res early unless this is set, and
        # run_code does the same thing before its own Newton polish.
        self.assertIn("boozer_surface.need_to_run_code = True", objective)

    def test_newton_trial_reuses_the_surface_own_solver_options(self):
        # SSOT: do not invent tolerances that can drift from the ones the
        # BoozerSurface was constructed with.
        source = driver_source()
        objective = source.split("    def fun(x):", 1)[1].split("    def callback(", 1)[0]
        for option in (
            "constraint_weight=boozer_surface.constraint_weight",
            'tol=boozer_surface.options["newton_tol"]',
            'maxiter=boozer_surface.options["newton_maxiter"]',
            'weight_inv_modB=boozer_surface.options["weight_inv_modB"]',
        ):
            self.assertIn(option, objective)

    def test_initial_solve_still_uses_the_full_bfgs_plus_newton_path(self):
        # Only the outer trials go Newton-only; the first solve has no warm start
        # to Newton from, so initialize_boozer_surface must keep run_code.
        source = BOOZER.read_text(encoding="utf-8")
        self.assertIn("res = boozer_surface.run_code(iota, G0)", source)


class FieldPolarityTests(unittest.TestCase):
    def test_G_sign_requires_a_single_polarity_across_the_base_bundle(self):
        # sign(tf_coils[0]) is only the field polarity if the independent bundle
        # agrees; mixed signs would silently produce a wrong-signed G.
        source = driver_source()
        self.assertIn("base TF currents must share one physical polarity", source)
        self.assertIn("base TF currents must be finite and non-zero", source)
        self.assertIn("G_sign = int(base_signs[0])", source)

    def test_derived_G_and_polarity_reach_the_artifact(self):
        source = driver_source()
        self.assertIn('"field_polarity": G_sign', source)
        self.assertIn('"initial_G": G0', source)


class BoozerGuardTests(unittest.TestCase):
    def test_self_intersection_screen_is_not_disabled(self):
        # The original overwrote the check result with True on the next line,
        # making the raise dead code.
        source = BOOZER.read_text(encoding="utf-8")
        self.assertNotIn("success2 = True", source)
        self.assertIn("is_self_intersecting(angle=angle)", source)

    def test_self_intersection_screen_covers_more_than_one_angle(self):
        # simsopt: "if this function returns False, the surface may still be
        # self-intersecting away from angle."
        source = BOOZER.read_text(encoding="utf-8")
        self.assertIn("(0.0, 0.25, 0.5, 0.75)", source)

    def test_boozer_solve_grid_is_matched_to_the_mode_numbers(self):
        # The solve inherited surf_prev's grid -- 512x128 = 65536 points from a
        # Stage-II artifact, 105x the 25x25 a mpol=ntor=6 solve needs.
        source = BOOZER.read_text(encoding="utf-8")
        self.assertIn("nphi_s, ntheta_s = 4 * ntor + 1, 4 * mpol + 1", source)
        self.assertNotIn("quadpoints_theta=surf_prev.quadpoints_theta", source)
        self.assertNotIn("quadpoints_phi=surf_prev.quadpoints_phi", source)

    def test_publication_uses_the_active_boozer_grid(self):
        source = driver_source()
        self.assertIn("nphi_b = boozer_surface.surface.quadpoints_phi.size", source)
        self.assertIn("B = bs.B().reshape((nphi_b, ntheta_b, 3))", source)
        self.assertNotIn("B = bs.B().reshape((nPhi, nTheta, 3))", source)


class GeneratorTests(unittest.TestCase):
    def test_single_generator_defines_its_surface_resolution(self):
        # Raised NameError at module scope, so it could not produce an --init-dir.
        source = (ROOT / "run_optimize_single.py").read_text(encoding="utf-8")
        self.assertIn("plas_nPhi = 128", source)
        self.assertIn("plas_nTheta = 64", source)

    def test_scan_generator_does_not_pass_unsupported_kwargs(self):
        # optimize() accepts none of these; passing them raised TypeError.
        source = (ROOT / "run_optimize_scan.py").read_text(encoding="utf-8")
        for kwarg in ("plas_nPhi=plas_nPhi", "plas_nTheta=plas_nTheta",
                      "numquadpoints=numquadpoints"):
            self.assertNotIn(kwarg, source)


class CliContractTests(unittest.TestCase):
    def test_iota_threshold_help_states_that_it_gates_publication(self):
        source = driver_source()
        self.assertNotIn("Reporting only", source)
        self.assertIn("Publication gate", source)

    def test_iota_knobs_reject_nonpositive_values(self):
        environment = os.environ.copy()
        environment.update({
            "MPLBACKEND": "Agg",
            "HWLOC_COMPONENTS": "-gl",
            "OMP_NUM_THREADS": "1",
        })
        for flag in ("--iota-penalty-weight", "--iota-scale"):
            completed = subprocess.run(
                [
                    sys.executable, str(DRIVER),
                    "--init-dir", "/tmp/missing-stage2",
                    "--iota-target", "0.05",
                    "--f-cp-threshold", "150000",
                    "--resolutions", "6",
                    "--fb-thresholds", "1e-4",
                    flag, "0",
                ],
                env=environment, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0, msg=flag)
            self.assertIn(f"{flag} must be finite and positive", completed.stderr)
            # The validator must fire before the expensive --init-dir load.
            self.assertNotIn("missing-stage2", completed.stderr)


if __name__ == "__main__":
    unittest.main()
