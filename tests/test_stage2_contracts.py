import json
from hashlib import sha256
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simsopt.configs import get_data

from bounded_bfgs import rejected_trial_value_and_gradient
from simsopt.geo import BoozerSurface, SurfaceRZFourier, SurfaceXYZTensorFourier, Volume

from stage2_contracts import (
    BoozerCertificate,
    BoozerCertificateThresholds,
    BoozerSolveError,
    PhaseJournal,
    MU0,
    apply_field_polarity,
    build_boozer_solve_surface,
    check_self_intersection_angles,
    certify_boozer_solution,
    create_phase_journal,
    derive_signed_G,
    normalize_polarity,
    publish_boozer_artifact,
    restore_accepted_boozer_state,
    set_boozer_constraint_weight,
    require_boozer_certificate,
    require_valid_boozer_result,
    solve_boozer_initialization,
    solve_boozer_outer_trial,
    validate_checkpoint_manifest,
    prolong_boozer_surface,
)


class _Current:
    def __init__(self, value):
        self.value = float(value)

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = float(value)


class _ScaledCurrent:
    def __init__(self, root, scale):
        self.current_to_scale = root
        self.scale = float(scale)

    def get_value(self):
        return self.scale * self.current_to_scale.get_value()


class _Coil:
    def __init__(self, current):
        self.current = current


class _Surface:
    def __init__(self, self_intersecting=False):
        self.self_intersecting = self_intersecting

    def is_self_intersecting(self):
        return self.self_intersecting


class _RestorableSurface:
    def __init__(self):
        self.x = np.zeros(2)


class _RestorableBoozer:
    def __init__(self):
        self.surface = _RestorableSurface()
        self.res = {"iota": 0.0, "G": 0.0}
        self.need_to_run_code = True


class _RestorableObjective:
    def __init__(self, boozer):
        self.boozer = boozer
        self._x = np.zeros(2)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value).copy()
        self.boozer.need_to_run_code = True


class _UncertifiableSurface:
    def is_self_intersecting(self):
        raise RuntimeError("missing geometry dependency")


class _OptimizerInfo:
    nfev = 7


class _BoozerSolver:
    def __init__(self, *, bfgs_success=True):
        self.constraint_weight = 1.0
        self.need_to_run_code = True
        self.bfgs_success = bfgs_success
        self.calls = []
        self.surface = _Surface()

    def minimize_boozer_penalty_constraints_LBFGS(self, **kwargs):
        self.calls.append(("bfgs", kwargs))
        return {
            "success": self.bfgs_success,
            "iter": 5,
            "info": _OptimizerInfo(),
            "iota": 0.2,
            "G": 3.0,
        }

    def minimize_boozer_penalty_constraints_newton(self, **kwargs):
        self.calls.append(("newton", kwargs))
        return {"success": True, "iter": 2, "iota": 0.21, "G": 3.1}


class _SerializableBoozerSurface:
    def save(self, filename):
        Path(filename).write_text('{"surface": "test"}\n')


def _symmetric_coils(value):
    roots = [_Current(value) for _ in range(2)]
    positive = [_Coil(root) for root in roots]
    negative = [_Coil(_ScaledCurrent(root, -1.0)) for root in roots]
    return positive + negative, roots


class SignedGContractTests(unittest.TestCase):
    def test_positive_and_negative_G_follow_base_tf_current(self):
        for current in (8.0e4, -8.0e4):
            coils, _ = _symmetric_coils(current)
            actual = derive_signed_G(
                coils,
                ntf=2,
                nfp=1,
                stellsym=True,
            )
            self.assertEqual(actual, MU0 * 4 * current)

    def test_global_polarity_flip_mutates_each_shared_root_once(self):
        tf_coils, tf_roots = _symmetric_coils(8.0e4)
        wp_root = _Current(2.0e4)
        all_coils = tf_coils + [
            _Coil(wp_root),
            _Coil(_ScaledCurrent(wp_root, -1.0)),
        ]
        G = apply_field_polarity(
            all_coils,
            tf_coils,
            polarity=-1,
            ntf=2,
            nfp=1,
            stellsym=True,
        )
        self.assertLess(G, 0.0)
        self.assertEqual([root.get_value() for root in tf_roots], [-8.0e4, -8.0e4])
        self.assertEqual(wp_root.get_value(), -2.0e4)

    def test_mixed_base_tf_polarity_is_rejected(self):
        coils = [_Coil(_Current(1.0)), _Coil(_Current(-1.0))]
        with self.assertRaisesRegex(ValueError, "one physical polarity"):
            derive_signed_G(coils, ntf=2, nfp=1, stellsym=False)

    def test_invalid_polarity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"-1 or \+1"):
            normalize_polarity(0)


class BoozerResultContractTests(unittest.TestCase):
    def test_valid_result_is_returned(self):
        result = {"success": True, "G": -0.3, "iota": 0.04}
        self.assertIs(
            require_valid_boozer_result(
                result,
                _Surface(),
                expected_G_sign=-1,
            ),
            result,
        )

    def test_solver_failure_is_not_silently_accepted(self):
        with self.assertRaisesRegex(BoozerSolveError, "did not converge"):
            require_valid_boozer_result(
                {"success": False, "G": 0.3, "iota": 0.04},
                _Surface(),
                expected_G_sign=1,
            )

    def test_G_sign_change_is_rejected(self):
        with self.assertRaisesRegex(BoozerSolveError, "violated requested polarity"):
            require_valid_boozer_result(
                {"success": True, "G": 0.3, "iota": 0.04},
                _Surface(),
                expected_G_sign=-1,
            )

    def test_self_intersection_is_rejected(self):
        with self.assertRaisesRegex(BoozerSolveError, "self-intersecting"):
            require_valid_boozer_result(
                {"success": True, "G": 0.3, "iota": 0.04},
                _Surface(self_intersecting=True),
                expected_G_sign=1,
            )

    def test_self_intersection_check_failure_is_not_swallowed(self):
        with self.assertRaisesRegex(
            BoozerSolveError,
            "self-intersection certification failed",
        ):
            require_valid_boozer_result(
                {"success": True, "G": 0.3, "iota": 0.04},
                _UncertifiableSurface(),
                expected_G_sign=1,
            )

    def test_self_intersection_screen_scales_with_field_period(self):
        angles = []

        class AngleSurface:
            def is_self_intersecting(self, angle=0.0):
                angles.append(angle)
                return False

        checked = check_self_intersection_angles(AngleSurface(), nfp=4)
        np.testing.assert_allclose(
            checked,
            (0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0),
        )
        np.testing.assert_allclose(angles, checked)


class BoozerSolveGridTests(unittest.TestCase):
    def test_resolution_six_uses_oversampled_field_period_grid(self):
        source = SurfaceRZFourier(
            mpol=1,
            ntor=0,
            nfp=2,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 0.5, 32, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 32, endpoint=False),
        )
        source.set_rc(0, 0, 1.0)
        source.set_rc(1, 0, 0.2)
        source.set_zs(1, 0, 0.2)

        surface = build_boozer_solve_surface(source, mpol=6, ntor=6)

        # 6*order+1: critical (4x) sampling is exploitable by the outer
        # optimizer (ablation E5f); the solve grid must match the
        # certification refinement's density.
        self.assertEqual(len(surface.quadpoints_phi), 37)
        self.assertEqual(len(surface.quadpoints_theta), 37)
        self.assertLess(max(surface.quadpoints_phi), 1.0 / source.nfp)
        self.assertAlmostEqual(surface.volume(), source.volume(), places=10)

    def test_boozer_prolongation_preserves_pointwise_surface(self):
        source = SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            nfp=2,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 0.5, 9, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 9, endpoint=False),
        )
        source.x = np.linspace(-0.2, 0.2, source.x.size)
        target = prolong_boozer_surface(source, mpol=4, ntor=4)
        np.testing.assert_allclose(target.gamma(), source.gamma(), rtol=0.0, atol=1.0e-13)


class BoozerSolvePolicyTests(unittest.TestCase):
    def test_phase_journals_are_unique_per_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = create_phase_journal(temporary, TF_a=0.4)
            second = create_phase_journal(temporary, TF_a=0.4)

            self.assertNotEqual(first.path, second.path)
            self.assertTrue(first.path.exists())
            self.assertTrue(second.path.exists())

    def test_constraint_weight_change_invalidates_cached_solve(self):
        boozer = _RestorableBoozer()
        boozer.need_to_run_code = False

        set_boozer_constraint_weight(boozer, 100.0)

        self.assertEqual(boozer.constraint_weight, 100.0)
        self.assertTrue(boozer.need_to_run_code)

    def test_initialization_runs_bfgs_once_then_newton(self):
        solver = _BoozerSolver()

        result, reports = solve_boozer_initialization(solver, 0.1, 2.0)

        self.assertEqual([name for name, _ in solver.calls], ["bfgs", "newton"])
        self.assertEqual(result["iota"], 0.21)
        self.assertEqual([report.phase for report in reports], [
            "initial_bfgs",
            "initial_newton",
        ])
        self.assertEqual(reports[0].function_evaluations, 7)
        self.assertTrue(reports[0].success)

    def test_phase_journal_flushes_start_and_end_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = PhaseJournal(Path(temporary) / "phase_records.jsonl")
            solve_boozer_initialization(_BoozerSolver(), 0.1, 2.0, phase_journal=journal)
            rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
            self.assertEqual(
                [(row["phase"], row["event"]) for row in rows],
                [
                    ("initial_bfgs", "start"),
                    ("initial_bfgs", "end"),
                    ("initial_newton", "start"),
                    ("initial_newton", "end"),
                ],
            )

    def test_newton_can_recover_a_nonconverged_initial_bfgs(self):
        solver = _BoozerSolver(bfgs_success=False)

        result, reports = solve_boozer_initialization(solver, 0.1, 2.0)

        self.assertEqual([name for name, _ in solver.calls], ["bfgs", "newton"])
        self.assertFalse(reports[0].success)
        self.assertTrue(reports[1].success)
        self.assertEqual(result["iota"], 0.21)

    def test_outer_trials_use_newton_without_repeating_bfgs(self):
        solver = _BoozerSolver()

        for _ in range(3):
            result, report = solve_boozer_outer_trial(
                solver,
                0.2,
                3.0,
            )

            self.assertTrue(result["success"])
            self.assertEqual(report.phase, "outer_newton")

        self.assertEqual([name for name, _ in solver.calls], ["newton"] * 3)

    def test_failed_trial_penalty_has_its_matching_gradient(self):
        trial = np.asarray([1.25, -0.5])
        accepted = np.asarray([1.0, -1.0])
        accepted_J = 0.25
        value, gradient = rejected_trial_value_and_gradient(
            trial,
            accepted,
            accepted_J,
        )
        epsilon = 1.0e-6
        finite_difference = np.empty_like(trial)
        for index in range(len(trial)):
            step = np.zeros_like(trial)
            step[index] = epsilon
            plus, _ = rejected_trial_value_and_gradient(
                trial + step,
                accepted,
                accepted_J,
            )
            minus, _ = rejected_trial_value_and_gradient(
                trial - step,
                accepted,
                accepted_J,
            )
            finite_difference[index] = (plus - minus) / (2.0 * epsilon)

        self.assertGreater(value, accepted_J)
        np.testing.assert_allclose(gradient, finite_difference, rtol=1.0e-9)
        at_accepted_value, at_accepted_gradient = rejected_trial_value_and_gradient(
            accepted,
            accepted,
            accepted_J,
        )
        self.assertEqual(at_accepted_value, accepted_J)
        np.testing.assert_array_equal(at_accepted_gradient, np.zeros_like(accepted))

    def test_restored_accepted_state_does_not_schedule_implicit_run_code(self):
        boozer = _RestorableBoozer()
        objective = _RestorableObjective(boozer)

        restore_accepted_boozer_state(
            boozer,
            objective,
            optimizer_dofs=np.asarray([1.0, 2.0]),
            surface_dofs=np.asarray([3.0, 4.0]),
            iota=0.05,
            G=-3.0,
        )

        np.testing.assert_array_equal(objective.x, [1.0, 2.0])
        np.testing.assert_array_equal(boozer.surface.x, [3.0, 4.0])
        self.assertEqual(boozer.res, {"iota": 0.05, "G": -3.0})
        self.assertFalse(boozer.need_to_run_code)


class BoozerCertificateTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = BoozerCertificateThresholds(
            max_offgrid_residual=1.0e-4,
            max_relative_volume_error=1.0e-3,
            max_iota_error=2.5e-3,
            max_iota_refinement_delta=2.5e-3,
            max_relative_G_refinement_delta=1.0e-3,
        )

    def test_all_passing_publication_gates_are_accepted(self):
        certificate = BoozerCertificate(
            offgrid_residual=1.0e-5,
            relative_volume_error=1.0e-4,
            iota_target_error=1.0e-4,
            iota_refinement_delta=1.0e-4,
            relative_G_refinement_delta=1.0e-4,
            refinement_nphi=37,
            refinement_ntheta=37,
            certificate_nphi=49,
            certificate_ntheta=49,
            fourier_mpol=6,
            fourier_ntor=6,
            spectral_order_certified=False,
        )

        require_boozer_certificate(certificate, self.thresholds)

    def test_any_failed_publication_gate_is_rejected(self):
        certificate = BoozerCertificate(
            offgrid_residual=2.0e-4,
            relative_volume_error=1.0e-4,
            iota_target_error=1.0e-4,
            iota_refinement_delta=1.0e-4,
            relative_G_refinement_delta=1.0e-4,
            refinement_nphi=37,
            refinement_ntheta=37,
            certificate_nphi=49,
            certificate_ntheta=49,
            fourier_mpol=6,
            fourier_ntor=6,
            spectral_order_certified=False,
        )
        with self.assertRaisesRegex(BoozerSolveError, "off-grid Boozer residual"):
            require_boozer_certificate(certificate, self.thresholds)

    def test_nonfinite_threshold_cannot_disable_a_gate(self):
        certificate = BoozerCertificate(
            offgrid_residual=1.0e9,
            relative_volume_error=1.0e9,
            iota_target_error=1.0e9,
            iota_refinement_delta=1.0e9,
            relative_G_refinement_delta=1.0e9,
            refinement_nphi=37,
            refinement_ntheta=37,
            certificate_nphi=49,
            certificate_ntheta=49,
            fourier_mpol=6,
            fourier_ntor=6,
            spectral_order_certified=False,
        )
        invalid_thresholds = BoozerCertificateThresholds(
            max_offgrid_residual=float("nan"),
            max_relative_volume_error=float("nan"),
            max_iota_error=float("nan"),
            max_iota_refinement_delta=float("nan"),
            max_relative_G_refinement_delta=float("nan"),
        )

        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            require_boozer_certificate(certificate, invalid_thresholds)


class PublicationContractTests(unittest.TestCase):
    def test_surface_and_hash_bound_metadata_publish_together(self):
        with tempfile.TemporaryDirectory() as output_directory:
            surface_file, metadata_file, publication_digest = (
                publish_boozer_artifact(
                    _SerializableBoozerSurface(),
                    {"field_polarity": -1},
                    output_directory=output_directory,
                    TF_a=0.5,
                )
            )

            metadata = json.loads(metadata_file.read_text())
            surface_bytes = surface_file.read_bytes()
            metadata_bytes = metadata_file.read_bytes()
            repeated_surface, repeated_metadata, repeated_digest = (
                publish_boozer_artifact(
                    _SerializableBoozerSurface(),
                    {"field_polarity": -1},
                    output_directory=output_directory,
                    TF_a=0.5,
                )
            )
            published_directories = list(Path(output_directory).iterdir())

            self.assertEqual(
                metadata["surface_sha256"],
                sha256(surface_bytes).hexdigest(),
            )
            self.assertEqual(
                publication_digest,
                sha256(surface_bytes + metadata_bytes).hexdigest(),
            )
            self.assertTrue(
                published_directories[0].name.endswith(publication_digest[:16])
            )

        self.assertTrue(surface_file.name.endswith(".json"))
        self.assertEqual(metadata["field_polarity"], -1)
        self.assertEqual(len(metadata["surface_sha256"]), 64)
        self.assertEqual(len(publication_digest), 64)
        self.assertEqual(repeated_surface, surface_file)
        self.assertEqual(repeated_metadata, metadata_file)
        self.assertEqual(repeated_digest, publication_digest)
        self.assertEqual(len(published_directories), 1)
        self.assertFalse(published_directories[0].name.startswith(".stage2-staging-"))

    def test_checkpoint_manifest_rejects_mixed_or_incompatible_generation(self):
        with tempfile.TemporaryDirectory() as output_directory:
            root = Path(output_directory)
            generation = root / "checkpoints" / "generation-1"
            generation.mkdir(parents=True)
            files = {}
            for filename, content in {
                "bs_opt.json": b"bs",
                "surf_opt.json": b"surface",
                "results.json": b"results",
                "iterations.json": b"iterations",
            }.items():
                path = generation / filename
                path.write_bytes(content)
                files[filename] = sha256(content).hexdigest()
            manifest_path = root / "checkpoint_manifest.json"
            manifest_path.write_text(json.dumps({
                "generation": "generation-1",
                "field_polarity": -1,
                "files": files,
            }))
            validated = validate_checkpoint_manifest(
                manifest_path,
                expected={"field_polarity": -1},
            )
            self.assertIsNotNone(validated)
            manifest_path.write_text(json.dumps({
                "generation": "generation-1",
                "field_polarity": 1,
                "files": files,
            }))
            with self.assertRaisesRegex(ValueError, "policy mismatch"):
                validate_checkpoint_manifest(
                    manifest_path,
                    expected={"field_polarity": -1},
                )


class BoozerPipelineIntegrationTests(unittest.TestCase):
    def test_real_bfgs_seed_is_newton_polished_and_offgrid_failure_is_loud(self):
        _, base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
        surface = SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 9, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 9, endpoint=False),
        )
        surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
        G = MU0 * nfp * sum(abs(current.get_value()) for current in base_currents)
        boozer_surface = BoozerSurface(
            biotsavart,
            surface,
            Volume(surface),
            surface.volume(),
            1.0,
        )
        result, reports = solve_boozer_initialization(boozer_surface, -0.4, G)

        original_intersection_check = SurfaceXYZTensorFourier.is_self_intersecting
        SurfaceXYZTensorFourier.is_self_intersecting = lambda self, angle=0.0: False
        try:
            with self.assertRaisesRegex(
                BoozerSolveError,
                "off-grid Boozer residual",
            ):
                certify_boozer_solution(
                    boozer_surface,
                    biotsavart,
                    expected_G_sign=1,
                    iota_target=float(result["iota"]),
                )
        finally:
            SurfaceXYZTensorFourier.is_self_intersecting = (
                original_intersection_check
            )

        self.assertFalse(reports[0].success)
        self.assertTrue(reports[1].success)


class LauncherContractTests(unittest.TestCase):
    def test_launcher_passes_the_production_iota_target(self):
        launcher = Path(__file__).parents[1] / "stage2_to_single_stage.sh"
        environment = {
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_ID": "test-job",
            "PYTHON_BIN": "/bin/echo",
            "DRIVER_PATH": "driver.py",
        }
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--iota-target 0.05", result.stdout)

    def test_launcher_propagates_driver_failures(self):
        launcher = Path(__file__).parents[1] / "stage2_to_single_stage.sh"
        environment = {
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_ID": "test-job",
            "PYTHON_BIN": "/bin/false",
            "DRIVER_PATH": "unused.py",
        }
        with tempfile.TemporaryDirectory() as working_directory:
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Finished at", result.stdout)


if __name__ == "__main__":
    unittest.main()
