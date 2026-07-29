import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
SUBMIT = ROOT / "launchers" / "submit_bridge_cases.sh"
SBATCH = ROOT / "launchers" / "run_stage2_bridge.sbatch"
MAKE_CASES = ROOT / "launchers" / "make_cases.sh"


class BridgeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.stage2_root = self.work / "stage2"
        self.output_root = self.work / "output"
        self.capture = self.work / "capture.txt"
        self.python_bin = self.work / "python-bin"
        self.python_bin.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${CAPTURE}\"\n",
            encoding="utf-8",
        )
        self.python_bin.chmod(0o755)
        self.driver = self.work / "driver.py"
        self.driver.write_text("# launcher fixture\n", encoding="utf-8")

    def _seed(self, tf_a):
        seed = self.stage2_root / f"TF_a_{tf_a:.3f}"
        seed.mkdir(parents=True)
        for filename in ("bs_opt.json", "surf_opt.json", "results.json"):
            (seed / filename).write_text("{}\n", encoding="utf-8")

    def _environment(self):
        return {
            **os.environ,
            "PYTHON_BIN": str(self.python_bin),
            "DRIVER_PATH": str(self.driver),
            "STAGE2_ROOT": str(self.stage2_root),
            "OUTPUT_ROOT": str(self.output_root),
            "CAPTURE": str(self.capture),
        }

    def _run_submit(self, cases, mode="--dry-run", environment=None):
        cases_path = self.work / "cases.tsv"
        cases_path.write_text(cases, encoding="utf-8")
        return subprocess.run(
            [str(SUBMIT), mode],
            cwd=self.work,
            env={**self._environment(), "CASES": str(cases_path), **(environment or {})},
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_propagates_every_case_field(self):
        self._seed(0.5)
        result = self._run_submit(
            "0.500\t-1\t0.05\t140000\t8\t7\t25\t0.2\t0.31\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--field-polarity -1", result.stdout)
        self.assertIn("--iota-target 0.05", result.stdout)
        self.assertIn("--current-limit-amperes 140000", result.stdout)
        self.assertIn("--mpol 8 --ntor 7 --maxiter 25", result.stdout)
        self.assertIn("--outer-step-radius 0.2", result.stdout)
        self.assertIn("--volume-target 0.31", result.stdout)
        self.assertIn(f"--stage2-root {self.stage2_root}", result.stdout)
        self.assertIn(f"--output-root {self.output_root}", result.stdout)

    def test_malformed_column_counts_fail_preflight(self):
        self._seed(0.5)
        rows = {
            "short": "0.500\t-1\t0.05\t150000\t6\t6\t150\t0.15\n",
            "extra": "0.500\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\textra\n",
        }
        for name, row in rows.items():
            with self.subTest(name=name):
                result = self._run_submit(row)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("expected 9 tab-separated columns", result.stdout)

    def test_invalid_numeric_fields_fail_preflight(self):
        self._seed(0.5)
        rows = {
            "invalid_float": "e\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\n",
            "overflow_float": "1e999\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\n",
            "negative_limit": "0.500\t-1\t0.05\t-1\t6\t6\t150\t0.15\t-\n",
            "fractional_order": "0.500\t-1\t0.05\t150000\t6.5\t6\t150\t0.15\t-\n",
            "zero_iterations": "0.500\t-1\t0.05\t150000\t6\t6\t0\t0.15\t-\n",
            "nonfinite_volume": "0.500\t-1\t0.05\t150000\t6\t6\t150\t0.15\tnan\n",
        }
        for name, row in rows.items():
            with self.subTest(name=name):
                result = self._run_submit(row)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_local_mode_executes_the_validated_row(self):
        self._seed(0.5)
        result = self._run_submit(
            "0.500\t1\t-0.05\t150000\t6\t6\t2\t0.1\t-\n",
            mode="--local",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        invocation = self.capture.read_text(encoding="utf-8")
        self.assertIn("-u", invocation)
        self.assertIn(str(self.driver), invocation)
        self.assertIn("--field-polarity 1", invocation)
        self.assertIn("--iota-target -0.05", invocation)
        self.assertNotIn("--volume-target", invocation)

    def test_submit_snapshots_only_validated_rows_and_caps_concurrency(self):
        self._seed(0.5)
        self._seed(0.6)
        fake_bin = self.work / "bin"
        fake_bin.mkdir()
        sbatch_capture = self.work / "sbatch.txt"
        cases_capture = self.work / "submitted-cases.txt"
        sbatch = fake_bin / "sbatch"
        sbatch.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" > \"${SBATCH_CAPTURE}\"\n"
            "cp \"${CASES}\" \"${CASES_CAPTURE}\"\n",
            encoding="utf-8",
        )
        sbatch.chmod(0o755)
        result = self._run_submit(
            "# comment\n"
            "0.500\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\n"
            "\n"
            "  # indented comment\n"
            "0.600\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\n",
            mode="",
            environment={
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "SBATCH_CAPTURE": str(sbatch_capture),
                "CASES_CAPTURE": str(cases_capture),
                "MAX_CONCURRENT": "2",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--array=1-2%2", sbatch_capture.read_text(encoding="utf-8"))
        submitted_rows = cases_capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(submitted_rows), 2)
        self.assertTrue(all(len(row.split("\t")) == 9 for row in submitted_rows))

    def test_array_cap_and_missing_seed_fail_before_submission(self):
        self._seed(0.5)
        two_rows = (
            "0.500\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\n"
            "0.600\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\n"
        )
        capped = self._run_submit(two_rows, environment={"MAX_ARRAY": "1"})
        self.assertNotEqual(capped.returncode, 0)
        self.assertIn("refusing to queue 2 cases", capped.stdout)

        missing_seed = self._run_submit(two_rows)
        self.assertNotEqual(missing_seed.returncode, 0)
        self.assertIn("missing", missing_seed.stdout)
        self.assertIn("TF_a_0.600", missing_seed.stdout)

    def test_case_generator_emits_three_valid_rows(self):
        generated = self.work / "generated.tsv"
        result = subprocess.run(
            [str(MAKE_CASES), str(generated)],
            cwd=self.work,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [
            line
            for line in generated.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")
        ]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row.split("\t")) == 9 for row in rows))

    def test_slurm_worker_binds_its_24_thread_allocation(self):
        self._seed(0.5)
        cases = self.work / "validated.tsv"
        cases.write_text(
            "0.500\t-1\t0.05\t150000\t6\t6\t150\t0.15\t-\n",
            encoding="utf-8",
        )
        fake_bin = self.work / "bin"
        fake_bin.mkdir()
        srun_capture = self.work / "srun.txt"
        srun = fake_bin / "srun"
        srun.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s|%s|%s|%s\\n' "
            "\"${OMP_NUM_THREADS}\" \"${MKL_NUM_THREADS}\" "
            "\"${OPENBLAS_NUM_THREADS}\" \"$*\" > \"${SRUN_CAPTURE}\"\n",
            encoding="utf-8",
        )
        srun.chmod(0o755)
        environment = {
            **self._environment(),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CASES": str(cases),
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_CPUS_PER_TASK": "24",
            "SRUN_CAPTURE": str(srun_capture),
        }

        result = subprocess.run(
            ["bash", str(SBATCH)],
            cwd=self.work,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        capture = srun_capture.read_text(encoding="utf-8")
        self.assertTrue(capture.startswith("24|24|24|"))
        self.assertIn("--cpu-bind=cores", capture)
        self.assertIn("--field-polarity -1", capture)


if __name__ == "__main__":
    unittest.main()
