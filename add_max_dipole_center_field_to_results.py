#!/usr/bin/env python3
"""
Recursively update results.json files with dipole center-field metrics.

For each run directory under a user-specified root:
  1) load bs_opt.json (a BiotSavart object),
  2) identify dipole coils (exclude leading TF coils when count is available),
  3) for each dipole coil, evaluate self-field at its own coil center,
  4) compute max(|B|) over dipole centers,
  5) scale by the run's on-axis field, if available,
  6) write results back into results.json.

The script supports both "flat" results.json dictionaries and SIMSON-style
serialized dictionaries with payload in results["graph"].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from simsopt._core.optimizable import load
from simsopt.field import BiotSavart


MU0 = 4.0 * np.pi * 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find run directories under ROOT that contain both "
            "results.json and bs_opt.json, then add dipole max center-field metrics."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to recursively scan for results.json files.",
    )
    parser.add_argument(
        "--field-on-axis",
        type=float,
        default=None,
        help=(
            "Override field_on_axis [T] for all runs. If omitted, the script uses "
            "field_on_axis from results.json or from init_dir/results.json when available."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print values without modifying any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file details and warnings.",
    )
    return parser.parse_args()


def _payload(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return mutable payload dictionary where run-level fields are stored.
    """
    if isinstance(results.get("graph"), dict):
        return results["graph"]
    return results


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_field_on_axis(
    results: Dict[str, Any],
    run_dir: Path,
    cli_field_on_axis: Optional[float],
    verbose: bool = False,
) -> Optional[float]:
    if cli_field_on_axis is not None:
        return float(cli_field_on_axis)

    payload = _payload(results)

    # Try local keys first.
    for key in ("field_on_axis", "on_axis_field", "B_axis"):
        val = _as_float(payload.get(key))
        if val is not None:
            return val
        val = _as_float(results.get(key))
        if val is not None:
            return val

    # Try init_dir/results.json for single-stage runs.
    init_dir = payload.get("init_dir")
    if isinstance(init_dir, str):
        init_path = (run_dir / init_dir).resolve()
        init_results = init_path / "results.json"
        if init_results.is_file():
            try:
                with init_results.open("r") as f:
                    init_data = json.load(f)
                init_payload = _payload(init_data)
                for key in ("field_on_axis", "on_axis_field", "B_axis"):
                    val = _as_float(init_payload.get(key))
                    if val is not None:
                        return val
                    val = _as_float(init_data.get(key))
                    if val is not None:
                        return val
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if verbose:
                    print(f"[WARN] Could not read init_dir results for {run_dir}: {exc}")

    return None


def _tf_count(results: Dict[str, Any]) -> int:
    payload = _payload(results)
    for key in ("# TF coils", "num_tf_coils"):
        val = payload.get(key)
        if val is None:
            val = results.get(key)
        try:
            return int(val)
        except (TypeError, ValueError):
            continue

    # Fallback using ntf and nfp when explicit count is missing.
    ntf = payload.get("ntf", results.get("ntf"))
    nfp = payload.get("surf_nfp", results.get("surf_nfp"))
    try:
        return 2 * int(ntf) * int(nfp)
    except (TypeError, ValueError):
        return 0


def _coil_center_from_dofs(coil: Any) -> np.ndarray:
    curve = coil.curve
    if hasattr(curve, "get") and hasattr(curve, "dof_names"):
        dof_names = set(curve.dof_names)
        if {"X", "Y", "Z"}.issubset(dof_names):
            return np.array([curve.get("X"), curve.get("Y"), curve.get("Z")], dtype=float)
    # Geometric fallback
    return np.mean(curve.gamma(), axis=0)


def _max_self_field_at_dipole_centers(
    bs_obj: Any,
    tf_count: int,
) -> Tuple[float, List[float]]:
    if not hasattr(bs_obj, "coils"):
        raise AttributeError("Loaded bs_opt.json object has no 'coils' attribute.")
    coils = list(bs_obj.coils)
    if len(coils) == 0:
        raise ValueError("No coils found in loaded BiotSavart object.")
    if tf_count < 0:
        tf_count = 0
    if tf_count >= len(coils):
        raise ValueError(
            f"TF coil count ({tf_count}) leaves no dipole coils out of total {len(coils)}."
        )

    dipole_coils = coils[tf_count:]
    center_field_magnitudes: List[float] = []

    for coil in dipole_coils:
        center = _coil_center_from_dofs(coil).reshape((1, 3))
        # Self-field only at this coil center.
        bs_self = BiotSavart([coil])
        bs_self.set_points(center)
        b_vec = bs_self.B().reshape((3,))
        center_field_magnitudes.append(float(np.linalg.norm(b_vec)))

    return float(np.max(center_field_magnitudes)), center_field_magnitudes


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    for results_path in root.rglob("results.json"):
        run_dir = results_path.parent
        if (run_dir / "bs_opt.json").is_file():
            yield run_dir


def update_one_run(
    run_dir: Path,
    cli_field_on_axis: Optional[float],
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[bool, str]:
    results_path = run_dir / "results.json"
    bs_path = run_dir / "bs_opt.json"

    try:
        with results_path.open("r") as f:
            results = json.load(f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"Failed reading results.json: {exc}"

    try:
        bs = load(str(bs_path))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"Failed loading bs_opt.json: {exc}"

    tf_count = _tf_count(results)
    try:
        max_center_field, all_center_fields = _max_self_field_at_dipole_centers(bs, tf_count)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"Failed computing center fields: {exc}"

    field_on_axis = _resolve_field_on_axis(results, run_dir, cli_field_on_axis, verbose=verbose)
    scaled = None if field_on_axis in (None, 0.0) else max_center_field / float(field_on_axis)

    payload = _payload(results)
    payload["max_dipole_center_field"] = max_center_field
    payload["max_dipole_center_field_over_on_axis"] = scaled
    payload["num_dipole_coils_used_for_center_field"] = len(all_center_fields)
    payload["center_field_metric_note"] = (
        "max over dipole coils of |B_self(center)| using BiotSavart([coil])"
    )

    if not dry_run:
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")

    if scaled is None:
        if field_on_axis == 0.0:
            msg = (
                f"updated max_dipole_center_field={max_center_field:.6e} T, "
                "could not scale (field_on_axis=0)"
            )
        else:
            msg = (
                f"updated max_dipole_center_field={max_center_field:.6e} T, "
                "could not scale (field_on_axis missing)"
            )
    else:
        msg = (
            f"updated max_dipole_center_field={max_center_field:.6e} T, "
            f"max_dipole_center_field_over_on_axis={scaled:.6e}"
        )
    return True, msg


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    run_dirs = sorted(set(_iter_run_dirs(root)))
    if len(run_dirs) == 0:
        print(f"No run directories with both results.json and bs_opt.json found under: {root}")
        return

    print(f"Found {len(run_dirs)} run directories under: {root}")
    updated = 0
    failed = 0

    for run_dir in run_dirs:
        ok, msg = update_one_run(
            run_dir,
            cli_field_on_axis=args.field_on_axis,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if ok:
            updated += 1
            if args.verbose or args.dry_run:
                print(f"[OK] {run_dir}: {msg}")
        else:
            failed += 1
            print(f"[FAIL] {run_dir}: {msg}")

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {updated} runs; {failed} failed.")


if __name__ == "__main__":
    main()

