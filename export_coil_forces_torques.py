#!/usr/bin/env python3
"""
For each run directory under a root that contains bs_opt.json:

  1) Load the optimized Biot-Savart state from bs_opt.json.
  2) Rebuild every coil as a CircularRegularizedCoil with separate radii for
     TF coils (leading segment of the coil list) and windowpane coils.
  3) Overwrite the same VTK as single-stage runs (default basename ``coils_opt`` →
     ``coils_opt.vtu`` via pyevtk): geometry matches the prior file; point data adds
     currents plus net/pointwise forces and torques from the regularized coils.
  4) Merge scalar summaries into results.json (max net / pointwise norms, etc.).

TF vs windowpane split uses the same conventions as add_max_dipole_center_field_to_results.py
(ntf, surf_nfp, or num_wps fallbacks).

With --no-sparse, any run directory whose path contains a folder name including
``_sparse`` or ``_sparsity`` is skipped (e.g. continuation stages tagged sparse).

With --vol-target, only runs under a path segment ``iota_tar*`` that encodes that
Boozer volume target are kept (same naming as single_stage_epsilon_constraint.py).
The default volume (0.3) corresponds to directories named ``iota_tar*`` with no
``_vol`` suffix; non-default targets use ``iota_tar*_vol<value>``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from simsopt._core.optimizable import load
from simsopt.field import CircularRegularizedCoil, coils_to_vtk

# Must match single_stage_epsilon_constraint.py default for path segments without _vol*.
_DEFAULT_VOL_TARGET = 0.3
_IOTA_VOL_DIR_RE = re.compile(r"^iota_tar([0-9.eE+-]+)(?:_vol([0-9.eE+-]+))?$")


def _coils_to_vtk_paths(run_dir: Path, vtk_basename: str) -> Tuple[str, Path]:
    """
    Return (filename_arg, resolved_output_file) for coils_to_vtk / pyevtk.

    pyevtk appends ``.vtu`` when the filename has no extension; geometry is the
    same as the dipole scripts' ``coils_opt`` export, with extra pointData.
    """
    rel = run_dir / vtk_basename
    arg = str(rel)
    if rel.suffix.lower() in (".vtu", ".vtk", ".vtp"):
        return arg, rel
    return arg, rel.with_suffix(".vtu")


def _payload(results: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(results.get("graph"), dict):
        return results["graph"]
    return results


def _merge_postprocess_keys(results: Dict[str, Any], prefix: str, data: Dict[str, Any]) -> None:
    """Write scalar postprocess entries at the top level and mirror into graph when present."""
    for k, v in data.items():
        key = f"{prefix}_{k}"
        results[key] = v
    graph = results.get("graph")
    if isinstance(graph, dict):
        for k, v in data.items():
            graph[f"{prefix}_{k}"] = v


def _tf_count(results: Dict[str, Any], n_coils_total: int) -> int:
    """Number of TF coils in the expanded coil list (same order as optimize.py: TF then WP)."""
    payload = _payload(results)
    for key in ("# TF coils", "num_tf_coils"):
        val = payload.get(key)
        if val is None:
            val = results.get(key)
        try:
            return int(val)
        except (TypeError, ValueError):
            continue

    ntf = payload.get("ntf", results.get("ntf"))
    nfp = payload.get("surf_nfp", results.get("surf_nfp"))
    try:
        return 2 * int(ntf) * int(nfp)
    except (TypeError, ValueError):
        pass

    num_wps = payload.get("num_wps", results.get("num_wps"))
    try:
        tfc = int(n_coils_total) - int(num_wps)
        if tfc >= 0:
            return tfc
    except (TypeError, ValueError):
        pass

    raise ValueError(
        "Could not determine TF coil count: need (# TF coils | num_tf_coils) or "
        "(ntf and surf_nfp) or num_wps consistent with total coil count."
    )


def _coils_as_circular_regularized(
    coils: List[Any],
    n_tf: int,
    tf_radius: float,
    wp_radius: float,
) -> List[CircularRegularizedCoil]:
    out: List[CircularRegularizedCoil] = []
    for i, c in enumerate(coils):
        a = tf_radius if i < n_tf else wp_radius
        out.append(CircularRegularizedCoil(c.curve, c.current, a))
    return out


def _mean_net_stats(coils: List[CircularRegularizedCoil]) -> Dict[str, Any]:
    """Per-coil means (not stored redundantly at every VTK point)."""
    net_f = np.array([c.net_force(coils) for c in coils])
    net_t = np.array([c.net_torque(coils) for c in coils])
    return {
        "mean_net_force_norm_N": float(np.mean(np.linalg.norm(net_f, axis=1))),
        "mean_net_torque_norm_Nm": float(np.mean(np.linalg.norm(net_t, axis=1))),
        "num_coils": len(coils),
    }


def _maxima_from_coils_to_vtk_pointdata(point_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maxima from the dict returned by coils_to_vtk (vector components as three arrays).
    Net fields are piecewise-constant along each coil; max |·| over points equals
    max over coils.
    """
    def _stack_vec(key: str) -> np.ndarray:
        tup = point_data[key]
        return np.column_stack([np.asarray(tup[0]), np.asarray(tup[1]), np.asarray(tup[2])])

    out: Dict[str, Any] = {}
    if "NetForces" in point_data:
        nf = _stack_vec("NetForces")
        out["max_net_force_norm_N"] = float(np.max(np.linalg.norm(nf, axis=1)))
    if "NetTorques" in point_data:
        nt = _stack_vec("NetTorques")
        out["max_net_torque_norm_Nm"] = float(np.max(np.linalg.norm(nt, axis=1)))
    if "Pointwise_Forces" in point_data:
        pf = _stack_vec("Pointwise_Forces")
        out["max_pointwise_force_norm_N_per_m"] = float(np.max(np.linalg.norm(pf, axis=1)))
    if "Pointwise_Torques" in point_data:
        pt = _stack_vec("Pointwise_Torques")
        out["max_pointwise_torque_norm_N"] = float(np.max(np.linalg.norm(pt, axis=1)))
    return out


def _path_has_sparse_marker(path: Path) -> bool:
    """True if any path component contains '_sparse' or '_sparsity'."""
    return any(
        ("_sparse" in part) or ("_sparsity" in part) for part in path.parts
    )


def _deepest_iota_tar_vol(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Find the deepest path segment matching iota_tar*.

    Returns:
        (False, None) if no such segment exists.
        (True, None) if a segment matches and has no _vol suffix (default-volume naming).
        (True, str) if a segment matches and has _vol<value> (group 2 string).
    """
    for part in reversed(path.parts):
        m = _IOTA_VOL_DIR_RE.match(part)
        if m:
            return True, m.group(2)
    return False, None


def _path_matches_vol_target(path: Path, vol_target: float) -> bool:
    """
    True if the run path matches the requested Boozer volume bucket.

    For vol_target equal to the default (0.3), only paths whose deepest iota_tar*
    segment has no _vol suffix match. For other targets, that segment must be
    iota_tar*_vol<vol_target>.

    Paths with no iota_tar* segment are kept (layout does not encode volume).
    """
    found, vol_suff = _deepest_iota_tar_vol(path)
    if not found:
        return True
    if np.isclose(vol_target, _DEFAULT_VOL_TARGET, rtol=0.0, atol=1e-12):
        return vol_suff is None
    if vol_suff is None:
        return False
    try:
        eff = float(vol_suff)
    except ValueError:
        return False
    return bool(np.isclose(eff, vol_target, rtol=0.0, atol=1e-12))


def _iter_run_dirs(
    root: Path,
    recursive: bool,
    skip_sparse: bool,
    vol_target: Optional[float],
) -> Iterable[Path]:
    if recursive:
        for p in root.rglob("bs_opt.json"):
            parent = p.parent
            if skip_sparse and _path_has_sparse_marker(parent):
                continue
            if vol_target is not None and not _path_matches_vol_target(parent, vol_target):
                continue
            yield parent
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "bs_opt.json").is_file():
                if skip_sparse and _path_has_sparse_marker(child):
                    continue
                if vol_target is not None and not _path_matches_vol_target(child, vol_target):
                    continue
                yield child


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root",
        type=Path,
        help="Root directory to scan (default: recursive search for bs_opt.json).",
    )
    p.add_argument(
        "--wp-coil-radius",
        type=float,
        default=0.025,
        help="Circular cross-section radius (m) for windowpane coils (default: 0.025).",
    )
    p.add_argument(
        "--tf-coil-radius",
        type=float,
        default=0.05,
        help="Circular cross-section radius (m) for TF coils (default: 0.05).",
    )
    p.add_argument(
        "--vtk-basename",
        type=str,
        default="coils_opt",
        help=(
            "Filename stem passed to coils_to_vtk under each run dir (default: coils_opt). "
            "Replaces the same coils_opt.vtu written by single-stage scripts; add .vtu/.vtk "
            "to force that extension."
        ),
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan immediate subdirectories of ROOT for bs_opt.json.",
    )
    p.add_argument(
        "--no-sparse",
        action="store_true",
        help="Skip run directories under any path component containing _sparse or _sparsity.",
    )
    p.add_argument(
        "--vol-target",
        type=float,
        default=None,
        metavar="V",
        help=(
            "Keep only runs whose path matches this Boozer volume target: "
            f"{_DEFAULT_VOL_TARGET:g} means iota_tar* with no _vol suffix; "
            "other values require .../iota_tar*_vol<V>/... . "
            "Paths without any iota_tar* segment are not excluded."
        ),
    )
    p.add_argument(
        "--no-close",
        dest="close",
        action="store_false",
        help="Pass close=False to coils_to_vtk (default: close=True).",
    )
    p.set_defaults(close=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics and print paths without writing VTK or results.json.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-directory messages.",
    )
    return p.parse_args()


def process_run_dir(
    run_dir: Path,
    wp_radius: float,
    tf_radius: float,
    vtk_basename: str,
    close: bool,
    dry_run: bool,
    verbose: bool,
) -> Tuple[bool, str]:
    results_path = run_dir / "results.json"
    bs_path = run_dir / "bs_opt.json"

    if not results_path.is_file():
        return False, "missing results.json"

    try:
        with results_path.open("r") as f:
            results = json.load(f)
    except Exception as exc:
        return False, f"failed reading results.json: {exc}"

    try:
        bs = load(str(bs_path))
    except Exception as exc:
        return False, f"failed loading bs_opt.json: {exc}"

    if not hasattr(bs, "coils"):
        return False, "loaded object has no .coils"

    coils_in = list(bs.coils)
    if not coils_in:
        return False, "no coils in BiotSavart"

    try:
        n_tf = _tf_count(results, len(coils_in))
    except ValueError as exc:
        return False, str(exc)

    if n_tf > len(coils_in):
        return False, f"TF count {n_tf} exceeds total coils {len(coils_in)}"

    coils = _coils_as_circular_regularized(coils_in, n_tf, tf_radius, wp_radius)
    vtk_arg, vtk_out_file = _coils_to_vtk_paths(run_dir, vtk_basename)

    if not dry_run:
        point_data = coils_to_vtk(coils, filename=vtk_arg, close=close)
        if verbose:
            keys = sorted(point_data.keys()) if isinstance(point_data, dict) else []
            print(f"  coils_to_vtk pointData keys: {keys}")
        stats = {**_maxima_from_coils_to_vtk_pointdata(point_data), **_mean_net_stats(coils)}
    else:
        if verbose:
            print(f"  [dry-run] would overwrite VTK {vtk_out_file}")
        # No VTK pointData in dry-run; compute maxima the same way as the VTK path.
        net_f = np.array([c.net_force(coils) for c in coils])
        net_t = np.array([c.net_torque(coils) for c in coils])
        pf_all = np.concatenate(
            [np.linalg.norm(c.force(coils), axis=1) for c in coils]
        )
        pt_all = np.concatenate(
            [np.linalg.norm(c.torque(coils), axis=1) for c in coils]
        )
        stats = {
            "max_net_force_norm_N": float(np.max(np.linalg.norm(net_f, axis=1))),
            "max_net_torque_norm_Nm": float(np.max(np.linalg.norm(net_t, axis=1))),
            "max_pointwise_force_norm_N_per_m": float(np.max(pf_all)) if pf_all.size else float("nan"),
            "max_pointwise_torque_norm_N": float(np.max(pt_all)) if pt_all.size else float("nan"),
            **_mean_net_stats(coils),
        }
    prefix = "coil_force_torque_postprocess"
    meta = {
        "vtk_basename": vtk_basename,
        "vtk_path": str(vtk_out_file),
        "wp_coil_radius_m": float(wp_radius),
        "tf_coil_radius_m": float(tf_radius),
        "n_tf_coils": int(n_tf),
        "coils_to_vtk_close": bool(close),
    }
    if not dry_run:
        _merge_postprocess_keys(results, prefix, {**meta, **stats})
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")

    msg = (
        f"max |F_net|={stats['max_net_force_norm_N']:.6e} N, "
        f"max |T_net|={stats['max_net_torque_norm_Nm']:.6e} N*m, "
        f"max pointwise |F|={stats['max_pointwise_force_norm_N_per_m']:.6e} N/m"
    )
    return True, msg


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    recursive = not args.no_recursive
    run_dirs = sorted(
        set(
            _iter_run_dirs(
                root,
                recursive,
                skip_sparse=args.no_sparse,
                vol_target=args.vol_target,
            )
        )
    )
    if not run_dirs:
        extra = ""
        if args.vol_target is not None:
            extra = f" (after --vol-target {args.vol_target:g} filter)"
        print(f"No bs_opt.json found under {root} (recursive={recursive}){extra}.")
        return

    vmsg = ""
    if args.vol_target is not None:
        vmsg = f", vol_target={args.vol_target:g}"
    print(
        f"Found {len(run_dirs)} run director(y/ies) with bs_opt.json under {root}"
        f" (no_sparse={args.no_sparse}{vmsg})"
    )
    ok = 0
    for run_dir in run_dirs:
        success, message = process_run_dir(
            run_dir,
            wp_radius=args.wp_coil_radius,
            tf_radius=args.tf_coil_radius,
            vtk_basename=args.vtk_basename,
            close=args.close,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        try:
            rel = run_dir.relative_to(root)
        except ValueError:
            rel = run_dir
        if success:
            ok += 1
            print(f"[OK] {rel}: {message}")
        else:
            print(f"[SKIP] {rel}: {message}")

    print(f"Done. Updated {ok} / {len(run_dirs)} runs.")


if __name__ == "__main__":
    main()
