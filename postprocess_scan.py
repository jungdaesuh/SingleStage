#!/usr/bin/env python3
"""
Postprocess single-stage dipole scan outputs.

Loads all runs from a scan directory, filters by Boozer residual and
successful iterations, and produces plots of max dipole current vs. iota
vs. quasisymmetry error (non-QS ratio).

Usage:
  python postprocess_scan.py --scan-dir ../single_stage_scans --max-boozer-residual 1e-3
  python postprocess_scan.py --scan-dir /path/to/single_stage_scans --max-boozer-residual 5e-4 --out-dir ./scan_plots
"""

import argparse
import json
from pathlib import Path

import re

import numpy as np
import matplotlib.pyplot as plt


def infer_init_id(init_dir: str, _depth: int = 0):
    """
    Infer init_id from an init_dir path recorded in results.json.

    Supports multiple stage-2 naming schemes, and includes a fallback:
    if the init_dir basename doesn't start with an integer init_id, we try
    reading <init_dir>/results.json and inferring again from its init_dir.
    """
    if not isinstance(init_dir, str) or not init_dir:
        return None
    if _depth > 6:
        return None

    # Stage-2 short scheme:
    #   ".../<init_id>_rest_of_name"
    #
    # We only accept 1-3 digit init ids (e.g. 59, 76, 79, 160, 188).
    # Longer numeric prefixes (e.g. 260316_...) correspond to the new
    # higher-res single-stage naming and should *not* be treated as the
    # stage-2 init_id.
    init_basename = Path(init_dir).name
    first_token = init_basename.split("_", 1)[0]
    if first_token.isdigit() and len(first_token) <= 3:
        return first_token

    # Fallback: try results.json inside the init_dir directory.
    candidate = Path(init_dir) / "results.json"
    if candidate.is_file():
        try:
            with open(candidate, "r") as f:
                results = json.load(f)
            init_dir2 = results.get("init_dir")
            graph_block = results.get("graph", {})
            if init_dir2 is None and isinstance(graph_block, dict):
                init_dir2 = graph_block.get("init_dir")
                if init_dir2 is None:
                    inner_graph = graph_block.get("graph", {})
                    if isinstance(inner_graph, dict):
                        init_dir2 = inner_graph.get("init_dir")
            if isinstance(init_dir2, str):
                return infer_init_id(init_dir2, _depth=_depth + 1)
        except (json.JSONDecodeError, OSError):
            return None

    return None


def find_run_dirs(scan_root: Path):
    """Yield (run_dir_path, eq_name) for every run directory (mpol*_ntor*)."""
    scan_root = Path(scan_root).resolve()
    if not scan_root.is_dir():
        return
    for eq_dir in scan_root.iterdir():
        if not eq_dir.is_dir():
            continue
        eq_name = eq_dir.name
        for run_dir in eq_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # Look for mpol*_ntor* subdirs (typically exactly one per run)
            for res_dir in run_dir.iterdir():
                if not res_dir.is_dir():
                    continue
                if not res_dir.name.startswith("mpol"):
                    continue
                yield res_dir, eq_name
                break


def load_last_iteration(run_dir: Path):
    """
    Load the last iteration record from iterations.json for a run directory.

    Returns the last element of the \"iterations\" list, or None if the file
    is missing, unreadable, or contains no iterations.
    """
    iter_path = run_dir / "iterations.json"
    if not iter_path.is_file():
        return None
    try:
        with open(iter_path, "r") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    iters = history.get("iterations", [])
    if not iters:
        return None
    return iters[-1]


def load_scan(scan_root: Path, max_boozer_residual: float):
    """
    Load and filter all runs under scan_root, using the final entry in
    iterations.json for each run (including runs that were terminated
    before writing results.json).

    Returns
    -------
    records : list[dict]
        Runs that passed all filters.
    stats : dict
        Summary counts for total runs discovered and reasons for rejection.
    """
    records = []
    stats = {
        "total_dirs": 0,
        # Rejection reasons (i.e., not included in records):
        "rejected_no_iter": 0,
        "rejected_missing_required_fields": 0,
        "rejected_boozer_residual": 0,
        # Bookkeeping / diagnostics:
        "loaded_records": 0,
        "loaded_unknown_init_id": 0,
        "loaded_iter_count_unknown": 0,
        "loaded_iter_count_eq_1": 0,
        "loaded_iter_count_gt_1": 0,
        # Cross-tabs for reconciling per-init summaries:
        "loaded_gt_1_unknown_init_id": 0,
        "loaded_gt_1_known_init_id": 0,
    }

    for run_dir, eq_name in find_run_dirs(scan_root):
        stats["total_dirs"] += 1

        last = load_last_iteration(run_dir)
        if last is None:
            stats["rejected_no_iter"] += 1
            continue

        boozer = last.get("J_Boozer")
        nonqs = last.get("J_nonQS")
        iota_val = last.get("iota")
        max_current = last.get("max_current")

        # Count how many successful iterations were recorded for this run.
        n_iterations = None
        iter_path = run_dir / "iterations.json"
        try:
            with open(iter_path, "r") as f:
                history = json.load(f)
            iters = history.get("iterations", [])
            n_iterations = len(iters) if isinstance(iters, list) else None
        except (json.JSONDecodeError, OSError):
            n_iterations = None

        # Try to infer an initial-condition identifier from results.json, if present.
        # We store init_id as a string to avoid mixed-type issues (int vs str).
        init_id = None
        results_path = run_dir / "results.json"
        if results_path.is_file():
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                # Different runs may save init_dir in different nesting layouts.
                # We've seen both:
                #   results["graph"]["init_dir"]
                # and
                #   results["graph"]["graph"]["init_dir"]
                init_dir = results.get("init_dir")
                graph_block = results.get("graph", {})
                if init_dir is None and isinstance(graph_block, dict):
                    init_dir = graph_block.get("init_dir")
                    if init_dir is None:
                        inner_graph = graph_block.get("graph", {})
                        if isinstance(inner_graph, dict):
                            init_dir = inner_graph.get("init_dir")
                if isinstance(init_dir, str):
                    init_id = infer_init_id(init_dir)
            except (json.JSONDecodeError, OSError):
                init_id = None

        # Require essential fields from the final iteration record.
        if (
            boozer is None
            or nonqs is None
            or iota_val is None
            or max_current is None
        ):
            stats["rejected_missing_required_fields"] += 1
            continue

        # Filter by Boozer residual threshold.
        if boozer > max_boozer_residual:
            stats["rejected_boozer_residual"] += 1
            continue

        records.append(
            {
                "run_dir": str(run_dir),
                "eq_name": eq_name,
                "iota": iota_val,
                "max_current": max_current,
                "nonQS_ratio": nonqs,
                "boozer_residual": boozer,
                "init_id": init_id,
                "n_iterations": n_iterations,
                # For completeness; not used in plotting:
                "final_objective": last.get("J"),
                "optimization_success": None,
            }
        )

    # Drop any record with missing essential fields (defensive)
    records = [
        r
        for r in records
        if r["iota"] is not None
        and r["max_current"] is not None
        and r["nonQS_ratio"] is not None
    ]

    stats["loaded_records"] = len(records)
    for r in records:
        iid = r.get("init_id")
        if iid is None:
            stats["loaded_unknown_init_id"] += 1
        n_it = r.get("n_iterations")
        if n_it is None:
            stats["loaded_iter_count_unknown"] += 1
        elif n_it <= 1:
            stats["loaded_iter_count_eq_1"] += 1
        else:
            stats["loaded_iter_count_gt_1"] += 1
            if iid is None:
                stats["loaded_gt_1_unknown_init_id"] += 1
            else:
                stats["loaded_gt_1_known_init_id"] += 1

    return records, stats

def plot_scan(records: list, out_dir: Path):
    """Create plots of max dipole current vs. iota vs. quasisymmetry error.

    This function respects an optional global iota threshold (IOTA_MIN_FOR_PLOTS)
    if it is defined above main(); any record with iota below that threshold
    will be dropped before plotting. This is intended to declutter plots by
    removing clearly suboptimal low-iota runs.
    """
    if not records:
        print("No records to plot.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional global iota threshold for plotting only.
    try:
        from __main__ import IOTA_MIN_FOR_PLOTS  # type: ignore[attr-defined]
    except ImportError:
        IOTA_MIN_FOR_PLOTS = None

    # Convert to arrays / lists
    iota = np.array([r["iota"] for r in records])
    max_current = np.array([r["max_current"] for r in records]) / 1e3  # kA
    qs_err = np.array([r["nonQS_ratio"] for r in records])
    init_ids = np.array([r.get("init_id") for r in records], dtype=object)
    run_dirs = np.array([r["run_dir"] for r in records], dtype=object)
    eq_names = np.array([r["eq_name"] for r in records], dtype=object)
    n_iters = np.array([r.get("n_iterations") for r in records], dtype=object)

    # Apply iota cut for plotting if requested.
    if IOTA_MIN_FOR_PLOTS is not None:
        mask = iota >= IOTA_MIN_FOR_PLOTS
        iota = iota[mask]
        max_current = max_current[mask]
        qs_err = qs_err[mask]
        init_ids = init_ids[mask].tolist()
        run_dirs = run_dirs[mask].tolist()
        eq_names = eq_names[mask].tolist()
        n_iters = n_iters[mask].tolist()
    else:
        init_ids = init_ids.tolist()
        run_dirs = run_dirs.tolist()
        eq_names = eq_names.tolist()
        n_iters = n_iters.tolist()

    # For plots: drop any run with unknown init_id entirely, so the Pareto
    # overlay and scatter series can't include non-plotted points.
    # However, if dropping would leave nothing (e.g. for midres scans where
    # init_dir doesn't encode the stage-2 init index), fall back to plotting
    # with unknown init runs.
    known_mask = np.array([iid is not None for iid in init_ids], dtype=bool)
    n_unknown = int(len(init_ids) - known_mask.sum())

    # Snapshot arrays in case we need to fall back.
    iota_all = iota
    max_current_all = max_current
    qs_err_all = qs_err
    init_ids_all = init_ids
    run_dirs_all = run_dirs
    eq_names_all = eq_names
    n_iters_all = n_iters

    if known_mask.any():
        if n_unknown > 0:
            print(f"Dropping {n_unknown} runs with unknown init_id from plots.")
        iota = iota[known_mask]
        max_current = max_current[known_mask]
        qs_err = qs_err[known_mask]
        init_ids = [iid for iid, keep in zip(init_ids, known_mask) if keep]
        run_dirs = [d for d, keep in zip(run_dirs, known_mask) if keep]
        eq_names = [e for e, keep in zip(eq_names, known_mask) if keep]
        n_iters = [ni for ni, keep in zip(n_iters, known_mask) if keep]
    else:
        print(
            "Warning: all runs have unknown init_id; "
            "keeping them for plots so you can still review Pareto front."
        )
        # leave arrays unchanged

    # Report how many successful runs (after Boozer filtering) had more than one
    # iteration, grouped by initial condition.
    if any(i is not None for i in init_ids):
        print("Successful runs with >1 iteration per initial condition:")
        counts_by_init = {}
        for iid, n_it in zip(init_ids, n_iters):
            if iid is None or n_it is None or n_it <= 1:
                continue
            counts_by_init[iid] = counts_by_init.get(iid, 0) + 1
        for iid in sorted(counts_by_init):
            print(f"  init {iid}: {counts_by_init[iid]} runs")

    # Unique initial-condition identifiers (excluding None).
    # All init_ids are strings (or None). Keep deterministic ordering.
    unique_inits = sorted({iid for iid in init_ids if iid is not None})
    has_unknown_init = any(iid is None for iid in init_ids)

    plot_inits = list(unique_inits)
    if has_unknown_init:
        plot_inits.append(None)
    if not plot_inits:
        plot_inits = [None]

    # Cycle through a list of marker styles for different initial conditions.
    marker_styles = ["o", "s", "1", "P", "+"]

    # Convenience function to get marker for a given init_id.
    def marker_for_init(iid, default="H"):
        if iid is None or not unique_inits:
            return default
        idx = unique_inits.index(iid) % len(marker_styles)
        return marker_styles[idx]

    # ---- Pareto front for (iota, max_current) ----
    # We treat this as: maximize iota, minimize max_current.
    if len(iota) > 1:
        # Sort by decreasing iota so we can sweep and keep the lowest current seen.
        order_desc_iota = np.argsort(-iota)
        best_current = np.inf
        pareto_idx = []
        for idx in order_desc_iota:
            c = max_current[idx]
            if c <= best_current:
                pareto_idx.append(idx)
                best_current = c
        pareto_idx = np.array(pareto_idx, dtype=int)

        # Sort Pareto points by increasing iota for plotting a clean line.
        pareto_sorted = pareto_idx[np.argsort(iota[pareto_idx])]

        print("Pareto front for iota vs max current (maximize iota, minimize current):")
        for idx in pareto_sorted:
            # Extract run_id as the first token of the parent directory name of run_dir,
            # which is typically "<run_id>_.../mpol_ntor".
            parent_name = Path(run_dirs[idx]).parent.name
            run_id_token = parent_name.split("_iota_tar", 1)[0]
            init_str = (
                f", init_id={init_ids[idx]}"
                if init_ids[idx] is not None
                else ", init_id=unknown"
            )
            print(
                f"  eq={eq_names[idx]}, run_id={run_id_token}"
                f"{init_str}, max_current_kA={max_current[idx]:.0f}, iota={iota[idx]:.3f}"
            )
    else:
        pareto_sorted = np.arange(len(iota))

    # ---- 3D scatter ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    vmin_qs, vmax_qs = qs_err.min(), qs_err.max()
    sc = None
    # Plot one scatter per initial condition so markers differ.
    for idx, iid in enumerate(plot_inits):
        mask = np.array([j == iid for j in init_ids])
        if not mask.any():
            continue
        m = marker_for_init(iid)
        sc_i = ax.scatter(
            iota[mask],
            max_current[mask],
            qs_err[mask],
            c=qs_err[mask],
            cmap="winter",
            s=50,
            alpha=0.8,
            vmin=vmin_qs,
            vmax=vmax_qs,
            marker=m,
            label=f"init {iid}" if iid is not None else "init unknown",
        )
        if sc is None:
            sc = sc_i
    # If there were no non-None init_ids, ensure we still have a scatter for colorbar.
    if sc is None:
        sc = ax.scatter(
            iota,
            max_current,
            qs_err,
            c=qs_err,
            cmap="winter",
            s=50,
            alpha=0.8,
        )
    ax.set_xlabel("ι (iota)")
    ax.set_ylabel("Max dipole current (kA)")
    ax.set_zlabel("Quasisymmetry error (non-QS ratio)")
    plt.colorbar(sc, ax=ax, label="Quasisymmetry error")
    if unique_inits or has_unknown_init:
        ax.legend(title="Initial condition", loc="best")
    ax.set_title("Scan: iota vs max current vs QS error")
    plt.tight_layout()
    fig.savefig(out_dir / "scan_3d_iota_current_qs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2D: iota vs max current, colored by QS error ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = None
    for idx, iid in enumerate(plot_inits):
        mask = np.array([j == iid for j in init_ids])
        if not mask.any():
            continue
        m = marker_for_init(iid)
        sc_i = ax.scatter(
            iota[mask],
            max_current[mask],
            c=qs_err[mask],
            cmap="winter",
            s=50,
            alpha=0.85,
            vmin=vmin_qs,
            vmax=vmax_qs,
            marker=m,
            label=f"init {iid}" if iid is not None else "init unknown",
        )
        if sc is None:
            sc = sc_i
    if sc is None:
        sc = ax.scatter(
            iota, max_current, c=qs_err, cmap="winter", s=50, alpha=0.85
        )
    ax.set_xlabel("ι (iota)")
    ax.set_ylabel("Max dipole current (kA)")
    plt.colorbar(sc, ax=ax, label="Quasisymmetry error (non-QS ratio)")

    # Overlay Pareto front as a line.
    if len(pareto_sorted) > 1:
        ax.plot(
            iota[pareto_sorted],
            max_current[pareto_sorted],
            color="k",
            linewidth=2.0,
            label="Pareto front",
        )

    if unique_inits or has_unknown_init or len(pareto_sorted) > 1:
        ax.legend(title="Initial condition", loc="best")
    ax.set_title("Max dipole current vs iota (color = QS error)")
    plt.tight_layout()
    fig.savefig(out_dir / "scan_iota_vs_current.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2D: iota vs QS error, colored by max current ----
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin_curr, vmax_curr = max_current.min(), max_current.max()
    sc = None
    for idx, iid in enumerate(plot_inits):
        mask = np.array([j == iid for j in init_ids])
        if not mask.any():
            continue
        m = marker_for_init(iid)
        sc_i = ax.scatter(
            iota[mask],
            qs_err[mask],
            c=max_current[mask],
            cmap="winter",
            s=50,
            alpha=0.85,
            vmin=vmin_curr,
            vmax=vmax_curr,
            marker=m,
            label=f"init {iid}" if iid is not None else "init unknown",
        )
        if sc is None:
            sc = sc_i
    if sc is None:
        sc = ax.scatter(
            iota, qs_err, c=max_current, cmap="winter", s=50, alpha=0.85
        )
    ax.set_xlabel("ι (iota)")
    ax.set_ylabel("Quasisymmetry error (non-QS ratio)")
    plt.colorbar(sc, ax=ax, label="Max dipole current (kA)")
    if unique_inits or has_unknown_init:
        ax.legend(title="Initial condition", loc="best")
    ax.set_title("QS error vs iota (color = max current)")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(out_dir / "scan_iota_vs_qs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2D: max current vs QS error, colored by iota ----
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin_iota, vmax_iota = iota.min(), iota.max()
    sc = None
    for idx, iid in enumerate(plot_inits):
        mask = np.array([j == iid for j in init_ids])
        if not mask.any():
            continue
        m = marker_for_init(iid)
        sc_i = ax.scatter(
            max_current[mask],
            qs_err[mask],
            c=iota[mask],
            cmap="winter",
            s=50,
            alpha=0.85,
            vmin=vmin_iota,
            vmax=vmax_iota,
            marker=m,
            label=f"init {iid}" if iid is not None else "init unknown",
        )
        if sc is None:
            sc = sc_i
    if sc is None:
        sc = ax.scatter(
            max_current, qs_err, c=iota, cmap="winter", s=50, alpha=0.85
        )
    ax.set_xlabel("Max dipole current (kA)")
    ax.set_ylabel("Quasisymmetry error (non-QS ratio)")
    plt.colorbar(sc, ax=ax, label="ι (iota)")
    if unique_inits or has_unknown_init:
        ax.legend(title="Initial condition", loc="best")
    ax.set_title("QS error vs max current (color = iota)")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(out_dir / "scan_current_vs_qs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess single-stage scan: filter runs and plot iota vs current vs QS error.",
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default="../single_stage_scans",
        help="Root directory containing scan outputs (eq subdirs with run/mpol_ntor/iterations.json).",
    )
    parser.add_argument(
        "--max-boozer-residual",
        type=float,
        default=1e-3,
        help="Exclude runs with Boozer residual above this value (default: 1e-3).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for plot outputs (default: <scan-dir>/postprocess_plots).",
    )

    parser.add_argument(
        "--min-iota",
        type=float,
        default=None,
        help="If set, exclude runs with final iota below this value from plots.",
    )
    args = parser.parse_args()

    scan_root = Path(args.scan_dir).resolve()
    if not scan_root.is_dir():
        raise SystemExit(f"Scan directory not found: {scan_root}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = scan_root / "postprocess_plots"
    out_dir = Path(out_dir).resolve()

    print(f"Scan root: {scan_root}")
    print(f"Max Boozer residual: {args.max_boozer_residual}")
    if args.min_iota is not None:
        print(f"Min iota for plots: {args.min_iota}")
    records, stats = load_scan(scan_root, args.max_boozer_residual)

    accounted = (
        stats["loaded_records"]
        + stats["rejected_no_iter"]
        + stats["rejected_missing_required_fields"]
        + stats["rejected_boozer_residual"]
    )
    print(
        "Run-folder summary:\n"
        f"  Total run folders: {stats['total_dirs']}\n"
        f"  Loaded (passed filters): {stats['loaded_records']}\n"
        "  Rejected:\n"
        f"    - No successful iteration (missing/unreadable/empty iterations.json): {stats['rejected_no_iter']}\n"
        f"    - Missing required fields in final iteration record: {stats['rejected_missing_required_fields']}\n"
        f"    - Boozer residual > {args.max_boozer_residual:g}: {stats['rejected_boozer_residual']}\n"
        "  Loaded-run diagnostics:\n"
        f"    - init_id unknown (missing/unparseable results.json graph.init_dir): {stats['loaded_unknown_init_id']}\n"
        f"    - iteration count unknown: {stats['loaded_iter_count_unknown']}\n"
        f"    - runs with n_iterations <= 1: {stats['loaded_iter_count_eq_1']}\n"
        f"    - runs with n_iterations > 1: {stats['loaded_iter_count_gt_1']}\n"
        "  Reconciling the per-init (>1 iteration) table:\n"
        f"    - runs with n_iterations > 1 AND init_id known: {stats['loaded_gt_1_known_init_id']}\n"
        f"    - runs with n_iterations > 1 AND init_id unknown: {stats['loaded_gt_1_unknown_init_id']}\n"
    )
    if accounted != stats["total_dirs"]:
        print(
            f"Warning: accounting mismatch (accounted {accounted} folders "
            f"vs total {stats['total_dirs']})."
        )

    if records:
        # Expose optional plotting threshold as a global so plot_scan can read it.
        global IOTA_MIN_FOR_PLOTS
        IOTA_MIN_FOR_PLOTS = args.min_iota
        plot_scan(records, out_dir)
        print(f"Plots written to {out_dir}")
    else:
        print("No runs passed filters; nothing to plot.")

if __name__ == "__main__":
    main()