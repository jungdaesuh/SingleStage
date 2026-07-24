#!/usr/bin/env python3
"""
Scatter plot of achieved iota vs max dipole current (or dipole center field)
from single-stage scan runs (sparse-aware version).

This script is intended for scan outputs under the epsilon-constraint updated tree:
  examples/single_stage_scans_epsilon_constraint_updated/<eq_or_init_dir>/iota_tar*/stage*/mpol*_ntor*/

It recognises both the new ``_sparse`` stage suffix (from --sparse in
single_stage_epsilon_constraint.py) and the legacy ``_sparsity`` /
``_sparsity_r0_source`` naming from single_stage_add_sparsity.py.

It loads the final record from each `iterations.json`, filters by a Boozer residual
threshold, then plots achieved iota vs a selected metric:
  - max dipole current (kA), or
  - max dipole center field (T) from results.json.

Run folders may be named ``iota_tarX`` (implicit default volume 0.3) or
``iota_tarX_volY`` when a non-default ``--vol-target`` was used in the optimization.

It saves two copies of the same scatter plot:
  1) colored by Boozer residual (J_Boozer)
  2) colored by QS error (J_nonQS)

By default (--marker-mode auto), the legend uses one of init_id, vol_target, or
current_weight, whichever first has multiple distinct values (that order).

Optional: --exclude-precision-loss omits runs from the plot when results.json
reports scipy's precision-loss termination (does not delete any files).

With --best-per-iota-init, bucketing is (iota, init) unless --vol-target is omitted
and multiple volumes appear in the scan; then bucketing is (iota, init, volume).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

# Must match single_stage_epsilon_constraint.py default for path segments without _vol*.
_DEFAULT_VOL_TARGET = 0.3

# Hard-coded on-axis field used to normalize dipole center field.
_FIELD_ON_AXIS_T = 0.5

# scipy.optimize.minimize (e.g. BFGS) may exit with this message in results.json.
_PRECISION_LOSS_SUBSTRING = "precision loss"


def optimization_message_from_results(run_dir: Path) -> str | None:
    """Return optimization_message from results.json if present."""
    path = run_dir / "results.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    graph = data.get("graph")
    if isinstance(graph, dict):
        msg = graph.get("optimization_message")
        if isinstance(msg, str):
            return msg
    msg = data.get("optimization_message")
    if isinstance(msg, str):
        return msg
    return None


def run_dir_failed_precision_loss(run_dir: Path) -> bool:
    """True if results.json reports scipy's precision-loss termination."""
    msg = optimization_message_from_results(run_dir)
    if msg is None:
        return False
    return _PRECISION_LOSS_SUBSTRING in msg.lower()


def max_dipole_center_field_from_results(run_dir: Path) -> float | None:
    """Return graph.max_dipole_center_field from results.json if present."""
    path = run_dir / "results.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    graph = data.get("graph")
    if isinstance(graph, dict):
        val = graph.get("max_dipole_center_field")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def parse_init_id_list(raw: str) -> set[int]:
    """Parse comma-separated init ids into a set of ints."""
    items = [x.strip() for x in raw.split(",")]
    out: set[int] = set()
    for item in items:
        if not item:
            continue
        out.add(int(item))
    if not out:
        raise ValueError("No valid init ids parsed.")
    return out


def find_run_dirs(scan_root: Path):
    """Yield run directories that look like `mpol*_ntor*` and contain iterations."""
    scan_root = Path(scan_root).resolve()
    if not scan_root.is_dir():
        return
    for run_dir in scan_root.rglob("mpol*_ntor*"):
        if not run_dir.is_dir():
            continue
        if (run_dir / "iterations.json").is_file():
            yield run_dir


def load_last_iteration(run_dir: Path):
    """Return the last iteration dict, or None if unavailable."""
    iter_path = run_dir / "iterations.json"
    if not iter_path.is_file():
        return None
    try:
        with open(iter_path, "r") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    iterations = history.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        return None
    return iterations[-1]


def load_scan(
    scan_root: Path,
    max_boozer_residual: float,
    sparse_mode: str,
    exclude_precision_loss: bool = False,
):
    """Load and filter final records for all runs under `scan_root`."""
    records = []
    stats = {
        "total_dirs": 0,
        "rejected_no_iter": 0,
        "rejected_missing_required_fields": 0,
        "rejected_boozer_residual": 0,
        "rejected_non_sparse": 0,
        "rejected_sparse": 0,
        "rejected_precision_loss": 0,
    }

    for run_dir in find_run_dirs(scan_root):
        stats["total_dirs"] += 1
        # Match stage folders containing "sparse" (catches both new _sparse and legacy _sparsity).
        is_sparse_path = any("sparse" in p for p in run_dir.parts)
        if sparse_mode == "sparse" and not is_sparse_path:
            stats["rejected_non_sparse"] += 1
            continue
        if sparse_mode == "dense" and is_sparse_path:
            stats["rejected_sparse"] += 1
            continue
        last = load_last_iteration(run_dir)
        if last is None:
            stats["rejected_no_iter"] += 1
            continue

        iota = last.get("iota")
        max_current = last.get("max_current")
        boozer = last.get("J_Boozer")
        nonqs = last.get("J_nonQS")
        if iota is None or boozer is None or nonqs is None:
            stats["rejected_missing_required_fields"] += 1
            continue

        if boozer > max_boozer_residual:
            stats["rejected_boozer_residual"] += 1
            continue

        if exclude_precision_loss and run_dir_failed_precision_loss(run_dir):
            stats["rejected_precision_loss"] += 1
            continue

        # Try to read CURRENT_WEIGHT from the top-level weights block, if present.
        current_weight = None
        weights = None
        try:
            with open(run_dir / "iterations.json", "r") as f:
                top = json.load(f)
            weights = top.get("weights")
        except (json.JSONDecodeError, OSError):
            weights = None
        if isinstance(weights, dict):
            cw = weights.get("CURRENT_WEIGHT")
            if isinstance(cw, (int, float)):
                current_weight = float(cw)

        # Target iota / volume from folder names like "iota_tar0.16" or "iota_tar0.16_vol0.35".
        iota_target = None
        vol_target = None
        _iota_vol_re = re.compile(r"iota_tar([\d.]+)(?:_vol([\d.]+))?$")
        for part in run_dir.parts:
            m = _iota_vol_re.fullmatch(part)
            if m:
                iota_target = float(m.group(1))
                vol_target = float(m.group(2)) if m.group(2) else _DEFAULT_VOL_TARGET
                break

        # Try to extract an init_id from path components like "init_dir91".
        init_id = None
        for part in run_dir.parts:
            m = re.search(r"init_dir(\d+)", part)
            if m:
                init_id = int(m.group(1))
                break

        max_current_kA = None
        if isinstance(max_current, (int, float)):
            max_current_kA = float(max_current) / 1e3

        max_center_field = max_dipole_center_field_from_results(run_dir)
        center_field_over_on_axis = None
        if max_center_field is not None:
            center_field_over_on_axis = float(max_center_field) / _FIELD_ON_AXIS_T

        records.append(
            {
                "run_dir": str(run_dir),
                "iota": float(iota),
                "max_current_kA": max_current_kA,
                "max_dipole_center_field_T": max_center_field,
                "field_on_axis_T": _FIELD_ON_AXIS_T,
                "max_dipole_center_field_over_on_axis": center_field_over_on_axis,
                "boozer_residual": float(boozer),
                "qs_error": float(nonqs),
                "current_weight": current_weight,
                "init_id": init_id,
                "iota_target": iota_target,
                "vol_target": vol_target,
            }
        )

    return records, stats


def choose_auto_marker_mode(records: list[dict]) -> str:
    """
    Pick a single legend dimension (marker style), priority:
      init_id > vol_target > current_weight > none.

    "Multiple" means at least two distinct non-null values for that field.
    """
    inits = [r.get("init_id") for r in records]
    nonnull_init = [x for x in inits if x is not None]
    if len(set(nonnull_init)) >= 2:
        return "init_id"

    vols = [r.get("vol_target") for r in records]
    nonnull_vol = [x for x in vols if x is not None]
    if len({round(float(v), 12) for v in nonnull_vol}) >= 2:
        return "vol_target"

    cws = [r.get("current_weight") for r in records]
    nonnull_cw = [x for x in cws if x is not None]
    if len({round(float(v), 12) for v in nonnull_cw}) >= 2:
        return "current_weight"

    return "none"


def _distinct_vol_target_count(records: list[dict]) -> int:
    """Number of distinct non-null vol_target values (rounded)."""
    vols = [r.get("vol_target") for r in records if r.get("vol_target") is not None]
    if not vols:
        return 0
    return len({round(float(v), 12) for v in vols})


def select_best_per_iota_volume(
    records: list[dict],
    vol_target_filter: float | None,
    objective_key: str = "max_current_kA",
) -> tuple[list[dict], int, str]:
    """
    For each bucket, keep the run that minimizes the chosen objective metric
    (tie-break: lower QS error).

    Bucketing:
      - If ``vol_target_filter`` is set (CLI ``--vol-target``), data are already one volume;
        bucket by (iota_target).
      - If ``vol_target_filter`` is None and there are at least two distinct volumes in
        ``records``, bucket by (iota_target, vol_target).
      - Otherwise bucket by (iota_target).

    Skips records missing required path tags for the chosen bucket key.
    """
    use_volume_bucket = vol_target_filter is None and _distinct_vol_target_count(records) >= 2

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    skipped = 0
    for r in records:
        it = r.get("iota_target")
        if it is None:
            skipped += 1
            continue
        if use_volume_bucket:
            vt = r.get("vol_target")
            if vt is None:
                skipped += 1
                continue
            key = (float(it), round(float(vt), 12))
        else:
            key = (float(it),)
        buckets[key].append(r)

    selected = []
    for group in buckets.values():
        group_with_metric = [x for x in group if x.get(objective_key) is not None]
        if not group_with_metric:
            skipped += len(group)
            continue
        best = min(group_with_metric, key=lambda x: (x[objective_key], x["qs_error"]))
        selected.append(best)

    mode = "(iota, vol)" if use_volume_bucket else "(iota)"
    return selected, skipped, mode


def make_plot(
    iota: np.ndarray,
    y_values: np.ndarray,
    y_label: str,
    color_values: np.ndarray,
    color_label: str,
    current_weight: np.ndarray,
    init_id: np.ndarray,
    vol_target: np.ndarray,
    marker_mode: str,
    y_max: float | None,
    out_path: Path,
):
    """
    Create one iota-vs-metric scatter.

    Marker style can be controlled by `marker_mode`:
      - "none": single marker for all points.
      - "init_id": marker encodes init_id parsed from run path.
      - "vol_target": marker encodes Boozer volume from path (iota_tar*_vol*).
      - "current_weight": marker encodes CURRENT_WEIGHT.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Map distinct marker values to marker styles.
    if marker_mode == "current_weight":
        marker_values = np.array(current_weight, dtype=object)
        label_prefix = "current_weight"
    elif marker_mode == "init_id":
        marker_values = np.array(init_id, dtype=object)
        label_prefix = "init_id"
    elif marker_mode == "vol_target":
        marker_values = np.array(vol_target, dtype=object)
        label_prefix = "vol_target"
    else:
        marker_values = np.array([None] * len(iota), dtype=object)
        label_prefix = ""

    def _sort_key(v):
        if v is None:
            return (2, 0.0)
        if isinstance(v, (int, np.integer)):
            return (0, float(v))
        return (1, float(v))

    unique_marker_values = sorted({v for v in marker_values if v is not None}, key=_sort_key)
    # Use high-contrast shapes (with black edges) so categories are easy to tell apart.
    marker_cycle = ["o", "*", "P", "X", "h", "8", "p", "H", "D", "s", "^", "v"]

    handles = []
    labels = []

    # If no marker grouping values are present, use a single marker style.
    if not unique_marker_values:
        sc = ax.scatter(
            iota,
            y_values,
            c=color_values,
            cmap="viridis",
            s=66,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.8,
        )
    else:
        sc = None
        for j, marker_value in enumerate(unique_marker_values):
            m = marker_cycle[j % len(marker_cycle)]
            mask = marker_values == marker_value
            if not np.any(mask):
                continue
            sc_part = ax.scatter(
                iota[mask],
                y_values[mask],
                c=color_values[mask],
                cmap="viridis",
                s=66,
                alpha=0.9,
                marker=m,
                edgecolors="k",
                linewidths=0.8,
            )
            # Remember one scatter handle for the colorbar.
            if sc is None:
                sc = sc_part
            handles.append(sc_part)
            if isinstance(marker_value, (bool, np.bool_)):
                lab = f"{label_prefix}={marker_value}"
            elif isinstance(marker_value, (int, np.integer)):
                lab = f"{label_prefix}={int(marker_value)}"
            else:
                lab = f"{label_prefix}={float(marker_value):g}"
            labels.append(lab)

    plt.colorbar(sc, ax=ax, label=color_label)

    ax.set_xlabel("Rotational transform (ι)")
    ax.set_ylabel(y_label)
    if y_max is not None:
        ax.set_ylim(top=y_max)

    if handles:
        for h, lab in zip(handles, labels):
            h.set_label(lab)
        ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of achieved iota vs max dipole current from a scan directory; "
            "save two copies with different colorbars."
        )
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default="../single_stage_scans_epsilon_constraint_updated",
        help="Scan root directory to search (default: ../single_stage_scans_epsilon_constraint_updated).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <scan-dir>/postprocess_plots).",
    )
    parser.add_argument(
        "--max-boozer-residual",
        type=float,
        default=1e-3,
        help="Exclude runs with J_Boozer above this threshold (default: 1e-3).",
    )
    sparse_group = parser.add_mutually_exclusive_group()
    sparse_group.add_argument(
        "--sparse",
        action="store_true",
        help=(
            "Only include runs under a subdirectory whose basename contains "
            "'sparse' (e.g. stage01_cw0.5_sparse or legacy stage04_cw2_sparsity)."
        ),
    )
    sparse_group.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Only include non-sparse runs (exclude paths whose subdirectory basename "
            "contains 'sparse')."
        ),
    )
    parser.add_argument(
        "--best-per-iota-init",
        action="store_true",
        help=(
            "Collapse to one point per bucket: lowest max dipole current (tie-break: lower "
            "QS error). Buckets are (iota target). If --vol-target is not set "
            "and the scan has multiple distinct volumes (iota_tar*_vol* paths), buckets are "
            "(iota, volume) instead. Selection is done from runs that already pass "
            "--max-boozer-residual. Omitting this flag plots every run (all stages)."
        ),
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        default=None,
        metavar="X",
        help="Only include runs under an iota_tarX folder matching this value (e.g. 0.16).",
    )
    parser.add_argument(
        "--init-id",
        type=str,
        default=None,
        metavar="N[,M,...]",
        help=(
            "Only include runs whose path contains init_dirN. Accepts a single value "
            "(e.g. 90) or a comma-separated list (e.g. 90,91,95)."
        ),
    )
    parser.add_argument(
        "--y-metric",
        choices=["max_current", "max_dipole_center_field"],
        default="max_current",
        help=(
            "Y-axis metric for Pareto plots: max dipole current (kA) or "
            "max dipole center field / field_on_axis from results.json."
        ),
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=None,
        metavar="V",
        help=(
            "Only include runs whose iota_tar folder matches this Boozer volume target "
            f"(paths without _vol are treated as {_DEFAULT_VOL_TARGET:g}). "
            "Example: 0.35 selects only .../iota_tar*_vol0.35/..."
        ),
    )
    parser.add_argument(
        "--marker-mode",
        choices=["auto", "none", "current_weight", "init_id", "vol_target"],
        default="auto",
        help=(
            "Legend / marker grouping: 'auto' picks one of init_id, vol_target, or "
            "current_weight when that field has multiple distinct values (priority in "
            "that order); otherwise a single marker. Or force a specific field."
        ),
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help=(
            "Optional y-axis upper cutoff in kA. "
            "If omitted, matplotlib autoscaling is used."
        ),
    )
    parser.add_argument(
        "--exclude-precision-loss",
        action="store_true",
        help=(
            "Omit runs from the scatter plot when results.json reports scipy's "
            'precision-loss termination (message contains "precision loss"). '
            "Does not delete or modify run directories. Default: include those runs."
        ),
    )
    args = parser.parse_args()

    scan_root = Path(args.scan_dir).resolve()
    if not scan_root.is_dir():
        raise SystemExit(f"Scan directory not found: {scan_root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else scan_root / "postprocess_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    sparse_mode = "all"
    if args.sparse:
        sparse_mode = "sparse"
    elif args.dense:
        sparse_mode = "dense"

    records, stats = load_scan(
        scan_root,
        args.max_boozer_residual,
        sparse_mode,
        exclude_precision_loss=args.exclude_precision_loss,
    )

    print(f"Scan root: {scan_root}")
    print(f"Hard-coded on-axis field for center-field scaling: {_FIELD_ON_AXIS_T:g} T")
    print(f"Max Boozer residual threshold: {args.max_boozer_residual:g}")
    print(f"Sparse filter mode: {sparse_mode}")
    print(f"Exclude precision-loss runs from plot: {args.exclude_precision_loss}")
    print(f"Total run directories discovered: {stats['total_dirs']}")
    print(f"Rejected (non-sparse path, --sparse only): {stats['rejected_non_sparse']}")
    print(f"Rejected (sparse path, --dense only): {stats['rejected_sparse']}")
    print(f"Rejected (no valid iterations.json): {stats['rejected_no_iter']}")
    print(f"Rejected (missing required final fields): {stats['rejected_missing_required_fields']}")
    print(f"Rejected (Boozer residual above threshold): {stats['rejected_boozer_residual']}")
    print(f"Rejected (precision loss, plot only): {stats['rejected_precision_loss']}")
    print(f"Loaded (used for plotting): {len(records)}")

    if args.iota_target is not None:
        before = len(records)
        records = [
            r
            for r in records
            if r.get("iota_target") is not None and abs(r["iota_target"] - args.iota_target) < 1e-9
        ]
        print(f"After --iota-target {args.iota_target:g}: {len(records)} (was {before})")

    if args.init_id is not None:
        try:
            init_ids = parse_init_id_list(args.init_id)
        except ValueError as e:
            raise SystemExit(f"Invalid --init-id value: {args.init_id!r} ({e})") from e
        before = len(records)
        records = [r for r in records if r.get("init_id") in init_ids]
        ids_text = ",".join(str(x) for x in sorted(init_ids))
        print(f"After --init-id {ids_text}: {len(records)} (was {before})")

    if args.vol_target is not None:
        before = len(records)
        records = [
            r
            for r in records
            if r.get("vol_target") is not None
            and np.isclose(r["vol_target"], args.vol_target, rtol=0.0, atol=1e-12)
        ]
        print(f"After --vol-target {args.vol_target:g}: {len(records)} (was {before})")

    if args.best_per_iota_init:
        objective_key = (
            "max_dipole_center_field_over_on_axis"
            if args.y_metric == "max_dipole_center_field"
            else "max_current_kA"
        )
        records, skipped_best, best_bucket_mode = select_best_per_iota_volume(
            records, args.vol_target, objective_key=objective_key
        )
        print(
            f"--best-per-iota-init: {len(records)} points, buckets {best_bucket_mode} "
            f"(skipped {skipped_best} without iota/volume path tags)"
        )

    if not records:
        print("No runs passed filters; no plots written.")
        return

    if args.y_metric == "max_dipole_center_field":
        y_key = "max_dipole_center_field_over_on_axis"
        y_label = "Max dipole center field / field on axis"
    else:
        y_key = "max_current_kA"
        y_label = "Max dipole current (kA)"

    before_metric = len(records)
    records = [r for r in records if r.get(y_key) is not None]
    if len(records) != before_metric:
        print(
            f"After --y-metric {args.y_metric}: {len(records)} (was {before_metric}); "
            f"dropped runs missing {y_key}"
        )
    if not records:
        print("No runs remain after y-metric filtering; no plots written.")
        return

    iota = np.array([r["iota"] for r in records], dtype=float)
    y_values = np.array([r[y_key] for r in records], dtype=float)
    boozer = np.array([r["boozer_residual"] for r in records], dtype=float)
    qs_error = np.array([r["qs_error"] for r in records], dtype=float)
    current_weight = np.array([r.get("current_weight") for r in records], dtype=object)
    init_id = np.array([r.get("init_id") for r in records], dtype=object)
    vol_target = np.array([r.get("vol_target") for r in records], dtype=object)

    if args.marker_mode == "auto":
        marker_mode = choose_auto_marker_mode(records)
        print(f"Legend (--marker-mode auto): using {marker_mode}")
    else:
        marker_mode = args.marker_mode

    metric_suffix = "center_field" if args.y_metric == "max_dipole_center_field" else "current"
    boozer_plot = out_dir / f"pareto_iota_vs_{metric_suffix}_color_boozer.png"
    qs_plot = out_dir / f"pareto_iota_vs_{metric_suffix}_color_qs_error.png"
    make_plot(
        iota=iota,
        y_values=y_values,
        y_label=y_label,
        color_values=boozer,
        color_label="Boozer residual",
        current_weight=current_weight,
        init_id=init_id,
        vol_target=vol_target,
        marker_mode=marker_mode,
        y_max=args.y_max,
        out_path=boozer_plot,
    )
    make_plot(
        iota=iota,
        y_values=y_values,
        y_label=y_label,
        color_values=qs_error,
        color_label="QS error",
        current_weight=current_weight,
        init_id=init_id,
        vol_target=vol_target,
        marker_mode=marker_mode,
        y_max=args.y_max,
        out_path=qs_plot,
    )

    print(f"Wrote: {boozer_plot}")
    print(f"Wrote: {qs_plot}")


if __name__ == "__main__":
    main()

