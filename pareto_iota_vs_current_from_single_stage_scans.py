#!/usr/bin/env python3
"""
Scatter plot of achieved iota vs max dipole current, dipole center field, or
coil force/torque metrics (from ``export_coil_forces_torques.py`` postprocessing)
from single-stage scan runs.

This script is intended for scan outputs like:
  examples/single_stage_scans_no_sparsity_epsilon_constraint/<eq_or_init_dir>/iota_tar*/stage*/mpol*_ntor*/

It loads the final record from each `iterations.json`, filters by a Boozer residual
threshold, then plots achieved iota vs a selected metric (``--y-metric``):
  - max dipole current (kA), or
  - max dipole center field / on-axis field from results.json, or
  - max/mean net or pointwise force/torque scalars from
    ``coil_force_torque_postprocess_*`` in results.json.

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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})

# Must match single_stage_epsilon_constraint.py default for path segments without _vol*.
_DEFAULT_VOL_TARGET = 0.3

# Hard-coded on-axis field used to normalize dipole center field.
_FIELD_ON_AXIS_T = 0.5

# scipy.optimize.minimize (e.g. BFGS) may exit with this message in results.json.
_PRECISION_LOSS_SUBSTRING = "precision loss"

# Stage-2 / init dirs often encode the windowpane grid as npol10_ntor8 in path strings.
_NPOL_NTOR_COIL_RE = re.compile(r"npol[_]?(\d+)[_]?ntor[_]?(\d+)", re.IGNORECASE)


def _collect_json_strings(obj) -> list[str]:
    """Recursively collect all string values from JSON-like dict/list structures."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_json_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_json_strings(v))
    return out


def poloidal_dipole_count_from_graph(graph: dict | None, run_dir: Path) -> int | None:
    """
    Number of poloidal windowpane dipoles (npol in npol x ntor coil grids).

    Prefer ``graph['npoloidal']`` when present; otherwise parse ``npol*_ntor*`` from
    ``graph`` string fields and/or the run path. When ``# dipole coils`` and
    ``surf_nfp`` are available, verify ``npol * ntor * 2 * nfp == # dipole coils``.
    """
    if isinstance(graph, dict):
        v = graph.get("npoloidal")
        if isinstance(v, (int, float)):
            return int(v)

    candidates: list[tuple[int, int]] = []
    for s in _collect_json_strings(graph) if graph is not None else []:
        for m in _NPOL_NTOR_COIL_RE.finditer(s):
            candidates.append((int(m.group(1)), int(m.group(2))))
    for m in _NPOL_NTOR_COIL_RE.finditer(str(run_dir)):
        candidates.append((int(m.group(1)), int(m.group(2))))

    if not candidates:
        return None

    n_dip = None
    nfp = None
    if isinstance(graph, dict):
        nd = graph.get("# dipole coils")
        if isinstance(nd, (int, float)):
            n_dip = int(nd)
        nf = graph.get("surf_nfp")
        if isinstance(nf, (int, float)):
            nfp = int(nf)

    if n_dip is not None and nfp is not None:
        for npol, ntor in candidates:
            if npol * ntor * 2 * nfp == n_dip:
                return npol

    return candidates[0][0]


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


def max_dipole_center_field_from_graph(graph: dict | None) -> float | None:
    """Return graph.max_dipole_center_field from a results.json ``graph`` dict if present."""
    if graph is None:
        return None
    val = graph.get("max_dipole_center_field")
    if isinstance(val, (int, float)):
        return float(val)
    return None


# Keys written by examples/dipoles/export_coil_forces_torques.py
_CFTP_PREFIX = "coil_force_torque_postprocess_"


def _float_from_results_json(data: dict, key: str) -> float | None:
    """Read a numeric field from top-level results.json or nested ``graph`` dict."""
    v = data.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    g = data.get("graph")
    if isinstance(g, dict):
        v = g.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def coil_force_torque_metrics_from_results(data: dict) -> dict[str, float | None]:
    """Scalars from postprocessed coil VTK / force pass (may be missing if not run)."""
    keys = (
        ("max_net_force_N", "max_net_force_norm_N"),
        ("max_net_torque_Nm", "max_net_torque_norm_Nm"),
        ("max_pointwise_force_N_per_m", "max_pointwise_force_norm_N_per_m"),
        ("max_pointwise_torque_N", "max_pointwise_torque_norm_N"),
        ("mean_net_force_N", "mean_net_force_norm_N"),
        ("mean_net_torque_Nm", "mean_net_torque_norm_Nm"),
    )
    out: dict[str, float | None] = {}
    for short, suff in keys:
        out[short] = _float_from_results_json(data, _CFTP_PREFIX + suff)
    return out


# --y-metric: record field, axis label, output filename token, --best-per description snippet
_Y_METRIC_SPECS: dict[str, dict[str, str]] = {
    "max_current": {
        "record_key": "max_current_kA",
        "ylabel": "Max current (kA)",
        "file_token": "current",
        "best_desc": "lowest max dipole current (kA)",
    },
    "max_dipole_center_field": {
        "record_key": "max_dipole_center_field_over_on_axis",
        "ylabel": r"$\max(B_{\mathrm{dipole,0}}) \,/\, B_T$",
        "file_token": "center_field",
        "best_desc": r"lowest $B_{\mathrm{dipole}}^{\max}/B_0$",
    },
    "max_net_force": {
        "record_key": "max_net_force_N",
        "ylabel": r"max $\|\mathbf{F}_{\mathrm{net}}\|$ (N)",
        "file_token": "max_net_force",
        "best_desc": r"lowest max $\|\mathbf{F}_{\mathrm{net}}\|$ (N)",
    },
    "max_net_torque": {
        "record_key": "max_net_torque_Nm",
        "ylabel": r"max $\|\mathbf{T}_{\mathrm{net}}\|$ (N$\cdot$m)",
        "file_token": "max_net_torque",
        "best_desc": r"lowest max $\|\mathbf{T}_{\mathrm{net}}\|$ (N$\cdot$m)",
    },
    "max_pointwise_force": {
        "record_key": "max_pointwise_force_N_per_m",
        "ylabel": r"max $\|\mathrm{d}\mathbf{F}/\mathrm{d}\ell\|$ (N/m)",
        "file_token": "max_pointwise_force",
        "best_desc": r"lowest max pointwise $\|\mathrm{d}\mathbf{F}/\mathrm{d}\ell\|$ (N/m)",
    },
    "max_pointwise_torque": {
        "record_key": "max_pointwise_torque_N",
        "ylabel": r"max $\|\mathrm{d}\mathbf{T}/\mathrm{d}\ell\|$ (N)",
        "file_token": "max_pointwise_torque",
        "best_desc": r"lowest max pointwise $\|\mathrm{d}\mathbf{T}/\mathrm{d}\ell\|$ (N)",
    },
    "mean_net_force": {
        "record_key": "mean_net_force_N",
        "ylabel": r"mean $\|\mathbf{F}_{\mathrm{net}}\|$ over coils (N)",
        "file_token": "mean_net_force",
        "best_desc": r"lowest mean $\|\mathbf{F}_{\mathrm{net}}\|$ (N)",
    },
    "mean_net_torque": {
        "record_key": "mean_net_torque_Nm",
        "ylabel": r"mean $\|\mathbf{T}_{\mathrm{net}}\|$ over coils (N$\cdot$m)",
        "file_token": "mean_net_torque",
        "best_desc": r"lowest mean $\|\mathbf{T}_{\mathrm{net}}\|$ (N$\cdot$m)",
    },
}


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
        # Match stage folders like stage03_cw3_sparsity, not *_no_sparsity_* in scan root names.
        is_sparse_path = any("sparsity" in p for p in run_dir.parts)
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

        results_data: dict | None = None
        results_graph = None
        try:
            with open(run_dir / "results.json", "r") as f:
                results_data = json.load(f)
            _g = results_data.get("graph")
            if isinstance(_g, dict):
                results_graph = _g
        except (json.JSONDecodeError, OSError):
            results_data = None
            results_graph = None

        max_center_field = max_dipole_center_field_from_graph(results_graph)
        center_field_over_on_axis = None
        if max_center_field is not None:
            center_field_over_on_axis = float(max_center_field) / _FIELD_ON_AXIS_T

        npoloidal = poloidal_dipole_count_from_graph(results_graph, run_dir)

        force_torque_fields = coil_force_torque_metrics_from_results(
            results_data if results_data is not None else {}
        )

        rec = {
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
            "npoloidal": npoloidal,
        }
        rec.update(force_torque_fields)
        records.append(rec)

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
    title: str | None = None,
):
    """
    Create one iota-vs-metric scatter.

    Marker style can be controlled by `marker_mode`:
      - "none": single marker for all points.
      - "init_id": marker encodes init_id parsed from run path.
      - "vol_target": marker encodes Boozer volume from path (iota_tar*_vol*).
      - "current_weight": marker encodes CURRENT_WEIGHT.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Map distinct marker values to marker styles.
    if marker_mode == "current_weight":
        marker_values = np.array(current_weight, dtype=object)
        label_prefix = r"$w_{\mathrm{CP}}$"
    elif marker_mode == "init_id":
        marker_values = np.array(init_id, dtype=object)
        label_prefix = "init"
    elif marker_mode == "vol_target":
        marker_values = np.array(vol_target, dtype=object)
        label_prefix = r"$V_{\mathrm{target}}$"
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
    marker_cycle = ["o", "s", "D", "^", "v", "P", "*", "X", "h", "8", "p", "H"]

    handles = []
    labels = []

    scatter_kw = dict(
        cmap="viridis",
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.6,
        zorder=3,
    )

    # If no marker grouping values are present, use a single marker style.
    if not unique_marker_values:
        sc = ax.scatter(iota, y_values, c=color_values, **scatter_kw)
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
                marker=m,
                **scatter_kw,
            )
            # Remember one scatter handle for the colorbar.
            if sc is None:
                sc = sc_part
            handles.append(sc_part)
            if isinstance(marker_value, (bool, np.bool_)):
                lab = f"{label_prefix} = {marker_value}"
            elif isinstance(marker_value, (int, np.integer)):
                lab = f"{label_prefix} = {int(marker_value)}"
            else:
                lab = f"{label_prefix} = {float(marker_value):g}"
            labels.append(lab)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(color_label)
    _cbar_fmt = ScalarFormatter(useMathText=True)
    _cbar_fmt.set_scientific(True)
    _cbar_fmt.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(_cbar_fmt)

    ax.set_xlabel(r"Rotational transform $\iota$")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if y_max is not None:
        ax.set_ylim(top=y_max)

    if handles:
        for h, lab in zip(handles, labels):
            h.set_label(lab)
        ax.legend(loc="best", framealpha=0.9, edgecolor="0.7")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of achieved iota vs a selectable y-metric from a scan directory; "
            "save two copies with different colorbars (Boozer residual and QS error)."
        )
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default="../single_stage_scans_no_sparsity_epsilon_constraint",
        help="Scan root directory to search (default: ../single_stage_scans_no_sparsity_epsilon_constraint).",
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
            "'sparsity' (e.g. stage03_cw3_sparsity)."
        ),
    )
    sparse_group.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Only include non-sparse runs (exclude paths whose subdirectory basename "
            "contains 'sparsity')."
        ),
    )
    parser.add_argument(
        "--best-per-iota-init",
        action="store_true",
        help=(
            "Collapse to one point per bucket: minimize the selected --y-metric (tie-break: "
            "lower QS error). Buckets are (iota target). If --vol-target is not set "
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
        choices=list(_Y_METRIC_SPECS.keys()),
        default="max_current",
        help=(
            "Y-axis metric: max_current or max_dipole_center_field from iterations/results; "
            "force/torque options require coil_force_torque_postprocess_* keys in results.json "
            "(from export_coil_forces_torques.py)."
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
            "Optional y-axis upper limit (same units as --y-metric). "
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

    y_spec = _Y_METRIC_SPECS[args.y_metric]
    print(f"--y-metric {args.y_metric!r} (record field {y_spec['record_key']!r})")
    if args.best_per_iota_init:
        records, skipped_best, best_bucket_mode = select_best_per_iota_volume(
            records, args.vol_target, objective_key=y_spec["record_key"]
        )
        print(
            f"--best-per-iota-init: {len(records)} points, buckets {best_bucket_mode}, "
            f"objective={y_spec['best_desc']} "
            f"(skipped {skipped_best} without iota/volume path tags)"
        )

    if not records:
        print("No runs passed filters; no plots written.")
        return

    y_key = y_spec["record_key"]
    y_label = y_spec["ylabel"]

    before_metric = len(records)
    records = [r for r in records if r.get(y_key) is not None]
    if len(records) != before_metric:
        print(
            f"After --y-metric {args.y_metric}: {len(records)} (was {before_metric}); "
            f"dropped runs missing {y_key!r}"
        )
    if not records:
        print("No runs remain after y-metric filtering; no plots written.")
        return

    npol_vals = sorted(
        {r["npoloidal"] for r in records if r.get("npoloidal") is not None}
    )
    if len(npol_vals) == 1:
        plot_title = f"# poloidal coils = {npol_vals[0]}"
    elif len(npol_vals) > 1:
        plot_title = "# poloidal coils = " + ", ".join(str(x) for x in npol_vals)
    else:
        plot_title = None

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

    metric_suffix = y_spec["file_token"]
    boozer_plot = out_dir / f"pareto_iota_vs_{metric_suffix}_color_boozer.png"
    qs_plot = out_dir / f"pareto_iota_vs_{metric_suffix}_color_qs_error.png"
    make_plot(
        iota=iota,
        y_values=y_values,
        y_label=y_label,
        color_values=boozer,
        color_label=r"$J_{\mathrm{B}}$",
        current_weight=current_weight,
        init_id=init_id,
        vol_target=vol_target,
        marker_mode=marker_mode,
        y_max=args.y_max,
        out_path=boozer_plot,
        title=plot_title,
    )
    make_plot(
        iota=iota,
        y_values=y_values,
        y_label=y_label,
        color_values=qs_error,
        color_label=r"$f_{\mathrm{QS}}$",
        current_weight=current_weight,
        init_id=init_id,
        vol_target=vol_target,
        marker_mode=marker_mode,
        y_max=args.y_max,
        out_path=qs_plot,
        title=plot_title,
    )

    print(f"Wrote: {boozer_plot}")
    print(f"Wrote: {qs_plot}")


if __name__ == "__main__":
    main()

