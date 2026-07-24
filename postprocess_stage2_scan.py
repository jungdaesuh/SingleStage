#!/usr/bin/env python3
"""
Postprocess stage-2 dipole scan outputs.

Loads results.json from each run directory, groups by (eq_name, npoloidal),
computes Pareto fronts of avg_Bnormal vs max_wp_current, generates plots,
and reports the minimum-current geometry under a field-error threshold.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_stage2_scan(scan_root):
    scan_root = Path(scan_root).resolve()
    records = []
    for run_dir in sorted(scan_root.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "results.json"
        if not results_path.is_file():
            continue
        try:
            with open(results_path, "r") as f:
                res = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        avg_bn = res.get("avg_Bnormal")
        max_bn = res.get("max_Bnormal")
        max_wp = res.get("max_wp_current")
        eq = res.get("eq_name")
        npol = res.get("npoloidal")
        ntor_val = res.get("ntoroidal")
        if avg_bn is None or max_wp is None or eq is None or npol is None:
            continue

        records.append(
            {
                "run_dir": str(run_dir),
                "eq_name": eq,
                "npoloidal": npol,
                "ntoroidal": ntor_val,
                "VV_R0": res.get("VV_R0"),
                "VV_a": res.get("VV_a"),
                "VV_b": res.get("VV_b"),
                "poloidal_radius": res.get("poloidal_radius"),
                "toroidal_radius_inboard": res.get("toroidal_radius_inboard"),
                "toroidal_radius_outboard": res.get("toroidal_radius_outboard"),
                "avg_Bnormal": avg_bn,
                "max_Bnormal": max_bn,
                "max_wp_current": max_wp,
            }
        )
    return records


def compute_pareto_front(field_errors, currents):
    order = np.argsort(field_errors)
    pareto_idx = []
    best_current = np.inf
    for idx in order:
        if currents[idx] <= best_current:
            pareto_idx.append(idx)
            best_current = currents[idx]
    return np.array(pareto_idx, dtype=int)


def _short_run_dir(run_dir_value, scan_root):
    run_dir_path = Path(run_dir_value).resolve()
    try:
        return run_dir_path.relative_to(scan_root)
    except ValueError:
        return run_dir_path


def plot_and_report(records, scan_root, out_dir, field_error_threshold=None, error_metric="avg_Bnormal"):
    scan_root = Path(scan_root).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in records:
        groups[(r["eq_name"], r["npoloidal"])].append(r)

    for (eq_name, npol), group_records in sorted(groups.items()):
        fe = np.array([r[error_metric] for r in group_records])
        mc = np.array([r["max_wp_current"] for r in group_records])
        mc_kA = mc / 1e3
        pareto_idx = compute_pareto_front(fe, mc)
        pareto_sorted = pareto_idx[np.argsort(fe[pareto_idx])]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(fe, mc_kA, c="steelblue", s=40, alpha=0.6, label="All samples")
        if len(pareto_sorted) > 0:
            ax.plot(fe[pareto_sorted], mc_kA[pareto_sorted], "k-o", linewidth=2, markersize=6, label="Pareto front")
        if field_error_threshold is not None:
            ax.axvline(field_error_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Error threshold = {field_error_threshold:.2e}")
        ax.set_xlabel(f"Field error ({error_metric})")
        ax.set_ylabel("Max WP current (kA)")
        ax.set_title(f"Pareto: {eq_name}, npol={npol}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_name = f"pareto_{eq_name}_npol{npol}.png"
        fig.savefig(out_dir / plot_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / plot_name}")

        print(f"\nPareto front for eq={eq_name}, npol={npol} ({len(pareto_sorted)} points):")
        for idx in pareto_sorted:
            r = group_records[idx]
            print(
                f"  {error_metric}={r[error_metric]:.4e}, "
                f"max_wp_current={r['max_wp_current']/1e3:.2f} kA, "
                f"VV_R0={r['VV_R0']:.3f}, VV_a={r['VV_a']:.3f}, VV_b={r['VV_b']:.3f}, "
                f"run_dir={_short_run_dir(r['run_dir'], scan_root)}"
            )

        if field_error_threshold is not None:
            qualified = [(i, group_records[i]) for i in range(len(group_records)) if group_records[i][error_metric] <= field_error_threshold]
            if qualified:
                _, best = min(qualified, key=lambda x: x[1]["max_wp_current"])
                print(f"\n  ** Best geometry under {error_metric} <= {field_error_threshold:.2e}:")
                print(f"     {error_metric} = {best[error_metric]:.4e}")
                print(f"     max_wp_current = {best['max_wp_current']/1e3:.2f} kA")
                print(f"     VV_R0 = {best['VV_R0']:.3f}, VV_a = {best['VV_a']:.3f}, VV_b = {best['VV_b']:.3f}")
                pol_r = best.get("poloidal_radius")
                tin = best.get("toroidal_radius_inboard")
                tout = best.get("toroidal_radius_outboard")
                if pol_r is not None and tin is not None and tout is not None:
                    print(f"     poloidal_radius={100*pol_r:.3f}, toroidal_inboard={100*tin:.3f}, toroidal_outboard={100*tout:.3f} [cm]")
                print(f"     run_dir={_short_run_dir(best['run_dir'], scan_root)}")
            else:
                print(f"\n  No samples satisfy {error_metric} <= {field_error_threshold:.2e} for eq={eq_name}, npol={npol}.")


def main():
    parser = argparse.ArgumentParser(description="Postprocess stage-2 dipole scan.")
    parser.add_argument("--scan-dir", type=str, required=True)
    parser.add_argument("--field-error-threshold", type=float, default=5e-3)
    parser.add_argument("--error-metric", type=str, default="avg_Bnormal", choices=["avg_Bnormal", "max_Bnormal"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    scan_root = Path(args.scan_dir).resolve()
    if not scan_root.is_dir():
        raise SystemExit(f"Scan directory not found: {scan_root}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else (scan_root / "postprocess_plots").resolve()

    print(f"Scan root: {scan_root}")
    records = load_stage2_scan(scan_root)
    print(f"Loaded {len(records)} runs with valid results.json")
    if not records:
        print("No valid runs found; nothing to plot.")
        return

    plot_and_report(records, scan_root, out_dir, args.field_error_threshold, args.error_metric)
    print(f"\nPlots written to {out_dir}")


if __name__ == "__main__":
    main()
