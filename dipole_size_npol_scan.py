"""
File: fixed_vv_npol_scan.py
Description:
    Scan poloidal windowpane count at fixed VV-plasma clearance and fixed
    toroidal count.  Coil size is determined automatically by
    generate_windowpane_array from the VV arc length and the requested npol
    (smaller coils for larger npol).

    Produces one optimization per poloidal count in npol_list, then generates
    a publication-quality figure (twin y-axes) of max WP current and mean
    |B_N/|B|| versus poloidal dipole count.

    Set PLOT_ONLY = True to skip optimizations and just regenerate the figure
    from existing results.json files in OUTPUT_DIR.
"""

import os
import sys
import shutil
import json
import glob
import numpy as np
from scipy.special import ellipe
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simsopt.geo import SurfaceRZFourier

# ===========================================================================
# USER PARAMETERS — edit these directly
# ===========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# Equilibrium
eq_name = "wout_nfp22ginsburg_000_000281"   # VMEC wout file stem (no .nc)
eq_dir = os.path.join(script_dir, "equilibria")

# Output
output_subfolder = "dipole_size_npol_scan_ntor8"
OUTPUT_DIR = os.path.join(script_dir, f"../tradeoff_scans/{output_subfolder}")

# VV-plasma clearance (fixed for the entire scan) [m]
vv_plasma_delta = 0.12

# Filament / panel spacing (fixed across the scan)
fil_distance = 0.05        # filament spacing [m]
half_per_distance = 0.05   # half-period panel spacing [m]

# TF coils
ntf = 4                    # TF coils per half field period
field_on_axis = 0.5        # on-axis toroidal field [T]
TF_a = 0.4                # TF coil semi-axis a [m]

# Fixed toroidal windowpane count
ntor_wp = 8

# Poloidal counts to scan
npol_list = list(range(7, 15))

# Plasma surface
surf_s = 1.0
surf_dof_scale = 1.0

# Optimization
CURRENT_THRESHOLD = 1e8
CURRENT_WEIGHT = 1e-12
verbose = False

# Set True to skip optimizations and just re-plot from existing results
PLOT_ONLY = False

# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def ellipse_perimeter(a, b):
    """Perimeter of an ellipse with semi-axes *a*, *b* via complete elliptic
    integral (same formula as helper_functions.generate_windowpane_array)."""
    if a < b:
        a, b = b, a
    return 4.0 * a * ellipe(1.0 - (b / a) ** 2)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results_list, output_dir):
    """Single axes with left/right y-axes: max WP current and avg |B_N/|B|| vs npol."""

    results_list = sorted(results_list, key=lambda r: r["npoloidal"])
    npol = np.array([r["npoloidal"] for r in results_list])
    max_I = np.array([r["max_wp_current"] for r in results_list])
    avg_Bn = np.array([r["avg_Bnormal"] for r in results_list])

    # Publication rcParams
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.25))
    ax2 = ax.twinx()

    ax.plot(
        npol, max_I * 1e-3, "o-", color="C0", linewidth=1.5, markersize=5,
        label="Max WP current [kA]",
    )
    ax2.plot(
        npol, avg_Bn * 100, "s-", color="C1", linewidth=1.5, markersize=5,
        label=r"$\langle |B_N| / |B| \rangle$ [%]",
    )

    ax.set_xlabel("# poloidal coils")
    ax.set_xticks(npol)
    ax.set_ylabel("Max WP current  [kA]", color="C0")
    ax2.set_ylabel(r"$\langle |B_N| / |B| \rangle$  [%]", color="C1")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.grid(True, linewidth=0.4, alpha=0.6)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", framealpha=0.95)

    fig.suptitle(
        f"VV-plasma distance = {100*vv_plasma_delta:.0f} cm",
        fontsize=13,
    )

    for ext in ("pdf", "png"):
        fname = os.path.join(output_dir, f"dipole_size_npol_tradeoff.{ext}")
        fig.savefig(fname)
        print(f"Saved {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Plot-only mode
# ------------------------------------------------------------------
if PLOT_ONLY:
    json_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "**/results.json"), recursive=True))
    if not json_files:
        print(f"No results.json found under {OUTPUT_DIR}")
        sys.exit(1)
    results_list = []
    for jf in json_files:
        with open(jf) as f:
            results_list.append(json.load(f))
    print(f"Loaded {len(results_list)} result files")
    plot_results(results_list, OUTPUT_DIR)
    sys.exit(0)

# ------------------------------------------------------------------
# Extract plasma-bounding ellipse from VMEC equilibrium
# ------------------------------------------------------------------
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
surf = SurfaceRZFourier.from_wout(
    eq_name_full, s=surf_s, range="full torus", nphi=128, ntheta=64,
)
surf.set_dofs(surf_dof_scale * surf.get_dofs())
gamma = surf.gamma()
R = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
Z = gamma[:, :, 2]
Rmin, Rmax = np.min(R), np.max(R)
Zmin, Zmax = np.min(Z), np.max(Z)
VV_R0 = 0.5 * (Rmin + Rmax)
VV_amin = 0.5 * (Rmax - Rmin)
VV_bmin = 0.5 * (Zmax - Zmin)

VV_a = VV_amin + vv_plasma_delta
VV_b = VV_bmin + vv_plasma_delta
print(f"Plasma bounding ellipse: VV_R0={VV_R0:.4f}, VV_amin={VV_amin:.4f}, VV_bmin={VV_bmin:.4f}")
print(f"VV with delta={vv_plasma_delta}: VV_a={VV_a:.4f}, VV_b={VV_b:.4f}")

# TF geometry (same convention as run_optimize_scan.py)
TF_R0 = VV_R0
TF_b = TF_a * VV_bmin / VV_amin

# ------------------------------------------------------------------
# Feasibility check: filter npol_list to values that give positive Rpol
# Rpol = arc_length / (2 * npol) - fil_distance / 2  must be > 0
# ------------------------------------------------------------------
arc_length = ellipse_perimeter(VV_a, VV_b)
max_npol = int(arc_length / fil_distance)  # Rpol > 0 requires npol < arc / fil_distance
print(f"VV ellipse arc length = {arc_length:.4f} m  ->  max feasible npol = {max_npol}")

feasible_npols = []
for k in npol_list:
    Rpol_check = arc_length / (2 * k) - fil_distance / 2
    if Rpol_check <= 0:
        print(f"  npol={k:3d}  ->  SKIPPED (Rpol={Rpol_check:.4f} <= 0)")
    else:
        feasible_npols.append(k)
        print(f"  npol={k:3d}  ->  Rpol={Rpol_check:.4f} m")

if not feasible_npols:
    print("No feasible npol values. Adjust npol_list or vv_plasma_delta.")
    sys.exit(1)

# ------------------------------------------------------------------
# Run optimizations
# ------------------------------------------------------------------
# Import optimize() here so the module-level imports in optimize.py
# (helper_functions, simsopt) are only pulled in when actually needed.
from optimize import optimize

current_date = datetime.now().strftime("%Y%m%d")
results_list = []

for idx, npol in enumerate(feasible_npols):
    run_name = f"{idx + 1:02}_{current_date}_npol_{npol:03d}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Copy source files for reproducibility
    for src in ("optimize.py", "helper_functions.py"):
        src_path = os.path.join(script_dir, src)
        if os.path.isfile(src_path):
            shutil.copy(src_path, os.path.join(run_dir, src))

    print(f"\n{'='*60}")
    print(f"Run {idx + 1}/{len(feasible_npols)}:  npol={npol}, ntor={ntor_wp}, VV_a={VV_a:.4f}, VV_b={VV_b:.4f}")
    print(f"{'='*60}")

    optimize(
        fil_distance=fil_distance,
        half_per_distance=half_per_distance,
        dipole_radius=None,
        VV_a=VV_a,
        VV_b=VV_b,
        VV_R0=VV_R0,
        surf_s=surf_s,
        surf_dof_scale=surf_dof_scale,
        eq_dir=eq_dir,
        eq_name=eq_name,
        ntf=ntf,
        num_fixed=ntf,
        field_on_axis=field_on_axis,
        TF_R0=TF_R0,
        TF_a=TF_a,
        TF_b=TF_b,
        fixed_geo_TFs=True,
        CURRENT_THRESHOLD=CURRENT_THRESHOLD,
        CURRENT_WEIGHT=CURRENT_WEIGHT,
        output_dir=run_dir,
        verbose=verbose,
        wp_npol_target=npol,
        wp_ntor_target=ntor_wp,
    )

    # Collect result for plotting
    rjson = os.path.join(run_dir, "results.json")
    if os.path.isfile(rjson):
        with open(rjson) as f:
            results_list.append(json.load(f))

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
if results_list:
    plot_results(results_list, OUTPUT_DIR)
else:
    print("No results collected — skipping plot.")
