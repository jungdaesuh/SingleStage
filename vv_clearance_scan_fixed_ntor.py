"""
File: vv_clearance_scan_fixed_ntor.py
Author: auto-generated from plan
Description:
    Scan vacuum-vessel clearance (VV-to-plasma distance) at fixed dipole size
    so that the number of *poloidal* windowpane coils increases discretely while
    the toroidal count stays fixed at ntor_wp.

    Produces one optimization per distinct poloidal dipole count, then generates
    a publication-quality figure (twin y-axes) of max WP current and mean |B_N/|B||
    versus poloidal dipole count.

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
from scipy.optimize import brentq
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
output_subfolder = "vv_dist_scan_ntor8"
OUTPUT_DIR = os.path.join(script_dir, f"../tradeoff_scans/{output_subfolder}")

# Dipole parameters (fixed across the scan)
dipole_radius = 0.05       # inboard dipole radius [m]
fil_distance = 0.05        # filament spacing [m]
half_per_distance = 0.05   # half-period panel spacing [m]

# TF coils
ntf = 4                    # TF coils per half field period
field_on_axis = 0.5        # on-axis toroidal field [T]
TF_a = 0.4                # TF coil semi-axis a [m]

# Fixed toroidal windowpane count
ntor_wp = 8

# Poloidal count scan range (set to None for automatic from delta bounds)
npol_min = None
npol_max = None

# VV-plasma clearance search bounds [m]
delta_min = 0.02
delta_max = 0.20

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
# Geometry helpers (mirror the formula in generate_windowpane_array)
# ---------------------------------------------------------------------------

def ellipse_perimeter(a, b):
    """Approximate perimeter of an ellipse with semi-axes *a*, *b*
    using the complete elliptic integral (same formula as helper_functions)."""
    if a < b:
        a, b = b, a
    return 4.0 * a * ellipe(1.0 - (b / a) ** 2)


def npol_from_delta(delta, VV_amin, VV_bmin, dipole_radius, fil_distance):
    """Integer poloidal dipole count for a given VV clearance *delta*."""
    VV_a = VV_amin + delta
    VV_b = VV_bmin + delta
    arc = ellipse_perimeter(VV_a, VV_b)
    pitch = 2.0 * dipole_radius + fil_distance
    return int(arc / pitch)


def find_delta_for_npol(k, VV_amin, VV_bmin, dipole_radius, fil_distance,
                        delta_lo=0.0, delta_hi=1.0):
    """Return the midpoint of the *delta* interval that yields *npol == k*.

    Uses bisection to locate the lower and upper boundaries of the plateau
    where ``int(arc / pitch) == k``, then returns the midpoint.
    """
    pitch = 2.0 * dipole_radius + fil_distance

    def _npol(d):
        return int(ellipse_perimeter(VV_amin + d, VV_bmin + d) / pitch)

    # Quick feasibility check
    if _npol(delta_hi) < k:
        return None  # unreachable with current delta range
    if _npol(delta_lo) > k:
        return None  # already past this count even at smallest delta

    # Lower boundary: smallest delta where npol >= k
    # continuous function f(d) = arc/pitch - k  crosses zero at boundary
    def _cont_lower(d):
        return ellipse_perimeter(VV_amin + d, VV_bmin + d) / pitch - k

    if _cont_lower(delta_lo) >= 0:
        d_lower = delta_lo
    else:
        d_lower = brentq(_cont_lower, delta_lo, delta_hi)

    # Upper boundary: smallest delta where npol >= k+1
    def _cont_upper(d):
        return ellipse_perimeter(VV_amin + d, VV_bmin + d) / pitch - (k + 1)

    if _cont_upper(delta_hi) < 0:
        d_upper = delta_hi
    else:
        d_upper = brentq(_cont_upper, delta_lo, delta_hi)

    d_mid = 0.5 * (d_lower + d_upper)

    # Sanity: make sure we actually get k
    if _npol(d_mid) != k:
        return None
    return d_mid


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

    fig.suptitle(f"Coil radius = {100*dipole_radius:.0f} cm", fontsize=13)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", framealpha=0.95)

    for ext in ("pdf", "png"):
        fname = os.path.join(output_dir, f"vv_clearance_tradeoff.{ext}")
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
print(f"Plasma bounding ellipse: VV_R0={VV_R0:.4f}, VV_amin={VV_amin:.4f}, VV_bmin={VV_bmin:.4f}")

# TF geometry (same convention as run_optimize_scan.py)
TF_R0 = VV_R0
TF_b = TF_a * VV_bmin / VV_amin

# ------------------------------------------------------------------
# Determine discrete VV offsets that each give a unique poloidal count
# ------------------------------------------------------------------
npol_lo = npol_from_delta(delta_min, VV_amin, VV_bmin, dipole_radius, fil_distance)
npol_hi = npol_from_delta(delta_max, VV_amin, VV_bmin, dipole_radius, fil_distance)
k_min = npol_min if npol_min is not None else max(npol_lo, 1)
k_max = npol_max if npol_max is not None else npol_hi
print(f"Poloidal count range: {k_min} .. {k_max}  (delta {delta_min} .. {delta_max} m)")

scan_points = []
for k in range(k_min, k_max + 1):
    d = find_delta_for_npol(k, VV_amin, VV_bmin,
                            dipole_radius, fil_distance,
                            delta_min, delta_max)
    if d is not None:
        scan_points.append((k, d))
        print(f"  npol={k:3d}  ->  delta={d:.5f} m   VV_a={VV_amin + d:.4f}  VV_b={VV_bmin + d:.4f}")
    else:
        print(f"  npol={k:3d}  ->  SKIPPED (infeasible)")

if not scan_points:
    print("No feasible scan points. Adjust delta_min / delta_max.")
    sys.exit(1)

# ------------------------------------------------------------------
# Run optimizations
# ------------------------------------------------------------------
# Import optimize() here so the module-level imports in optimize.py
# (helper_functions, simsopt) are only pulled in when actually needed.
from optimize import optimize

current_date = datetime.now().strftime("%Y%m%d")
results_list = []

for idx, (k, delta) in enumerate(scan_points):
    run_name = f"{idx + 1:02}_{current_date}_npol_{k:03d}_delta_{delta:.4f}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Copy source files for reproducibility
    for src in ("optimize.py", "helper_functions.py"):
        src_path = os.path.join(script_dir, src)
        if os.path.isfile(src_path):
            shutil.copy(src_path, os.path.join(run_dir, src))

    VV_a = VV_amin + delta
    VV_b = VV_bmin + delta

    print(f"\n{'='*60}")
    print(f"Run {idx + 1}/{len(scan_points)}:  npol_target={k}, delta={delta:.5f}, VV_a={VV_a:.4f}, VV_b={VV_b:.4f}")
    print(f"{'='*60}")

    optimize(
        fil_distance=fil_distance,
        half_per_distance=half_per_distance,
        dipole_radius=dipole_radius,
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
