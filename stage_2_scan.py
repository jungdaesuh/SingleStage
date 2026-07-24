"""
File: stage_2_scan.py
Author: Jake Halpern
Last Edit Date: 03/26/2026
Description: This script sends off an optimization scan over various geometric parameters for
             windowpane coil currents on an axisymmetric surface for a specified plasma equilibrium.
             Uses fixed coil counts (ntor=8, user-specified npol) with geometry-derived coil sizes.
             Samples R0 and a with constraints R0+a < 1.3 and R0-a > 0.7, and
             b in [VV_b_min + 0.08, VV_b_min + 0.16].
"""

import os
import numpy as np
from datetime import datetime
from optimize import *
from simsopt.geo import SurfaceRZFourier

# Set script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

### Campaign Settings (edit per run) ###
eq_name = "wout_nfp22ginsburg_000_000281"  # name of the wout file from vmec
npol = 10        # number of poloidal windowpane coils (user-specified per run)
ntor = 8        # number of toroidal windowpane coils (always 8)
CURRENT_THRESHOLD = 1e6   # threshold for current penalty
CURRENT_WEIGHT = 1e-12    # weight on current penalty
n_samples = 50           # number of geometry samples to run

# Define output parent folder
output_subfolder = f"stage_2_npol{npol}_ntor{ntor}_{eq_name}_sparse"
parent_run_dir = os.path.join(script_dir, f"../outputs/{output_subfolder}")
os.makedirs(parent_run_dir, exist_ok=True)

### Base Simulation Parameters ###
# Dipole parameters
fil_distance = 0.05       # distance between dipole filaments for finite coil winding pack [m]
half_per_distance = 0.05  # distance between dipole panels between half field periods [m]

# Plasma Surface
surf_s = 1                # value of s to cut the surface at (1 if already HBT sized)
surf_dof_scale = 1        # used to scale the dofs of the surface (1 if already HBT sized)
eq_dir = os.path.join(script_dir, "equilibria")

# Extract the minimum axisymmetric VV size from equilibrium
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
surf = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range="full torus", nphi=128, ntheta=64)
surf.set_dofs(surf_dof_scale * surf.get_dofs())
R = np.sqrt(surf.gamma()[:, :, 0]**2 + surf.gamma()[:, :, 1]**2)
Rmin = np.min(R); Rmax = np.max(R)
Zmin = np.min(surf.gamma()[:, :, 2]); Zmax = np.max(surf.gamma()[:, :, 2])
VV_Ravg = (Rmin + Rmax) / 2
VV_a_min = (Rmax - Rmin) / 2
VV_b_min = (Zmax - Zmin) / 2
print(f"VV_Ravg = {VV_Ravg:.4f}, VV_a_min = {VV_a_min:.4f}, VV_b_min = {VV_b_min:.4f}")

# TF coils parameters
ntf = 4
num_fixed = ntf           # all currents are fixed in TF coils
field_on_axis = 0.5       # on-axis magnetic field (Tesla)
fixed_geo_TFs = True

# Directory naming
current_date = datetime.now().strftime("%Y%m%d")
existing_runs = [d for d in os.listdir(parent_run_dir) if d.split("_")[0].isdigit()]
if existing_runs:
    next_run_number = max(int(d.split("_")[0]) for d in existing_runs) + 1
else:
    next_run_number = 1

### Constrained Sampling ###
# Bounds for offsets from equilibrium-derived minimums
VV_a_dist_min, VV_a_dist_max = 0.08, 0.16
VV_b_dist_min, VV_b_dist_max = 0.08, 0.16
VV_R0_dist_min, VV_R0_dist_max = -0.03, 0.03

# Hard constraints on absolute geometry
R0_PLUS_A_MAX = 1.3
R0_MINUS_A_MIN = 0.7

rng = np.random.default_rng()
accepted = 0
max_attempts = n_samples * 20

for attempt in range(max_attempts):
    if accepted >= n_samples:
        break

    # Randomly sample VV_a, VV_R0, and VV_b offsets from equilibrium-derived minimums
    VV_a = VV_a_min + rng.uniform(VV_a_dist_min, VV_a_dist_max)
    VV_R0 = VV_Ravg + rng.uniform(VV_R0_dist_min, VV_R0_dist_max)
    VV_b = VV_b_min + rng.uniform(VV_b_dist_min, VV_b_dist_max)

    # Reject samples that are outside the HBT design range to the edge of the allowed range
    hit_upper = VV_R0 + VV_a >= R0_PLUS_A_MAX
    hit_lower = VV_R0 - VV_a <= R0_MINUS_A_MIN
    if hit_upper or hit_lower:
        hit_bounds = []
        if hit_upper:
            hit_bounds.append(f"R0+a={VV_R0 + VV_a:.3f} >= {R0_PLUS_A_MAX:.3f}")
        if hit_lower:
            hit_bounds.append(f"R0-a={VV_R0 - VV_a:.3f} <= {R0_MINUS_A_MIN:.3f}")
        print(
            f"Skipping sample attempt {attempt + 1}: "
            f"VV_R0={VV_R0:.3f}, VV_a={VV_a:.3f}, VV_b={VV_b:.3f} | "
            f"bound(s) hit: {', '.join(hit_bounds)}"
        )
        continue

    accepted += 1

    # For this scan, we keep the distance between TF coils and VV constant
    # Pros: physical because this distance is likely to be constant
    # Cons: this changes things like ripple and BdotN, which we don't want to change
    TF_R0 = VV_R0
    TF_a = VV_a + 0.15
    TF_b = VV_b + 0.15

    # Create a unique directory name
    run_dir_name = (
        f"{next_run_number:02}"
        f"_npol_{npol}_ntor_{ntor}"
        f"_VV_a_{VV_a:.3f}_VV_b_{VV_b:.3f}_VV_R0_{VV_R0:.3f}"
    )
    run_dir = os.path.join(parent_run_dir, run_dir_name)
    # Create directory and copy files into it
    os.makedirs(run_dir, exist_ok=True)

    print(
        f"\nSample {accepted}/{n_samples}: npol={npol}, ntor={ntor}, "
        f"VV_a={VV_a:.3f}, VV_b={VV_b:.3f}, VV_R0={VV_R0:.3f}, "
        f"TF_a={TF_a:.2f}, TF_b={TF_b:.2f}"
    )

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
        num_fixed=num_fixed,
        field_on_axis=field_on_axis,
        TF_R0=TF_R0,
        TF_a=TF_a,
        TF_b=TF_b,
        fixed_geo_TFs=fixed_geo_TFs,
        CURRENT_THRESHOLD=CURRENT_THRESHOLD,
        CURRENT_WEIGHT=CURRENT_WEIGHT,
        output_dir=run_dir,
        verbose=False,
        wp_npol_target=npol,
        wp_ntor_target=ntor,
    )

    next_run_number += 1

if accepted < n_samples:
    print(f"\nWarning: only {accepted}/{n_samples} samples satisfied constraints after {max_attempts} attempts.")
else:
    print(f"\nCompleted all {n_samples} samples.")
