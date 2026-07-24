#!/usr/bin/env python3
"""
Plot the vacuum vessel cross-section and the R,Z locations where dipole coils
in one toroidal row intersect their local toroidal plane.

For each coil, the toroidal plane is defined by the coil center's phi angle.
The intersection is found by interpolating between quadrature points that
straddle the plane.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from simsopt._core.optimizable import load
from helper_functions import find_toroidal_plane_intersections

# ===== Hardcoded path =====
RUN_DIR = (
    "/global/homes/j/jhalpern/codes/simsopt/examples/"
    "single_stage_scans_epsilon_constraint_updated/"
    "wout_nfp22ginsburg_000_000281_init_dir90/"
    "iota_tar0.1/stage03_cw2/mpol6_ntor6"
)
TOROIDAL_IDX = 0  # which toroidal column to plot (0-indexed)

# ===== Load results =====
with open(os.path.join(RUN_DIR, "results.json")) as f:
    results_raw = json.load(f)
results = results_raw["graph"]

VV_R0 = results["VV_R0"]
VV_a = results["VV_a"]
VV_b = results["VV_b"]
nfp = results["surf_nfp"]
ntf = results["ntf"]

# ===== Load coils =====
bs = load(os.path.join(RUN_DIR, "bs_opt.json"))
coils = bs.coils
num_tf_coils = ntf * 2 * nfp
dipole_coils = coils[num_tf_coils:]
n_dipoles = len(dipole_coils)

# coils_via_symmetries stores base coils first (k=0, flip=False),
# so the first n_base entries are the original half-field-period coils.
n_base = n_dipoles // (2 * nfp)

print(f"Total coils: {len(coils)} ({num_tf_coils} TF, {n_dipoles} dipole)")
print(f"Base dipole coils per half-period: {n_base}")

# ===== Determine grid dimensions from the phi pattern of base coils =====
# Base coils are created as:
#   for ii in range(nwps_poloidal):
#       for jj in range(nwps_toroidal):
# Within a poloidal group (fixed ii) phi increases with jj.
# Detect nwps_toroidal by finding where phi first decreases.
base_phis = np.array([
    np.arctan2(dipole_coils[i].curve.center[1],
               dipole_coils[i].curve.center[0])
    for i in range(n_base)
])

nwps_toroidal = n_base
for i in range(1, n_base):
    if base_phis[i] < base_phis[i - 1]:
        nwps_toroidal = i
        break

nwps_poloidal = n_base // nwps_toroidal
assert nwps_poloidal * nwps_toroidal == n_base, (
    f"Grid dimensions {nwps_poloidal}x{nwps_toroidal} != {n_base}"
)
print(f"Detected grid: {nwps_poloidal} poloidal x {nwps_toroidal} toroidal")

# ===== Select one toroidal column (fixed toroidal index, all poloidal positions) =====
toroidal_idx = min(TOROIDAL_IDX, nwps_toroidal - 1)
row_base_indices = [ii * nwps_toroidal + toroidal_idx for ii in range(nwps_poloidal)]

print(f"Selected toroidal index {toroidal_idx} ({nwps_poloidal} coils in column)")


# ===== Find intersections with the toroidal plane =====
rz_pairs = []  # each entry is ((R_bot, Z_bot), (R_top, Z_top))

for base_idx in row_base_indices:
    curve = dipole_coils[base_idx].curve
    phi_cut = np.arctan2(curve.center[1], curve.center[0])
    crossings = find_toroidal_plane_intersections(curve, phi_cut)

    if len(crossings) >= 2:
        crossings.sort(key=lambda p: p[1])
        rz_pairs.append((crossings[0], crossings[-1]))
    elif len(crossings) == 1:
        print(f"  Warning: coil at base index {base_idx} has only 1 crossing")
        rz_pairs.append((crossings[0], crossings[0]))
    else:
        print(f"  Warning: coil at base index {base_idx} has no crossings")

print(f"Found {len(rz_pairs)} coil intersection pairs")

print("-- rz_pairs --")
for r, z in rz_pairs:
    print(f"({r}, {z})")

# ===== Build VV cross-section (elliptical, same at every phi) =====
theta = np.linspace(0, 2 * np.pi, 200)
R_vv = VV_R0 + VV_a * np.cos(theta)
Z_vv = VV_b * np.sin(theta)

# ===== Plot =====
fig, ax = plt.subplots(figsize=(8, 10))
ax.plot(R_vv, Z_vv, 'k-', linewidth=2, label='Vacuum Vessel')

n_pairs = len(rz_pairs)
cmap = plt.cm.tab20 if n_pairs <= 20 else plt.cm.hsv
colors = [cmap(i / max(n_pairs - 1, 1)) for i in range(n_pairs)]

for i, ((R_bot, Z_bot), (R_top, Z_top)) in enumerate(rz_pairs):
    label = 'Coil intersections' if i == 0 else None
    ax.scatter([R_bot, R_top], [Z_bot, Z_top],
               color=colors[i], s=60, zorder=5,
               edgecolors='k', linewidths=0.5, label=label)

ax.set_xlabel('R [m]', fontsize=14)
ax.set_ylabel('Z [m]', fontsize=14)
ax.set_aspect('equal')
ax.set_title(
    f'VV Cross-Section with Dipole Coil Intersections\n'
    f'(toroidal index {toroidal_idx}, {nwps_poloidal} poloidal x {nwps_toroidal} toroidal grid)',
    fontsize=14,
)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

output_path = os.path.join(RUN_DIR, 'coil_vv_cross_section.png')
plt.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
plt.show()
