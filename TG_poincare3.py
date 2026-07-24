"""
File: run_optimize_scan.py
Author: Jake Halpern
Last Edit Date: 02/2025
Description: This script sends off a single optimization run of windowpane coil currents on 
             an axisymmetric surface for a specified plasma equilibrium
"""
import os
import shutil
import numpy as np
from optimize import optimize
from simsopt.geo import SurfaceRZFourier

script_dir = os.path.dirname(os.path.abspath(__file__))
###### set this to wherever you'd like all the outputs from this script to go
run_dir = os.path.join(script_dir, '../outputs/stage2_initial_conditions')

### Simulation parameters ###
# Dipole parameters
fil_distance = 0.05 # distance between dipole filaments for finite coil winding pack [m]
half_per_distance = 0.05 # distance between dipole panels between half field periods of the device [m]
dipole_radius = 0.05 # target radius of dipoles (poloidally constant, will vary toroidally so this is at inboard midplane) [m]
# Plasma Surface
surf_s = 0.9                # value of s to cut the surface at (if already HBT sized, these will both be 1)
surf_dof_scale = 1.0      # used to scale the dofs of the surface
eq_name = 'wout_nfp22ginsburg_000_001242'  # name of the wout file from vmec
eq_dir = os.path.join(script_dir, 'equilibria') # equilibria should be in this folder
# Vacuum Vessel
VV_plas_dist = 0.12
# Extract the minimum axisymmetric VV size
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")

#TG: Added missing initialisations
plas_nPhi = 512
plas_nTheta = 128

surf = SurfaceRZFourier.from_wout(
    eq_name_full, s=surf_s, range="full torus", nphi=plas_nPhi, ntheta=plas_nTheta
)
R = np.sqrt(surf.gamma()[:, :, 0]**2 + surf.gamma()[:, :, 1]**2)
Rmin = np.min(R)
Rmax = np.max(R)
Zmin = np.min(surf.gamma()[:, :, 2])
Zmax = np.max(surf.gamma()[:, :, 2])
VV_R0 = (Rmin + Rmax) / 2 + 0.03
VV_amin = (Rmax - Rmin) / 2
VV_bmin = (Zmax - Zmin) / 2
VV_a = VV_amin + VV_plas_dist               # minor radius of vacuum vessel (horizontal)
VV_b = VV_bmin + VV_plas_dist               # minor radius of vacuum vessel (vertical)
# TF coils parameters (radius current set as 1.6 * VV_b)
n_tf = 4                       # number of TF coils per half field period
num_fixed = 4 #n_tf                  # number of TF coil currents to fix during combined optimization
field_on_axis = 0.5            # on-axis magnetic field (Tesla)
TF_R0 = VV_R0
TF_a = 0.40
TF_b = TF_a * VV_b / VV_a
fixed_geo_TFs = False
CC_THRESHOLD = 0.1
CC_WEIGHT = 100
CS_THRESHOLD = 0.1
CS_WEIGHT = 100
if fixed_geo_TFs:
    CC_THRESHOLD = None
    CC_WEIGHT = None 
    CS_THRESHOLD = None
    CS_WEIGHT = None
# Optimization parameters
CURRENT_THRESHOLD = 1000000.0      # Current penality threshold and weight
CURRENT_WEIGHT = 1E-12       # make sure weight is appropriate for the current threshold
verbose=True



import argparse
import matplotlib.pyplot as plt

from simsopt._core import load
from simsopt.field import (
    InterpolatedField,
    MaxRStoppingCriterion,
    MinRStoppingCriterion,
    MaxZStoppingCriterion,
    MinZStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    compute_fieldlines
)

DELTA_R = 0.01 # interpolated field grid spacing for R in meters
DELTA_PHI = np.pi/360 # interpolated field grid spacing for phi in radians
DELTA_Z = 0.01 # interpolated field grid spacing for Z in meters

def build_parser():
    parser = argparse.ArgumentParser(description="Template Poincare tracer script.")
    parser.add_argument("biotsavart_file", type=str, help="Path to the BiotSavart SIMSOPT JSON file.")
    parser.add_argument("surface_file", type=str, help="Path to the Surface SIMSOPT JSON file.")
    parser.add_argument("--nfieldlines", type=int, default=20, help="Number of field lines to trace. Default: 10")
    parser.add_argument("--tmax", type=float, default=5000, help="Maximum time for tracing a field line. Default: 1000")
    parser.add_argument("--ntransits", type=int, default=2500, help="Number of toroidal transits to trace for each field line. Default: 700")
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance for field line tracing. Default: 1e-7")
    parser.add_argument("--nphis", type=int, default=4, help="Number of phi locations to record field line intersections. Default: 4")
    parser.add_argument("--degree", type=int, default=3, help="Degree of piecewise polynomial interpolant for InterpolatedField. Default: 3")
    parser.add_argument("--use-original-field", action="store_true", help="If set, will use the original BiotSavart field for field line tracing instead of building an InterpolatedField.")
    return parser

def plot_poincare(res_phi_hits, surface, phis):
    nphis = len(phis)
    nrows = int(np.floor(np.sqrt(nphis)))
    ncols = int(np.ceil(nphis / nrows))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs[np.newaxis, :]
    elif ncols == 1:
        axs = axs[:, np.newaxis]
    for irow in range(nrows):
        axs[irow, 0].set_ylabel("Z (m)", fontsize=12)
    for icol in range(ncols):
        axs[-1, icol].set_xlabel("R (m)", fontsize=12)
    axs = axs.flatten()

    nfp = surface.nfp
    for iax, phi in enumerate(phis):
        ax = axs[iax]
        phi_deg = np.degrees(phi)
        frac_of_fp = (phi / (2*np.pi)) * nfp
        title = f"φ = {phi_deg:.2f}° ({frac_of_fp:.2f} field period)"
        ax.text(0.5, 0.98, title, transform=ax.transAxes, ha="center", va="top", fontsize=14)

        cs = surface.cross_section(phi/(2*np.pi))
        cs = np.append(cs, cs[:1], axis=0) # close the curve
        r = np.linalg.norm(cs[:, :2], axis=-1)
        z = cs[:, 2]
        ax.plot(r, z, c="k", lw=2, label="Boundary surface")

        idx = iax
        for line in res_phi_hits:
            _, _, x, y, z = line[line[:, 1] == idx].T
            r = np.sqrt(x**2 + y**2)
            ax.scatter(r, z, s=2)
        ax.set_box_aspect(1)
    
    axs[0].legend()
    
    return fig, axs

def main():
    args = build_parser().parse_args()

    # biotsavart = load(args.biotsavart_file) # BiotSavart object
    # surface = load(args.surface_file) # Surface object --> typically a SurfaceRZFourier or SurfaceXYZTensorFourier object
    TF_a_vals =[0.400] #, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650,0.675, 0.700,0.725,0.750,0.775,0.800,0.825,0.850,0.875,0.900]

    for TF_a_cur in TF_a_vals:
        #TG: current TF_a and TF_b are changed in optimize function
        J_cur, R0s_cur, r_rotations_cur, tf_currents_cur, wp_currents_cur, BdotN_norm_max_cur, BdotN_norm_avg_cur, BdotN_norm, biotsavart = optimize(fil_distance=fil_distance, half_per_distance=half_per_distance, dipole_radius=dipole_radius, # dipole parameters
                VV_a=VV_a, VV_b=VV_b, VV_R0=VV_R0,  # vessel parameters
                surf_s=surf_s, surf_dof_scale=surf_dof_scale, eq_dir=eq_dir, eq_name=eq_name,  # equilibrium parameters
                ntf=n_tf, num_fixed=num_fixed, field_on_axis=field_on_axis, TF_R0=TF_R0, TF_a=TF_a_cur, TF_b=TF_a_cur * VV_b / VV_a, fixed_geo_TFs=fixed_geo_TFs,
                CC_THRESHOLD=CC_THRESHOLD, CC_WEIGHT=CC_WEIGHT, CS_THRESHOLD=CS_THRESHOLD, CS_WEIGHT=CS_WEIGHT, # TF parameters
                CURRENT_THRESHOLD=CURRENT_THRESHOLD, CURRENT_WEIGHT=CURRENT_WEIGHT, 
                output_dir=run_dir, verbose=verbose)
    path = '/Users/tianlanggong/simsopt/examples/outputs/stage2_initial_conditions/wout_nfp22ginsburg_000_001242/06_ntf4_diprad_0.05_VVa_0.26_VVb_0.27_VV_R0_1.03/'
    print(f"Loading field and surface from {path}...", flush=True)
    # biotsavart = load(path + 'bs_opt.json')
    surface = load(path + 'surf_opt.json')

    # Build initial points for field line tracing.
    # Here we use the outboard midplane at phi = 0.
    phi_start = 0.0
    cs_x, cs_y, _ = surface.cross_section(phi_start).T # Surface.cross_section takes in 2pi normalized units, e.g. phi=0.5 corresponds to pi.
    cs_r = np.sqrt(cs_x**2 + cs_y**2)
    r_start_max = cs_r.max()
    r_start_min = surface.major_radius()
    r_start = np.linspace(r_start_min, r_start_max, args.nfieldlines)
    z_start = np.zeros_like(r_start)

    g_x, g_y, g_z = surface.gamma().T
    g_r = np.sqrt(g_x**2 + g_y**2)
    g_r_min, g_r_max = g_r.min(), g_r.max()
    g_z_min, g_z_max = g_z.min(), g_z.max()
    zrange_min = 0 if surface.stellsym else g_z_min

    nfp = surface.nfp
    if args.use_original_field:
        field = biotsavart
    else:
        # Build InterpolatedField for field line tracing using the maxima of the boundary surface.
        g_phi_min, g_phi_max = 0.0, 2*np.pi/nfp
        n_r = int((g_r_max - g_r_min) / DELTA_R)
        n_phi = int((g_phi_max - g_phi_min) / DELTA_PHI)
        n_z = int((g_z_max - zrange_min) / DELTA_Z)
        print(f"Building InterpolatedField with grid size (n_r, n_phi, n_z) = ({n_r}, {n_phi}, {n_z})")
        degree   = args.degree
        rrange   = (g_r_min, g_r_max, n_r)
        phirange = (g_phi_min, g_phi_max, n_phi) # Only span one field period.
        zrange   = (zrange_min, g_z_max, n_z)
        field = InterpolatedField(
            biotsavart,
            degree,
            rrange,
            phirange,
            zrange,
            extrapolate=True,
            nfp=nfp,
            stellsym=surface.stellsym,
        )

    nphis = args.nphis
    phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)

    stopping_criteria = [
        MaxRStoppingCriterion(g_r_max + 2*DELTA_R),
        MinRStoppingCriterion(g_r_min - 2*DELTA_R),
        MaxZStoppingCriterion(g_z_max + 2*DELTA_Z),
        MinZStoppingCriterion(g_z_min - 2*DELTA_Z),
        ToroidalTransitStoppingCriterion(args.ntransits, False),
    ]

    _, res_phi_hits = compute_fieldlines(field, r_start, z_start, tmax=args.tmax, tol=args.tol, phis=phis, stopping_criteria=stopping_criteria, comm=None)

    fig, _ = plot_poincare(res_phi_hits, surface, phis)

    import os

    fig.savefig(os.path.join(path, "poincare_plot.png"), dpi=150)

    np.savez(
        os.path.join(path, "poincare_res_phi_hits.npz"),
        res_phi_hits=np.array(res_phi_hits, dtype=object),
        phis=phis
    ) 

    # np.savez("poincare_res_phi_hits.npz", res_phi_hits=res_phi_hits, phis=phis)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())