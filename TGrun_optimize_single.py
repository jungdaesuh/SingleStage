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
num_fixed = 0 #n_tf                  # number of TF coil currents to fix during combined optimization
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

extra = ""
# only add these parameters to output file if they are unusual runs, can add more as needed
if VV_a != 0.25:
    extra = extra + f"_VVa_{VV_a:.2f}"
if VV_b != 0.25:
    extra = extra + f"_VVb_{VV_b:.2f}"
if VV_R0 != 1.0:
    extra = extra + f"_VV_R0_{VV_R0:.2f}"


# Name the directory the next optimization number
if not os.path.exists(os.path.join(run_dir, f'{eq_name}')) or not os.listdir(os.path.join(run_dir, f'{eq_name}')):
    num = 1
else:
    num = max(int(f.split('_')[0]) for f in os.listdir(os.path.join(run_dir, f'{eq_name}')) if f.split('_')[0].isdigit()) + 1
run_dir = os.path.join(run_dir, f'{eq_name}/{num:02}_ntf{n_tf}_diprad_{dipole_radius}{extra}/')

###### Change directory
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

# Copy all helpful files to document the optimization here
shutil.copy(os.path.join(script_dir, "optimize.py"), os.path.join(run_dir, "optimize.py"), follow_symlinks=True)
shutil.copy(os.path.join(script_dir, 'helper_functions.py'), os.path.join(run_dir, 'helper_functions.py'), follow_symlinks=True)
""" 
optimize(fil_distance=fil_distance, half_per_distance=half_per_distance, dipole_radius=dipole_radius, # dipole parameters
            VV_a=VV_a, VV_b=VV_b, VV_R0=VV_R0,  # vessel parameters
            surf_s=surf_s, surf_dof_scale=surf_dof_scale, eq_dir=eq_dir, eq_name=eq_name,  # equilibrium parameters
            ntf=n_tf, num_fixed=num_fixed, field_on_axis=field_on_axis, TF_R0=TF_R0, TF_a=TF_a, TF_b=TF_b, fixed_geo_TFs=fixed_geo_TFs,
            CC_THRESHOLD=CC_THRESHOLD, CC_WEIGHT=CC_WEIGHT, CS_THRESHOLD=CS_THRESHOLD, CS_WEIGHT=CS_WEIGHT, # TF parameters
            CURRENT_THRESHOLD=CURRENT_THRESHOLD, CURRENT_WEIGHT=CURRENT_WEIGHT, 
            output_dir=run_dir, verbose=verbose) """

#TG: We saved the vtu (coils) and vtk (plasma) before and after the optimisation in the optimize function.

#TG: Heatmap of BdotN_norm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
TF_a_vals =[0.400] #, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650,0.675, 0.700,0.725,0.750,0.775,0.800,0.825,0.850,0.875,0.900]

for TF_a_cur in TF_a_vals:
    #TG: current TF_a and TF_b are changed in optimize function
    J_cur, R0s_cur, r_rotations_cur, tf_currents_cur, wp_currents_cur, BdotN_norm_max_cur, BdotN_norm_avg_cur, BdotN_norm, bs = optimize(fil_distance=fil_distance, half_per_distance=half_per_distance, dipole_radius=dipole_radius, # dipole parameters
            VV_a=VV_a, VV_b=VV_b, VV_R0=VV_R0,  # vessel parameters
            surf_s=surf_s, surf_dof_scale=surf_dof_scale, eq_dir=eq_dir, eq_name=eq_name,  # equilibrium parameters
            ntf=n_tf, num_fixed=num_fixed, field_on_axis=field_on_axis, TF_R0=TF_R0, TF_a=TF_a_cur, TF_b=TF_a_cur * VV_b / VV_a, fixed_geo_TFs=fixed_geo_TFs,
            CC_THRESHOLD=CC_THRESHOLD, CC_WEIGHT=CC_WEIGHT, CS_THRESHOLD=CS_THRESHOLD, CS_WEIGHT=CS_WEIGHT, # TF parameters
            CURRENT_THRESHOLD=CURRENT_THRESHOLD, CURRENT_WEIGHT=CURRENT_WEIGHT, 
            output_dir=run_dir, verbose=verbose)

    phi_1d = surf.quadpoints_phi
    theta_1d = surf.quadpoints_theta

    PHI, THETA = np.meshgrid(phi_1d, theta_1d, indexing='ij')
    vmax = np.max(np.abs(BdotN_norm))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.seismic
    fig, ax = plt.subplots(figsize=(8,6))
    heatmap = ax.pcolormesh(PHI, THETA, BdotN_norm, cmap=cmap, norm=norm, shading='gouraud')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel("BdotN_norm", fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 0.95)
    ax.set_xlim(-0.005, 0.255) 
    ax.set_title(f'BdotN_norm Heatmap (TF_a = {TF_a_cur:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
plt.show()