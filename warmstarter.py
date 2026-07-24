import numpy as np

from scipy.optimize import minimize

from simsopt._core import load
from simsopt.field import CurrentPenalty

from simsopt.geo import (
    BoozerSurface,
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume
)
from simsopt.objectives import QuadraticPenalty


MU0 = 4e-7 * np.pi

#----------calculating initialistions from stage 2 optimisation------

import os
from optimize import optimize
from simsopt.geo import SurfaceRZFourier

### Simulation parameters ###
# Dipole parameters
fil_distance = 0.05 # distance between dipole filaments for finite coil winding pack [m]
half_per_distance = 0.05 # distance between dipole panels between half field periods of the device [m]
dipole_radius = None # target radius of dipoles (poloidally constant, will vary toroidally so this is at inboard midplane) [m]
# Plasma Surface
surf_s = 0.9                # value of s to cut the surface at (if already HBT sized, these will both be 1)
surf_dof_scale = 1.0      # used to scale the dofs of the surface
eq_name = 'wout_nfp22ginsburg_000_001242'  # name of the wout file from vmec
eq_dir = "/burg-archive/home/tg2998/simsopt/examples/dipoles/equilibria"
# Vacuum Vessel
VV_plas_dist = 0.12 #[0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
# Extract the minimum axisymmetric VV size
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")

#TG: Added missing initialisations
plas_nPhi = 64
plas_nTheta = 64

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
fixed_geo_TFs = False                 #TG: Unfixed geometry, num_fixed is also set to 0
CC_THRESHOLD = 0.1
CC_WEIGHT = 100        # [0.1, 1.0, 10.0, 100] 
CS_THRESHOLD = 0.1   
CS_WEIGHT = 100        # [0]
if fixed_geo_TFs:
    CC_THRESHOLD = None
    CC_WEIGHT = None 
    CS_THRESHOLD = None
    CS_WEIGHT = None
# Optimization parameters
CURRENT_THRESHOLD = 5000000.0      # Current penality threshold and weight
CURRENT_WEIGHT = 1E-12       # make sure weight is appropriate for the current threshold
verbose=True


base_run_dir = "/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed1"

TF_a_vals = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650,0.675, 0.700,0.725,0.750,0.775,0.800,0.825,0.850,0.875,0.900]
#CURRENT_THRESHOLD = [1e1] #, 1e5, 1e4, 1e3, 1e2, 1e1
#CURRENT_WEIGHT = [1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

for TF_a_cur in TF_a_vals:
    #for CURRENT_THRESHOLD_cur in CURRENT_THRESHOLD:
        #for CURRENT_WEIGHT_cur in CURRENT_WEIGHT:

    run_dir = os.path.join(base_run_dir, f"TF_a_{TF_a_cur:.3f}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n---> Running optimization for TF_a = {TF_a_cur:.3f}")
    print(f"---> Saving outputs to: {run_dir}\n")
    #TG: current TF_a and TF_b are changed in optimize function
    J_cur, R0s_cur, r_rotations_cur, tf_currents_cur, wp_currents_cur, BdotN_norm_max_cur, BdotN_norm_avg_cur, BdotN_norm, bs, surf_full, results = optimize(fil_distance=fil_distance, half_per_distance=half_per_distance, dipole_radius=dipole_radius, # dipole parameters
            VV_a=VV_a, VV_b=VV_b, VV_R0=VV_R0,  # vessel parameters
            surf_s=surf_s, surf_dof_scale=surf_dof_scale, eq_dir=eq_dir, eq_name=eq_name,  # equilibrium parameters
            ntf=n_tf, num_fixed=num_fixed, field_on_axis=field_on_axis, TF_R0=TF_R0, TF_a=TF_a_cur, TF_b=TF_a_cur * VV_b / VV_a, fixed_geo_TFs=fixed_geo_TFs,
            CC_THRESHOLD=CC_THRESHOLD, CC_WEIGHT=CC_WEIGHT, CS_THRESHOLD=CS_THRESHOLD, CS_WEIGHT=CS_WEIGHT, # TF parameters
            CURRENT_THRESHOLD=CURRENT_THRESHOLD, CURRENT_WEIGHT=CURRENT_WEIGHT, 
            output_dir=run_dir, verbose=verbose, wp_npol_target=10, wp_ntor_target=8)





# 0.625 is running the optimisation from fixed current, unfixed geometry, 
# 0.650 is running the optimisation from fixed current, unfixed geometry, but then we unfix the currents
# 0.675 is running the optimisation from fixed current, fixed geometry, but then we unfix the currents
# 0.700 is running the optimisation from fixed current, fixed geomery, but then we unfix both currents and geometry

'''TF_VALUES=(
    0.300
    0.325
    0.350
    0.375
    0.400
    0.425
    0.450
    0.475
    0.500
    0.525
)''' # batch 1

# batch 2 is 0.550 to 0.775