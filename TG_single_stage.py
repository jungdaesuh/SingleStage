# Example of a single-stage optimization driver script using the Boozer surface
# method for stellarator optimization using SIMSOPT.

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
dipole_radius = 0.05 # target radius of dipoles (poloidally constant, will vary toroidally so this is at inboard midplane) [m]
# Plasma Surface
surf_s = 0.9                # value of s to cut the surface at (if already HBT sized, these will both be 1)
surf_dof_scale = 1.0      # used to scale the dofs of the surface
eq_name = 'wout_nfp22ginsburg_000_001242'  # name of the wout file from vmec
eq_dir = "/burg-archive/home/tg2998/simsopt/examples/dipoles/equilibria"
# Vacuum Vessel
VV_plas_dist = 0.12
# Extract the minimum axisymmetric VV size
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")

#TG: Added missing initialisations
plas_nPhi = 16
plas_nTheta = 8

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
fixed_geo_TFs = False                 #TG: Unfixed geometry, num_fixed is also set to 0
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


run_dir = "/burg-archive/home/tg2998/simsopt/examples/outputs/stage2_initial_conditions"

TF_a_vals =[0.400] #, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650,0.675, 0.700,0.725,0.750,0.775,0.800,0.825,0.850,0.875,0.900]

for TF_a_cur in TF_a_vals:
    #TG: current TF_a and TF_b are changed in optimize function
    J_cur, R0s_cur, r_rotations_cur, tf_currents_cur, wp_currents_cur, BdotN_norm_max_cur, BdotN_norm_avg_cur, BdotN_norm, bs, surf_full, results = optimize(fil_distance=fil_distance, half_per_distance=half_per_distance, dipole_radius=dipole_radius, # dipole parameters
            VV_a=VV_a, VV_b=VV_b, VV_R0=VV_R0,  # vessel parameters
            surf_s=surf_s, surf_dof_scale=surf_dof_scale, eq_dir=eq_dir, eq_name=eq_name,  # equilibrium parameters
            ntf=n_tf, num_fixed=num_fixed, field_on_axis=field_on_axis, TF_R0=TF_R0, TF_a=TF_a_cur, TF_b=TF_a_cur * VV_b / VV_a, fixed_geo_TFs=fixed_geo_TFs,
            CC_THRESHOLD=CC_THRESHOLD, CC_WEIGHT=CC_WEIGHT, CS_THRESHOLD=CS_THRESHOLD, CS_WEIGHT=CS_WEIGHT, # TF parameters
            CURRENT_THRESHOLD=CURRENT_THRESHOLD, CURRENT_WEIGHT=CURRENT_WEIGHT, 
            output_dir=run_dir, verbose=verbose)


#---------stage2 optimisation ends here----------






 # Running the Boozer solve requires an initial guess for the rotational
 # transform on the surface.
init_iota_guess = 0.3

# There are two methods for solving the Boozer surface partial differential
# equation (PDE). The first method BoozerLS minimizes the PDE. The second
# method BoozerExact uses a Newton solve to find an exact solution to the PDE.
# For BoozerLS, set constraint_weight to a positive value. The magnitude of
# constraint_weight determines how much the PDE constraint is enforced, where
# the PDE constraint is the surface volume.
# For BoozerExact, set constraint_weight to None.
constraint_weight = 1.0
use_boozer_exact = constraint_weight is None

biotsavart = bs
# In addition to init_iota_guess, an initial guess for G is also required. G is
# proportional to the poloidal current outside the surface. This is usually
# taken to be the total current from all coils. A reasonable guess for this
# case is the total current from the TF coils. While the sign of G could be
# inferred from the component currents, it is not always guaranteed to be
# correct.

n_tf_coils = results["ntf"] * 2 * results["surf_nfp"]  # ntf is the number of TF coils per half-period, so this is the total number
sign_g = -1
coils = biotsavart.coils
tf_coils = coils[:n_tf_coils]
total_current = sum(abs(coil.current.get_value()) for coil in tf_coils)
init_G_guess = sign_g * total_current * MU0
dipole_coils = coils[n_tf_coils:]
dipole_curves = [c.curve for c in dipole_coils]


for c in dipole_curves:
    c.fix_all()
for c in tf_coils:
    c.current.fix_all()

# The BoozerSurface class requires the surface to be SurfaceXYZTensorFourier.
# The stage 2 surface is typically a SurfaceRZFourier class so it needs to be
# converted.
init_surface = surf_full
if not isinstance(init_surface, SurfaceXYZTensorFourier):
    surface = SurfaceXYZTensorFourier(
        mpol=init_surface.mpol,
        ntor=init_surface.ntor,
        nfp=init_surface.nfp,
        stellsym=init_surface.stellsym,
        quadpoints_phi=init_surface.quadpoints_phi,
        quadpoints_theta=init_surface.quadpoints_theta
    )
    surface.least_squares_fit(init_surface.gamma())
else:
    surface = init_surface

# For BoozerExact, the number of surface quadpoints has to be related to mpol
# and ntor: number of phi quadpoints = 2*ntor + 1, number of theta quadpoints =
# 2*mpol + 1.
if use_boozer_exact:
    mpol = surface.mpol
    ntor = surface.ntor
    dofs = surface.x.copy()
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_phi=2*ntor + 1,
        quadpoints_theta=2*mpol + 1
    )
    surface.x = dofs

label = Volume(surface)
targetlabel = surface.volume()



print("Initialising Boozer Surface: ")

# custom_options = {
#     "verbose": True,
#     "bfgs_tol": 1e-2,       # Default is 1e-10 (way too strict for an initial guess)
#     "newton_tol": 1e-2,     # Default is 1e-11
#     "bfgs_maxiter": 200     # Default is 1500 (stops it from hanging forever)
# }

boozersurface = BoozerSurface(
    biotsavart, surface, label, targetlabel, constraint_weight #, options=custom_options
    
)
res = boozersurface.run_code(init_iota_guess, init_G_guess)
iota = res['iota']
G = res['G']
res_success = res['success']
try:
    not_self_intersecting = not surface.is_self_intersecting()      
except Exception as e:
    print(f"    Error checking self-intersection: {e}")
    not_self_intersecting = False
solve_success = res_success and not_self_intersecting
print(f"Solve success: {solve_success}")
print(f"    iota: {iota}")
print(f"    G: {G}")
if not res_success:
    print("    Residual solve failed.")
if not not_self_intersecting:
    print("    Surface is self-intersecting.")

J_non_qs_ratio = NonQuasiSymmetricRatio(boozersurface, biotsavart)

iota_target = init_iota_guess
J_iotas = QuadraticPenalty(Iotas(boozersurface), iota_target)

# The CurrentPenalty is a custom objective class from Jake specifically for
# optimizing the dipole currents.
dipole_index = n_tf_coils # Assumes coils are TF coils + dipoles
dipole_coils = coils[dipole_index:]
current_penalty_p_norm = 12
J_current = CurrentPenalty(
    [coil.current for coil in dipole_coils],
    p=current_penalty_p_norm
)

NONQS_WEIGHT = 1.0
IOTAS_WEIGHT = 1.0
CURRS_WEIGHT = 1.0
BZRES_WEIGHT = 1.0

if not use_boozer_exact:
    J_boozer_residual = BoozerResidual(boozersurface, biotsavart)
    JF = (
        NONQS_WEIGHT * J_non_qs_ratio +
        IOTAS_WEIGHT * J_iotas +
        CURRS_WEIGHT * J_current +
        BZRES_WEIGHT * J_boozer_residual
    )
else:
    JF = (
        NONQS_WEIGHT * J_non_qs_ratio +
        IOTAS_WEIGHT * J_iotas +
        CURRS_WEIGHT * J_current
    )

run_dict = dict(
    surface_dofs=boozersurface.surface.x.copy(),
    iota=iota,
    G=G,
    J=JF.J(),
    dJ=JF.dJ().copy(),
)

def func(x):
    boozersurface.surface.x   = run_dict["surface_dofs"]
    boozersurface.res["iota"] = run_dict["iota"]
    boozersurface.res["G"]    = run_dict["G"]

    JF.x = x
    res = boozersurface.run_code(run_dict["iota"], run_dict["G"])
    res_success = res['success']
    try:
        not_self_intersecting = not surface.is_self_intersecting()
    except Exception as e:
        print(f"    Error checking self-intersection: {e}")
        not_self_intersecting = False
    solve_success = res_success and not_self_intersecting
    print(f"Solve success: {solve_success}")
    print(f"    iota: {res['iota']}")
    print(f"    G: {res['G']}")
    if not res_success:
        print("    Residual solve failed.")
    if not not_self_intersecting:
        print("    Surface is self-intersecting.")

    if solve_success:
        J = JF.J()
        dJ = JF.dJ().copy()
    else:
        J = run_dict["J"]
        dJ = -run_dict["dJ"]
        boozersurface.surface.x   = run_dict["surface_dofs"]
        boozersurface.res["iota"] = run_dict["iota"]
        boozersurface.res["G"]    = run_dict["G"]

    return J, dJ

def callback(x):
    run_dict["surface_dofs"] = boozersurface.surface.x.copy()
    run_dict["iota"]         = boozersurface.res["iota"]
    run_dict["G"]            = boozersurface.res["G"]
    run_dict["J"]            = JF.J()
    run_dict["dJ"]           = JF.dJ().copy()
    print(f"Callback: ")
    print(f"    iota: {run_dict['iota']}")
    print(f"       G: {run_dict['G']}")
    print(f"       J: {run_dict['J']}")
    print(f"    |dJ|: {np.linalg.norm(run_dict['dJ'])}")

maxiter = 150
gtol = 1e-2

min_res = minimize(
    func, 
    JF.x.copy(), 
    jac=True, 
    callback=callback, 
    method="BFGS",
    options={'maxiter': maxiter, 'gtol': gtol}
)
print(min_res.message)

output_dir = "/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs"
savefile = os.path.join(output_dir, f"optimized_boozer_surface_{TF_a_cur}.json")
boozersurface.save(savefile)
print(f"Optimized Boozer surface successfully saved to: {savefile}")

