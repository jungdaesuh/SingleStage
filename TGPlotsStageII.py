from pathlib import Path
import json
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

base_dir = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_unfixed_True")

tf_vals = []
J_vals = []
able_to_initialise = []

for folder in sorted(base_dir.glob("TF_a_*")):
    if not folder.is_dir():
        continue

    tf_a_tmp = float(folder.name.split("_")[-1])

    print(f"Processing TF_A = {tf_a_tmp}")

    results_file = folder / "results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    J_cur = results["final_squared_flux"]
    tf_a_cur = results["TF_a"]

    tf_vals.append(tf_a_cur)
    J_vals.append(J_cur)
    bs = load(folder / "bs_opt.json")
    surf_full  = load(folder / "surf_opt.json")

    #------------------- Using Wataru's template script--------------
    
    #Next, we check if we can call successfully initialise the boozer surface for TF_a_cur
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

    able_to_initialise.append(int(solve_success))


import matplotlib.pyplot as plt

tf_vals, J_vals, able_to_initialise = zip(*sorted(
    zip(tf_vals, J_vals, able_to_initialise),
    key=lambda x: x[0]
))

colors = ["green" if v == 1 else "red" for v in able_to_initialise]


plt.figure()
plt.scatter(tf_vals, J_vals, c=colors, marker="o")
plt.plot(tf_vals, J_vals, linestyle="--", alpha=0.5)  # line

plt.xlabel("TF_A")
plt.ylabel("Final squared flux (J)")
plt.title("J vs TF_A")
plt.grid(True)
plt.savefig("/burg-archive/home/tg2998/simsopt/examples/outputs/fixed_True/TF_vs_J_withcolour.png",
            dpi=300, bbox_inches="tight")
#plt.savefig("/burg-archive/home/tg2998/simsopt/examples/outputs/TF_vs_J.pdf", bbox_inches="tight")