from simsopt.geo import SurfaceRZFourier
from simsopt.field import coils_to_vtk, Current, ScaledCurrent
from simsopt.objectives import SquaredFlux
import numpy as np
import os
import json
from helper_functions import *

def optimize(
    fil_distance,
    half_per_distance,
    dipole_radius, # dipole parameters
    VV_a,
    VV_b,
    VV_R0,  # vessel parameters
    surf_s,
    surf_dof_scale,
    eq_dir,
    eq_name,  # equilibrium parameters
    ntf,
    num_fixed,
    field_on_axis,
    TF_R0,
    TF_a,
    TF_b,
    fixed_geo_TFs,
    CURRENT_THRESHOLD,
    CURRENT_WEIGHT,
    output_dir,
    verbose=False,
    CC_THRESHOLD=None,
    CC_WEIGHT=None,
    CS_THRESHOLD=None,
    CS_WEIGHT=None,
    wp_npol_target=None,
    wp_ntor_target=None,
):
    
    # When dipole_radius is None, both windowpane targets must be provided so
    # generate_windowpane_array can derive coil sizes from the VV geometry.
    if dipole_radius is None and (wp_npol_target is None or wp_ntor_target is None):
        raise ValueError("dipole_radius is None — must provide both wp_npol_target and wp_ntor_target")

    # Create plot configuration
    plot_config = PlotConfig(
        dpi=100,
        titlefontsize=18,
        axisfontsize=16,
        legendfontsize=14,
        ticklabelfontsize=14,
        cbarfontsize=18
    )

    # ============================================================================
    # Initialization
    # ============================================================================
    # Create the plasma surface
    eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
    surf = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range="half period", nphi=128, ntheta=64)
    surf.set_dofs(surf_dof_scale * surf.get_dofs())
    # Create a surface representing the vacuum vessel that dipoles will be placed on
    VV = SurfaceRZFourier(nfp=surf.nfp)
    VV.set_rc(0, 0, VV_R0)
    VV.set_rc(1, 0, VV_a)
    VV.set_zs(1, 0, VV_b)

    # Coil regularization radii (meters)
    tf_coil_radius = 0  # TF coil filament radius
    wp_coil_radius = 0  # WP coil filament radius, gives 5cm spacing in between coils

    # Initialize TF Coils
    # Compute I from toroidal solenoid approximation, I = B_T * 2 * pi * R0 / mu0 / (2 * nfp * ntf)
    mu0 = 4.0 * np.pi * 1e-7
    TF_current = 2.0 * np.pi * surf.get_rc(0, 0) * field_on_axis / mu0 / (2 * ntf * surf.nfp)
    base_tf_curves, base_tf_coils = generate_tf_array(
        winding_surface=VV,
        ntf=ntf,
        TF_R0=TF_R0,
        TF_a=TF_a,
        TF_b=TF_b,
        TF_current=TF_current,
        fixed_geo_tfs=fixed_geo_TFs,
        numquadpoints=64,
        tf_coil_radius=tf_coil_radius,
    )
    tf_regularizations = [c.regularization for c in base_tf_coils] if hasattr(base_tf_coils[0], "regularization") else None
    tf_coils = coils_via_symmetries(
        [c.curve for c in base_tf_coils],
        [c.current for c in base_tf_coils],
        surf.nfp,
        True,
        regularizations=tf_regularizations,
    )

    from simsopt.geo import create_equally_spaced_curves, \
    CurveLength, curves_to_vtk

    #TG: We save the vtu (coils) and the vtk (plasma) before the optimisation.
    coils_to_vtk(tf_coils, filename=os.path.join(output_dir, "IO_coils"), close=True);
    
    bs_tf = BiotSavart(tf_coils)
    bs_tf.save(os.path.join(output_dir, "bs_initial.json"))
    # plot_relBfinal_norm_modB(bs_tf, surf, output_dir, "initial", plot_config)
    if not fixed_geo_TFs:
        optimize_tfs(
            base_tf_coils=base_tf_coils,
            surf_plasma=surf,
            winding_surface=VV,
            CC_THRESHOLD=CC_THRESHOLD,
            CC_WEIGHT=CC_WEIGHT,
            CS_THRESHOLD=CS_THRESHOLD,
            CS_WEIGHT=CS_WEIGHT,
            num_fixed=num_fixed,
            definition="local",
            maxiter=2500,
            verbose=verbose,
        )
        plot_relBfinal_norm_modB(bs_tf, surf, output_dir, "post_tf_optimization", plot_config)
        
    # Initialize dipoles
    base_wp_coils, Rpol, Rtor_min, Rtor_max, nwps_poloidal, nwps_toroidal = generate_windowpane_array(
        winding_surface=VV,
        inboard_radius=dipole_radius,
        wp_fil_spacing=fil_distance,
        half_per_spacing=half_per_distance,
        wp_n=4,
        numquadpoints=64,
        order=12,
        verbose=verbose,
        wp_coil_radius=wp_coil_radius,
        nwps_poloidal_target=wp_npol_target,
        nwps_toroidal_target=wp_ntor_target,
    )
    print(f"Initialized {nwps_poloidal}x{nwps_toroidal} (npol x ntor) windowpane coils with Rpol={Rpol:.3f}, Rtor_min={Rtor_min:.3f}, Rtor_max={Rtor_max:.3f}")
    nwptot = nwps_poloidal * nwps_toroidal * 2 * surf.nfp
    plot_cross_section(surf, VV, output_dir, "stage_2", plot_config, base_dipole_coils=base_wp_coils)

    # ============================================================================
    # Optimization
    # ============================================================================
    if verbose: print(f"\n===== Starting optimization =====")
    res, bs = optimize_windowpane_currents(
        base_wp_coils=base_wp_coils,
        base_tf_coils=base_tf_coils,
        surf_plasma=surf,
        definition="local",
        precomputed=True,
        current_threshold=CURRENT_THRESHOLD,
        current_weight=CURRENT_WEIGHT,
        maxiter=2500,
        num_fixed=num_fixed,
        verbose=verbose,
    )

    # ============================================================================
    # Post-processing
    # ============================================================================
    print(f"Saving results to {output_dir}...")
    # Final Bnormal
    relBfinal_norm, mean_abs_relBfinal_norm, max_relBfinal_norm = plot_relBfinal_norm_modB(bs, surf, output_dir, "optimized", plot_config)
    Jf = SquaredFlux(surf, bs, definition="local")
    # plots currents on surface
    plot_coil_currents_on_theta_phi_grid(base_wp_coils, VV, output_dir, "optimized", plot_config)

    # Prep coil data and convert to ScaledCurrents for saving
    tf_regularizations = [c.regularization for c in base_tf_coils] if hasattr(base_tf_coils[0], "regularization") else None
    wp_regularizations = [c.regularization for c in base_wp_coils] if hasattr(base_wp_coils[0], "regularization") else None

    # Represent optimized currents as ScaledCurrent objects relative to unit base currents
    tf_base_scaled_currents = [
        ScaledCurrent(Current(1.0), c.current.get_value()) for c in base_tf_coils
    ]
    for i in range(num_fixed):
        tf_base_scaled_currents[i].fix_all()

    wp_base_scaled_currents = [
        ScaledCurrent(Current(1.0), c.current.get_value()) for c in base_wp_coils
    ]

    tf_coils = coils_via_symmetries(
        [c.curve for c in base_tf_coils],
        tf_base_scaled_currents,
        surf.nfp,
        True,
        regularizations=tf_regularizations,
    )
    wp_coils = coils_via_symmetries(
        [c.curve for c in base_wp_coils],
        wp_base_scaled_currents,
        surf.nfp,
        True,
        regularizations=wp_regularizations,
    )
    coils = tf_coils + wp_coils
    bs = BiotSavart(coils)
    tf_currents = [c.current.get_value() for c in tf_coils]
    wp_currents = [c.current.get_value() for c in wp_coils]
    

    #TG: We save the vtu (coils) and vtk (vacuum vessel denoted by VV) files after the optimisation.
    VV.to_vtk(os.path.join(output_dir, "FO_vacuum_vessel"))
    coils_to_vtk(coils, filename=os.path.join(output_dir, "FO_coils"), close=True);

    #TG: Missing intialisations added:
    plas_nPhi = 128
    plas_nTheta = 128


    bs.save(os.path.join(output_dir, "bs_opt.json"))
    # BdotN on the full torus surface
    surf_full = SurfaceRZFourier.from_wout(
        eq_name_full,
        s=surf_s,
        range="full torus",
        nphi=2 * surf.nfp * plas_nPhi,
        ntheta=plas_nTheta,
    )
    #TG: We also save the full torus surface
    surf_full.save(os.path.join(output_dir, "surf_opt.json"))



    bs.set_points(surf_full.gamma().reshape(-1, 3))
    Bdotn = np.sum(bs.B().reshape(surf_full.unitnormal().shape) * surf_full.unitnormal(), axis=2)
    modB = bs.AbsB().reshape((2 * surf.nfp * plas_nPhi, plas_nTheta))
    BdotN_norm = Bdotn / modB

    #TG: Calculate max and average of Bdotn_norm:
    BdotN_norm_max = np.max(np.abs(BdotN_norm))
    BdotN_norm_avg = np.mean(np.abs(BdotN_norm))

    surf_full.to_vtk(os.path.join(output_dir, "surf_full"), extra_data={"B_N": BdotN_norm[:, :, None]})
    bs.set_points(surf.gamma().reshape(-1, 3))

    # Extract TF optimized geometric dofs
    if not fixed_geo_TFs:
        R0s = np.zeros_like(base_tf_curves)
        r_rotations = np.zeros_like(base_tf_curves)
        for i, c in enumerate(base_tf_curves):
            R0s[i] = c.get("R0")
            r_rotations[i] = c.get("r_rotation")
    else:
        R0s = None
        r_rotations = None


    #TG: We print the desired values to see if everything has been saved properly:
    """ print(Jf.J())
    for i in range(len(R0s)):
        print(str(i + 1) +  ": " + str(R0s[i]))
    for i in range(len(r_rotations)):
        print(str(i + 1) +  ": " + str(r_rotations[i])) """
    


    results = {
        # input parameters
        "filament_distance": fil_distance,
        "half_period_distance": half_per_distance,
        "poloidal_radius": Rpol,
        "toroidal_radius_inboard": Rtor_min,
        "toroidal_radius_outboard": Rtor_max,
        "VV_a": VV_a,
        "VV_b": VV_b,
        "VV_R0": VV_R0,
        "surf_s": surf_s,
        "surf_dof_scale": surf_dof_scale,
        "eq_dir": eq_dir,
        "eq_name": eq_name,
        "ntf": ntf,
        "num_fixed": num_fixed,
        "TF_R0": TF_R0,
        "TF_a": TF_a,
        "TF_b": TF_b,
        "fixed_geo_TFs": fixed_geo_TFs,
        "CC_THRESHOLD": CC_THRESHOLD,
        "CC_WEIGHT": CC_WEIGHT,
        "CS_THRESHOLD": CS_THRESHOLD,
        "CS_WEIGHT": CS_WEIGHT,
        "field_on_axis": field_on_axis,
        "squared_flux_def": "local",
        "max_iterations": 2500,
        "current_threshold": CURRENT_THRESHOLD,
        "current_weight": CURRENT_WEIGHT,
        # derived quantities
        "surf_nfp": surf.nfp,
        "surf_major_radius": surf.major_radius(),
        "surf_minor_radius": surf.minor_radius(),
        "surf_aspect_ratio": surf.aspect_ratio(),
        "surf_volume": surf.volume(),
        "initial_tf_current": TF_current,
        "num_wps": nwptot,
        "ntoroidal": nwps_toroidal,
        "npoloidal": nwps_poloidal,
        # optimization results
        "message":                  res.message,
        "success":                  res.success,
        "iterations":               res.nit,
        "function_evaluations":     res.nfev,
        "max_tf_current": np.max(np.abs(np.array(tf_currents))),
        "min_tf_current": np.min(np.abs(np.array(tf_currents))),
        "max_wp_current": np.max(np.abs(np.array(wp_currents))),
        "min_wp_current": np.min(np.abs(np.array(wp_currents))),
        "final_squared_flux": Jf.J(),
        "avg_Bnormal": mean_abs_relBfinal_norm,
        "max_Bnormal": max_relBfinal_norm,
        "peak_wp_field": np.max(np.abs(np.array(wp_currents))) * mu0 / 2 / (dipole_radius if dipole_radius is not None else Rpol),
        "MA_meters": get_total_amp_meters(base_tf_coils, base_wp_coils, VV) / 1e6,
        "maxR0": np.max(R0s) if R0s is not None else None,
        "minR0": np.min(R0s) if R0s is not None else None,
        "avgR0": np.mean(R0s) if R0s is not None else None,
        "maxtilt": np.max(r_rotations) if r_rotations is not None else None,
        "mintilt": np.min(r_rotations) if r_rotations is not None else None,
        "avgtilt": np.mean(np.abs(r_rotations)) if r_rotations is not None else None,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)

    #TG: return J, R0, r_rotation for each iteration in driver script
    return Jf.J(), R0s, r_rotations, tf_currents, wp_currents, BdotN_norm_max, BdotN_norm_avg, BdotN_norm, bs, surf_full, results