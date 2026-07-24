from pathlib import Path
import numpy as np

from simsopt._core import load
from simsopt.field import CurrentPenalty
from simsopt.geo import BoozerSurface, BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.objectives import QuadraticPenalty

TF_a_list = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425]
n_tf = 4  # must match n_tf_coils used at optimization time

NONQS_WEIGHT = 1.0
IOTAS_WEIGHT = 1.0
CURRS_WEIGHT = 1.0
BZRES_WEIGHT = 1.0
current_penalty_p_norm = 12
iota_target = 0.3       # matches init_iota_guess used originally
init_iota_guess = 0.3
sign_g = -1              # matches sign_g used at optimization time
MU0 = 4e-7 * np.pi

output_dir = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")

for TF_a in TF_a_list:
    print(f"\n=== TF_a = {TF_a:.3f} ===")
    boozersurface = load(output_dir / f"optimized_boozer_surface_{TF_a:.3f}.json")
    biotsavart = boozersurface.biotsavart
    coils = biotsavart.coils

    tf_coils = coils[:n_tf]
    dipole_coils = coils[n_tf:]

    # --- Re-populate boozersurface.res, which is not preserved by save/load ---
    total_current = sum(abs(coil.current.get_value()) for coil in tf_coils)
    init_G_guess = sign_g * total_current * MU0
    res = boozersurface.run_code(init_iota_guess, init_G_guess)
    print(f"  re-solved Boozer surface: success={res['success']}, "
          f"iota={res['iota']:.6f}, G={res['G']:.6f}")
    if not res['success']:
        print("  WARNING: Boozer surface re-solve did not converge, "
              "gradient below may be unreliable.")

    # --- Rebuild JF exactly as in the optimization script ---
    J_non_qs_ratio = NonQuasiSymmetricRatio(boozersurface, biotsavart)
    J_iotas = QuadraticPenalty(Iotas(boozersurface), iota_target)
    J_current = CurrentPenalty(
        [coil.current for coil in dipole_coils],
        p=current_penalty_p_norm
    )
    J_boozer_residual = BoozerResidual(boozersurface, biotsavart)

    JF = (
        NONQS_WEIGHT * J_non_qs_ratio +
        IOTAS_WEIGHT * J_iotas +
        CURRS_WEIGHT * J_current +
        BZRES_WEIGHT * J_boozer_residual
    )

    grad = JF.dJ()
    dof_names = JF.dof_names

    tf_curve_names = [coil.curve.name for coil in tf_coils]

    tf_idx = [i for i, name in enumerate(dof_names)
              if any(name.startswith(cn) for cn in tf_curve_names)]
    other_idx = [i for i in range(len(dof_names)) if i not in tf_idx]

    grad_tf = grad[tf_idx]
    grad_other = grad[other_idx]

    print(f"  # TF-coil DOFs: {len(tf_idx)}, # other DOFs: {len(other_idx)}")
    print(f"  max|grad| TF coils   : {np.max(np.abs(grad_tf)):.6e}")
    print(f"  max|grad| other DOFs : {np.max(np.abs(grad_other)):.6e}")
    ratio = np.max(np.abs(grad_other)) / max(np.max(np.abs(grad_tf)), 1e-300)
    print(f"  ratio (other/TF)     : {ratio:.3e}")

    # Per-coil, per-DOF-type breakdown for TF coils (rotation is dof index 3 within each curve)
    for i, coil in enumerate(tf_coils):
        cn = coil.curve.name
        idx = [j for j, n in enumerate(dof_names) if n.startswith(cn)]
        if len(idx) >= 4:
            rot_idx = idx[3]
            other_curve_idx = [j for j in idx if j != rot_idx]
            max_other = np.max(np.abs(grad[other_curve_idx])) if other_curve_idx else float('nan')
            print(f"    TF coil {i}: |grad_rotation| = {abs(grad[rot_idx]):.3e}, "
                  f"max|grad_other_curve_dofs| = {max_other:.3e}")
        else:
            print(f"    TF coil {i}: only {len(idx)} dofs found, skipping rotation breakdown")