import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from simsopt._core import load

from simsopt.geo import (
    BoozerSurface,
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    Volume
)
from simsopt.objectives import QuadraticPenalty

from stage2_contracts import (
    BFGS_MAXITER,
    BFGS_TOLERANCE,
    CERTIFICATE_POLICY_ID,
    CERTIFICATE_THRESHOLDS,
    BOOZER_RESIDUAL_THRESHOLD,
    CURRENT_P_NORM,
    CURRENT_SCALE_AMPERES,
    NEWTON_MAXITER,
    NEWTON_TOLERANCE,
    BoozerSolveError,
    apply_field_polarity,
    build_boozer_solve_surface,
    boozer_residual_guard,
    certify_boozer_solution,
    create_phase_journal,
    current_limit_penalty,
    current_p_norm_amperes,
    polarity_output_directory,
    publish_boozer_artifact,
    iota_penalty_curvature,
    restore_accepted_boozer_state,
    require_valid_boozer_result,
    solve_boozer_initialization,
    solve_boozer_newton,
    solve_boozer_outer_trial,
)
from bounded_bfgs import minimize_bounded_bfgs, rejected_trial_value_and_gradient
from coil_layout import layout_for_nfp

DEFAULT_STAGE2_ROOT = Path(
    "/burg-archive/home/tg2998/simsopt/examples/outputs/"
    "TG_warmstarter_stage2results_fixed"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/burg-archive/home/tg2998/simsopt/examples/outputs/"
    "TG_single_stage_outputs"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("TF_a", type=float)
    parser.add_argument("--field-polarity", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--initial-iota", type=float, default=0.045)
    parser.add_argument("--iota-target", type=float, required=True)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument(
        "--outer-step-radius",
        type=float,
        default=0.15,
        help="Maximum L2 change in the outer DOF vector per evaluated trial. "
        "The outer DOFs are the dipole ScaledCurrent multipliers "
        "(dimensionless fractions of each dipole's Stage-2 current), so the "
        "radius is dimensionless (default: 0.15).",
    )
    parser.add_argument(
        "--current-limit-amperes",
        type=float,
        default=CURRENT_SCALE_AMPERES,
        help="Soft current p-norm threshold in AMPERES; currents below it cost "
        f"nothing in the objective (default: {CURRENT_SCALE_AMPERES:.0f}, the "
        "production fcp150kA policy).",
    )
    parser.add_argument(
        "--current-pnorm-p",
        type=int,
        default=CURRENT_P_NORM,
        help=f"Order of the current p-norm, which must be > 1 "
        f"(default: {CURRENT_P_NORM}); higher "
        "approaches max|I|.",
    )
    parser.add_argument(
        "--mpol",
        type=int,
        default=6,
        help="Poloidal Fourier order of the Boozer solve surface (default: 6). "
        "Quadrature and certification densities derive from it.",
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=6,
        help="Toroidal Fourier order of the Boozer solve surface (default: 6). "
        "Quadrature and certification densities derive from it.",
    )
    parser.add_argument(
        "--volume-target",
        type=float,
        default=None,
        help="Volume label (m^3) pinning the Boozer surface. Default: the "
        "fitted Stage-2 boundary's own volume. Overriding solves a DIFFERENT "
        "flux surface than the seed boundary — recorded in metadata.",
    )
    parser.add_argument("--stage2-root", type=Path, default=DEFAULT_STAGE2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if not np.isfinite(args.TF_a) or args.TF_a <= 0.0:
        parser.error("TF_a must be finite and positive")
    if not np.isfinite(args.initial_iota):
        parser.error("--initial-iota must be finite")
    if not np.isfinite(args.iota_target):
        parser.error("--iota-target must be finite")
    if args.maxiter < 1:
        parser.error("--maxiter must be a positive integer")
    if not np.isfinite(args.current_limit_amperes) or args.current_limit_amperes <= 0.0:
        parser.error("--current-limit-amperes must be finite and positive")
    if args.current_pnorm_p <= 1:
        parser.error("--current-pnorm-p must be an integer greater than 1")
    if args.mpol < 1 or args.ntor < 1:
        parser.error("--mpol and --ntor must be positive integers")
    if args.volume_target is not None and not (
        np.isfinite(args.volume_target) and args.volume_target > 0.0
    ):
        parser.error("--volume-target must be finite and positive")
    if not np.isfinite(args.outer_step_radius) or args.outer_step_radius <= 0.0:
        parser.error("--outer-step-radius must be finite and positive")
    return args


args = parse_args()
TF_a_cur = args.TF_a

#TF_a_vals = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650,0.675, 0.700,0.725,0.750,0.775,0.800,0.825,0.850,0.875,0.900]

#for TF_a_cur in TF_a_vals: 
base_dir = args.stage2_root

folder = base_dir / f"TF_a_{TF_a_cur:.3f}"
TF_a_tmp = float(folder.name.split("_")[-1])
output_dir = polarity_output_directory(args.output_root, args.field_polarity)
output_dir.mkdir(parents=True, exist_ok=True)
phase_journal = create_phase_journal(
    output_dir,
    TF_a=TF_a_cur,
)

print(f"Processing TF_A = {TF_a_tmp}")

results_file = folder / "results.json"

with open(results_file, "r") as f:
    results = json.load(f)

bs = load(folder / "bs_opt.json")
surf_full  = load(folder / "surf_opt.json")




#------------------- Initialising using Wataru's template script--------------

#Next, we check if we can call successfully initialise the boozer surface for TF_a_cur
init_iota_guess = args.initial_iota

# A positive constraint_weight selects BoozerLS and weights the separate
# surface-label and z(0, 0) gauge penalties. Here the label is surface volume;
# the Boozer PDE residual itself remains the least-squares objective.
constraint_weight = 1.0


biotsavart = bs
n_tf_coils = results["ntf"] * 2 * results["surf_nfp"]
layout = layout_for_nfp(results["surf_nfp"])
if n_tf_coils != layout.tf_physical_count:
    raise ValueError("Stage-II TF count disagrees with coil-layout SSOT")
coils = biotsavart.coils
tf_coils = coils[:n_tf_coils]
dipole_coils = coils[n_tf_coils:]
if len(dipole_coils) != layout.dipole_physical_count:
    raise ValueError("Stage-II dipole count disagrees with coil-layout SSOT")
init_G_guess = apply_field_polarity(
    coils,
    tf_coils,
    polarity=args.field_polarity,
    ntf=results["ntf"],
    nfp=results["surf_nfp"],
    stellsym=surf_full.stellsym,
)

for coil in dipole_coils:
    coil.curve.fix_all()
    coil.current.unfix_all()

# TF curves are FIXED: the outer DOF vector is dipole-current multipliers only
# (dimensionless, homogeneous), which is what makes the L2 step bound below
# physically meaningful — mixing free TF curve coordinates (metres) into the
# same vector has no repo precedent for a length scale. Same policy as
# single_stage_dipoles.py. Measured 2026-07-29 (ablation E5a/E5b): with TF
# curves free the normalized outer optimization accepts zero steps; with
# currents only it descends monotonically. Freeing TF geometry again is an
# open physics question owned by Tian (fix plan, "TF coil DOF scope").
for coil in tf_coils:
    coil.curve.fix_all()
    coil.current.fix_all()


# This bridge uses SurfaceXYZTensorFourier for its BoozerLS solve, while the
# Stage-2 surface is SurfaceRZFourier and must be fitted into that representation.

mpol = args.mpol
ntor = args.ntor
lossy_seed_fit = mpol < surf_full.mpol or ntor < surf_full.ntor
if lossy_seed_fit:
    print(
        f"WARNING: solve orders (mpol={mpol}, ntor={ntor}) are below the "
        f"Stage-2 boundary's ({surf_full.mpol}, {surf_full.ntor}); the fit is "
        "lossy and the volume gate cannot detect it."
    )
surface = build_boozer_solve_surface(surf_full, mpol=mpol, ntor=ntor)

label = Volume(surface)
targetlabel = (
    surface.volume() if args.volume_target is None else args.volume_target
)



print("Initialising Boozer Surface: ")

boozersurface = BoozerSurface(
    biotsavart,
    surface,
    label,
    targetlabel,
    constraint_weight,
)
res, initialization_reports = solve_boozer_initialization(
    boozersurface,
    init_iota_guess,
    init_G_guess,
    phase_journal=phase_journal,
)
require_valid_boozer_result(
    res,
    surface,
    expected_G_sign=args.field_polarity,
    nfp=surface.nfp,
)
iota = res['iota']
G = res['G']
print("Solve success: True")
print(f"    iota: {iota}")
print(f"    G: {G}")

print(
    f"Surface resolution: mpol={surface.mpol}, ntor={surface.ntor}, "
    f"nphi={len(surface.quadpoints_phi)}, ntheta={len(surface.quadpoints_theta)}"
)


J_non_qs_ratio = NonQuasiSymmetricRatio(boozersurface, biotsavart)

iota_target = args.iota_target
J_iotas = QuadraticPenalty(Iotas(boozersurface), iota_target)

NONQS_WEIGHT = 1.0
IOTAS_WEIGHT = 1.0
CURRS_WEIGHT = 1.0
BZRES_WEIGHT = 1.0

# Every weighted term below is dimensionless and O(1) at its reference
# deviation. nonQS is a ratio; the iota penalty is scaled so an error of one
# IOTA_SCALE (= |iota_target|) contributes 0.5. BoozerResidual is a
# residual-squared objective, so it enters through a normalized one-sided
# guard that is zero below the fixed-order residual threshold.
# The current term is one-sided by measurement, not by taste — measured
# 2026-07-29 (ablations E5g vs E5h): minimizing the raw p-norm holds iota at
# 0.0238 against a 0.05 target; the threshold form reaches 0.0498 in 10 outer
# iterations with the current p-norm settling near the soft threshold.
J_boozer_residual = BoozerResidual(boozersurface, biotsavart)
JF = (
    NONQS_WEIGHT * J_non_qs_ratio +
    (IOTAS_WEIGHT * iota_penalty_curvature(iota_target)) * J_iotas +
    CURRS_WEIGHT * current_limit_penalty(
        [coil.current for coil in dipole_coils],
        p=args.current_pnorm_p,
        limit_amperes=args.current_limit_amperes,
    ) +
    BZRES_WEIGHT * boozer_residual_guard(J_boozer_residual)
)

run_dict = dict(
    optimizer_dofs=JF.x.copy(),
    surface_dofs=boozersurface.surface.x.copy(),
    iota=iota,
    G=G,
    J=JF.J(),
    dJ=JF.dJ().copy(),
    rejected_boozer_trials=0,
)
boozer_phase_reports = list(initialization_reports)

def func(x):
    boozersurface.surface.x   = run_dict["surface_dofs"]
    boozersurface.res["iota"] = run_dict["iota"]
    boozersurface.res["G"]    = run_dict["G"]

    JF.x = x
    try:
        res, phase_report = solve_boozer_outer_trial(
            boozersurface,
            run_dict["iota"],
            run_dict["G"],
            phase_journal=phase_journal,
        )
        boozer_phase_reports.append(phase_report)
        require_valid_boozer_result(
            res,
            surface,
            expected_G_sign=args.field_polarity,
            nfp=surface.nfp,
        )
    except BoozerSolveError as error:
        run_dict["rejected_boozer_trials"] += 1
        J, dJ = rejected_trial_value_and_gradient(
            x,
            run_dict["optimizer_dofs"],
            run_dict["J"],
        )
        restore_accepted_boozer_state(
            boozersurface,
            JF,
            optimizer_dofs=run_dict["optimizer_dofs"],
            surface_dofs=run_dict["surface_dofs"],
            iota=run_dict["iota"],
            G=run_dict["G"],
        )
        print(f"Boozer trial rejected: {error}")
        return J, dJ, False

    print("Solve success: True")
    print(f"    iota: {res['iota']}")
    print(f"    G: {res['G']}")
    return JF.J(), JF.dJ().copy(), True

def callback(x):
    if not np.array_equal(np.asarray(JF.x), np.asarray(x)):
        raise RuntimeError("optimizer callback does not match the accepted state")
    run_dict["optimizer_dofs"] = np.asarray(x).copy()
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

maxiter = args.maxiter
gtol = 1e-2

min_res = minimize_bounded_bfgs(
    func,
    JF.x.copy(),
    initial_value=run_dict["J"],
    initial_gradient=run_dict["dJ"],
    callback=callback,
    maxiter=maxiter,
    gtol=gtol,
    max_step_norm=args.outer_step_radius,
)
print(min_res.message)
if not min_res.success:
    raise RuntimeError(
        "Outer single-stage optimization failed; no artifact was saved: "
        f"{min_res.message}"
    )
# The gate re-verifies the optimizer's own convergence contract, which is an
# INF-norm criterion (bounded_bfgs, like scipy BFGS, terminates on
# ||grad||_inf <= gtol). An L2 check here — the template's original form — is
# strictly tighter by up to sqrt(dim) and rejects runs the optimizer
# legitimately converged.
outer_gradient_norm = float(np.linalg.norm(min_res.jac, ord=np.inf))
if not np.isfinite(outer_gradient_norm) or outer_gradient_norm > gtol:
    raise RuntimeError(
        "Outer optimization gradient gate failed; no artifact was saved: "
        f"|dJ|_inf={outer_gradient_norm:.6e}, threshold={gtol:.6e}"
    )

# Reapply and certify the optimizer's reported solution before publication.
boozersurface.surface.x = run_dict["surface_dofs"]
boozersurface.res["iota"] = run_dict["iota"]
boozersurface.res["G"] = run_dict["G"]
JF.x = np.asarray(min_res.x).copy()
boozersurface.need_to_run_code = True
final_result, final_report = solve_boozer_newton(
    boozersurface,
    run_dict["iota"],
    run_dict["G"],
    phase="final_search_grid_newton",
    phase_journal=phase_journal,
)
boozer_phase_reports.append(final_report)
require_valid_boozer_result(
    final_result,
    surface,
    expected_G_sign=args.field_polarity,
    nfp=surface.nfp,
)
certified_boozer_surface, certificate, refinement_report = certify_boozer_solution(
    boozersurface,
    biotsavart,
    expected_G_sign=args.field_polarity,
    iota_target=iota_target,
    phase_journal=phase_journal,
)
boozer_phase_reports.append(refinement_report)
certified_result = certified_boozer_surface.res
final_current_p_norm = current_p_norm_amperes(
    [coil.current for coil in dipole_coils],
    p=args.current_pnorm_p,
)

phase_journal_bytes = phase_journal.path.read_bytes()
metadata = {
    "metadata_schema_version": 2,
    "TF_a": TF_a_cur,
    "field_polarity": args.field_polarity,
    "input_hashes": {
        filename: sha256((folder / filename).read_bytes()).hexdigest()
        for filename in ("bs_opt.json", "surf_opt.json", "results.json")
    },
    "phase_journal": str(phase_journal.path),
    "phase_journal_sha256": sha256(phase_journal_bytes).hexdigest(),
    "initial_iota": init_iota_guess,
    "iota_target": iota_target,
    "objective_scales": {
        "boozer_residual_soft_threshold": BOOZER_RESIDUAL_THRESHOLD,
        "current_p_norm": args.current_pnorm_p,
        "current_scale_amperes": args.current_limit_amperes,
        "current_threshold_kind": "soft-objective-threshold",
        "volume_target_source": (
            "fitted-seed-volume" if args.volume_target is None else "cli-override"
        ),
        "volume_target_m3": float(targetlabel),
        "iota_penalty_curvature": iota_penalty_curvature(iota_target),
    },
    "final_current_p_norm_amperes": final_current_p_norm,
    "final_iota": float(certified_result["iota"]),
    "initial_G": float(init_G_guess),
    "final_G": float(certified_result["G"]),
    "mpol": certified_boozer_surface.surface.mpol,
    "ntor": certified_boozer_surface.surface.ntor,
    "source_mpol": surf_full.mpol,
    "source_ntor": surf_full.ntor,
    "lossy_seed_fit": lossy_seed_fit,
    "nphi": len(certified_boozer_surface.surface.quadpoints_phi),
    "ntheta": len(certified_boozer_surface.surface.quadpoints_theta),
    "search_nphi": len(surface.quadpoints_phi),
    "search_ntheta": len(surface.quadpoints_theta),
    "solve_grid_points": len(surface.quadpoints_phi) * len(surface.quadpoints_theta),
    "coil_count": len(coils),
    "outer_optimizer": "bounded_bfgs",
    "outer_maxiter": maxiter,
    "outer_gradient_tolerance": gtol,
    "outer_step_radius": args.outer_step_radius,
    "outer_dof_policy": "dipole-current-multipliers-only; TF curves fixed",
    "optimizer_success": bool(min_res.success),
    "optimizer_message": str(min_res.message),
    "optimizer_iterations": int(min_res.nit),
    "outer_objective_evaluations": int(min_res.nfev),
    "outer_gradient_norm": outer_gradient_norm,
    "outer_gradient_norm_kind": "infinity",
    "boozer_phase_reports": [report.to_dict() for report in boozer_phase_reports],
    "boozer_total_iterations": sum(
        report.iterations for report in boozer_phase_reports
    ),
    "boozer_total_seconds": sum(
        report.elapsed_seconds for report in boozer_phase_reports
    ),
    "rejected_boozer_trials": run_dict["rejected_boozer_trials"],
    "boozer_solver_policy": {
        "initial_bfgs_maxiter": BFGS_MAXITER,
        "initial_bfgs_tolerance": BFGS_TOLERANCE,
        "newton_maxiter": NEWTON_MAXITER,
        "newton_tolerance": NEWTON_TOLERANCE,
    },
    "certificate": certificate.to_dict(),
    "certificate_policy_id": CERTIFICATE_POLICY_ID,
    "certificate_thresholds": CERTIFICATE_THRESHOLDS.to_dict(),
}
savefile, metadata_file, publication_digest = publish_boozer_artifact(
    certified_boozer_surface,
    metadata,
    output_directory=output_dir,
    TF_a=TF_a_cur,
)
print(f"Fixed-order validated Boozer surface saved to: {savefile}")
print(f"Run metadata saved to: {metadata_file}")
print(f"Publication digest: {publication_digest}")
