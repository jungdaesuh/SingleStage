"""
Single-iota soft-penalty optimization at one fixed surface resolution.

The run is initialized from ``INIT_DIR`` and writes its output under
  <output-root>/<eq_name>/iota{X}_fcp{Y}kA_vt{Z}_G_{positive|negative}/mpol{M}_ntor{N}/
so opposite physical field polarities cannot overwrite one another.

A RuntimeWarning (not an error) is raised if the run finishes
with a Boozer residual that is more than 5 % above its target fb_threshold.

Usage:
  python single_stage_dipoles.py \\
      --init-dir /burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results/TF_a_0.400 \\
      --iota-target 0.15 \\
      --f-cp-threshold 150000 \\
      --field-polarity 1 \\
      --fb-threshold 5e-5
      
Re-running with the same settings skips a run that already has a converged
results.json; pass --new to force a full re-run.

Pass --sparse to remove outboard-midplane dipoles before optimization.
"""

import os
import sys
import json
import argparse
import warnings
import time
import numpy as np
from datetime import datetime
from scipy.optimize import OptimizeResult

from simsopt.geo import SurfaceRZFourier
from simsopt.geo.surfaceobjectives import BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.field import BiotSavart, coils_to_vtk, CurrentPenalty
from simsopt.objectives import QuadraticPenalty
from simsopt._core.optimizable import load, save
import matplotlib.pyplot as plt
from helper_functions import *
from boozer_functions import *
from bounded_bfgs import minimize_bounded_bfgs, rejected_trial_value_and_gradient
from run_configuration import RunConfiguration, prepare_output_generation


def coil_center_theta(coil, major_radius):
    gamma = coil.curve.gamma()
    center = np.mean(gamma, axis=0)
    x0, y0, z0 = center
    r0 = np.sqrt(x0 * x0 + y0 * y0)
    return np.mod(np.arctan2(z0, r0 - major_radius), 2 * np.pi)


def angular_distance_to_zero(theta):
    return np.minimum(theta, 2 * np.pi - theta)


# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(
    description=(
        "Single-iota soft-penalty run at one fixed mpol=ntor resolution."
    )
)
parser.add_argument("--init-dir", type=str, required=True,
    help="Directory with bs_opt.json, results.json, and surf_opt.json used to start "
         "the optimization. Can be either a stage-2 output directory or "
         "a previous single-stage run directory.")
parser.add_argument("--iota-target", type=float, required=True,
    help="Single iota target for this run.")
parser.add_argument("--f-cp-threshold", type=float, required=True,
    help="Current p-norm upper bound [A].")
parser.add_argument("--field-polarity", type=int, choices=(-1, 1), required=True,
    help="Expected physical TF-current polarity. Must match the Stage-2 input.")
parser.add_argument("--fb-threshold", type=float, required=True,
    help="Boozer residual upper bound at the fixed mpol=ntor=8 resolution. A "
         "RuntimeWarning is raised (not an error) if a "
         "step finishes with residual > threshold * 1.05.")
parser.add_argument("--iota-threshold", type=float, default=0.0025,
    help="Absolute iota tolerance (default: 0.0025). Publication gate: a run whose final iota is outside this band writes no final artifact.")
parser.add_argument("--iota-penalty-weight", type=float, default=1.0,
    help="Weight of the soft iota penalty (default: 1.0).")
parser.add_argument("--iota-scale", type=float, default=None,
    help="Iota scale in the penalty denominator. Defaults to |--iota-target|. "
         "Penalty curvature is weight/scale**2; do NOT set this to the "
         "tolerance -- that is the original bug (curvature 1.6e+07).")
parser.add_argument("--maxiter", type=int, default=200,
    help="Max accepted BFGS iterations, counting "
         "any prior progress when resuming from checkpoint (default: 200). "
         "Example: --maxiter 200 with 100 rows in iterations.json runs at most "
         "100 more iterations.")
parser.add_argument("--outer-step-radius", type=float, default=0.15,
    help="Maximum L2 change in normalized current variables per accepted outer "
         "step (default: 0.15).")
parser.add_argument("--volume-target", type=float, default=0.3,
    help="Target volume of the Boozer surface (default: 0.3).")
parser.add_argument("--output-root", type=str,
    default="/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs",
    
    help="Top-level output directory; run outputs are placed under "
         "<output-root>/<eq_name>/iota<X>_fcp<Y>kA_vt<Z>_G_<sign>/"
         "mpol<M>_ntor<N>/.")
parser.add_argument(
    "--new",
    action="store_true",
    default=False,
    help=(
        "Start from scratch: do not skip a run that already has a "
        "converged results.json.  Default behaviour resumes a partially finished run."
    ),
)
parser.add_argument(
    "--sparse",
    action="store_true",
    default=False,
    help="Remove outboard-midplane dipoles before optimization.",
)
args = parser.parse_args()

# ---------- unpack remaining args ----------
INIT_DIR       = args.init_dir
IOTA_TARGET    = args.iota_target
FCP_THRESHOLD  = args.f_cp_threshold
EXPECTED_FIELD_POLARITY = args.field_polarity
IOTA_THRESHOLD = args.iota_threshold
IOTA_WEIGHT = args.iota_penalty_weight
IOTA_SCALE = (
    abs(float(args.iota_target)) or 1.0
    if args.iota_scale is None
    else float(args.iota_scale)
)
for _name, _value in (("--iota-penalty-weight", IOTA_WEIGHT), ("--iota-scale", IOTA_SCALE)):
    if not np.isfinite(_value) or _value <= 0.0:
        raise ValueError(f"{_name} must be finite and positive")
IOTA_PENALTY_CURVATURE = IOTA_WEIGHT / IOTA_SCALE**2
MAXITER        = args.maxiter
OUTER_STEP_RADIUS = args.outer_step_radius
VOL_TARGET     = args.volume_target
OUTPUT_ROOT    = args.output_root
START_FRESH    = args.new
SPARSE         = args.sparse
MPOL = NTOR = 8
FB_THRESHOLD = args.fb_threshold

METHOD_NAME = (
    "soft_penalty_bounded_bfgs_sparse"
    if SPARSE else "soft_penalty_bounded_bfgs"
)

# Fixed internal parameters (same as the single-resolution sequential script).
BOOZER_CW        = 1.0
PENALTY_WEIGHT   = 100.0
CURRENT_P_NORM   = 20.0
GTOL             = 1e-3
CURRENT_SCALE    = 100.0 # penalize the current more strongly than the residual
THETA_TOL        = 0.01  # outboard dipole removal tolerance [rad] when --sparse
if not np.isfinite(OUTER_STEP_RADIUS) or OUTER_STEP_RADIUS <= 0.0:
    raise ValueError("--outer-step-radius must be finite and positive")
if not np.isfinite(FB_THRESHOLD) or FB_THRESHOLD <= 0.0:
    raise ValueError("--fb-threshold must be finite and positive")

# Tolerance for the convergence warning (5 % above threshold).
FB_WARN_MARGIN   = 0.05

RUN_CONFIG = RunConfiguration(
    schema_version=1,
    init_dir=os.path.realpath(INIT_DIR),
    method=METHOD_NAME,
    mpol=MPOL,
    ntor=NTOR,
    field_polarity=EXPECTED_FIELD_POLARITY,
    iota_target=IOTA_TARGET,
    f_b_threshold=FB_THRESHOLD,
    f_cp_threshold=FCP_THRESHOLD,
    iota_threshold=IOTA_THRESHOLD,
    iota_penalty_weight=IOTA_WEIGHT,
    iota_scale=IOTA_SCALE,
    volume_target=VOL_TARGET,
    maxiter=MAXITER,
    outer_step_radius=OUTER_STEP_RADIUS,
    boozer_constraint_weight=BOOZER_CW,
    penalty_weight=PENALTY_WEIGHT,
    current_pnorm_p=CURRENT_P_NORM,
    current_scale=CURRENT_SCALE,
    gtol=0.1 * GTOL,
    sparse=SPARSE,
    theta_tol=THETA_TOL,
)

plot_config = PlotConfig(
    dpi=100, titlefontsize=16, axisfontsize=16,
    legendfontsize=14, ticklabelfontsize=14, cbarfontsize=16,
)




def plot_objective_vs_iterations(out_dir, config):
    """
    Plot total objective J and each contribution vs accepted iteration using
    iterations.json, plus ||grad J|| on the lower panel.  Identical to the
    single-resolution sequential script so all downstream postprocessing works.
    """
    history_path = os.path.join(out_dir, "iterations.json")
    if not os.path.isfile(history_path):
        return
    with open(history_path, "r") as f:
        history = json.load(f)
    rows = history.get("iterations") or []
    if len(rows) < 1:
        return

    it = np.array([r["iteration"] for r in rows], dtype=float)
    J_tot = np.array([r["J"] for r in rows], dtype=float)
    grad_norm = np.array([r["grad_norm"] for r in rows], dtype=float)

    contrib_keys = ("J_boozer_contrib", "J_iota_contrib", "J_current_contrib")

    def _row_has_contribs(r):
        return all(k in r for k in contrib_keys)

    any_components = any(_row_has_contribs(r) for r in rows)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), constrained_layout=True,
    )
    if any_components:
        J_qs = np.array([r["nonQS_ratio"] for r in rows], dtype=float)
        J_b = np.array(
            [float(r["J_boozer_contrib"]) if "J_boozer_contrib" in r else np.nan for r in rows],
            dtype=float,
        )
        J_i = np.array(
            [float(r["J_iota_contrib"]) if "J_iota_contrib" in r else np.nan for r in rows],
            dtype=float,
        )
        J_c = np.array(
            [float(r["J_current_contrib"]) if "J_current_contrib" in r else np.nan for r in rows],
            dtype=float,
        )
        ax1.plot(it, J_tot, color="k", lw=2.0, label=r"$J$ (total)")
        ax1.plot(it, J_qs, label=r"$J_{\mathrm{nonQS}}$")
        ax1.plot(it, J_b, label=r"$w_{\mathrm{pen}}\,J_{\mathrm{Boozer\,guard}}$")
        ax1.plot(it, J_i, label=r"$w_{\mathrm{pen}}\,J_{\iota}$")
        ax1.plot(it, J_c, label=r"$w_{\mathrm{pen}}\,J_{\mathrm{current\,guard}}$")
    else:
        ax1.plot(it, J_tot, color="k", lw=2.0, label=r"$J$ (total)")

    ax1.set_ylabel("Contribution to $J$", fontsize=config.axisfontsize)
    ax1.set_title("Objective and components vs iteration", fontsize=config.titlefontsize)
    ax1.legend(fontsize=config.legendfontsize, loc="best")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.tick_params(labelsize=config.ticklabelfontsize)
    ax1.set_yscale("symlog", linthresh=1e-4)

    ax2.plot(it, grad_norm, color="C5", lw=1.5)
    ax2.set_ylabel(r"$\|\nabla J\|_2$", fontsize=config.axisfontsize)
    ax2.set_xlabel("Iteration (accepted)", fontsize=config.axisfontsize)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.tick_params(labelsize=config.ticklabelfontsize)
    ax2.set_yscale("symlog", linthresh=1e-3)

    out_path = os.path.join(out_dir, "objective_vs_iteration.png")
    fig.savefig(out_path, dpi=config.dpi)
    plt.close(fig)
    print(f"Saved objective history plot to {out_path}")


# ==============================================================================
# LOAD INIT STATE
# ==============================================================================
init_results = load(os.path.join(INIT_DIR, "results.json"))
eq_name = init_results["eq_name"]

EQ_OUT_ROOT = os.path.join(OUTPUT_ROOT, eq_name)

def _per_res_out_dir(mpol, ntor):
    """Directory for a single (iota, mpol, ntor) point -- same naming as the
    single-resolution scripts so downstream postprocessing is unchanged."""
    parent = os.path.join(
        EQ_OUT_ROOT,
        f"iota{IOTA_TARGET:g}_fcp{FCP_THRESHOLD / 1e3:g}kA_"
        f"vt{VOL_TARGET:g}_{POLARITY_DIR}",
    )
    return os.path.join(parent, f"mpol{mpol}_ntor{ntor}")


def _is_completed(out_dir):
    """True only for a successful artifact with this exact configuration."""
    rpath = os.path.join(out_dir, "results.json")
    if not os.path.isfile(rpath):
        return False
    try:
        rj = load(rpath)
    except Exception:
        return False
    RUN_CONFIG.require_match(rj, out_dir)
    return bool(rj.get("optimization_success", False))


def _has_checkpoint(out_dir):
    """Whether a complete checkpoint exists for this exact configuration."""
    needed = ["bs_opt.json", "surf_opt.json", "results.json"]
    if not all(os.path.isfile(os.path.join(out_dir, f)) for f in needed):
        return False
    checkpoint = load(os.path.join(out_dir, "results.json"))
    RUN_CONFIG.require_match(checkpoint, out_dir)
    return True


def _next_iteration_index_from_history(out_dir):
    """1-based callback index after existing iterations.json rows."""
    path = os.path.join(out_dir, "iterations.json")
    if not os.path.isfile(path):
        return 1
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 1
    iters = data.get("iterations")
    if not isinstance(iters, list) or not iters:
        return 1
    last = iters[-1]
    if not isinstance(last, dict) or "iteration" not in last:
        return 1
    return int(last["iteration"]) + 1


# ==============================================================================
# Initial coils + surface
# ==============================================================================
print(f"Single-resolution soft-penalty run: "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  init-dir:    {INIT_DIR}")
print(f"  iota target: {IOTA_TARGET:g}")
# Measured on order 12: curvature 1.6e+07 amplified the ~2.7e-11 Boozer Newton
# wobble into 1.562979e-04 of outer-gradient noise; ablating it gave 1.06e-12.
_IOTA_NOISE = IOTA_PENALTY_CURVATURE * (1.562979e-04 / 1.6e07)
print(f"  iota penalty: weight={IOTA_WEIGHT:g} scale={IOTA_SCALE:g} "
      f"curvature={IOTA_PENALTY_CURVATURE:.3e} est. gradient noise={_IOTA_NOISE:.3e}")
if _IOTA_NOISE > 0.1 * GTOL:
    print(f"  WARNING: iota penalty curvature {IOTA_PENALTY_CURVATURE:.3e} puts its own "
          f"gradient noise (~{_IOTA_NOISE:.3e}) above the tightest tolerance "
          f"({0.1 * GTOL:.3e}); expect the gradient to plateau on noise. "
          f"Lower --iota-penalty-weight or raise --iota-scale.")
print(f"  sparse:      {SPARSE}")
if SPARSE:
    print(f"  theta_tol:   {THETA_TOL:g}")
print(f"  f_CP:        {FCP_THRESHOLD:.0f} A")
print(f"  vt:          {VOL_TARGET:g}")
print(f"  resolution:  mpol=ntor={MPOL}  fb_threshold={FB_THRESHOLD:.2e}")
print(f"  penalty:     {PENALTY_WEIGHT}, gtol={0.1 * GTOL:g}, "
      f"maxiter/step={MAXITER} (cumulative per step incl. resume)")
print(f"  outer step:  L2 radius {OUTER_STEP_RADIUS:g}")

bs = load(os.path.join(INIT_DIR, "bs_opt.json"))
surf_opt_path = os.path.join(INIT_DIR, "surf_opt.json")
if os.path.exists(surf_opt_path):
    surf = load(surf_opt_path)
    nPhi = surf.quadpoints_phi.size
    nTheta = surf.quadpoints_theta.size
else:
    nPhi, nTheta = 128, 64
    eq_path = os.path.join(init_results["eq_dir"], init_results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_path, s=init_results["surf_s"], range="half period", nphi=nPhi, ntheta=nTheta,
    )
    surf.set_dofs(init_results["surf_dof_scale"] * surf.get_dofs())

shared_meta = {
    "eq_name":  init_results["eq_name"],
    "eq_dir":   init_results["eq_dir"],
    "surf_nfp": init_results["surf_nfp"],
    "surf_s":   init_results["surf_s"],
    "VV_R0":    init_results["VV_R0"],
    "VV_a":     init_results["VV_a"],
    "VV_b":     init_results["VV_b"],
    "ntf":      init_results["ntf"],
}
if "surf_dof_scale" in init_results:
    shared_meta["surf_dof_scale"] = init_results["surf_dof_scale"]

VV = SurfaceRZFourier(nfp=init_results["surf_nfp"])
VV.set_rc(0, 0, init_results["VV_R0"])
VV.set_rc(1, 0, init_results["VV_a"])
VV.set_zs(1, 0, init_results["VV_b"])

num_tf = init_results["ntf"] * 2 * init_results["surf_nfp"]
coils_dense = bs.coils
tf_coils = coils_dense[:num_tf]
dipole_coils_dense = coils_dense[num_tf:]

removed_dipoles = []
if SPARSE:
    dipole_coils = []
    for idx, coil in enumerate(dipole_coils_dense):
        theta = coil_center_theta(coil, init_results["VV_R0"])
        if angular_distance_to_zero(theta) <= THETA_TOL:
            removed_dipoles.append((idx, theta))
        else:
            dipole_coils.append(coil)
    if len(dipole_coils) == 0:
        raise RuntimeError("All dipole coils were removed; relax THETA_TOL.")
    coils = tf_coils + dipole_coils
    bs = BiotSavart(coils)
    print(
        f"  dipoles: kept {len(dipole_coils)} / {len(dipole_coils_dense)}, "
        f"removed {len(removed_dipoles)}"
    )
else:
    coils = coils_dense
    dipole_coils = dipole_coils_dense

FIELD_POLARITY = EXPECTED_FIELD_POLARITY
recorded_polarity = init_results.get("field_polarity")
if recorded_polarity is not None and int(recorded_polarity) != FIELD_POLARITY:
    raise ValueError(
        f"--field-polarity={FIELD_POLARITY} does not match the Stage-2 "
        f"artifact metadata field_polarity={recorded_polarity}"
    )
POLARITY_DIR = "G_positive" if FIELD_POLARITY > 0 else "G_negative"
EQ_OUT_ROOT = os.path.join(OUTPUT_ROOT, eq_name)
os.makedirs(EQ_OUT_ROOT, exist_ok=True)
print(f"  field:       {POLARITY_DIR}")
print(f"  output root: {EQ_OUT_ROOT}")




# ==============================================================================
# OPTIMIZE A SINGLE (IOTA, RESOLUTION) STEP
# ==============================================================================
def optimize_one_iota(iota_target, prev_load_dir, mpol, ntor, fb_threshold,
                      gtol=None, resume_this_step=False):
    """Run BFGS for one (iota, mpol/ntor) point, warm-started from current bs/surf.

    Parameters
    ----------
    iota_target : float
        Rotational transform target for this step.
    prev_load_dir : str
        Directory that provided the warm start (recorded in results.json for
        provenance).
    mpol : int
        Boozer surface poloidal resolution.
    ntor : int
        Boozer surface toroidal resolution.
    fb_threshold : float
        Boozer residual upper bound.
    gtol : float, optional
        Gradient norm convergence tolerance for BFGS.  Defaults to the global
        Gradient tolerance for the fixed order-8 optimization.
    resume_this_step : bool
        If True, append to the existing log and continue from the checkpoint.
        The global ``MAXITER`` is a cumulative cap for this (iota, resolution)
        directory: remaining BFGS iterations are ``MAXITER`` minus accepted
        iterations already stored in ``iterations.json``.
    """
    if gtol is None:
        gtol = GTOL
    out_dir = _per_res_out_dir(mpol, ntor)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Boozer surface for this (iota, resolution) ----
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
    G_sign = FIELD_POLARITY
    G0 = G_sign * 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
    boozer_surface = initialize_boozer_surface(
        surf, mpol, ntor, bs, VOL_TARGET, BOOZER_CW, iota_target, G0,
    )
    print(f"[iota={iota_target:g}, mpol={mpol}] Initial Boozer volume: "
          f"{boozer_surface.surface.volume():.4f}")

    # ---- Objective + constraints ----
    bs_obj = BiotSavart(coils)
    JnonQSRatio    = sum([NonQuasiSymmetricRatio(boozer_surface, bs_obj)])
    JBoozerResidual = sum([BoozerResidual(boozer_surface, bs_obj)])

    JBoozerGuard = (1.0 / fb_threshold**2) * QuadraticPenalty(
        JBoozerResidual, fb_threshold, f="max",
    )
    iota = Iotas(boozer_surface)
    Jiota = IOTA_PENALTY_CURVATURE * QuadraticPenalty(iota, iota_target)
    Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=CURRENT_P_NORM)
    JCurrentGuard = (CURRENT_SCALE / FCP_THRESHOLD**2) * QuadraticPenalty(
        Jcurrent, FCP_THRESHOLD, f="max",
    )
    JF = JnonQSRatio + PENALTY_WEIGHT * (JBoozerGuard + JCurrentGuard) + Jiota

    # ---- Per-step log file ----
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    log_path = os.path.join(out_dir, "log.txt")
    log_mode = "a" if resume_this_step else "w"
    log_file = open(log_path, log_mode, buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    start_time = time.time()
    print(f"\n{'=' * 70}")
    print(f"Single-resolution step -- {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}")
    print(f"Output:     {out_dir}")
    print(f"warm-start: {prev_load_dir}")
    print(f"iota_target={iota_target:g},  f_b={fb_threshold:.1e},  f_CP={FCP_THRESHOLD:.0f} A")
    if resume_this_step:
        next_it = _next_iteration_index_from_history(out_dir)
        n_done = max(0, next_it - 1)
    else:
        next_it = 1
        n_done = 0
    maxiter_scipy = max(0, MAXITER - n_done)

    print(f"mpol={mpol}, ntor={ntor}, maxiter_cap={MAXITER}, gtol={gtol:.2e}")
    if resume_this_step:
        print(
            f"  resume: {n_done} accepted iteration(s) on disk -> "
            f"up to {maxiter_scipy} further BFGS iteration(s) (cap {MAXITER})."
        )
    else:
        print(f"  fresh step: up to {maxiter_scipy} BFGS iteration(s) (cap {MAXITER}).")
    print(f"Resolution: mpol=ntor={MPOL}  fb_threshold={FB_THRESHOLD}")
    if SPARSE:
        print(
            f"Dipole sparsity: kept={len(dipole_coils)} "
            f"removed={len(removed_dipoles)} theta_tol={THETA_TOL:g}"
        )
    if resume_this_step:
        print(f"Resuming in-place from existing checkpoint at callback it={next_it}.")
    print(f"# TF: {len(tf_coils)},  # dipole: {len(dipole_coils)}")
    print(f"Objective: nonQS + {PENALTY_WEIGHT} * (Boozer_guard + current_guard)"
          f" + {IOTA_PENALTY_CURVATURE:.3e} * 0.5*(iota - {iota_target:g})^2\n")

    print(f"Initial nonQS ratio:      {JnonQSRatio.J():.6e}")
    print(f"Initial Boozer residual:  {JBoozerResidual.J():.6e}  (threshold {fb_threshold:.1e})")
    print(f"Initial iota:             {iota.J():.6f}  (target {iota_target:g} +/- {IOTA_THRESHOLD:g})")
    print(f"Initial current p-norm:   {Jcurrent.J():.1f} A  (threshold {FCP_THRESHOLD:.0f} A)")

    # Plot initial cross-section only for the very first (iota, res) step.
    is_first_step = (prev_load_dir == INIT_DIR) and (not resume_this_step)
    if is_first_step:
        pass
        # plot_cross_section(
        #     boozer_surface.surface, VV, out_dir, "initial", plot_config,
        #     base_dipole_coils=dipole_coils,
        # )

    # ---- Optimizer state ----
    x0 = JF.x.copy()
    run_dict = {
        "sdofs": boozer_surface.surface.x.copy(),
        "iota": boozer_surface.res["iota"],
        "G": boozer_surface.res["G"],
        "J": JF.J(),
        "dJ": JF.dJ().copy(),
        "res": boozer_surface.res.copy(),
        "it": next_it,
        "lscount": 0,
        "x_prev": x0.copy(),
        "x_accepted": x0.copy(),
        "failed_boozer_solves": 0,
    }

    def fun(x):
        """Evaluate objective and gradient; reject bad Boozer solves."""
        dx = np.linalg.norm(x - run_dict["x_prev"])
        run_dict["x_prev"] = x.copy()
        run_dict["lscount"] += 1

        # Reset to last accepted Boozer state
        boozer_surface.surface.x = run_dict["sdofs"].copy()
        boozer_surface.res = run_dict["res"].copy()

        JF.x = x
        # Newton only. run_code()'s BoozerLS path is "BFGS followed by Newton"
        # (simsopt BoozerSurface.run_code), and every outer trial starts from the
        # last ACCEPTED surface, which is already converged -- so the BFGS stage
        # re-derived ~300 iterations of work that Newton then finished in one
        # step. Measured at order 6: 10 solves, 2976 inner iterations, for 7
        # accepted outer iterations. Parameters come from the surface's own
        # options so they cannot drift from the ones it was built with; this
        # mirrors run_code's Newton polish exactly, minus the BFGS pre-stage.
        # Newton returns self.res early unless this flag is set, and clearing it
        # is also what stops the objectives (Iotas, NonQuasiSymmetricRatio,
        # BoozerResidual) re-entering run_code themselves.
        boozer_surface.need_to_run_code = True
        try:
            boozer_surface.minimize_boozer_penalty_constraints_newton(
                constraint_weight=boozer_surface.constraint_weight,
                iota=run_dict["iota"],
                G=run_dict["G"],
                tol=boozer_surface.options["newton_tol"],
                maxiter=boozer_surface.options["newton_maxiter"],
                verbose=boozer_surface.options["verbose"],
                weight_inv_modB=boozer_surface.options["weight_inv_modB"],
            )
        except np.linalg.LinAlgError as error:
            print(f"/!\\ Boozer Newton linear solve failed: {error}")
            ok = False
        else:
            try:
                ok = (
                    boozer_surface.res["success"]
                    and not boozer_surface.surface.is_self_intersecting()
                )
            except Exception:
                ok = False

        if ok:
            run_dict["failed_boozer_solves"] = 0
            J, dJ = JF.J(), JF.dJ()
        else:
            run_dict["failed_boozer_solves"] += 1
            print(f"/!\\ Boozer rejected (consecutive: {run_dict['failed_boozer_solves']})")
            J, dJ = rejected_trial_value_and_gradient(
                x, run_dict["x_accepted"], run_dict["J"],
            )
            JF.x = run_dict["x_accepted"].copy()
            boozer_surface.surface.x = run_dict["sdofs"].copy()
            boozer_surface.res = run_dict["res"].copy()
            boozer_surface.need_to_run_code = False

        max_I = float(np.max(np.abs([c.current.get_value() for c in dipole_coils])))
        print(
            f"  step={dx:.2e}  J={J:.6e}  ||grad J||={np.linalg.norm(dJ):.6e}  "
            f"QS={JnonQSRatio.J():.6e}  fb={JBoozerResidual.J():.6e}  "
            f"iota={iota.J():.4f}  Imax={max_I:.0f}A"
        )
        return J, dJ, ok

    def callback(x):
        """Accept step: cache state, log diagnostics, save partial outputs."""
        run_dict["lscount"] = 0
        run_dict["sdofs"] = boozer_surface.surface.x.copy()
        run_dict["iota"] = boozer_surface.res["iota"]
        run_dict["G"] = boozer_surface.res["G"]
        run_dict["res"] = boozer_surface.res.copy()
        run_dict["J"] = JF.J()
        run_dict["dJ"] = JF.dJ().copy()
        run_dict["x_accepted"] = x.copy()

        J = run_dict["J"]
        grad = run_dict["dJ"]
        qs_val = float(JnonQSRatio.J())
        fb_val = float(JBoozerResidual.J())
        iota_val = float(iota.J())
        cp_val = float(Jcurrent.J())
        max_I = float(np.max([abs(c.current.get_value()) for c in dipole_coils]))

        nphi_b = boozer_surface.surface.quadpoints_phi.size
        ntheta_b = boozer_surface.surface.quadpoints_theta.size
        BdotN = float(np.mean(np.abs(np.sum(
            bs.B().reshape((nphi_b, ntheta_b, 3)) * boozer_surface.surface.unitnormal(), axis=2,
        ))))
        vol = float(boozer_surface.surface.volume())

        fb_flag = "  [!>fb]" if fb_val > fb_threshold else ""
        cp_flag = "  [!>fcp]" if cp_val > FCP_THRESHOLD else ""
        print(f"\n{'=' * 60}")
        print(f"ITER {run_dict['it']:3d}  J={J:.6e}  ||grad J||={np.linalg.norm(grad):.6e}")
        print(f"  nonQS={qs_val:.6e}")
        print(f"  Boozer={fb_val:.6e}{fb_flag}")
        print(f"  iota={iota_val:.4f} (target {iota_target:g})")
        print(f"  <|B.n|>={BdotN:.6e}")
        print(f"  I_pnorm={cp_val:.0f}A (limit {FCP_THRESHOLD:.0f}A){cp_flag}")
        print(f"  I_max={max_I:.0f}A")
        print(f"  volume={vol:.4f}\n")

        # ---- Iteration history JSON ----
        history_path = os.path.join(out_dir, "iterations.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = {
                "config": {
                    **RUN_CONFIG.to_dict(),
                    "warm_start_from": prev_load_dir,
                    **(
                        {
                            "sparse_run": True,
                            "theta_tol": THETA_TOL,
                            "removed_outboard_dipoles": len(removed_dipoles),
                        }
                        if SPARSE else {}
                    ),
                },
                "iterations": [],
            }
        jb_g = float(JBoozerGuard.J())
        ji_g = float(Jiota.J())
        jc_g = float(JCurrentGuard.J())
        history["iterations"].append({
            "iteration": int(run_dict["it"]),
            "J": float(J),
            "grad_norm": float(np.linalg.norm(grad)),
            "nonQS_ratio": qs_val,
            "boozer_residual": fb_val,
            "iota": iota_val,
            "current_pnorm": cp_val,
            "max_current": max_I,
            "BdotN": BdotN,
            "volume": vol,
            "J_boozer_contrib": float(PENALTY_WEIGHT * jb_g),
            "J_iota_contrib": float(ji_g),
            "J_current_contrib": float(PENALTY_WEIGHT * jc_g),
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # ---- Partial saves ----
        bs.save(os.path.join(out_dir, "bs_opt.json"))
        boozer_surface.surface.save(os.path.join(out_dir, "surf_opt.json"))

        partial_results = {
            "run_config": RUN_CONFIG.to_dict(),
            **RUN_CONFIG.to_dict(),
            "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
            "optimization_success": None,
            "optimization_message": "partial snapshot from callback",
            "final_objective": float(J),
            "nonQS_ratio": qs_val,
            "boozer_residual": fb_val,
            "final_iota": iota_val,
            "max_current": max_I,
            "# TF coils": len(tf_coils),
            "# dipole coils": len(dipole_coils),
            **(
                {
                    "sparse_run": True,
                    "theta_tol": THETA_TOL,
                    "removed_outboard_dipoles": len(removed_dipoles),
                }
                if SPARSE else {}
            ),
            **shared_meta,
        }
        save(partial_results, os.path.join(out_dir, "results.json"))

        run_dict["it"] += 1

    # ---- Run BFGS for this (iota, resolution) step ----
    # MAXITER is a cumulative cap vs accepted iterations already in iterations.json
    # when resuming (n_done); not an additional 200 on top of prior work.
    if maxiter_scipy <= 0:
        J0, dJ0 = float(JF.J()), JF.dJ()
        res = OptimizeResult(
            x=x0,
            success=False,
            status=0,
            message=(
                f"Iteration budget exhausted: {n_done} accepted iteration(s) "
                f"already recorded (cap {MAXITER}); no further BFGS iterations run."
            ),
            fun=J0,
            jac=dJ0,
            nit=0,
            nfev=0,
            njev=0,
        )
    else:
        res = minimize_bounded_bfgs(
            fun,
            x0,
            initial_value=run_dict["J"],
            initial_gradient=run_dict["dJ"],
            callback=callback,
            maxiter=maxiter_scipy,
            gtol=gtol,
            max_step_norm=OUTER_STEP_RADIUS,
        )
    print(f"\nOptimizer: {res.message}")

    max_I = float(np.max([abs(c.current.get_value()) for c in dipole_coils]))
    final_qs = float(JnonQSRatio.J())
    final_fb = float(JBoozerResidual.J())
    final_iota = float(iota.J())
    final_vol = float(boozer_surface.surface.volume())
    final_cp = float(Jcurrent.J())

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS (iota_target={iota_target:g}, mpol={mpol}, ntor={ntor})")
    print(f"{'=' * 70}")
    print(f"  nonQS ratio:      {final_qs:.6e}")
    print(f"  Boozer residual:  {final_fb:.6e}  (threshold {fb_threshold:.1e})")
    print(f"  Iota:             {final_iota:.6f}  (target {iota_target:g})")
    print(f"  Max current:      {max_I:.0f} A  (threshold {FCP_THRESHOLD:.0f} A)")
    print(f"  Current p-norm:   {final_cp:.0f} A")
    print(f"  Volume:           {final_vol:.4f}")

    # ---- Publication gates: fail closed ----
    # These run BEFORE any final artifact is written. The callback still
    # checkpoints bs_opt/surf_opt/results.json on every accepted step so a run
    # can resume, but that snapshot carries optimization_success=None and is
    # overwritten by the final save below only once these pass. Measured before
    # the gates existed: a run published iota=0.046760 (1.30x outside
    # --iota-threshold) and a current p-norm of 150037 A against a 150000 A
    # limit, with optimization_success False and no error at all. The Boozer
    # residual stays a RuntimeWarning, as the module docstring documents.
    for gate, ok, detail in (
        ("optimizer convergence", bool(res.success), str(res.message).rstrip(".")),
        (
            "iota",
            np.isfinite(final_iota) and abs(final_iota - iota_target) <= IOTA_THRESHOLD,
            f"{final_iota:.6f} vs target {iota_target:g} +/- {IOTA_THRESHOLD:g}",
        ),
        (
            "current p-norm",
            np.isfinite(final_cp) and final_cp <= FCP_THRESHOLD,
            f"{final_cp:.0f} A exceeds {FCP_THRESHOLD:.0f} A",
        ),
    ):
        if not ok:
            raise RuntimeError(
                f"{gate} gate failed at mpol=ntor={mpol}: {detail}. No final "
                f"artifact was written; the resume checkpoint under {out_dir} "
                f"stays marked optimization_success=None."
            )

    # ==========================================================================
    # SAVE FINAL OUTPUTS  (only reachable once every gate above has passed)
    # ==========================================================================
    coils_to_vtk(coils, filename=os.path.join(out_dir, "coils_opt"), close=True)
    bs.save(os.path.join(out_dir, "bs_opt.json"))
    VV.to_vtk(os.path.join(out_dir, "vacuum_vessel"))

    bs.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
    nphi_b = boozer_surface.surface.quadpoints_phi.size
    ntheta_b = boozer_surface.surface.quadpoints_theta.size
    B = bs.B().reshape((nphi_b, ntheta_b, 3))
    modB = np.sqrt(np.sum(B**2, axis=2))[:, :, None]
    BdotN_surf = np.sum(B * boozer_surface.surface.unitnormal(), axis=2)[:, :, None]
    pointData = {"B_N/B": BdotN_surf / modB}
    boozer_surface.surface.to_vtk(os.path.join(out_dir, "surf_opt"), extra_data=pointData)
    boozer_surface.surface.save(os.path.join(out_dir, "surf_opt.json"))

    plot_relBfinal_norm_modB(bs, boozer_surface.surface, out_dir, "optimized", plot_config)
    # plot_cross_section(
    #     boozer_surface.surface, VV, out_dir, "optimized", plot_config,
    #     base_dipole_coils=dipole_coils,
    # )
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, out_dir, "optimized", plot_config)
    plot_objective_vs_iterations(out_dir, plot_config)

    results_output = {
        "run_config": RUN_CONFIG.to_dict(),
        # Flatten the same immutable configuration for existing postprocessors.
        **RUN_CONFIG.to_dict(),
        "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
        "iota_penalty_curvature": IOTA_PENALTY_CURVATURE,
        "initial_G": G0,
        **(
            {
                "sparse_run": True,
                "theta_tol": THETA_TOL,
                "removed_outboard_dipoles": len(removed_dipoles),
            }
            if SPARSE else {}
        ),

        # Optimization results
        "optimization_success": bool(res.success),
        "optimization_message": str(res.message),
        "final_objective": float(JF.J()),
        "nonQS_ratio": final_qs,
        "boozer_residual": final_fb,
        "final_iota": final_iota,
        "final_volume": final_vol,
        "iota_penalty": float(Jiota.J()),
        "current_pnorm": final_cp,
        "max_current": max_I,

        # Coil counts
        "# TF coils": len(tf_coils),
        "# dipole coils": len(dipole_coils),

        # Inherited from upstream
        **shared_meta,
    }
    save(results_output, os.path.join(out_dir, "results.json"))
    print(f"\nResults saved to {os.path.join(out_dir, 'results.json')}")

    elapsed = time.time() - start_time
    print(f"Wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")

    log_file.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr

    return out_dir, boozer_surface


# ==============================================================================
# RUN THE FIXED ORDER-8 OPTIMIZATION
# ==============================================================================
walk_start = time.time()
prev_load_dir = INIT_DIR
mpol, ntor, fb_thresh = MPOL, NTOR, FB_THRESHOLD
out_dir = _per_res_out_dir(mpol, ntor)
step_label = f"[order 8] iota={IOTA_TARGET:g}"
prepare_output_generation(out_dir, START_FRESH)

if (not START_FRESH) and _is_completed(out_dir):
    print(f"\n{step_label}: already completed at {out_dir}.")
    sys.exit(0)

resume_this_step = False
if (not START_FRESH) and _has_checkpoint(out_dir):
    print(f"\n{step_label}: found compatible checkpoint at {out_dir}; resuming.")
    bs = load(os.path.join(out_dir, "bs_opt.json"))
    surf = load(os.path.join(out_dir, "surf_opt.json"))
    coils = bs.coils
    tf_coils = coils[:num_tf]
    dipole_coils = coils[num_tf:]
    for c in dipole_coils:
        c.curve.fix_all()
    for c in tf_coils:
        c.current.fix_all()
    resume_this_step = True

print(f"\n{step_label}: starting (warm start from {prev_load_dir})")
step_t0 = time.time()
out_dir, boozer_surface = optimize_one_iota(
    IOTA_TARGET, prev_load_dir, mpol, ntor, fb_thresh,
    gtol=0.1 * GTOL,
    resume_this_step=resume_this_step,
)

try:
    step_results = load(os.path.join(out_dir, "results.json"))
    final_fb = float(step_results.get("boozer_residual", float("inf")))
except Exception:
    final_fb = float("inf")

if final_fb > fb_thresh * (1.0 + FB_WARN_MARGIN):
    warnings.warn(
        f"{step_label}: Boozer residual {final_fb:.3e} exceeds threshold "
        f"{fb_thresh:.3e} by more than {FB_WARN_MARGIN * 100:.0f}%.",
        RuntimeWarning,
        stacklevel=2,
    )

print(f"{step_label}: done in {(time.time() - step_t0) / 60:.1f} min -> {out_dir}")

print(f"\nSingle-resolution run complete in {(time.time() - walk_start) / 60:.1f} min.")
