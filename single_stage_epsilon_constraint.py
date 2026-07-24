import os
import io
import re
import sys
import json
import argparse
import time
from datetime import datetime
import numpy as np
from scipy.optimize import minimize

# SIMSOPT imports
from simsopt.geo import SurfaceRZFourier, curves_to_vtk, RotatedCurve
from simsopt.geo.surfaceobjectives import BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.field import BiotSavart, Coil, Current, ScaledCurrent, coils_to_vtk, CurrentPenalty, coils_via_symmetries
from simsopt.objectives import QuadraticPenalty
from simsopt._core.optimizable import load, save
import matplotlib.pyplot as plt
from helper_functions import *
from boozer_functions import *

# Default Boozer volume target (also used for output path naming when unchanged).
_DEFAULT_VOL_TARGET = 0.3

# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================
parser = argparse.ArgumentParser(
    description="Single-stage dipole optimization with configurable targets and weights."
)

parser.add_argument(
    "--init-dir",
    type=str,
    default="",
    help="Directory containing results from previous optimization (bs_opt.json, surf_opt.json, results.json).",
)

parser.add_argument(
    "--iota-target",
    type=float,
    default=0.1,
    help="Target iota value on the Boozer surface.",
)
parser.add_argument(
    "--iota-weight",
    type=float,
    default=1,
    help="Weight for the iota penalty term.",
)
parser.add_argument(
    "--qs-weight",
    type=float,
    default=1,
    help="Weight for the quasi-symmetry (nonQS ratio) term.",
)
parser.add_argument(
    "--qs-target",
    type=float,
    default=5e-4,
    help="Target upper bound for nonQS ratio (penalize only when nonQS exceeds this value).",
)
parser.add_argument(
    "--qs-rel-tol",
    type=float,
    default=0.10,
    help="Relative QS tolerance used to normalize the QS penalty around qs-target (default: 0.10 = 10%%).",
)
parser.add_argument(
    "--iota-rel-tol",
    type=float,
    default=0.05,
    help="Relative iota tolerance used to normalize the iota penalty (default: 0.005 = 0.5%%).",
)
# Comma-separated continuation schedule for current weights.
# If provided, the optimization will run a warm-started continuation loop over
# these weights, producing one solution per weight.
parser.add_argument(
    "--current-weight-schedule",
    type=str,
    default="1",
    help="Comma-separated list of current weights for continuation (default: '0.1,0.3,1').",
)
parser.add_argument(
    "--vol-target",
    type=float,
    default=_DEFAULT_VOL_TARGET,
    help="Target plasma volume for Boozer surface initialization (default: 0.3). "
    "If not the default, output uses an extra _vol<value> suffix on the iota_tar directory name.",
)
parser.add_argument(
    "--sparse",
    action="store_true",
    help="Remove outboard-midplane dipole coils before optimization. "
    "When set, --init-dir should point to an existing stage iteration directory "
    "(e.g. .../iota_tar0.15/stage00_cw0/mpol6_ntor6/) and output is written into "
    "the same iota_tar parent with '_sparse' stage tags.",
)
parser.add_argument(
    "--mpol",
    type=int,
    default=6,
    help="Boozer surface poloidal mode resolution (default: 6).",
)
parser.add_argument(
    "--ntor",
    type=int,
    default=6,
    help="Boozer surface toroidal mode resolution (default: 6).",
)
parser.add_argument(
    "--maxiter",
    type=int,
    default=150,
    help="Maximum optimizer iterations per continuation step (default: 150).",
)

# Allow unknown args so this script can coexist with external launchers that add flags
_args, _unknown = parser.parse_known_args()

# Configuration parameters (can be overridden from the command line)
INIT_DIR = _args.init_dir
IOTA_TARGET = _args.iota_target
IOTA_WEIGHT = _args.iota_weight
QS_WEIGHT = _args.qs_weight
QS_TARGET = _args.qs_target
QS_REL_TOL = _args.qs_rel_tol
IOTA_REL_TOL = _args.iota_rel_tol
CURRENT_WEIGHT_SCHEDULE = [
    float(x) for x in _args.current_weight_schedule.split(",") if x.strip() != ""
]
# Boozer surface volume target (previously fixed at 0.3 here).
VOL_TARGET = _args.vol_target
SPARSE = _args.sparse
MAXITER = _args.maxiter
THETA_TOL = 0.01
if not CURRENT_WEIGHT_SCHEDULE:
    raise ValueError("current-weight-schedule must contain at least one value.")
if IOTA_REL_TOL <= 0:
    raise ValueError("iota-rel-tol must be > 0.")
if QS_TARGET <= 0:
    raise ValueError("qs-target must be > 0.")
if QS_REL_TOL <= 0:
    raise ValueError("qs-rel-tol must be > 0.")
if VOL_TARGET <= 0:
    raise ValueError("vol-target must be > 0.")
if _args.mpol < 1:
    raise ValueError("mpol must be >= 1.")
if _args.ntor < 1:
    raise ValueError("ntor must be >= 1.")
if _args.maxiter < 1:
    raise ValueError("maxiter must be >= 1.")

# Other parameters that are not set from the command line
CONSTRAINT_WEIGHT = 1.0

# Create plot configuration
plot_config = PlotConfig(
    dpi=100,
    titlefontsize=16,
    axisfontsize=16,
    legendfontsize=14,
    ticklabelfontsize=14,
    cbarfontsize=16
)


def coil_center_theta(coil, major_radius):
    """Poloidal angle of a coil's centroid relative to the torus major radius."""
    gamma = coil.curve.gamma()
    center = np.mean(gamma, axis=0)
    x0, y0, z0 = center
    r0 = np.sqrt(x0 * x0 + y0 * y0)
    return np.mod(np.arctan2(z0, r0 - major_radius), 2 * np.pi)


def angular_distance_to_zero(theta):
    """Shortest angular distance from theta to 0 (i.e. outboard midplane)."""
    return np.minimum(theta, 2 * np.pi - theta)


def plot_objective_vs_iterations(out_dir, r0, i0, current_weight, qs_weight, iota_weight, config):
    """
    Plot total objective J and each scaled term vs accepted iteration using iterations.json.

    Scaling matches JF = (1/r0)*JBoozer + (current_weight/i0)*Jcurrent
    + qs_weight*JQSGuard + iota_weight*Jiota.
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
    J_boozer = np.array([r["J_Boozer"] for r in rows], dtype=float) / r0
    J_qs = np.array([r["J_nonQS_guard"] for r in rows], dtype=float) * qs_weight
    J_iota = np.array([r["J_iota"] for r in rows], dtype=float) * iota_weight
    J_current = np.array([r["J_current"] for r in rows], dtype=float) * (current_weight / i0)
    grad_norm = np.array([r["grad_norm"] for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 8),
        constrained_layout=True,
    )
    ax1.plot(it, J_tot, color="k", lw=2.0, label=r"$J$ (total)")
    ax1.plot(it, J_boozer, label=r"$(1/R_0)\,J_{\mathrm{Boozer}}$")
    ax1.plot(it, J_qs, label=r"$w_{\mathrm{QS}} J_{\mathrm{QS\,guard}}$")
    ax1.plot(it, J_iota, label=r"$w_{\iota} J_{\iota}$")
    ax1.plot(it, J_current, label=r"$(w_I/I_0)\,J_{\mathrm{current}}$")
    ax1.set_ylabel("Contribution to $J$", fontsize=config.axisfontsize)
    ax1.set_title("Objective and components vs iteration", fontsize=config.titlefontsize)
    ax1.legend(fontsize=config.legendfontsize, loc="best")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.tick_params(labelsize=config.ticklabelfontsize)
    # Symlog handles wide range (e.g. early J ~ 1e3, late subterms ~ 1e-6) and exact zeros in QS guard.
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


def fun(x):
    """
    Objective function for L-BFGS-B optimization.

    Evaluates the total objective function and its gradient for a given set of
    degrees of freedom (coil parameters). Attempts to solve for a valid Boozer
    surface; if unsuccessful (solver failure or self-intersection), returns the
    last accepted objective value with negated gradient to reject the step.

    Args:
        x: Current degrees of freedom (coil parameters)

    Returns:
        J: Objective function value
        dJ: Gradient of objective function
    """
    dx = np.linalg.norm(x - run_dict['x_prev'])
    run_dict['x_prev'] = x.copy()
    print(f"Step size: {dx:.2e}")

    run_dict['lscount']+=1

    # initialize to last accepted surface values
    boozer_surface.surface.x = run_dict['sdofs']
    boozer_surface.res['iota'] = run_dict['iota']
    boozer_surface.res['G'] = run_dict['G']

    # Set new coil dofs
    JF.x = x

    # Run boozer surface
    res = boozer_surface.run_code(run_dict['iota'], run_dict['G'])

    # Check success
    try:
        success1 = boozer_surface.res['success']
        #success2 = not boozer_surface.surface.is_self_intersecting()
        success2 = True
    except Exception as e:
        print("Surface check failed:", e)
        success2 = False
    success = success1 and success2

    if success:
        # Reset failure counter on a successful Boozer solve
        run_dict['failed_boozer_solves'] = 0
        J = JF.J()
        dJ = JF.dJ()
    else:
        print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not success1:
            print("Boozer solver failed")
        if not success2:
            print("Surface is self-intersecting")

        # Increment counter of consecutive failed Boozer solves and abort if too many.
        run_dict['failed_boozer_solves'] += 1
        print(f"Consecutive failed Boozer solves: {run_dict['failed_boozer_solves']}")

        J = run_dict['J']
        dJ = -run_dict['dJ']
        boozer_surface.surface.x = run_dict['sdofs']
        boozer_surface.res['iota'] = run_dict['iota']
        boozer_surface.res['G'] = run_dict['G']

    print(f"Objective J: {J:.6e}, ||∇J||: {np.linalg.norm(dJ):.6e}")
    print(
        "Individual scaled terms -- "
        f"Boozer: {(1.0 / R0) * JBoozerResidual.J():.6e}, "
        f"QS_guard: {QS_WEIGHT * JQSGuard.J():.6e}, "
        f"iota: {IOTA_WEIGHT * Jiota.J():.6e}, "
        f"current: {(CURRENT_WEIGHT / I0) * Jcurrent.J():.6e}"
    )
    return J, dJ

def callback(x):
    """
    Callback function executed after each successful optimization iteration.

    Stores the accepted state (surface DOFs, iota, G), evaluates and prints
    detailed diagnostics for all objective function components, and logs the
    iteration summary to file. Used for monitoring optimization progress and
    recording convergence history.

    Args:
        x: Current degrees of freedom (coil parameters) from accepted step
    """
    # Update count for tracking
    run_dict['lscount'] = 0

    # Store last accepted state
    run_dict['sdofs'] = boozer_surface.surface.x.copy()
    run_dict['iota'] = boozer_surface.res['iota']
    run_dict['G'] = boozer_surface.res['G']
    run_dict['J'] = JF.J()
    run_dict['dJ'] = JF.dJ().copy()

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']

    J_QS = JnonQSRatio.J()
    dJ_QS = np.linalg.norm(JnonQSRatio.dJ())
    J_QS_guard = JQSGuard.J()
    J_Boozer = JBoozerResidual.J()
    dJ_Boozer = np.linalg.norm(JBoozerResidual.dJ())
    J_iota = Jiota.J()
    dJ_iota = np.linalg.norm(Jiota.dJ())
    J_curr = Jcurrent.J()
    dJ_curr = np.linalg.norm(Jcurrent.dJ())

    iota_str = f"{iota.J():.4f}"
    volume_str = f"{boozer_surface.surface.volume():.4f}"

    nphi = boozer_surface.surface.quadpoints_phi.size
    ntheta = boozer_surface.surface.quadpoints_theta.size
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * boozer_surface.surface.unitnormal(), axis=2)))

    currents = np.array([abs(c.current.get_value()) for c in dipole_coils])
    max_current = np.max(currents)

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Current Penalty':{width}} = {J_curr:.6e} (dJ = {dJ_curr:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    print(f"{'Max current':{width}} = {max_current:.2f} A", file=buffer)
    print("="*70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    # Save iteration diagnostics to JSON for postprocessing
    history_path = os.path.join(OUT_DIR_ITER, "iterations.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {
            "weights": {
                "QS_WEIGHT": QS_WEIGHT,
                "IOTA_WEIGHT": IOTA_WEIGHT,
                "CURRENT_WEIGHT": CURRENT_WEIGHT,
            },
            "targets": {
                "IOTA_TARGET": IOTA_TARGET,
                "QS_RATIO_INITIAL": QS_RATIO_INITIAL,
                "QS_RATIO_TARGET": QS_TARGET,
            },
            "tolerances": {
                "gtol": GTOL,
                "iota_rel_tol": IOTA_REL_TOL,
                "iota_scale": iota_scale,
                "qs_rel_tol": QS_REL_TOL,
                "qs_scale": qs_scale,
            },
            "max_iterations": MAXITER,
            "iterations": [],
        }

    iter_record = {
        "iteration": int(run_dict["it"]),
        "J": float(J),
        "grad_norm": float(np.linalg.norm(grad)),
        "J_nonQS": float(J_QS),
        "J_nonQS_guard": float(J_QS_guard),
        "J_Boozer": float(J_Boozer),
        "J_iota": float(J_iota),
        "J_current": float(J_curr),
        "J_nonQS_guard_scaled": float(QS_WEIGHT * J_QS_guard),
        "J_iota_scaled": float(IOTA_WEIGHT * J_iota),
        "J_current_scaled": float(CURRENT_WEIGHT * J_curr),
        "iota": float(iota.J()),
        "volume": float(boozer_surface.surface.volume()),
        "BdotN": float(BdotN),
        "max_current": float(max_current),
    }

    history["iterations"].append(iter_record)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save a partial results.json snapshot on every accepted iteration so that
    # runs that terminate early still leave usable metadata for postprocessing.
    partial_results = {
        "graph": {"init_dir": INIT_DIR},
        # Optimization configuration
        "mpol": mpol,
        "ntor": ntor,
        "maxiter": MAXITER,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "iota_target": IOTA_TARGET,
        "qs_weight": QS_WEIGHT,
        "iota_weight": IOTA_WEIGHT,
        "current_weight": CURRENT_WEIGHT,
        # Convergence tolerances used
        "gtol": GTOL,
        # Optimization results (partial)
        "optimization_success": None,
        "optimization_message": "partial snapshot from callback",
        "final_objective": float(J),
        "final_iota": float(iota.J()),
        "final_volume": float(boozer_surface.surface.volume()),
        # Diagnostic metrics
        "nonQS_ratio": float(JnonQSRatio.J()),
        "boozer_residual": float(JBoozerResidual.J()),
        "iota_penalty": float(Jiota.J()),
        "current_penalty": float(Jcurrent.J()),
        "max_current": float(max_current),
        # Coil information
        "# TF coils": len(tf_coils),
        "# dipole coils": len(dipole_coils),
        # Inherited from stage 2
        "eq_name": results["eq_name"],
        "eq_dir": results["eq_dir"],
        "surf_nfp": results["surf_nfp"],
        "surf_s": results["surf_s"],
        "VV_R0": results["VV_R0"],
        "VV_a": results["VV_a"],
        "VV_b": results["VV_b"],
        "ntf": results["ntf"],
    }
    save(partial_results, os.path.join(OUT_DIR_ITER, "results.json"))
    # Save the coils
    bs.save(OUT_DIR_ITER + "/bs_opt.json")

    # Advance iteration counter
    run_dict['it'] += 1

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# This is created by and contains the results from stage 2, which we use to initialize single stage
results = load(os.path.join(INIT_DIR, 'results.json'))

# Boozer surface resolution (--mpol / --ntor; default 6 each)
mpol = _args.mpol
ntor = _args.ntor

# EMPIRICAL convergence tolerances for different mpol values
GTOL = 1e-2

# Output directory setup
eq_name = results["eq_name"]
if SPARSE:
    # init-dir is .../iota_tar*/stageNN_cwX/mpolM_ntorN/; go up 2 levels.
    OUT_ROOT = os.path.dirname(os.path.dirname(INIT_DIR))
    # Read iota_target from source results when CLI is still at default.
    if _args.iota_target == parser.get_default("iota_target"):
        IOTA_TARGET = float(results.get("iota_target", IOTA_TARGET))
else:
    _iota_leaf = f"iota_tar{IOTA_TARGET:g}"
    if not np.isclose(VOL_TARGET, _DEFAULT_VOL_TARGET, rtol=0.0, atol=1e-15):
        _iota_leaf = f"{_iota_leaf}_vol{VOL_TARGET:g}"
    # init-dir is .../iota_tar*/stageNN_cwX/mpolM_ntorN/: write under that iota_tar* tree
    # (same layout as --sparse parent). Legacy layout: stage-2 folder as last path segment.
    _stage_parent = os.path.basename(os.path.dirname(os.path.abspath(os.path.expanduser(INIT_DIR))))
    if re.match(r"stage\d+_cw", _stage_parent):
        OUT_ROOT = os.path.abspath(
            os.path.join(os.path.expanduser(INIT_DIR), "..", "..")
        )
    else:
        OUT_ROOT = os.path.join(
            "..",
            "single_stage_scans_epsilon_constraint_updated",
            f"{eq_name}_init_dir{INIT_DIR.split('/')[-1].split('_')[0]}",
            _iota_leaf,
        )
os.makedirs(OUT_ROOT, exist_ok=True)

print("Starting single-stage optimization on: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Send this to terminal/slurm output
print(f"Output directory root: {OUT_ROOT}")

boozer_type = {'initial': 'least_squares', 'final': 'exact'}  # example
stage = 'initial'  # or 'final', depending on what you want

# ==============================================================================
# SURFACE GEOMETRY DEFINITIONS
# ==============================================================================
# Solely for visualization purposes
VV = SurfaceRZFourier(nfp=results["surf_nfp"])
VV.set_rc(0, 0, results["VV_R0"])
VV.set_rc(1, 0, results["VV_a"])
VV.set_zs(1, 0, results["VV_b"])

# ==============================================================================
# LOAD EQUILIBRIUM AND COILS
# ==============================================================================
# Load coil set from previous run
bs = load(os.path.join(INIT_DIR, 'bs_opt.json'))

# Initialize the boundary magnetic surface
surf_opt_path = os.path.join(INIT_DIR, 'surf_opt.json')
if os.path.exists(surf_opt_path): # load from single-stage
    surf = load(surf_opt_path)
    plas_nPhi = surf.quadpoints_phi.size
    plas_nTheta = surf.quadpoints_theta.size
else: # load and scale the surface from stage 2
    plas_nPhi = 64
    plas_nTheta = 32
    eq_name_full = os.path.join(results["eq_dir"], results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_name_full, s=results["surf_s"], range="half period", nphi=plas_nPhi, ntheta=plas_nTheta
    )
    surf.set_dofs(results["surf_dof_scale"] * surf.get_dofs())

# Extract coil information
num_tf_coils = results["ntf"] * 2 * results["surf_nfp"]  # ntf is the number of TF coils per half-period, so this is the total number
coils = bs.coils
curves = [c.curve for c in coils]
tf_coils = coils[:num_tf_coils]
tf_curves = [c.curve for c in tf_coils]
dipole_coils = coils[num_tf_coils:]
dipole_curves = [c.curve for c in dipole_coils]

# Sparse mode: remove outboard-midplane dipoles before optimization.
if SPARSE:
    removed_dipoles = []
    kept_dipoles = []
    for idx, coil in enumerate(dipole_coils):
        theta = coil_center_theta(coil, results["VV_R0"])
        if angular_distance_to_zero(theta) <= THETA_TOL:
            removed_dipoles.append((idx, theta))
        else:
            kept_dipoles.append(coil)
    if not kept_dipoles:
        raise RuntimeError(
            f"All {len(dipole_coils)} dipole coils were removed with "
            f"theta_tol={THETA_TOL}; increase --theta-tol."
        )
    print(f"Sparse: removed {len(removed_dipoles)}/{len(dipole_coils)} outboard dipoles "
          f"(theta_tol={THETA_TOL})")
    dipole_coils = kept_dipoles
    dipole_curves = [c.curve for c in dipole_coils]
    coils = tf_coils + dipole_coils
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)
else:
    removed_dipoles = []

# Just triple make sure they're fixed
for c in dipole_curves:
    c.fix_all()
for c in tf_coils:
    c.current.fix_all()

# ==============================================================================
# BEGIN SINGLE STAGE OPTIMIZATION
# ==============================================================================
# We run a warm-started continuation in CURRENT_WEIGHT_SCHEDULE. Each stage gets
# its own output directory with the current weight and resolution encoded.
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

# Initialize Boozer surface using the loaded in coils + surface
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
boozer_surface = initialize_boozer_surface(surf, mpol, ntor, bs, VOL_TARGET, CONSTRAINT_WEIGHT, IOTA_TARGET, G0)
print(f"Initial boozer surface volume: {boozer_surface.surface.volume()}")

# ==============================================================================
# DEFINE OBJECTIVE FUNCTION COMPONENTS
# ==============================================================================
# Biot-Savart field calculation
bs_obj = BiotSavart(coils)

# Quasi-symmetry and Boozer coordinate residuals
nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
if boozer_type[stage]=='exact':
    brs = [BoozerResidualExact(boozer_surface, bs_obj)]
else:
    brs = [BoozerResidual(boozer_surface, bs_obj)]

# Individual (normalized) objective terms
# Get iota to iota target with tolerance-normalized penalty.
iota = Iotas(boozer_surface)
Jiota_raw = QuadraticPenalty(iota, IOTA_TARGET)
iota_scale = IOTA_REL_TOL * IOTA_TARGET
Jiota = (1.0 / (iota_scale ** 2)) * Jiota_raw

# Keep non-QS ratio below a user-specified target to within some tolerance
JnonQSRatio = sum(nonQSs)
QS_RATIO_INITIAL = float(JnonQSRatio.J())
JQSGuard_raw = QuadraticPenalty(JnonQSRatio, QS_TARGET, f="max")
qs_scale = QS_REL_TOL * QS_TARGET
JQSGuard = (1.0 / (qs_scale ** 2)) * JQSGuard_raw

# Boozer residual (field quality metric)
JBoozerResidual = sum(brs)

# Penalize the p-norm of the total current vector (proxy for max current)
CURRENT_P_NORM = 12.0
Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=CURRENT_P_NORM)

# --------------------------------------------------------------------------
# Normalize Boozer residual and current objective by their initial values so
# current-weight continuation has consistent meaning across runs.
# When --sparse, inherit Boozer residual scaling from the source stage
# so current-weight retains consistent physical meaning across the continuation
# chain, even after the residual increases from removing outboard dipoles.
# --------------------------------------------------------------------------
if SPARSE:
    if results.get("R0_boozer_residual") is not None:
        R0 = float(results["R0_boozer_residual"])
    else:
        R0 = float(JBoozerResidual.J())
else:
    R0 = float(JBoozerResidual.J())

I0 = float(Jcurrent.J())
if not np.isfinite(R0) or R0 <= 0:
    raise ValueError(f"Invalid initial Boozer residual for normalization: R0={R0}")
if not np.isfinite(I0) or I0 <= 0:
    raise ValueError(f"Invalid initial current objective for normalization: I0={I0}")

print(
    f"Initial nonQS ratio: {QS_RATIO_INITIAL:.6e} "
    f"(target={QS_TARGET:.6e})"
)
print(f"Normalization: R0 (Boozer) = {R0:.6e}, I0 (p-norm current) = {I0:.6e}, p={CURRENT_P_NORM:g}")
print(f"Iota tolerance normalization: rel_tol={IOTA_REL_TOL:.4g}, scale={iota_scale:.6e}")
print(f"QS tolerance normalization: rel_tol={QS_REL_TOL:.4g}, scale={qs_scale:.6e}")
print(f"Continuation current weights: {CURRENT_WEIGHT_SCHEDULE}")

# --------------------------------------------------------------------------
# Continuation loop (warm-start): one solution per current weight.
# Output structure:
#   ../single_stage_scans.../<eq_name>/iota_tarX/   or iota_tarX_volY/ if vol-target != default
#       stage00_cw0.3/mpol6_ntor6/...
#       stage01_cw1/mpol6_ntor6/...
# --------------------------------------------------------------------------
# Stage numbering: when --sparse, continue from the source stage index.
if SPARSE:
    _stage_parent = os.path.basename(os.path.dirname(INIT_DIR))
    _m = re.match(r"stage(\d+)_", _stage_parent)
    _stage_offset = int(_m.group(1)) + 1 if _m else 0
else:
    _stage_offset = 0

x0 = None
for stage_idx, CURRENT_WEIGHT in enumerate(CURRENT_WEIGHT_SCHEDULE):
    effective_idx = stage_idx + _stage_offset
    stage_tag = f"stage{effective_idx:02d}_cw{CURRENT_WEIGHT:g}"
    if SPARSE:
        stage_tag += "_sparse"
    OUT_DIR_STAGE = os.path.join(OUT_ROOT, stage_tag)
    OUT_DIR_ITER = os.path.join(OUT_DIR_STAGE, f"mpol{mpol}_ntor{ntor}")
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # Redirect stdout/stderr per stage to keep logs separate
    log_path = os.path.join(OUT_DIR_ITER, "log.txt")
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    #plot_cross_section(boozer_surface.surface, VV, OUT_DIR_ITER, "initial", plot_config, base_dipole_coils=dipole_coils)

    start_time = time.time()
    print(f"\n===== Single-stage continuation: {stage_tag} =====")
    print(f"Output directory: {OUT_DIR_ITER}")
    print(f"# of TF coils: {len(tf_coils)}")
    print(f"# of shaping coils: {len(dipole_coils)}")
    print(f"Starting equilibrium = {eq_name}")
    print(f"Resolution: mpol={mpol}, ntor={ntor}")
    print(f"Target volume: {VOL_TARGET}")
    print(f"Target iota: {IOTA_TARGET}")
    print(f"QS guard: target nonQS <= {QS_TARGET:.6e}, scale={qs_scale:.6e} ({100*QS_REL_TOL:.2f}% rel)")
    print(f"Iota target/tolerance: target={IOTA_TARGET:.6g}, scale={iota_scale:.6e} ({100*IOTA_REL_TOL:.2f}% rel)")
    print(f"Weights: iota_weight={IOTA_WEIGHT:g}, qs_weight={QS_WEIGHT:g}, current_weight={CURRENT_WEIGHT:g}")
    print(f"Normalized objective: (1/R0)*Boozer + (CURRENT_WEIGHT/I0)*Current + guards")
    print(f"Normalization: R0={R0:.6e}, I0={I0:.6e}\n")

    # Combined (normalized) objective function for this stage
    JF = (1.0 / R0) * JBoozerResidual + (CURRENT_WEIGHT / I0) * Jcurrent + QS_WEIGHT * JQSGuard + IOTA_WEIGHT * Jiota

    # Determine initial dofs for this stage
    if x0 is None:
        dofs = JF.x
        x0 = dofs.copy()
    else:
        # Warm-start from the previous stage solution
        x0 = np.asarray(x0, dtype=float)

    # Initialize run_dict after JF and boozer_surface are ready
    run_dict = {
        'sdofs': boozer_surface.surface.x.copy(),
        'iota': boozer_surface.res['iota'],
        'G': boozer_surface.res['G'],
        'J': JF.J(),
        'dJ': JF.dJ().copy(),
        'it': 1,
        'lscount': 0,
        'x_prev': x0.copy(),
        'failed_boozer_solves': 0,
        'stage_idx': int(stage_idx),
        'current_weight': float(CURRENT_WEIGHT),
        'R0': float(R0),
        'I0': float(I0),
    }

    # Run optimization (BFGS)
    res = minimize(
        fun,
        x0,
        jac=True,
        method='BFGS',
        callback=callback,
        options={'maxiter': MAXITER, 'gtol': GTOL},
    )
    print(res.message)

    # Warm start next stage from this solution
    x0 = res.x.copy()

    # Save optimized coil configurations
    coils_to_vtk(coils, filename=OUT_DIR_ITER + "/coils_opt", close=True)
    bs.save(OUT_DIR_ITER + "/bs_opt.json")

    # Save vacuum vessel for visualization
    VV.to_vtk(os.path.join(OUT_DIR_ITER, "vacuum_vessel"))

    # Save optimized surface with magnetic field normal component data
    pointData = {"B_N/B": np.sum(bs.B().reshape((plas_nPhi, plas_nTheta, 3)) *
        boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((plas_nPhi, plas_nTheta, 3))**2, axis=2))[:, :, None]}
    boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_opt", extra_data=pointData)
    boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_opt.json")

    # Compute final diagnostics
    true_max_current = float(np.max([abs(c.current.get_value()) for c in dipole_coils]))
    final_nonqs = float(JnonQSRatio.J())
    final_boozer = float(JBoozerResidual.J())

    print("\n===== Final results (stage) =====\n")
    print(f"Boozer surface volume: {boozer_surface.surface.volume()}")
    print(f"Iota: {Iotas(boozer_surface).J()}")
    print(f"Max current (true): {true_max_current:.6e} A")
    print(f"nonQS ratio: {final_nonqs:.6e} (guard target {QS_TARGET:.6e})")
    print(f"Boozer residual: {final_boozer:.6e} (normalized {final_boozer/R0:.6e})")
    print(f"Iota penalty: {Jiota.J():.6e}")
    print(f"Current objective (p-norm): {Jcurrent.J():.6e} A (normalized {Jcurrent.J()/I0:.6e})\n")

    # Generate final diagnostic plots
    plot_relBfinal_norm_modB(bs, boozer_surface.surface, OUT_DIR_ITER, "optimized", plot_config)
    plot_cross_section(boozer_surface.surface, VV, OUT_DIR_ITER, "optimized", plot_config, base_dipole_coils=dipole_coils)
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, OUT_DIR_ITER, "optimized", plot_config)
    plot_objective_vs_iterations(
        OUT_DIR_ITER,
        R0,
        I0,
        CURRENT_WEIGHT,
        QS_WEIGHT,
        IOTA_WEIGHT,
        plot_config,
    )

    # Save results dictionary for reproducibility and downstream use
    results_output = {
        # Stage 2 directory (in a "graph" sub-dict so postprocessing can parse init_id)
        "graph": {
            "init_dir": INIT_DIR,
        },

        # Continuation metadata
        "continuation_stage": int(effective_idx),
        "current_weight_schedule": [float(v) for v in CURRENT_WEIGHT_SCHEDULE],
        "current_weight_stage": float(CURRENT_WEIGHT),

        # Normalization metadata
        "normalized_objective": True,
        "R0_boozer_residual": float(R0),
        "I0_current_pnorm": float(I0),
        "current_pnorm": float(CURRENT_P_NORM),

        # Optimization configuration
        "mpol": mpol,
        "ntor": ntor,
        "maxiter": MAXITER,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "iota_target": IOTA_TARGET,
        "qs_weight": QS_WEIGHT,
        "iota_weight": IOTA_WEIGHT,

        # QS guard thresholds
        "QS_RATIO_INITIAL": float(QS_RATIO_INITIAL),
        "QS_RATIO_TARGET": float(QS_TARGET),
        "iota_rel_tol": float(IOTA_REL_TOL),
        "iota_scale": float(iota_scale),
        "qs_rel_tol": float(QS_REL_TOL),
        "qs_scale": float(qs_scale),

        # Convergence tolerances used
        "gtol": GTOL,

        # Optimization results
        "optimization_success": bool(res.success),
        "optimization_message": str(res.message),
        "final_objective": float(JF.J()),
        "final_iota": float(Iotas(boozer_surface).J()),
        "final_volume": float(boozer_surface.surface.volume()),

        # Diagnostic metrics
        "nonQS_ratio": float(final_nonqs),
        "boozer_residual": float(final_boozer),
        "iota_penalty": float(Jiota.J()),
        "current_pnorm": float(Jcurrent.J()),
        "max_current": float(true_max_current),

        # Coil information
        "# TF coils": len(tf_coils),
        "# dipole coils": len(dipole_coils),

        # Inherited from stage 2
        "eq_name": results["eq_name"],
        "eq_dir": results["eq_dir"],
        "surf_nfp": results["surf_nfp"],
        "surf_s": results["surf_s"],
        "VV_R0": results["VV_R0"],
        "VV_a": results["VV_a"],
        "VV_b": results["VV_b"],
        "ntf": results["ntf"],
    }
    if SPARSE:
        results_output["sparse"] = True
        results_output["theta_tol"] = float(THETA_TOL)
        results_output["removed_outboard_dipoles"] = len(removed_dipoles)

    # Save to file
    save(results_output, os.path.join(OUT_DIR_ITER, "results.json"))
    print(f"Results saved to {os.path.join(OUT_DIR_ITER, 'results.json')}")

    # Print total wall time for the stage
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Stage wall time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")

    # Close log file and restore streams
    log_file.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr

print("All continuation stages completed.")