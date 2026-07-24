import argparse
import io
import json
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from simsopt._core.optimizable import load, save
from simsopt.field import BiotSavart, CurrentPenalty, coils_to_vtk
from simsopt.geo import SurfaceRZFourier
from simsopt.geo.surfaceobjectives import BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.objectives import QuadraticPenalty

from boozer_functions import initialize_boozer_surface
from helper_functions import (
    PlotConfig,
    plot_coil_currents_on_theta_phi_grid,
    plot_cross_section,
    plot_relBfinal_norm_modB,
)


def coil_center_theta(coil, major_radius):
    gamma = coil.curve.gamma()
    center = np.mean(gamma, axis=0)
    x0, y0, z0 = center
    r0 = np.sqrt(x0 * x0 + y0 * y0)
    return np.mod(np.arctan2(z0, r0 - major_radius), 2 * np.pi)


def angular_distance_to_zero(theta):
    return np.minimum(theta, 2 * np.pi - theta)


def parse_stage_name(stage_name):
    # Baseline continuation: stage03_cw1 — sparsity continuation: stage04_cw1_sparsity
    match = re.match(r"^stage(\d+)_cw([0-9eE+\-.]+)(?:_sparsity)?$", stage_name)
    if not match:
        return None
    return int(match.group(1)), float(match.group(2))


def is_sparsity_stage_name(stage_name):
    return stage_name.endswith("_sparsity")


def parse_mpol_ntor(dirname):
    match = re.match(r"^mpol(\d+)_ntor(\d+)$", dirname)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def find_latest_stage_iter_dir(iota_dir, include_sparsity=True, source_stage=None):
    stage_candidates = []
    for stage_name in os.listdir(iota_dir):
        stage_path = os.path.join(iota_dir, stage_name)
        if not os.path.isdir(stage_path):
            continue
        parsed = parse_stage_name(stage_name)
        if parsed is None:
            continue
        stage_idx, cw = parsed
        if (not include_sparsity) and is_sparsity_stage_name(stage_name):
            continue
        if source_stage is not None and stage_idx != source_stage:
            continue
        iter_dirs = []
        for sub in os.listdir(stage_path):
            sub_path = os.path.join(stage_path, sub)
            if not os.path.isdir(sub_path):
                continue
            mpol_ntor = parse_mpol_ntor(sub)
            if mpol_ntor is None:
                continue
            iter_dirs.append((mpol_ntor[0], mpol_ntor[1], sub_path))
        if not iter_dirs:
            continue
        iter_dirs.sort(key=lambda x: (x[0], x[1]))
        mpol, ntor, iter_dir = iter_dirs[-1]
        stage_candidates.append((stage_idx, cw, stage_name, mpol, ntor, iter_dir))
    if not stage_candidates:
        return None
    stage_candidates.sort(key=lambda x: (x[0], x[1]))
    return stage_candidates[-1]


def plot_objective_vs_iterations(out_dir, r0, i0, current_weight, qs_weight, iota_weight, config):
    history_path = os.path.join(out_dir, "iterations.json")
    if not os.path.isfile(history_path):
        return
    with open(history_path, "r") as f:
        history = json.load(f)
    rows = history.get("iterations") or []
    if len(rows) < 1:
        return

    it = np.array([r["iteration"] for r in rows], dtype=float)
    j_tot = np.array([r["J"] for r in rows], dtype=float)
    j_boozer = np.array([r["J_Boozer"] for r in rows], dtype=float) / r0
    j_qs = np.array([r["J_nonQS_guard"] for r in rows], dtype=float) * qs_weight
    j_iota = np.array([r["J_iota"] for r in rows], dtype=float) * iota_weight
    j_current = np.array([r["J_current"] for r in rows], dtype=float) * (current_weight / i0)
    grad_norm = np.array([r["grad_norm"] for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), constrained_layout=True
    )
    ax1.plot(it, j_tot, color="k", lw=2.0, label=r"$J$ (total)")
    ax1.plot(it, j_boozer, label=r"$(1/R_0)\,J_{\mathrm{Boozer}}$")
    ax1.plot(it, j_qs, label=r"$w_{\mathrm{QS}} J_{\mathrm{QS\,guard}}$")
    ax1.plot(it, j_iota, label=r"$w_{\iota} J_{\iota}$")
    ax1.plot(it, j_current, label=r"$(w_I/I_0)\,J_{\mathrm{current}}$")
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

    fig.savefig(os.path.join(out_dir, "objective_vs_iteration.png"), dpi=config.dpi)
    plt.close(fig)


def resolve_root_dir(init_dir_root):
    if os.path.isdir(init_dir_root):
        return init_dir_root
    fallbacks = [
        os.path.join("..", "single_stage_scans_epsilon_constraint_updated", init_dir_root),
        # os.path.join("..", "single_stage_scans_no_sparsity_epsilon_constraint", init_dir_root),
    ]
    for fallback in fallbacks:
        if os.path.isdir(fallback):
            return fallback
    raise FileNotFoundError(
        "Could not find init-dir root at "
        f"'{init_dir_root}' or under ../single_stage_scans_epsilon_constraint_updated/ "
        f"or ../single_stage_scans_no_sparsity_epsilon_constraint/ (tried: {fallbacks})."
    )


_DEFAULT_VOL_TARGET = 0.3


def _parse_iota_vol_from_leaf(leaf: str):
    """
    Parse iota and optional Boozer volume suffix from a directory name, e.g.
    iota_tar0.1 -> (0.1, None); iota_tar0.1_vol0.4 -> (0.1, 0.4).
    Returns None if the name does not match.
    """
    m = re.match(r"^iota_tar([0-9.eE+-]+)(?:_vol([0-9.eE+-]+))?$", leaf)
    if not m:
        return None
    iota_val = float(m.group(1))
    vol_val = float(m.group(2)) if m.group(2) is not None else None
    return iota_val, vol_val


def select_iota_dirs(root_dir, iota_target=None, vol_target=None):
    all_iota_dirs = sorted(
        [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if d.startswith("iota_tar") and os.path.isdir(os.path.join(root_dir, d))
        ]
    )
    if not all_iota_dirs:
        raise RuntimeError(f"No iota_tar* directories found in {root_dir}")

    if iota_target is None:
        if vol_target is not None:
            raise ValueError("--vol-target requires --iota-target when selecting runs.")
        return all_iota_dirs

    matched = []
    for d in all_iota_dirs:
        base = os.path.basename(d)
        parsed = _parse_iota_vol_from_leaf(base)
        if parsed is None:
            continue
        val, vol_suff = parsed
        if not np.isclose(val, iota_target, rtol=0.0, atol=1e-12):
            continue
        if vol_target is None:
            matched.append(d)
        else:
            eff_vol = _DEFAULT_VOL_TARGET if vol_suff is None else vol_suff
            if np.isclose(eff_vol, vol_target, rtol=0.0, atol=1e-12):
                matched.append(d)

    if not matched:
        raise RuntimeError(
            f"Could not find iota_tar directory matching iota={iota_target:g}"
            + (f", vol_target={vol_target:g}" if vol_target is not None else "")
            + f" in {root_dir}"
        )
    if vol_target is None and len(matched) > 1:
        raise RuntimeError(
            f"Multiple directories match iota={iota_target:g} in {root_dir}: "
            f"{[os.path.basename(p) for p in matched]}. "
            "Pass --vol-target (e.g. 0.3 for plain iota_tar*, or 0.35/0.4 for _vol*) to pick one."
        )
    return matched


def run_one_iota_dir(
    iota_dir,
    theta_tol,
    maxiter,
    constraint_weight,
    gtol,
    verbose,
    source_stage=None,
    current_weight_override=None,
):
    # Match single_stage_epsilon_constraint.py: Lp exponent on dipole current vector.
    current_penalty_p = 12.0
    latest = find_latest_stage_iter_dir(
        iota_dir, include_sparsity=False, source_stage=source_stage
    )
    # Backward-compatible fallback if only sparsity stages are present.
    if latest is None and source_stage is None:
        latest = find_latest_stage_iter_dir(iota_dir, include_sparsity=True, source_stage=None)
    if latest is None:
        if source_stage is None:
            print(f"Skipping {iota_dir}: no stageXX_cwY/mpol*_ntor* data found.")
        else:
            print(
                f"Skipping {iota_dir}: no matching stage{source_stage:02d}_cw*/mpol*_ntor* found."
            )
        return
    prev_stage_idx, source_current_weight, prev_stage_name, mpol, ntor, prev_iter_dir = latest
    current_weight = (
        float(source_current_weight)
        if current_weight_override is None
        else float(current_weight_override)
    )
    next_stage_idx = prev_stage_idx + 1
    next_stage_tag = f"stage{next_stage_idx:02d}_cw{current_weight:g}_sparsity_r0_source"
    out_dir_stage = os.path.join(iota_dir, next_stage_tag)
    out_dir_iter = os.path.join(out_dir_stage, f"mpol{mpol}_ntor{ntor}")
    os.makedirs(out_dir_iter, exist_ok=True)

    prev_results_path = os.path.join(prev_iter_dir, "results.json")
    prev_bs_path = os.path.join(prev_iter_dir, "bs_opt.json")
    prev_surf_path = os.path.join(prev_iter_dir, "surf_opt.json")
    if not os.path.isfile(prev_results_path):
        raise FileNotFoundError(f"Missing {prev_results_path}")
    if not os.path.isfile(prev_bs_path):
        raise FileNotFoundError(f"Missing {prev_bs_path}")
    if not os.path.isfile(prev_surf_path):
        raise FileNotFoundError(f"Missing {prev_surf_path}")

    results = load(prev_results_path)
    bs_dense = load(prev_bs_path)
    surf = load(prev_surf_path)

    if gtol is None:
        gtol = float(results.get("gtol", 1e-3))
    constraint_weight = float(results.get("constraint_weight", constraint_weight))

    leaf = os.path.basename(iota_dir)
    parsed_iv = _parse_iota_vol_from_leaf(leaf)
    if parsed_iv is None:
        vol_target = _DEFAULT_VOL_TARGET
    else:
        _iot_p, vol_suff = parsed_iv
        vol_target = _DEFAULT_VOL_TARGET if vol_suff is None else vol_suff

    eq_name = results["eq_name"]
    if results.get("iota_target") is not None:
        iota_target = float(results["iota_target"])
    elif parsed_iv is not None:
        iota_target = float(parsed_iv[0])
    else:
        raise ValueError(f"Could not determine iota_target for directory {leaf}")
    qs_weight = float(results.get("qs_weight", 1e2))
    iota_weight = float(results.get("iota_weight", 1e2))
    qs_target = float(results.get("QS_RATIO_TARGET", 1e-3))
    qs_rel_tol = float(results.get("qs_rel_tol", 0.10))
    iota_rel_tol = float(results.get("iota_rel_tol", 0.01))

    num_tf_coils = results["ntf"] * 2 * results["surf_nfp"]
    coils_dense = bs_dense.coils
    tf_coils = coils_dense[:num_tf_coils]
    dipole_coils = coils_dense[num_tf_coils:]

    vv = SurfaceRZFourier(nfp=results["surf_nfp"])
    vv.set_rc(0, 0, results["VV_R0"])
    vv.set_rc(1, 0, results["VV_a"])
    vv.set_zs(1, 0, results["VV_b"])

    removed = []
    kept_dipoles = []
    for idx, coil in enumerate(dipole_coils):
        theta = coil_center_theta(coil, results["VV_R0"])
        if angular_distance_to_zero(theta) <= theta_tol:
            removed.append((idx, theta))
        else:
            kept_dipoles.append(coil)
    if len(kept_dipoles) == 0:
        raise RuntimeError(f"All dipole coils were removed in {iota_dir}; increase/decrease theta_tol.")

    coils = tf_coils + kept_dipoles
    bs = BiotSavart(coils)

    # Ensure the same fixed/free structure as baseline single-stage script.
    for c in kept_dipoles:
        c.curve.fix_all()
    for c in tf_coils:
        c.current.fix_all()

    # Boozer initialization from sparse coil set (volume matches iota_tar*_vol* path / default 0.3).
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
    g0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
    iota_init = float(results.get("final_iota", iota_target))
    boozer_surface = initialize_boozer_surface(
        surf, mpol, ntor, bs, vol_target, constraint_weight, iota_init, g0
    )

    boozer_type = "least_squares"
    non_qs = [NonQuasiSymmetricRatio(boozer_surface, bs)]
    if boozer_type == "least_squares":
        brs = [BoozerResidual(boozer_surface, bs)]
    else:
        raise ValueError("Unsupported boozer residual type.")

    iota_obj = Iotas(boozer_surface)
    jiota_raw = QuadraticPenalty(iota_obj, iota_target)
    iota_scale = iota_rel_tol * iota_target
    jiota = (1.0 / (iota_scale ** 2)) * jiota_raw

    j_nonqs_ratio = sum(non_qs)
    qs_ratio_initial = float(j_nonqs_ratio.J())
    j_qs_guard_raw = QuadraticPenalty(j_nonqs_ratio, qs_target, f="max")
    qs_scale = qs_rel_tol * qs_target
    j_qs_guard = (1.0 / (qs_scale ** 2)) * j_qs_guard_raw

    j_boozer = sum(brs)
    j_current = CurrentPenalty([c.current for c in kept_dipoles], p=current_penalty_p)

    r0_source = "source_stage"
    if results.get("R0_boozer_residual") is not None:
        r0 = float(results["R0_boozer_residual"])
    else:
        # Fallback for older source results lacking normalized-objective metadata.
        r0_source = "sparse_init_fallback"
        r0 = float(j_boozer.J())
    i0 = float(j_current.J())
    if not np.isfinite(r0) or r0 <= 0:
        raise ValueError(f"Invalid initial Boozer residual normalization in {iota_dir}: R0={r0}")
    if not np.isfinite(i0) or i0 <= 0:
        raise ValueError(f"Invalid initial current normalization in {iota_dir}: I0={i0}")

    jf = (1.0 / r0) * j_boozer + (current_weight / i0) * j_current + qs_weight * j_qs_guard + iota_weight * jiota
    x0 = jf.x.copy()

    run_dict = {
        "sdofs": boozer_surface.surface.x.copy(),
        "iota": boozer_surface.res["iota"],
        "G": boozer_surface.res["G"],
        "J": jf.J(),
        "dJ": jf.dJ().copy(),
        "it": 1,
        "lscount": 0,
        "x_prev": x0.copy(),
        "failed_boozer_solves": 0,
    }

    plot_config = PlotConfig(
        dpi=100,
        titlefontsize=16,
        axisfontsize=16,
        legendfontsize=14,
        ticklabelfontsize=14,
        cbarfontsize=16,
    )
    plas_nphi = surf.quadpoints_phi.size
    plas_ntheta = surf.quadpoints_theta.size

    def fun(x):
        dx = np.linalg.norm(x - run_dict["x_prev"])
        run_dict["x_prev"] = x.copy()
        print(f"Step size: {dx:.2e}")

        run_dict["lscount"] += 1
        boozer_surface.surface.x = run_dict["sdofs"]
        boozer_surface.res["iota"] = run_dict["iota"]
        boozer_surface.res["G"] = run_dict["G"]

        jf.x = x
        boozer_surface.run_code(run_dict["iota"], run_dict["G"])

        try:
            success1 = boozer_surface.res["success"]
            succuss2 = True
            # success2 = not boozer_surface.surface.is_self_intersecting()
        except Exception as exc:
            print(f"Surface check failed: {exc}")
            success1 = False
            success2 = False
        success = success1 and success2

        if success:
            run_dict["failed_boozer_solves"] = 0
            j_val = jf.J()
            dj = jf.dJ()
        else:
            print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
            run_dict["failed_boozer_solves"] += 1
            j_val = run_dict["J"]
            dj = -run_dict["dJ"]
            boozer_surface.surface.x = run_dict["sdofs"]
            boozer_surface.res["iota"] = run_dict["iota"]
            boozer_surface.res["G"] = run_dict["G"]

        print(f"Objective J: {j_val:.6e}, ||∇J||: {np.linalg.norm(dj):.6e}")
        if success:
            print(
                "Individual scaled terms -- "
                f"Boozer: {(1.0 / r0) * j_boozer.J():.6e}, "
                f"QS_guard: {qs_weight * j_qs_guard.J():.6e}, "
                f"iota: {iota_weight * jiota.J():.6e}, "
                f"current: {(current_weight / i0) * j_current.J():.6e}"
            )
        return j_val, dj

    def callback(_x):
        run_dict["lscount"] = 0
        run_dict["sdofs"] = boozer_surface.surface.x.copy()
        run_dict["iota"] = boozer_surface.res["iota"]
        run_dict["G"] = boozer_surface.res["G"]
        run_dict["J"] = jf.J()
        run_dict["dJ"] = jf.dJ().copy()

        j_val = run_dict["J"]
        grad = run_dict["dJ"]
        j_qs = j_nonqs_ratio.J()
        j_qs_guard_val = j_qs_guard.J()
        j_boozer_val = j_boozer.J()
        j_iota_val = jiota.J()
        j_curr = j_current.J()
        bdotn = np.mean(
            np.abs(
                np.sum(
                    bs.B().reshape((plas_nphi, plas_ntheta, 3)) * boozer_surface.surface.unitnormal(),
                    axis=2,
                )
            )
        )
        currents = np.array([abs(c.current.get_value()) for c in kept_dipoles])
        max_current = np.max(currents)

        width = 35
        buffer = io.StringIO()
        print("=" * 70, file=buffer)
        print(f"ITERATION {run_dict['it']}", file=buffer)
        print(f"{'Objective J':{width}} = {j_val:.6e}", file=buffer)
        print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
        print(f"{'nonQS ratio':{width}} = {j_qs:.6e}", file=buffer)
        print(f"{'Boozer Residual':{width}} = {j_boozer_val:.6e}", file=buffer)
        print(f"{'ι Penalty':{width}} = {j_iota_val:.6e}", file=buffer)
        print(f"{'Current Penalty':{width}} = {j_curr:.6e}", file=buffer)
        print(f"{'⟨|B·n|⟩':{width}} = {bdotn:.6e}", file=buffer)
        print(f"{'Max current':{width}} = {max_current:.2f} A", file=buffer)
        print("=" * 70, file=buffer)
        print(buffer.getvalue())
        buffer.close()

        history_path = os.path.join(out_dir_iter, "iterations.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = {
                "weights": {
                    "QS_WEIGHT": qs_weight,
                    "IOTA_WEIGHT": iota_weight,
                    "CURRENT_WEIGHT": current_weight,
                },
                "targets": {
                    "IOTA_TARGET": iota_target,
                    "QS_RATIO_INITIAL": qs_ratio_initial,
                    "QS_RATIO_TARGET": qs_target,
                },
                "tolerances": {
                    "gtol": gtol,
                    "iota_rel_tol": iota_rel_tol,
                    "iota_scale": iota_scale,
                    "qs_rel_tol": qs_rel_tol,
                    "qs_scale": qs_scale,
                },
                "max_iterations": maxiter,
                "iterations": [],
            }

        history["iterations"].append(
            {
                "iteration": int(run_dict["it"]),
                "J": float(j_val),
                "grad_norm": float(np.linalg.norm(grad)),
                "J_nonQS": float(j_qs),
                "J_nonQS_guard": float(j_qs_guard_val),
                "J_Boozer": float(j_boozer_val),
                "J_iota": float(j_iota_val),
                "J_current": float(j_curr),
                "J_nonQS_guard_scaled": float(qs_weight * j_qs_guard_val),
                "J_iota_scaled": float(iota_weight * j_iota_val),
                "J_current_scaled": float(current_weight * j_curr),
                "iota": float(iota_obj.J()),
                "volume": float(boozer_surface.surface.volume()),
                "BdotN": float(bdotn),
                "max_current": float(max_current),
            }
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        partial_results = {
            "graph": {"init_dir": prev_iter_dir, "sparsity_parent": iota_dir},
            "continuation_stage": int(next_stage_idx),
            "current_weight_stage": float(current_weight),
            "r0_normalization_source": r0_source,
            "mpol": mpol,
            "ntor": ntor,
            "maxiter": maxiter,
            "constraint_weight": constraint_weight,
            "iota_target": iota_target,
            "qs_weight": qs_weight,
            "iota_weight": iota_weight,
            "gtol": gtol,
            "optimization_success": None,
            "optimization_message": "partial snapshot from callback",
            "final_objective": float(j_val),
            "final_iota": float(iota_obj.J()),
            "final_volume": float(boozer_surface.surface.volume()),
            "nonQS_ratio": float(j_nonqs_ratio.J()),
            "boozer_residual": float(j_boozer.J()),
            "iota_penalty": float(jiota.J()),
            "current_penalty_value": float(j_current.J()),
            "max_current": float(max_current),
            "# TF coils": len(tf_coils),
            "# dipole coils": len(kept_dipoles),
            "removed_outboard_dipoles": int(len(removed)),
            "eq_name": results["eq_name"],
            "eq_dir": results["eq_dir"],
            "surf_nfp": results["surf_nfp"],
            "surf_s": results["surf_s"],
            "VV_R0": results["VV_R0"],
            "VV_a": results["VV_a"],
            "VV_b": results["VV_b"],
            "ntf": results["ntf"],
        }
        save(partial_results, os.path.join(out_dir_iter, "results.json"))
        bs.save(os.path.join(out_dir_iter, "bs_opt.json"))
        run_dict["it"] += 1

    print(f"\n===== Running sparsity continuation for {iota_dir} =====")
    print(f"Boozer volume target (init): {vol_target:g} (from directory name)")
    print(f"Using source stage: {prev_stage_name}/{os.path.basename(prev_iter_dir)}")
    print(
        f"Current weight: source={source_current_weight:g}, run={current_weight:g}"
        + (" (overridden)" if current_weight_override is not None else " (inherited)")
    )
    print(f"R0 normalization source: {r0_source} (R0={r0:.6e})")
    print(f"Creating output stage: {next_stage_tag}/mpol{mpol}_ntor{ntor}")
    print(f"Removed outboard dipoles: {len(removed)} / {len(dipole_coils)}")
    if verbose and removed:
        print("Removed dipole indices/thetas:")
        for idx, theta in removed:
            print(f"  dipole_idx={idx:4d}, theta={theta:.6f}")

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_path = os.path.join(out_dir_iter, "log.txt")
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    start_time = time.time()

    try:
        plot_cross_section(boozer_surface.surface, vv, out_dir_iter, "initial", plot_config, base_dipole_coils=kept_dipoles)

        res = minimize(
            fun,
            x0,
            jac=True,
            method="BFGS",
            callback=callback,
            options={"maxiter": maxiter, "gtol": gtol},
        )
        print(res.message)

        coils_to_vtk(coils, filename=os.path.join(out_dir_iter, "coils_opt"), close=True)
        bs.save(os.path.join(out_dir_iter, "bs_opt.json"))
        vv.to_vtk(os.path.join(out_dir_iter, "vacuum_vessel"))

        point_data = {
            "B_N/B": np.sum(
                bs.B().reshape((plas_nphi, plas_ntheta, 3)) * boozer_surface.surface.unitnormal(),
                axis=2,
            )[:, :, None]
            / np.sqrt(np.sum(bs.B().reshape((plas_nphi, plas_ntheta, 3)) ** 2, axis=2))[:, :, None]
        }
        boozer_surface.surface.to_vtk(os.path.join(out_dir_iter, "surf_opt"), extra_data=point_data)
        boozer_surface.surface.save(os.path.join(out_dir_iter, "surf_opt.json"))

        true_max_current = float(np.max([abs(c.current.get_value()) for c in kept_dipoles]))
        final_nonqs = float(j_nonqs_ratio.J())
        final_boozer = float(j_boozer.J())

        plot_relBfinal_norm_modB(bs, boozer_surface.surface, out_dir_iter, "optimized", plot_config)
        plot_cross_section(boozer_surface.surface, vv, out_dir_iter, "optimized", plot_config, base_dipole_coils=kept_dipoles)
        plot_coil_currents_on_theta_phi_grid(kept_dipoles, vv, out_dir_iter, "optimized", plot_config)
        plot_objective_vs_iterations(out_dir_iter, r0, i0, current_weight, qs_weight, iota_weight, plot_config)

        results_output = {
            "graph": {"init_dir": prev_iter_dir, "sparsity_parent": iota_dir},
            "continuation_stage": int(next_stage_idx),
            "current_weight_stage": float(current_weight),
            "normalized_objective": True,
            "r0_normalization_source": r0_source,
            "R0_boozer_residual": float(r0),
            "I0_current_pnorm": float(i0),
            "mpol": mpol,
            "ntor": ntor,
            "maxiter": maxiter,
            "constraint_weight": constraint_weight,
            "iota_target": iota_target,
            "qs_weight": qs_weight,
            "iota_weight": iota_weight,
            "QS_RATIO_INITIAL": float(qs_ratio_initial),
            "QS_RATIO_TARGET": float(qs_target),
            "iota_rel_tol": float(iota_rel_tol),
            "iota_scale": float(iota_scale),
            "qs_rel_tol": float(qs_rel_tol),
            "qs_scale": float(qs_scale),
            "gtol": gtol,
            "optimization_success": bool(res.success),
            "optimization_message": str(res.message),
            "final_objective": float(jf.J()),
            "final_iota": float(Iotas(boozer_surface).J()),
            "final_volume": float(boozer_surface.surface.volume()),
            "nonQS_ratio": float(final_nonqs),
            "boozer_residual": float(final_boozer),
            "iota_penalty": float(jiota.J()),
            "current_pnorm": float(j_current.J()),
            "current_penalty_value": float(j_current.J()),
            "max_current": float(true_max_current),
            "# TF coils": len(tf_coils),
            "# dipole coils": len(kept_dipoles),
            "removed_outboard_dipoles": int(len(removed)),
            "eq_name": results["eq_name"],
            "eq_dir": results["eq_dir"],
            "surf_nfp": results["surf_nfp"],
            "surf_s": results["surf_s"],
            "VV_R0": results["VV_R0"],
            "VV_a": results["VV_a"],
            "VV_b": results["VV_b"],
            "ntf": results["ntf"],
        }
        save(results_output, os.path.join(out_dir_iter, "results.json"))

        elapsed = time.time() - start_time
        print(f"Stage wall time: {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")
    finally:
        log_file.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    print(f"Finished {iota_dir} -> {out_dir_iter}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each iota_tar* subdirectory in an init-dir root, continue from the highest "
            "stageXX_cwY, remove outboard-midplane dipoles, and run one new sparsity stage."
        )
    )
    parser.add_argument(
        "--init-dir-root",
        type=str,
        required=True,
        help=(
            "Root directory containing iota_tar* folders (e.g. wout_nfp..._init_dir90 or full path)."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=150,
        help="Maximum optimizer iterations for the new sparsity stage (single_stage_epsilon_constraint uses 150).",
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        default=None,
        help="If provided, only process this iota target (e.g. 0.15 for iota_tar0.15).",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=None,
        help=(
            "Boozer volume branch: use 0.3 for plain iota_tar* (default volume), or 0.35/0.4 for "
            "iota_tar*_vol* folders. Required when multiple directories share the same iota."
        ),
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=None,
        help="BFGS gtol (default: read from source results.json, else 1e-3 as in single_stage_epsilon_constraint).",
    )
    parser.add_argument(
        "--source-stage",
        type=int,
        default=None,
        help=(
            "Stage index to initialize from (e.g. 3 for stage03_*). "
            "Default: highest stage without '_sparsity'."
        ),
    )
    parser.add_argument(
        "--current-weight",
        type=float,
        default=None,
        help=(
            "Current-weight to use in the new sparsity run. "
            "Default: inherit source stage current weight."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print removed dipole indices/thetas per iota target.",
    )
    args = parser.parse_args()

    root_dir = resolve_root_dir(args.init_dir_root)
    iota_dirs = select_iota_dirs(root_dir, args.iota_target, args.vol_target)

    print(f"Root directory: {root_dir}")
    if args.iota_target is None:
        print(f"Found {len(iota_dirs)} iota targets.")
    else:
        msg = f"Selected iota target: {args.iota_target:g}"
        if args.vol_target is not None:
            msg += f", vol_target: {args.vol_target:g}"
        print(msg)
    for iota_dir in iota_dirs:
        run_one_iota_dir(
            iota_dir=iota_dir,
            theta_tol=0.01,
            maxiter=args.maxiter,
            constraint_weight=1.0,
            gtol=args.gtol,
            verbose=args.verbose,
            source_stage=args.source_stage,
            current_weight_override=args.current_weight,
        )
    print("All sparsity continuation stages completed.")


if __name__ == "__main__":
    main()
