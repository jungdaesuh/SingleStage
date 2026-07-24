"""
Load a stage-2 dipole optimization directory (bs_opt.json + results.json from
optimize.py), remove windowpane coils near the outboard midplane (theta = 0),
and test Boozer surface initialization.

Default run directory matches:
  examples/outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_000281/
  90_npol_10_ntor_8_VV_a_0.250_VV_b_0.282_VV_R0_1.042
"""

import argparse
import os
import numpy as np

from simsopt._core.optimizable import load
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart

from boozer_functions import initialize_boozer_surface
from helper_functions import PlotConfig, plot_relBfinal_norm_modB, plot_cross_section


def _default_stage2_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(
            here,
            "../outputs/stage_2_npol10_ntor8_wout_nfp22ginsburg_000_000281/"
            "90_npol_10_ntor_8_VV_a_0.250_VV_b_0.282_VV_R0_1.042",
        )
    )


def coil_center_theta(coil, major_radius):
    """
    Estimate coil center poloidal angle from geometric center.
    theta = atan2(Z, R - R0), with theta = 0 at outboard midplane.
    """
    gamma = coil.curve.gamma()
    center = np.mean(gamma, axis=0)
    x0, y0, z0 = center
    r0 = np.sqrt(x0 * x0 + y0 * y0)
    return np.mod(np.arctan2(z0, r0 - major_radius), 2 * np.pi)


def angular_distance_to_zero(theta):
    """Smallest periodic angular distance from theta to 0 on [0, 2*pi)."""
    return np.minimum(theta, 2 * np.pi - theta)


def load_plasma_surface_stage2(results):
    """Half-period plasma boundary from VMEC wout (matches optimize.py)."""
    eq_name_full = os.path.join(results["eq_dir"], results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_name_full,
        s=results["surf_s"],
        range="half period",
        nphi=128,
        ntheta=64,
    )
    surf.set_dofs(results["surf_dof_scale"] * surf.get_dofs())
    return surf


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 run: remove dipoles near outboard midplane (theta=0) and "
            "test Boozer surface initialization."
        )
    )
    parser.add_argument(
        "--stage2-dir",
        type=str,
        default=None,
        help=(
            "Directory with bs_opt.json and results.json from stage_2 optimize(). "
            f"Default: {_default_stage2_dir()}"
        ),
    )
    parser.add_argument(
        "--theta-tol",
        type=float,
        default=0.01,
        help="Angular tolerance [rad] for coils near theta=0 (default: 0.01).",
    )
    parser.add_argument("--mpol", type=int, default=6, help="Boozer surface mpol.")
    parser.add_argument("--ntor", type=int, default=6, help="Boozer surface ntor.")
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=1.0,
        help="Boozer least-squares constraint weight.",
    )
    parser.add_argument(
        "--iota-init",
        type=float,
        default=None,
        help="Initial iota guess. If omitted, use results['final_iota'] if present, else 0.1.",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=None,
        help="Volume constraint for Boozer init. Default: plasma surf.volume() or results['surf_volume'].",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for plots. Default: <stage2-dir>/sparse_midplane_check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-coil theta for removed dipoles.",
    )
    args = parser.parse_args()

    stage2_dir = args.stage2_dir if args.stage2_dir is not None else _default_stage2_dir()
    bs_path = os.path.join(stage2_dir, "bs_opt.json")
    results_path = os.path.join(stage2_dir, "results.json")

    if not os.path.isfile(bs_path):
        raise FileNotFoundError(f"Missing {bs_path}")
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Missing {results_path}")

    results = load(results_path)
    bs = load(bs_path)
    coils = bs.coils

    # Same convention as optimize.py: TF expanded via coils_via_symmetries.
    num_tf_coils = results["ntf"] * 2 * results["surf_nfp"]
    if num_tf_coils > len(coils):
        raise ValueError(
            f"Computed TF coil count ({num_tf_coils}) exceeds total coil count ({len(coils)})."
        )

    tf_coils = coils[:num_tf_coils]
    dipole_coils = coils[num_tf_coils:]

    surf = load_plasma_surface_stage2(results)

    vv_major_radius = results["VV_R0"]
    theta_tol = args.theta_tol
    plot_dir = args.plot_dir if args.plot_dir is not None else os.path.join(
        stage2_dir, "sparse_midplane_check"
    )
    os.makedirs(plot_dir, exist_ok=True)

    removed = []
    kept_dipoles = []
    for idx, coil in enumerate(dipole_coils):
        theta = coil_center_theta(coil, vv_major_radius)
        if angular_distance_to_zero(theta) <= theta_tol:
            removed.append((idx, coil, theta))
        else:
            kept_dipoles.append(coil)

    sparse_coils = tf_coils + kept_dipoles
    sparse_bs = BiotSavart(sparse_coils)

    if args.vol_target is not None:
        vol_target = float(args.vol_target)
    else:
        vol_target = float(results.get("surf_volume", surf.volume()))

    current_sum_tf = sum(abs(c.current.get_value()) for c in tf_coils)
    mu0_over_2pi = 4 * np.pi * 1e-7 / (2 * np.pi)
    g0 = 2.0 * np.pi * current_sum_tf * mu0_over_2pi
    iota_init = (
        args.iota_init
        if args.iota_init is not None
        else results.get("final_iota", 0.1)
    )

    print("=== Stage-2 sparse midplane Boozer initialization check ===")
    print(f"stage2_dir: {stage2_dir}")
    print(f"Total coils loaded: {len(coils)}")
    print(f"TF coils: {len(tf_coils)}")
    print(f"Dipole coils before removal: {len(dipole_coils)}")
    print(f"Dipole coils removed near theta=0: {len(removed)}")
    print(f"Dipole coils kept: {len(kept_dipoles)}")
    print(f"Total coils after removal: {len(sparse_coils)}")
    print(f"theta_tol [rad]: {theta_tol}")
    print(f"vol_target [m^3]: {vol_target:.6e}")
    print(f"Boozer init guesses: iota={iota_init:.6g}, G0={g0:.6e}")
    print(f"plot_dir: {plot_dir}")

    if args.verbose and removed:
        print("Removed dipole coil indices (within dipole block) and theta [rad]:")
        for idx, _, theta in removed:
            print(f"  dipole_idx={idx:4d}, theta={theta:.6f}")

    try:
        boozer_surface = initialize_boozer_surface(
            surf_prev=surf,
            mpol=args.mpol,
            ntor=args.ntor,
            bs=sparse_bs,
            vol_target=vol_target,
            constraint_weight=args.constraint_weight,
            iota=iota_init,
            G0=g0,
        )
        print("RESULT: SUCCESS")
        print(f"Initialized Boozer surface volume: {boozer_surface.surface.volume():.6e}")
        print(f"Initialized iota: {boozer_surface.res['iota']:.6e}")
        print(f"Initialized G: {boozer_surface.res['G']:.6e}")

        vv = SurfaceRZFourier(nfp=results["surf_nfp"])
        vv.set_rc(0, 0, results["VV_R0"])
        vv.set_rc(1, 0, results["VV_a"])
        vv.set_zs(1, 0, results["VV_b"])
        plot_config = PlotConfig()

        sparse_bs.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
        plot_relBfinal_norm_modB(
            sparse_bs,
            boozer_surface.surface,
            plot_dir,
            "boozer_init_sparse_midplane",
            plot_config,
        )
        plot_cross_section(
            boozer_surface.surface,
            vv,
            plot_dir,
            "boozer_init_sparse_midplane",
            plot_config,
            base_dipole_coils=kept_dipoles,
        )
        boozer_surface.surface.to_vtk(
            os.path.join(plot_dir, "boozer_surface_init_sparse_midplane")
        )
        print(f"Saved Boozer initialization plots to: {plot_dir}")
    except Exception as exc:
        print("RESULT: FAILURE")
        print(f"Boozer initialization failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
