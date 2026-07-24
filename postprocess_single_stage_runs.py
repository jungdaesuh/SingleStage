#!/usr/bin/env python3
"""
Batch postprocessing for single-stage epsilon-constraint scan outputs.

Runs VMEC (from a template input), BOOZXFORM, and plotting steps for every
``mpol*_ntor*`` run under a scan root, optionally filtered by the **exact**
stage subdirectory name (``--stage``) and by equilibrium / iota path globs.

All VMEC, Boozer, and figure outputs are written **only** under each run
directory (the script temporarily changes the working directory to that folder
for each step, matching how :class:`simsopt.mhd.vmec.Vmec` names outputs).

For a template named ``input.fixed``, VMEC writes ``wout.fixed_000_000000.nc``
(first run, group 0), not ``wout_fixed_...``. The helper
:func:`vmec_output_basenames` matches that convention.

The default ``--scan-root`` is ``../single_stage_scans_no_sparsity_epsilon_constraint``
(next to ``examples/``), matching :file:`single_stage_epsilon_constraint.py`.

Typical usage (from ``examples/dipoles``)::

    python postprocess_single_stage_runs.py --stage stage03_cw3 --dry-run
    python postprocess_single_stage_runs.py --dry-run

Omit ``--stage`` to process **all** stage subdirectories (including sparsity
stages if present). To postprocess only a sparsity stage, pass its full name,
e.g. ``--stage stage03_cw3_sparsity``.

If your trees live elsewhere, set ``--scan-root`` explicitly. Each run directory
must contain both ``surf_opt.json`` and ``bs_opt.json``.

To redo only some steps, open :func:`process_one_run` and comment out the
calls you do not need.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import booz_xform as bx
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt


def _configure_matplotlib_paper_style() -> None:
    """
    Defaults for print-ready figures: serif math, readable type, crisp raster output.
    Callers should not use figure/subplot titles (captions belong in the manuscript).
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Nimbus Roman",
                "Times New Roman",
                "Computer Modern Roman",
                "serif",
            ],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.axisbelow": True,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
        }
    )


_configure_matplotlib_paper_style()

from simsopt._core.optimizable import load
from simsopt.field import InterpolatedField, compute_fieldlines
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.mhd.vmec import Vmec
from simsoptpp import (
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    MinZStoppingCriterion,
)


# -----------------------------------------------------------------------------
# Paths resolved relative to this script (examples/dipoles/)
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FIXED = _SCRIPT_DIR / "input.fixed"
DEFAULT_SCAN_ROOT = _SCRIPT_DIR.parent / "single_stage_scans_no_sparsity_epsilon_constraint"


def vmec_output_basenames(
    input_fixed_path: os.PathLike,
    mpi_group: int = 0,
    iteration: int = 0,
) -> tuple[str, str]:
    """
    Basenames of ``wout`` and ``boozmn`` files after a VMEC run, following
    ``simsopt.mhd.vmec.Vmec.run()`` (first run: ``iteration=0``).
    """
    stem = Path(input_fixed_path).name
    tagged = f"{stem}_{mpi_group:03d}_{iteration:06d}"
    wout = tagged.replace("input.", "wout_") + ".nc"
    boozmn = tagged.replace("input.", "boozmn_") + ".nc"
    return wout, boozmn


# =============================================================================
#  Substeps (comment these out inside process_one_run to skip)
# =============================================================================


@contextmanager
def working_directory(path: os.PathLike):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_vmec(input_fixed_path: Path) -> None:
    """
    Run VMEC using ``input_fixed_path`` as the template, ``surf_opt.json`` and
    ``bs_opt.json`` in the **current** directory, and write all VMEC files here.
    """
    input_fixed_path = Path(input_fixed_path).resolve()
    if not input_fixed_path.is_file():
        raise FileNotFoundError(f"VMEC input template not found: {input_fixed_path}")

    surf = load("surf_opt.json")
    bs = load("bs_opt.json")
    tf = ToroidalFlux(surf, bs)

    vmec = Vmec(str(input_fixed_path))
    vmec.boundary = surf
    vmec.indata.phiedge = tf.J()
    vmec.indata.mpol = surf.mpol
    vmec.indata.ntor = surf.ntor
    vmec.indata.nfp = surf.nfp
    vmec.run()


def run_boozxform(wout_name: str, boozmn_name: str, mboz: int, nboz: int) -> None:
    """Run BOOZXFORM on ``wout_name`` in the current directory."""
    if not os.path.isfile(wout_name):
        raise FileNotFoundError(
            f"Missing {wout_name}; run the VMEC step first (cwd={os.getcwd()})."
        )

    b = bx.Booz_xform()
    b.read_wout(wout_name)
    b.mboz = mboz
    b.nboz = nboz
    b.run()
    b.write_boozmn(boozmn_name)


def plot_boozxform(wout_name: str, boozmn_name: str) -> None:
    """Write fqs_plot.png, iota.png, modB_plot.png, magwell.png in cwd."""
    if not os.path.isfile(boozmn_name):
        raise FileNotFoundError(
            f"Missing {boozmn_name}; run the Boozer step first (cwd={os.getcwd()})."
        )

    b = bx.Booz_xform()
    b.read_boozmn(boozmn_name)

    bmnc = b.bmnc_b
    xn = b.xn_b
    f = np.sqrt(np.sum(bmnc[xn != 0, :] ** 2, axis=0) / np.sum(bmnc**2, axis=0))
    print("Mean QA metric: ", np.mean(f))
    print("Mean iota: ", np.mean(b.iota))
    s = b.s_b

    fig, ax = plt.subplots(figsize=(4.0, 2.75), constrained_layout=True)
    ax.plot(s, f, color="C0")
    ax.set_xlabel(r"$s$ [normalized toroidal flux]")
    ax.set_ylabel(r"$f$ [quasisymmetry metric]")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    fig.savefig("fqs_plot.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 2.75), constrained_layout=True)
    ax.plot(s, b.iota, color="C0")
    ax.set_xlabel(r"$s$ [normalized toroidal flux]")
    ax.set_ylabel(r"$\iota$")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    fig.savefig("iota.png", dpi=300)
    plt.close(fig)

    bx.surfplot(b, js=-1)
    fig = plt.gcf()
    fig.set_size_inches(4.25, 3.5)
    for _ax in fig.axes:
        _ax.set_title("")
    fig.savefig("modB_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    vmec = Vmec(wout_name)
    fig, ax = plt.subplots(figsize=(4.0, 2.75), constrained_layout=True)
    ax.plot(vmec.s_half_grid, vmec.wout.vp[1::], color="C0")
    ax.set_xlabel(r"$s$ [normalized toroidal flux]")
    ax.set_ylabel(r"$V^\prime$ [radial derivative of volume]")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    fig.savefig("magwell.png", dpi=300)
    plt.close(fig)

    print("Volume: ", vmec.wout.volume_p)
    print("Aspect ratio: ", vmec.wout.aspect)
    print("Minor radius: ", vmec.wout.Aminor_p)
    print("Major radius: ", vmec.wout.Rmajor_p)
    print("Averaged field strength: ", vmec.wout.volavgB)


def _plot_poincare_data_colored(
    fieldlines_phi_hits,
    phis,
    filename,
    surf=None,
    r_axis=None,
    dpi=300,
    s=1.0,
    xlims=None,
    ylims=None,
    marker="o",
    aspect="equal",
):
    nrowcol = ceil(sqrt(len(phis)))
    fig, axs = plt.subplots(
        nrowcol,
        nrowcol,
        figsize=(3.0 * nrowcol, 3.0 * nrowcol),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04)
    axs = np.atleast_1d(axs).ravel()

    line_dist_from_axis = np.zeros(len(fieldlines_phi_hits))
    for j, line_hits in enumerate(fieldlines_phi_hits):
        data_phi0 = line_hits[np.where(line_hits[:, 1] == 0)[0], :]
        if data_phi0.size > 0:
            r0 = np.sqrt(data_phi0[0, 2] ** 2 + data_phi0[0, 3] ** 2)
        else:
            r0 = np.sqrt(line_hits[0, 2] ** 2 + line_hits[0, 3] ** 2)
        line_dist_from_axis[j] = np.abs(r0 - r_axis) if r_axis is not None else r0
    line_order = np.argsort(line_dist_from_axis)
    ranks = np.empty_like(line_order)
    ranks[line_order] = np.arange(len(line_order))

    base_colors = plt.get_cmap("tab20").colors

    for ax in axs:
        ax.set_aspect(aspect)

    for i, phi in enumerate(phis):
        ax = axs[i]
        row = i // nrowcol
        col = i % nrowcol

        if row == nrowcol - 1:
            ax.set_xlabel("$r$ [m]")
        if col == 0:
            ax.set_ylabel("$z$ [m]")
        if col != 0:
            ax.tick_params(labelleft=False)
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)

        for j, line_hits in enumerate(fieldlines_phi_hits):
            data_this_phi = line_hits[np.where(line_hits[:, 1] == i)[0], :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
            color_idx = ranks[j] % len(base_colors)
            ax.scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=[base_colors[color_idx]])

        ax.minorticks_on()
        ax.grid(True, which="major", linewidth=0.45, alpha=0.35)

        if surf is not None:
            cross_section = surf.cross_section(phi=phi / (2 * np.pi))
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            ax.plot(r_interp, z_interp, linewidth=1, c="k")

    for k in range(len(phis), len(axs)):
        axs[k].set_visible(False)

    fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_poincare(
    wout_name: str,
    nfieldlines: int = 50,
    tmax_fl: float = 10000.0,
    tol: float = 1e-12,
    interpolate: bool = True,
    nr: int = 30,
    nphi: int = 15,
    nz: int = 15,
    degree: int = 3,
    poincare_label: str = "batch",
    poincare_filename: str | None = None,
) -> None:
    """
    Poincaré plot using ``bs_opt.json`` and ``wout_name`` in the current directory.
    """
    if not os.path.isfile(wout_name):
        raise FileNotFoundError(
            f"Missing {wout_name}; run the VMEC step first (cwd={os.getcwd()})."
        )
    if not os.path.isfile("bs_opt.json"):
        raise FileNotFoundError(f"Missing bs_opt.json in {os.getcwd()}.")

    bs = load("bs_opt.json")
    vmec = Vmec(wout_name)
    surf = vmec.boundary
    r_axis = vmec.wout.raxis_cc[0]
    nfp = surf.nfp

    surf_extended = Vmec(wout_name).boundary
    surf_extended.extend_via_normal(0.02)
    gamma = surf_extended.gamma()
    R = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
    Z = gamma[:, :, 2]
    Zmin = np.min(Z)
    Rmin = np.min(R)
    Rmax = np.max(R)
    Zmax = np.max(Z)

    out_name = poincare_filename or f"poincare_fieldline_{poincare_label}.png"

    def trace_fieldlines(bfield):
        R0 = np.linspace(Rmin, Rmax, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        phis = [(i / 4) * (2 * np.pi / nfp) for i in range(4)]
        _tys, phi_hits = compute_fieldlines(
            bfield,
            R0,
            Z0,
            tmax=tmax_fl,
            tol=tol,
            phis=phis,
            stopping_criteria=[
                MinRStoppingCriterion(Rmin),
                MaxRStoppingCriterion(Rmax),
                MinZStoppingCriterion(Zmin),
                MaxZStoppingCriterion(Zmax),
            ],
        )
        _plot_poincare_data_colored(
            phi_hits,
            phis,
            out_name,
            dpi=300,
            surf=surf,
            r_axis=r_axis,
            s=0.9,
            xlims=(Rmin, Rmax),
            ylims=(Zmin, Zmax),
        )
        return phi_hits

    rrange = (Rmin, Rmax, nr)
    phirange = (0, 2 * np.pi / nfp, nphi)
    zrange = (0, Zmax, nz)

    if interpolate:
        bsh = InterpolatedField(bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True)
        bsh.set_points(surf.gamma().reshape((-1, 3)))
        bs.set_points(surf.gamma().reshape((-1, 3)))
        Bh = bsh.B()
        B = bs.B()
        print("Maximum field interpolation error: ", np.max(np.abs(B - Bh)))
    else:
        bsh = bs

    trace_fieldlines(bsh)


def iter_run_directories(
    scan_root: Path,
    stage: str | None,
    eq_glob: str | None,
    iota_glob: str | None,
):
    """
    Yield leaf run directories (``.../stage*/mpol*_ntor*``) under ``scan_root``.

    If ``stage`` is set, only directories whose immediate parent folder basename
    equals that string are included; if ``None``, every stage subdirectory is
    considered. Directories must contain ``surf_opt.json`` and ``bs_opt.json``.
    """
    scan_root = Path(scan_root).resolve()
    if not scan_root.is_dir():
        raise FileNotFoundError(f"scan root is not a directory: {scan_root}")

    for res_dir in sorted(scan_root.rglob("mpol*_ntor*")):
        if not res_dir.is_dir():
            continue
        stage_dir = res_dir.parent
        if stage is not None and stage_dir.name != stage:
            continue
        iota_dir = stage_dir.parent
        if iota_glob is not None and not fnmatch.fnmatch(iota_dir.name, iota_glob):
            continue
        eq_dir = iota_dir.parent
        if eq_glob is not None and not fnmatch.fnmatch(eq_dir.name, eq_glob):
            continue
        if not (res_dir / "surf_opt.json").is_file() or not (res_dir / "bs_opt.json").is_file():
            continue
        yield res_dir


def process_one_run(
    run_dir: Path,
    input_fixed_path: Path,
    mboz: int,
    nboz: int,
    poincare_kw: dict,
) -> None:
    """
    Execute postprocessing steps in ``run_dir``. Comment out any ``step_*`` call
    you want to skip.
    """
    run_dir = Path(run_dir).resolve()
    input_fixed_path = Path(input_fixed_path).resolve()
    wout_name, boozmn_name = vmec_output_basenames(input_fixed_path)
    with working_directory(run_dir):
        run_vmec(input_fixed_path)
        run_boozxform(wout_name, boozmn_name, mboz, nboz)
        plot_boozxform(wout_name, boozmn_name)
        plot_poincare(wout_name, **poincare_kw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-root",
        type=Path,
        default=DEFAULT_SCAN_ROOT,
        help=f"Scan root (default: {DEFAULT_SCAN_ROOT})",
    )
    p.add_argument(
        "--input-fixed",
        type=Path,
        default=DEFAULT_INPUT_FIXED,
        help=f"VMEC input template (default: {DEFAULT_INPUT_FIXED})",
    )
    p.add_argument(
        "--stage",
        default=None,
        metavar="NAME",
        help='Exact basename of the stage subdirectory (e.g. stage03_cw3). Default: all stages.',
    )
    p.add_argument(
        "--eq-glob",
        default=None,
        help='Optional fnmatch for equilibrium folder (under scan root), e.g. "*ginsburg*".',
    )
    p.add_argument(
        "--iota-glob",
        default=None,
        help='Optional fnmatch for iota folder name (e.g. "iota_tar0.1*").',
    )
    p.add_argument("--mboz", type=int, default=48, help="BOOZXFORM mboz")
    p.add_argument("--nboz", type=int, default=48, help="BOOZXFORM nboz")
    p.add_argument("--dry-run", action="store_true", help="List run directories and exit")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing after a run fails (prints traceback).",
    )
    p.add_argument("--nfieldlines", type=int, default=50)
    p.add_argument("--poincare-no-interpolate", action="store_true", help="Use raw Biot-Savart field")
    args = p.parse_args(argv)

    runs = list(
        iter_run_directories(
            args.scan_root,
            args.stage,
            args.eq_glob,
            args.iota_glob,
        )
    )
    if not runs:
        print(f"No run directories found under {args.scan_root.resolve()!s}", file=sys.stderr)
        return 1

    print(f"Found {len(runs)} run director(y|ies).")
    for d in runs:
        print(f"  {d}")
    if args.dry_run:
        return 0

    poincare_kw = {
        "nfieldlines": args.nfieldlines,
        "interpolate": not args.poincare_no_interpolate,
    }

    input_fixed = Path(args.input_fixed).resolve()
    exit_code = 0
    for run_dir in runs:
        print(f"\n=== Postprocess: {run_dir} ===", flush=True)
        try:
            process_one_run(run_dir, input_fixed, args.mboz, args.nboz, poincare_kw)
        except Exception:
            exit_code = 1
            import traceback

            traceback.print_exc()
            if not args.continue_on_error:
                break
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
