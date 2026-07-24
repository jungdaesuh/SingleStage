from simsopt._core.optimizable import load
from simsopt.field import InterpolatedField
from simsopt.field import SurfaceClassifier, \
    compute_fieldlines, LevelsetStoppingCriterion, IterationStoppingCriterion
import numpy as np
from simsopt.mhd.vmec import Vmec
from simsopt.geo import SurfaceRZFourier
from simsoptpp import MinRStoppingCriterion,MaxRStoppingCriterion,MinZStoppingCriterion,MaxZStoppingCriterion
from simsopt._core import load
import time

start_time = time.time()

nfieldlines = 60 # Number of field lines for integration 
tmax_fl = 8000 # Maximum toroidal angle for integration
tol = 1e-12 # Tolerance for field line integration
max_fieldline_iters = int(2E6) # Safety cap on adaptive ODE steps per field line
interpolate = True # If True, then the BiotSavart magnetic field is interpolated 
                   # on a grid for the magnetic field evaluation
nr = 30 # Number of radial points for interpolation
nphi = 15 # Number of toroidal angle points for interpolation
nz = 15 # Number of vertical points for interpolation
degree = 3 # Degree for interpolation

#path = '../single_stage_true_epsilon_adaptive_res_lower_residual/wout_nfp22ginsburg_000_000281/iota0.125_fcp150kA_vt0.3/mpol12_ntor12/'
#TG: replaced path with own
path = '/Users/tianlanggong/simsopt/examples/outputs/stage2_initial_conditions/wout_nfp22ginsburg_000_001242/06_ntf4_diprad_0.05_VVa_0.26_VVb_0.27_VV_R0_1.03/'
print(f"Loading field and surface from {path}...", flush=True)
bs = load(path + 'bs_opt.json')
surf = load(path + 'surf_opt.json')
r_axis = surf.get("rc(0,0)")

nfp = surf.nfp
print(f"Loaded {len(bs.coils)} coils, nfp={nfp}.", flush=True)

# Use extended surface to determine initial conditions
print("Building extended surface for initial conditions...", flush=True)
surf_extended = load(path + 'surf_opt.json')
surf_extended.extend_via_normal(0.015) # go 1.5cm outside the surface for tracing
gamma = surf_extended.gamma()
R = np.sqrt(gamma[:,:,0]**2 + gamma[:,:,1]**2)
Z = gamma[:,:,2]
Zmin = np.min(Z)
Rmin = np.min(R)
Rmax = np.max(R)
Zmax = np.max(Z)
print(f"Tracing box: R=[{Rmin:.4f}, {Rmax:.4f}], Z=[{Zmin:.4f}, {Zmax:.4f}].", flush=True)

def plot_poincare_data_colored(fieldlines_phi_hits, phis, filename, surf=None, r_axis=None, dpi=300, s=1.0,
                               xlims=None, ylims=None, marker='o', aspect='equal'):
    import matplotlib.pyplot as plt
    from math import ceil, sqrt

    nrowcol = ceil(sqrt(len(phis)))
    fig, axs = plt.subplots(
        nrowcol, nrowcol,
        figsize=(3.0 * nrowcol, 3.0 * nrowcol),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04)
    axs = np.atleast_1d(axs).ravel()

    # Assign one stable color per field line, increasing monotonically with
    # distance from the magnetic axis.
    line_dist_from_axis = np.zeros(len(fieldlines_phi_hits))
    for j, line_hits in enumerate(fieldlines_phi_hits):
        data_phi0 = line_hits[np.where(line_hits[:, 1] == 0)[0], :]
        if data_phi0.size > 0:
            r0 = np.sqrt(data_phi0[0, 2]**2 + data_phi0[0, 3]**2)
        else:
            # Fallback for rare cases with no phi=0 hit.
            r0 = np.sqrt(line_hits[0, 2]**2 + line_hits[0, 3]**2)
        line_dist_from_axis[j] = np.abs(r0 - r_axis) if r_axis is not None else r0
    line_order = np.argsort(line_dist_from_axis)
    ranks = np.empty_like(line_order)
    ranks[line_order] = np.arange(len(line_order))

    # Use a discrete qualitative palette and cycle through it.
    base_colors = plt.get_cmap('tab20').colors

    for ax in axs:
        ax.set_aspect(aspect)

    for i, phi in enumerate(phis):
        ax = axs[i]
        row = i // nrowcol
        col = i % nrowcol

        if i != len(phis) - 1:
            ax.set_title(f"$\\phi = {phi/np.pi:.2f}\\pi$ ", loc='left', y=0.0)
        else:
            ax.set_title(f"$\\phi = {phi/np.pi:.2f}\\pi$ ", loc='right', y=0.0)
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
            r = np.sqrt(data_this_phi[:, 2]**2 + data_this_phi[:, 3]**2)
            color_idx = ranks[j] % len(base_colors)
            ax.scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=[base_colors[color_idx]])

        plt.rc('axes', axisbelow=True)
        ax.grid(True, linewidth=0.5)

        if surf is not None:
            cross_section = surf.cross_section(phi=phi / (2 * np.pi))
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            ax.plot(r_interp, z_interp, linewidth=1, c='k')

    # Hide unused axes when number of phis is not a perfect square.
    for k in range(len(phis), len(axs)):
        axs[k].set_visible(False)

    plt.savefig(filename, dpi=dpi)
    plt.close(fig)

def trace_fieldlines(bfield):
    # Set up initial conditions 
    R0 = np.linspace(Rmin, Rmax, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    stopping_criteria = [
        MinRStoppingCriterion(Rmin), MaxRStoppingCriterion(Rmax),
        MinZStoppingCriterion(Zmin), MaxZStoppingCriterion(Zmax),
        IterationStoppingCriterion(max_fieldline_iters),
    ]
    print(f"Tracing {nfieldlines} fieldlines (tmax={tmax_fl}, tol={tol}, max_iters={max_fieldline_iters})...", flush=True)
    trace_start = time.time()
    fieldlines_tys = []
    fieldlines_phi_hits = []
    for i in range(nfieldlines):
        line_start = time.time()
        print(f"  Field line {i+1}/{nfieldlines} (R0={R0[i]:.4f})...", flush=True)
        tys, hits = compute_fieldlines(
            bfield, [R0[i]], [Z0[i]], tmax=tmax_fl, tol=tol,
            phis=phis, stopping_criteria=stopping_criteria)
        fieldlines_tys.append(tys[0])
        fieldlines_phi_hits.append(hits[0])
        print(f"    done in {time.time()-line_start:.1f}s ({len(tys[0])} steps, {len(hits[0])} hits)", flush=True)
    print(f"Field line tracing finished in {time.time()-trace_start:.1f}s.", flush=True)
    print(f"Plotting poincare data to {path}poincare.png...", flush=True)
    plot_poincare_data_colored(fieldlines_phi_hits, phis, path + f'poincare.png', dpi=300, surf=surf, r_axis=r_axis, s=0.9, xlims=(Rmin, Rmax), ylims=(Zmin, Zmax))
    print(f"Saved {path}poincare.png.", flush=True)
    return fieldlines_phi_hits

rrange = (Rmin, Rmax, nr)
phirange = (0, 2*np.pi/nfp, nphi)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, Zmax, nz)

if interpolate:
    print(f"Building InterpolatedField (nr={nr}, nphi={nphi}, nz={nz}, degree={degree})...", flush=True)
    interp_start = time.time()
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )

    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    print("Evaluating interpolated field (first B() builds the grid)...", flush=True)
    Bh = bsh.B()
    print(f"Interpolated field ready in {time.time()-interp_start:.1f}s.", flush=True)
    print("Evaluating direct Biot-Savart field on surface...", flush=True)
    B = bs.B()
    print("Maximum field interpolation error: ", np.max(np.abs(B-Bh)), flush=True)
else:
    bsh = bs

hits = trace_fieldlines(bsh)

end_time = time.time()
elapsed = end_time - start_time
print(f"Total wall time: {elapsed:.2f} seconds.", flush=True)

