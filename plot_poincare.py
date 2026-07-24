from simsopt._core.optimizable import load
from simsopt.field import InterpolatedField
from simsopt.field import SurfaceClassifier, \
    compute_fieldlines, LevelsetStoppingCriterion
import numpy as np
from simsopt.mhd.vmec import Vmec
from simsopt.geo import SurfaceRZFourier
from simsoptpp import MinRStoppingCriterion,MaxRStoppingCriterion,MinZStoppingCriterion,MaxZStoppingCriterion
from simsopt._core import load

nfieldlines = 50 # Number of field lines for integration 
tmax_fl = 10000 # Maximum toroidal angle for integration
tol = 1e-12 # Tolerance for field line integration
interpolate = True # If True, then the BiotSavart magnetic field is interpolated 
                   # on a grid for the magnetic field evaluation
nr = 30 # Number of radial points for interpolation
nphi = 15 # Number of toroidal angle points for interpolation
nz = 15 # Number of vertical points for interpolation
degree = 3 # Degree for interpolation

path = '../single_stage_scans_no_sparsity_epsilon_constraint/wout_nfp22ginsburg_000_000281_init_dir90/iota_tar0.15/stage02_cw3/mpol6_ntor6/'
bs = load(path + 'bs_opt.json')
vmec = Vmec('wout_fixed_000_000000.nc')
surf = vmec.boundary
r_axis = vmec.wout.raxis_cc[0]

nfp = surf.nfp

# Use extended surface to determine initial conditions
surf_extended = Vmec('wout_fixed_000_000000.nc').boundary
surf_extended.extend_via_normal(0.02)
gamma = surf_extended.gamma()
R = np.sqrt(gamma[:,:,0]**2 + gamma[:,:,1]**2)
Z = gamma[:,:,2]
Zmin = np.min(Z)
Rmin = np.min(R)
Rmax = np.max(R)
Zmax = np.max(Z)

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

def trace_fieldlines(bfield,label):
    # Set up initial conditions 
    R0 = np.linspace(Rmin, Rmax, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol,
        phis=phis, stopping_criteria=[MinRStoppingCriterion(Rmin),MaxRStoppingCriterion(Rmax),MinZStoppingCriterion(Zmin),MaxZStoppingCriterion(Zmax)])
    plot_poincare_data_colored(fieldlines_phi_hits, phis, f'poincare_fieldline_{label}.png', dpi=300, surf=surf, r_axis=r_axis, s=0.9, xlims=(Rmin, Rmax), ylims=(Zmin, Zmax))
    return fieldlines_phi_hits

rrange = (Rmin, Rmax, nr)
phirange = (0, 2*np.pi/nfp, nphi)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, Zmax, nz)

if interpolate:
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )

    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    print("Maximum field interpolation error: ", np.max(np.abs(B-Bh)))
else:
    bsh = bs

hits = trace_fieldlines(bsh, 'test')