import numpy as np
from simsopt._core.derivative import derivative_dec
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface
from simsopt.geo.surfaceobjectives import Volume
from simsopt._core.optimizable import Optimizable

class BoozerResidualExact(Optimizable):
    r"""
    This term returns the Boozer residual penalty term

    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.

    where

    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]

    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        # same number of points as on the solved surface
        nphis = in_surface.quadpoints_phi.size
        phis = np.linspace(0,1./in_surface.nfp,nphis*4,endpoint=False)
        nthetas = in_surface.quadpoints_theta.size
        thetas = np.linspace(0,1,nthetas*4,endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        print("warning: constraint weight set to 0")
        self.constraint_weight = 0.0
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """

        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=True)
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)

        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil

        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """

        surface = self.surface
        res = self.boozer_surface.res
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=True)

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)

        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB

def initialize_boozer_surface(surf_prev, mpol, ntor, bs, vol_target, constraint_weight, iota, G0):
    """
    This initializes the boozer surface, using either the boozer "exact" algorithm, or the boozer "least squares" algorithm

    surf_prev: Any instance of simsopt.geo.Surface. This is the initial guess for the boozer surface solver
    mpol: SurfaceXYZTensorFourier resolution (both toroidal and poloidal)
    bs: simsopt.field.BiotSavart instance
    vol_target: target volume to be enclosed by the boozer surface
    constraint_weight: Set to 1.0 to use Boozer least square, None to use Boozer exact
    iota: initial guess for iota value on the surface
    G0: Value of net current going through the torus hole
    """
    # FIX 6: solve on a grid matched to (mpol, ntor), not on surf_prev's
    # plotting grid (128x64 = 8192 points -> 25x25 = 625 at mpol=ntor=6).
    # surf_prev must be resampled onto the new grid before the fit.
    nphi_s, ntheta_s = 4 * ntor + 1, 4 * mpol + 1
    qp_phi = np.linspace(0.0, 1.0 / surf_prev.nfp, nphi_s, endpoint=False)
    qp_theta = np.linspace(0.0, 1.0, ntheta_s, endpoint=False)
    sampled = type(surf_prev)(
          mpol=surf_prev.mpol, ntor=surf_prev.ntor, nfp=surf_prev.nfp,
          stellsym=surf_prev.stellsym,
          quadpoints_phi=qp_phi, quadpoints_theta=qp_theta,
          )
    sampled.x = surf_prev.x.copy()
    surf = SurfaceXYZTensorFourier(
          mpol=mpol,ntor=ntor,nfp=surf_prev.nfp,stellsym=True,
          quadpoints_theta=qp_theta,
          quadpoints_phi=qp_phi
          )
    surf.least_squares_fit(sampled.gamma())

    if constraint_weight:
        # Boozer least square approach
        print("Generating Boozer least squares surface...")
        vol = Volume(surf)
        boozer_surface = BoozerSurface(bs, surf, vol, vol_target, constraint_weight, options={'verbose':True})
    else:
        # Boozer exact approach
        print("Generating Boozer exact surface...")
        surf_exact = SurfaceXYZTensorFourier(
              mpol=mpol,ntor=ntor,nfp=surf.nfp,stellsym=True,
              quadpoints_theta=np.linspace(0,1,2*mpol+1,endpoint=False),
              quadpoints_phi=np.linspace(0,1./surf.nfp,2*mpol+1,endpoint=False),
              dofs=surf.dofs
              )

        vol = Volume(surf_exact)
        boozer_surface = BoozerSurface(bs, surf_exact, vol, vol_target, None, options={'verbose':True})

    # Run boozer surface algorithm
    res = boozer_surface.run_code(iota, G0)

    # Check if boozer algo is successful
    success1 = res['success'] # True if the boozer surface algo converged
    # is_self_intersecting() inspects ONE cylindrical cross-section, and simsopt
    # warns in its own docstring that a False result does not rule out an
    # intersection elsewhere. Screen four angles across a field period instead;
    # each call costs ~10 ms. A raise from cross_section() is itself a failure --
    # it means the surface doubles back so the angle is not monotonic.
    angles = [(2.0 * np.pi / surf_prev.nfp) * f for f in (0.0, 0.25, 0.5, 0.75)]
    if not success1:
        raise RuntimeError("Boozer solver did not converge")
    for angle in angles:
        try:
            self_intersecting = boozer_surface.surface.is_self_intersecting(angle=angle)
        except Exception as error:
            raise RuntimeError(
                f"Boozer surface self-intersection check failed at angle "
                f"{angle:.6g}: {error}"
            ) from error
        if self_intersecting:
            raise RuntimeError(
                f"Boozer surface is self-intersecting at angle {angle:.6g}"
            )

    return boozer_surface
