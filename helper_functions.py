# this script has the bulk of functions necessary for the optimize.py script
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from dataclasses import dataclass
from simsopt.geo import CurvePlanarFourier, create_equally_spaced_curves, CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import apply_symmetries_to_curves, apply_symmetries_to_currents, coils_via_symmetries, BiotSavart, Coil, Current, CircularRegularizedCoil
from simsopt.objectives import SquaredFlux
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.optimize import root_scalar
import scipy.integrate as spi
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ellipe
import os

@dataclass
class PlotConfig:
    """Configuration container for plot formatting parameters."""
    dpi: int = 100
    titlefontsize: int = 18
    axisfontsize: int = 16
    legendfontsize: int = 14
    ticklabelfontsize: int = 14
    cbarfontsize: int = 18

# These four functions are used to compute the rotation quaternion for the coil
def quaternion_from_axis_angle(axis, theta):
    """Compute a quaternion from a rotation axis and angle."""
    axis = axis / np.linalg.norm(axis)
    q0 = np.cos(theta / 2)
    q_vec = axis * np.sin(theta / 2)
    return np.array([q0, *q_vec])
def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
def rotate_vector(v, q):
    """Rotate a vector v using quaternion q."""
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])  # q* (conjugate)
    v_quat = np.array([0, *v])  # Convert v to quaternion form
    v_rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conjugate)
    return v_rotated[1:]  # Return the vector part
def compute_quaternion(normal, tangent):
    """
    Compute the quaternion to rotate the upward direction [0, 0, 1] to the given unit normal vector.
    Then, compute the change needed to align the x direction [1, 0, 0] to the desired tangent after the first rotation.
    
    Parameters:
        normal (array-like): A 3-element array representing the unit normal vector [n_x, n_y, n_z].
        tangent (array-like): A 3-element array representing the unit tangent vector [t_x, t_y, t_z].
    
    Returns:
        np.array: Quaternion as [q0, qi, qj, qk].
    """
    # Ensure input vectors are numpy arrays and normalized
    normal = np.asarray(normal)
    tangent = np.asarray(tangent)
    if not np.isclose(np.linalg.norm(normal), 1.0):
        raise ValueError("The input normal must be a unit vector.")
    if not np.isclose(np.linalg.norm(tangent), 1.0):
        raise ValueError("The input tangent must be a unit vector.")
    # Step 1: Rotate +z (upward) to normal
    upward = np.array([0.0, 0.0, 1.0])
    cos_theta1 = np.dot(upward, normal)
    axis1 = np.cross(upward, normal)
    if np.allclose(axis1, 0):
        q1 = np.array([1.0, 0.0, 0.0, 0.0]) if np.allclose(normal, upward) else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        axis1 /= np.linalg.norm(axis1)
        theta1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
        q1 = quaternion_from_axis_angle(axis1, theta1)
    # Step 2: Rotate initial x-direction using q1
    initial_x = np.array([1.0, 0.0, 0.0])
    rotated_x = rotate_vector(initial_x, q1)
    # Step 3: Rotate rotated_x to match desired tangent
    axis2 = np.cross(rotated_x, tangent)
    if np.linalg.norm(axis2) < 1e-8:
        q2 = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation needed
    else:
        axis2 /= np.linalg.norm(axis2)
        theta2 = np.arccos(np.clip(np.dot(rotated_x, tangent), -1.0, 1.0))
        q2 = quaternion_from_axis_angle(axis2, theta2)
    # Final quaternion (q2 * q1)
    q_final = quaternion_multiply(q2, q1)
    return q_final

# These four functions compute the Fourier series coefficients for the superellipse
def rho(theta, a, b, n):
    return ( 1/(abs(np.cos(theta)/a)**(n) + abs(np.sin(theta)/b)**(n) ) ** (1/(n)))
# Define Fourier coefficient integrals
def a_m(m, a, b, n):
    integrand = lambda theta: rho(theta, a, b, n) * np.cos(m * theta)
    if m==0:
        return (1 / (2 * np.pi)) * spi.quad(integrand, 0, 2 * np.pi)[0]
    else:
        return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]
def b_m(m, a, b, n):
    integrand = lambda theta: rho(theta, a, b, n) * np.sin(m * theta)
    return (1 / np.pi) * spi.quad(integrand, 0, 2 * np.pi)[0]
# Compute Fourier coefficients up to a given order
def compute_fourier_coeffs(max_order, a, b, n):
    coeffs = {'a_m': [], 'b_m': []}
    for m in range(max_order + 1):
        coeffs['a_m'].append(a_m(m, a, b, n))
        coeffs['b_m'].append(b_m(m, a, b, n))
    return coeffs

# use this to evenly space coils on elliptical vacuum vessel
# since evenly spaced in poloidal angle won't work
# must use quadrature for elliptic integral
def generate_even_arc_angles(a, b, ntheta):
    def arc_length_diff(theta):
        return np.sqrt((a * np.sin(theta))**2 + (b * np.cos(theta))**2)
    # Total arc length of the ellipse
    total_arc_length, _ = quad(arc_length_diff, 0, 2 * np.pi)
    arc_lengths = np.linspace(0, total_arc_length, ntheta, endpoint=False)
    def arc_length_to_theta(theta, s_target):
        s, _ = quad(arc_length_diff, 0, theta)
        return s - s_target
    # Solve for theta corresponding to each arc length
    thetas = np.zeros(ntheta)
    for i, s in enumerate(arc_lengths):
        if i != 0:
            result = root_scalar(arc_length_to_theta, args=(s,), bracket=[thetas[i-1], 2*np.pi])
            thetas[i] = result.root
    return thetas

def generate_tf_array(winding_surface, ntf, TF_R0, TF_a, TF_b, TF_current, fixed_geo_tfs=False, numquadpoints=32, tf_coil_radius=None):
    """
    Initialize an array of planar toroidal field coils over a half field period
    Parameters:
        winding_surface: surface upon which to obtain field periodicity/stellarator symmetry for the coils
        ntf: number of TF coils per half field period
        TF_R0: major radius of the TF coils
        TF_a: minor radius of the TF coils (in R direction)
        TF_b: minor radius of the TF coils (in Z direction)
        TF_current: current in each TF coil
        fixed_geo_tfs: whether to fix the geometric degrees of freedom of the TF coils
        numquadpoints: number of quadrature points representing each coil
        tf_coil_radius: if set, initialize coils as CircularRegularizedCoil with this radius
    Returns:
        base_tf_curves: list of initialized curves (half field period)
        base_tf_coils: list of initialized coils (half field period)
    """  
    if not fixed_geo_tfs:
        try:
            #TG: 
            from simsopt.geo.curveplanarellipticalcylindrical import (
                create_equally_spaced_cylindrical_curves
            )
            base_tf_curves = create_equally_spaced_cylindrical_curves(
                ntf,
                winding_surface.nfp,
                stellsym=winding_surface.stellsym,
                R0=TF_R0,
                a=TF_a,
                b=TF_b,
                numquadpoints=numquadpoints,
            )
        except ImportError:
            raise ImportError("Need to be on the windowpane branch with the correct TF curve class to unfix TF geometry")
    else:
        # order=1 is fine for elliptical
        base_tf_curves = create_equally_spaced_curves(
            ncurves=ntf,
            nfp=winding_surface.nfp,
            stellsym=winding_surface.stellsym,
            R0=TF_R0,
            R1=TF_a,
            order=1,
            numquadpoints=numquadpoints,
        )
        # TF_b set to be same ellipticity as VV, i.e. TF_b/TF_a = VV_b/VV_a
        for c in base_tf_curves:
            c.set("zs(1)", -TF_b) # see create_equally_spaced_curves doc for minus sign info
            c.fix_all()
    # Now make the curves into coils
    if tf_coil_radius == 0:
        base_tf_coils = [Coil(curve, Current(TF_current)) for curve in base_tf_curves]
    else:
        base_tf_coils = [CircularRegularizedCoil(curve, Current(TF_current), tf_coil_radius) for curve in base_tf_curves]
    return base_tf_curves, base_tf_coils

def generate_windowpane_array(winding_surface, inboard_radius, wp_fil_spacing, half_per_spacing, wp_n, numquadpoints=32, order=12, verbose=False, wp_coil_radius=0.0, nwps_poloidal_target=None, nwps_toroidal_target=None):
    """
    Initialize an array of nwps_poloidal x nwps_toroidal planar windowpane coils on a winding surface
    Coils are initialized with a current of 1 in order to simplify the logic in the precomputed section
    Parameters:
        winding_surface: surface upon which to place the coils, with coil plane locally tangent to the surface normal
                         assumed to be an elliptical cross section
        inboard_radius: radius of dipoles at inboard midplane - constant poloidally, will increase toroidally.
                        Pass None when using nwps_poloidal_target/nwps_toroidal_target instead.
        wp_fil_spacing: spacing wp filaments
        half_per_spacing: spacing between half period segments
        wp_n: value of n for superellipse, see https://en.wikipedia.org/wiki/Superellipse
        numquadpoints: number of points representing each coil (see CurvePlanarFourier documentation)
        order: number of Fourier moments for the planar coil representation, 0 = circle 
               (see CurvePlanarFourier documentation), more for ellipse approximation
        wp_coil_radius: if set, initialize coils as CircularRegularizedCoil with this radius
        nwps_poloidal_target: if set, use this fixed poloidal coil count instead of deriving from inboard_radius
        nwps_toroidal_target: if set, use this fixed toroidal coil count instead of deriving from inboard_radius.
                              May also be used together with inboard_radius to override only the toroidal count
                              while the poloidal count is still derived from the ellipse perimeter.
    Returns:
        base_wp_coils: list of initialized coils (half field period)
    """    
    # Identify locations of windowpanes
    VV_a = winding_surface.get_rc(1,0)
    VV_b = winding_surface.get_zs(1,0)
    VV_R0 = winding_surface.get_rc(0,0)
    arc_length = 4 * VV_a * ellipe(1-(VV_b/VV_a)**2)
    if nwps_poloidal_target is not None and nwps_toroidal_target is not None:
        nwps_poloidal = nwps_poloidal_target
        nwps_toroidal = nwps_toroidal_target
    elif inboard_radius is not None:
        nwps_poloidal = int(arc_length / (2 * inboard_radius + wp_fil_spacing))
        if nwps_toroidal_target is not None:
            nwps_toroidal = int(nwps_toroidal_target)
        else:
            nwps_toroidal = int((np.pi/winding_surface.nfp*(VV_R0 - VV_a) - half_per_spacing + wp_fil_spacing) / (2 * inboard_radius + wp_fil_spacing))
    else:
        raise ValueError("Must provide either inboard_radius or both nwps_poloidal_target and nwps_toroidal_target")
    Rpol = arc_length / 2 / nwps_poloidal - wp_fil_spacing / 2
    # TODO: make this even more exact. Right now, find evenly spaced theta locations in arc length, and make that center of coil
    # but this doesn't account for difference in arc length on either side of the center of the coil. Ideally, figure out how many
    # can fit, their radius, then figure out a way to find the theta location which keeps filament distance constant
    theta_locs = generate_even_arc_angles(VV_a, VV_b, nwps_poloidal)
    #theta_locs += (theta_locs[1] - theta_locs[0]) / 2 # shift by half the angle spacing to open spacing at theta = 0
    if verbose:
        print(f'     Number of Toroidal Dipoles: {nwps_toroidal}')
        print(f'     Number of Poroidal Dipoles: {nwps_poloidal}')
    # Interpolate unit normal and gamma vectors of winding surface at location of windowpane centers
    unitn_interpolators =        [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.unitnormal()[..., i], method='linear') for i in range(3)]
    gamma_interpolators =        [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.gamma()[..., i], method='linear') for i in range(3)]
    dgammadtheta_interpolators = [RegularGridInterpolator((winding_surface.quadpoints_phi, winding_surface.quadpoints_theta), winding_surface.dgamma_dtheta()[..., i], method='linear') for i in range(3)]
    # Initialize curves
    base_wp_curves = []
    for ii in range(0, nwps_poloidal): # Can alter this to remove theta = 0 coil (ii = 0)
        for jj in range(nwps_toroidal):
            theta_coil = theta_locs[ii]
            r = VV_a*VV_b / np.sqrt((VV_b*np.cos(theta_coil))**2 + (VV_a*np.sin(theta_coil))**2)
            Rtor = (np.pi/winding_surface.nfp*(VV_R0 + r * np.cos(theta_coil)) - half_per_spacing - (nwps_toroidal-1) * wp_fil_spacing) / (2 * nwps_toroidal)
            # Calculate toroidal angle of center of coil
            dphi = (half_per_spacing/2 + Rtor) / (VV_R0 + r * np.cos(theta_coil)) # need to add buffer in phi for gaps in panels
            phi_coil = dphi + jj * (2 * Rtor + wp_fil_spacing) / (VV_R0 + r * np.cos(theta_coil))
            # # After calculating equally spaced phi positions, try keeping width for outboard coils constant 
            # if theta_coil < np.pi/2 or theta_coil > 3*np.pi/2:
            #     Rtor = (np.pi/winding_surface.nfp*(VV_R0-VV_a) - half_per_spacing - (nwps_toroidal-1) * wp_fil_spacing) / (2 * nwps_toroidal)
            # Interpolate coil center and rotation vectors
            unitn_interp =        np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in unitn_interpolators], axis=-1)
            gamma_interp =        np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in gamma_interpolators], axis=-1)
            dgammadtheta_interp = np.stack([interp((phi_coil/(2*np.pi), theta_coil/(2*np.pi))) for interp in dgammadtheta_interpolators], axis=-1)
            curve = CurvePlanarFourier(numquadpoints, order)
            # Compute Fourier coefficients for given super-ellipse
            coeffs = compute_fourier_coeffs(order, Rpol, Rtor, wp_n)
            for m in range(order + 1):
                curve.set(f"rc({m})", coeffs['a_m'][m])
            for m in range(1, order + 1):
                curve.set(f"rs({m})", coeffs['b_m'][m])
            # Align the coil normal with the surface normal and Rpol axis with dgamma/dtheta
            # Renormalize the vector because interpolation can slightly modify its norm
            quaternion = compute_quaternion(unitn_interp / np.linalg.norm(unitn_interp), dgammadtheta_interp / np.linalg.norm(dgammadtheta_interp))
            curve.set("q0", quaternion[0])
            curve.set("qi", quaternion[1])
            curve.set("qj", quaternion[2])
            curve.set("qk", quaternion[3])
            # Align the coil center with the winding surface gamma
            curve.set("X", gamma_interp[0])
            curve.set("Y", gamma_interp[1])
            curve.set("Z", gamma_interp[2])
            base_wp_curves.append(curve)

    # Make sure the dofs are all fixed for the curves
    for curve in base_wp_curves:
        curve.fix_all()

    # Now make the curves into coils
    # We must initialize the curves with 1A current because of the way we scale the BdotN in the precomputed
    # section of optimize_windowpane_currents - I did not notice any dependence on the intial windowpane current
    # but if we need to add variability here later we'll need to update the optimization or just use precomputed = false
    if wp_coil_radius == 0.0:
        base_wp_coils = [Coil(curve, Current(1)) for curve in base_wp_curves]
    else:
        base_wp_coils = [CircularRegularizedCoil(curve, Current(1), wp_coil_radius) for curve in base_wp_curves]
    
    Rtor_min = (np.pi/winding_surface.nfp*(VV_R0-VV_a) - half_per_spacing - (nwps_toroidal-1) * wp_fil_spacing) / (2 * nwps_toroidal)
    Rtor_max = (np.pi/winding_surface.nfp*(VV_R0+VV_a) - half_per_spacing - (nwps_toroidal-1) * wp_fil_spacing ) / (2 * nwps_toroidal)
    return base_wp_coils, Rpol, Rtor_min, Rtor_max, nwps_poloidal, nwps_toroidal

def optimize_tfs(base_tf_coils, surf_plasma, winding_surface, CC_THRESHOLD, CC_WEIGHT, CS_THRESHOLD, CS_WEIGHT, num_fixed, definition='local', maxiter=1000, verbose=False):
    """
    Optimize the toroidal field (TF) coils for a given plasma surface and winding surface.
    Parameters:
    base_tf_coils (list): list of TF coil objects for a half field period.
    surf_plasma (object): Plasma surface to optimize field error for.
    winding_surface (object): Surface that windowpane coils are placed on, used in Curve-Surface distance penalty
    CC_THRESHOLD (float): Threshold for curve-curve distance.
    CC_WEIGHT (float): Weight for curve-curve distance in the objective function.
    CS_THRESHOLD (float): Threshold for curve-surface distance.
    CS_WEIGHT (float): Weight for curve-surface distance in the objective function.
    num_fixed (int): Number of TF coils with fixed currents.
    definition (str, optional): Definition for the squared flux calculation.
    maxiter (int, optional): Maximum number of iterations for the optimizer.
    verbose (bool, optional): If True, print detailed optimization information. Default is False.
    """
    
    # Make sure only the major radius and r rotation is unfixed, and then unfix TF currents if needed
    for c in base_tf_coils:
        c.curve.fix_all()
        c.current.unfix_all()
        c.curve.unfix('R0')
        c.curve.unfix('r_rotation')
    for i in range(num_fixed):
        base_tf_coils[i].current.fix_all()

    base_tf_curves = [c.curve for c in base_tf_coils]
    tf_coils = coils_via_symmetries([c.curve for c in base_tf_coils], [c.current for c in base_tf_coils], surf_plasma.nfp, surf_plasma.stellsym)

    Jccdist = CurveCurveDistance(base_tf_curves, CC_THRESHOLD)
    Jcsdist = CurveSurfaceDistance(base_tf_curves, winding_surface, CS_THRESHOLD)
    
    bs_tf = BiotSavart(tf_coils)
    Jf = SquaredFlux(surf_plasma, bs_tf, definition=definition)
    JF = Jf + CC_WEIGHT * Jccdist + CS_WEIGHT* Jcsdist
    dofs = JF.x

    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        outstr = f"J={J:.4e}, Jf={Jf.J():.4e}, CC={Jccdist.J()}, CS={Jcsdist.J()}, |dJ|={np.linalg.norm(grad):.4e}"
        if verbose: print(outstr)
        return J, grad
    
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxcor': 300}, tol=1e-12)

# these next few functions are used for the currents optimization
# this should be a surface integral, int(f dS) = int(f|n| dtheta dphi)
def surf_int(f, n_norm, theta, phi):
    return spi.simpson(spi.simpson(f * n_norm, theta), phi)
# not 100% sure if I need this, but does surface integral with 3D array f
def surf_int_3D(f, n_norm, theta, phi): 
    return spi.simpson(spi.simpson(f * n_norm[None,:,:], theta), phi)
# differentiates between dipole and TF currents
def classify_current(item, threshold):
    # Extract the number after 'Current' and before ':'
    current_num = int(item.split(':')[0].replace('Current', ''))
    return 0 if current_num < threshold else 1
# Derivative of current penalty objective = 2*(|J| - J_threshold) * sign(J) for |J| > J_threshold, else 0
def derivativeJcp(current, CURRENT_THRESHOLD):
    diff = np.abs(current) - CURRENT_THRESHOLD
    mask = diff > 0
    grad = np.where(current > 0, 2 * (current - CURRENT_THRESHOLD), 2 * (current + CURRENT_THRESHOLD))
    return grad * mask

def optimize_windowpane_currents(base_wp_coils, base_tf_coils, surf_plasma, definition='local', precomputed=True, maxiter=1000, current_threshold=1e12, current_weight=1, num_fixed=1, verbose=False):
    
    '''Optimize the currents in a set of windowpane coils given optimized tf_coils and a plasma surface. 
    Note: this script assumes the coils are all initialized at the same current, i.e. cannot be used in an iterative loop
    Parameters:
        base_wp_coils: list of windowpane coils over half field period
        base_tf_coils: list of optimized tf_coils over half field period
        surf_plasma: half field period of plasma surface to optimize normal field on
        definition: definition of quadratic flux (https://simsopt.readthedocs.io/en/latest/simsopt.objectives.html#simsopt.objectives.fluxobjective.SquaredFlux)
        precomputed: whether precomputed and scaled (faster, slightly less accurate but should work 99% of time) 
                     or exact (slower, more accurate) quadratic flux is used in optimizer
        current_threshold: threshold to begin penalizing currents in the coils
        current_weight: weight on the current threshold penalty
        num_fixed: number of TF coils per half field period to fix current for, the rest will be optimized with the wp coils
        verbose: print things during optimization
    Returns:
        bs: optimized BiotSavart object
    '''
    
    nprint = 5 # print J every nprint steps
    if definition!='local' and definition!='normalized' and definition!='quadratic flux':
        raise ValueError('definition must either be "local", "normalized", or "quadratic flux"')
    base_coils = base_tf_coils + base_wp_coils

    # fix the currents in some TF coils (so that we avoid the trivial solution), choose either 1 or ntf
    for i in range(num_fixed):
        base_tf_coils[i].current.fix_all()
    # make sure the geometric dofs are fixed for the coils, and wp currents are unfixed
    for c in base_wp_coils:
        c.curve.fix_all()
        c.current.unfix_all()
    
    if precomputed:
        # This is basically a smart way of doing the optimization that accounts for
        # the linear scaling of B with current, and thus BdotN with current.
        # Same result as exact, but much faster since we only have to compute BdotN once per coil 
        # CANNOT handle TF geometric degrees of freedom
        for c in base_tf_coils:
            c.curve.fix_all()

        # I have to do this because the dofs are out of order and sorted like current18, current19, 2, 20... 29, 3, 30 etc., 
        # and I want them like 1, 2, 3 etc., this should allow me to index them correctly (don't name dofs or else this will get messed up)
        # shoutout chatGPT for this section
        # first, hacky way of obtaining starting dof number - this is needed when calling optimize in a scan since the 
        # dof numbers will just keep increasing through each iteration. If only one run, this will be 1 + num_fixed
        Jf_temp = SquaredFlux(surf_plasma, BiotSavart(base_wp_coils[0:1]), definition=definition)
        dof_start_num = int(Jf_temp.dof_names[0].split(':')[0].replace('Current', ''))
        # Now, create array of dof numbers and sort their strings lexicographically for later use
        ndofs = len(base_coils) - num_fixed
        numbers = list(range(dof_start_num, ndofs + dof_start_num))
        string_numbers = [str(num) for num in numbers]
        sorted_string_numbers = sorted(string_numbers)
        sorted_indices = [int(num) for num in sorted_string_numbers]

        surf_plasma_nphi = len(surf_plasma.quadpoints_phi)
        surf_plasma_ntheta = len(surf_plasma.quadpoints_theta)

        # Initialize arrays for Squared Flux calc 
        BdotNcoil = np.zeros((ndofs, surf_plasma_nphi, surf_plasma_ntheta))
        BdotNcoil_fixed = np.zeros((num_fixed, surf_plasma_nphi, surf_plasma_ntheta))
        # these two are only used if local or normalized
        if definition!='quadratic flux':
            Bcoil = np.zeros((ndofs, surf_plasma_nphi, surf_plasma_ntheta, 3))
            Bcoil_fixed = np.zeros((num_fixed, surf_plasma_nphi, surf_plasma_ntheta, 3))
        # now we will apply symmetries to the coils one at a time and calculate their respective BdotN contribution
        coils = []
        # need to add the fixed coils to the BdotN calc even though it isn't a dof
        for ii, c  in enumerate(base_tf_coils[0:num_fixed]):
            paired_curves_fixed = apply_symmetries_to_curves(base_curves=[c.curve], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
            paired_currents_fixed = apply_symmetries_to_currents(base_currents=[c.current], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
            paired_coils_fixed = [Coil(curve, current) for curve, current in zip(paired_curves_fixed, paired_currents_fixed)]
            bs_fixed = BiotSavart(paired_coils_fixed)
            bs_fixed.set_points(surf_plasma.gamma().reshape((-1,3)))
            BdotNcoil_fixed[ii, :, :] = np.sum(bs_fixed.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3)) * surf_plasma.unitnormal(), axis = 2) # this is BdotN for the unfixed coil
            if (definition=='local' or definition=='normalized'):
                Bcoil_fixed[ii, :, :, :] = bs_fixed.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3))
            coils += paired_coils_fixed
        # precompute normal field from coils at initial current, which just gets scaled up/down linearly during optization
        for ii, c  in enumerate(base_coils[num_fixed:]):
            paired_curves = apply_symmetries_to_curves(base_curves=[c.curve], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
            paired_currents = apply_symmetries_to_currents(base_currents=[c.current], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym)
            paired_coils = [Coil(curve, current) for curve, current in zip(paired_curves, paired_currents)]
            bs_coil = BiotSavart(paired_coils)
            bs_coil.set_points(surf_plasma.gamma().reshape((-1,3)))
            i = sorted_indices.index(ii+dof_start_num)
            BdotNcoil[i, :, :] = np.sum(bs_coil.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3)) * surf_plasma.unitnormal(), axis = 2) # this is BdotN for each coil
            if (definition=='local' or definition=='normalized'):
                Bcoil[i, :, :, :] = bs_coil.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3))
            coils += paired_coils

        bs = BiotSavart(coils)
        Jf = SquaredFlux(surf_plasma, bs, definition=definition) # we do this through the above now, just using this to get dofs
        dofs = Jf.x
        wp_scale_factor = base_wp_coils[0].current.get_value()

        # use this to only set dJ for dipole current dofs
        dJscale = np.array([classify_current(dof_name, len(base_tf_coils)+1) for dof_name in Jf.dof_names])
        for name, s in zip(Jf.dof_names, dJscale):
            print(name, s)
        def fun(dofs, info={'Nfeval':0}):
            info['Nfeval'] += 1
            n_norm = np.linalg.norm(surf_plasma.normal(), axis=2)
            phi = surf_plasma.quadpoints_phi
            theta = surf_plasma.quadpoints_theta
            BdotN = np.sum(dofs[:,None,None]*BdotNcoil, axis=0) + np.sum(BdotNcoil_fixed, axis=0)
            if definition=='local' or definition=='normalized':
                B = np.sum(dofs[:,None,None,None]*Bcoil, axis=0) + np.sum(Bcoil_fixed, axis=0)
                modB = np.linalg.norm(B, axis=2)
                BcoildotB = np.sum(Bcoil * B, axis = 3)
            # calculate the squared flux
            if definition=='local':
                SF = 0.5 * surf_int((BdotN / modB)**2, n_norm, theta, phi)
                gradSF = surf_int_3D((modB[None, :, :]**2 * BdotNcoil * BdotN[None,:,:] - BcoildotB * BdotN[None,:,:]**2) / modB[None, :, :]**4, n_norm, theta, phi)
            elif definition=='normalized':
                SF = 0.5 * surf_int(BdotN**2, n_norm, theta, phi) /  surf_int(modB**2, n_norm, theta, phi)
                gradSF = (surf_int_3D(modB[None, :, :]**2, n_norm, theta, phi) * surf_int_3D(BdotNcoil * BdotN[None,:,:], n_norm, theta, phi) - surf_int_3D(BcoildotB, n_norm, theta, phi) * surf_int_3D(BdotN[None,:,:]**2, n_norm, theta, phi)) / surf_int_3D(modB[None, :, :]**2 , n_norm, theta, phi)**2
            else: # regular quadratic flux definition
                SF = 0.5 * surf_int(BdotN**2, n_norm, theta, phi) 
                gradSF = surf_int_3D(BdotNcoil * BdotN[None,:,:],n_norm, theta, phi)
            # this is a hardcoded current threshold penalty
            Jcp = np.sum(dJscale*np.array([np.maximum(np.abs(dofs[i]*wp_scale_factor) - current_threshold, 0)**2 for i in range(len(dofs))]))
            dJcp = dJscale * np.array([derivativeJcp(dofs[i]*wp_scale_factor, current_threshold) for i in range(len(dofs))])
            #reg_weight = 1e-9  # tune this — start small and increase until min/max ratio shrinks
            #Jreg = np.sum(dJscale * (dofs * wp_scale_factor)**2)
            #dJreg = dJscale * 2 * dofs * wp_scale_factor**2

            #J = current_weight * Jcp + reg_weight * Jreg + SF
            #grad = current_weight * dJcp + reg_weight * dJreg + gradSF
            #outstr = f"Iteration {info['Nfeval']}: J={J:.4e}, Jf={SF:.4e}, Jcp={current_weight*Jcp:.4e}, Jreg={reg_weight*Jreg:.4e}, |dJ|={np.linalg.norm(grad):.4e}"
            J = current_weight * Jcp + SF
            grad = current_weight * dJcp + gradSF
            outstr = f"Iteration {info['Nfeval']}: J={J:.4e}, Jf={SF:.4e}, Jcp ={current_weight*Jcp:.4e}, |dJ|={np.linalg.norm(grad):.4e}, |dJcp|={np.linalg.norm(current_weight * dJcp):.4e}"
            if verbose and np.mod(info['Nfeval'], nprint) == 0: print(outstr)
            return J, grad
        # optimize currents and update in the coils
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxcor': 300}, tol=1e-20)
        if verbose: print(res.message)
        Jf.x = res.x
    else:
        coils = coils_via_symmetries([c.curve for c in base_tf_coils + base_wp_coils], [c.current for c in base_tf_coils + base_wp_coils], surf_plasma.nfp, surf_plasma.stellsym)
        bs = BiotSavart(coils)
        Jf = SquaredFlux(surf_plasma, bs, definition=definition)
        dofs = Jf.x
        def fun(dofs, info={'Nfeval':0}):
            info['Nfeval'] += 1
            Jf.x = dofs
            J = Jf.J()
            grad = Jf.dJ()
            outstr = f"Iteration {info['Nfeval']}: J={J:.4e}, |dJ|={np.linalg.norm(grad):.4e}"
            if verbose and np.mod(info['Nfeval'], 1) == 0: print(outstr)
            return J, grad
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'maxcor': 300}, tol=1e-20)
        if verbose: print(res.message)
    return res, bs 



def coil_currents_on_theta_phi_grid(base_wp_coils, winding_surface):
    # this function returns an array of wp coil currents at each theta, phi on the winding surface
    R0 = winding_surface.get_rc(0,0)
    currents_phis_thetas = np.zeros((len(base_wp_coils), 3))
    for i, wp in enumerate(base_wp_coils):
        wp.curve.unfix_all()
        # Prefer dofs if present; otherwise fall back to geometric center
        if hasattr(wp.curve, "dof_names") and "X" in wp.curve.dof_names:
            x0 = wp.curve.get("X")
            y0 = wp.curve.get("Y")
            z0 = wp.curve.get("Z")
        else:
            center = np.mean(wp.curve.gamma(), axis=0)
            x0, y0, z0 = center
        currents_phis_thetas[i, 0] = wp.current.get_value()
        currents_phis_thetas[i, 1] = np.arctan2(y0, x0)  # phi
        currents_phis_thetas[i, 2] = np.arctan2(z0, (np.sqrt(x0**2 + y0**2) - R0))  # theta
        wp.curve.fix_all()
    return currents_phis_thetas  

def get_total_amp_meters(base_tf_coils, base_wp_coils, winding_surface):
    """
    Estiamte total amp-meters of the coils for HTS length estimation.
    For the dipole coils, finds the maximum current at constant theta and assumes
    that current applies to all coils at that theta.
    Parameters:
    base_tf_coils: list of TF coil objects for a half field period.
    base_wp_coils: list of windowpane coil objects for a half field period.
    winding_surface: surface that windowpane coils are placed on.
    Returns:
        total_amp_meters: estimated total amp-meters of the coils.
    """
    total = 0
    theta_tol = 0.01
    # TFs are easy, just add current * arc length 
    for tf in base_tf_coils: 
        total += tf.current.get_value() * np.sum(np.linalg.norm(np.diff(tf.curve.gamma(), axis=0), axis=1))
    # WPs more complicated - find maximum current at constant theta, this will
    # then be the Amp-meter for the entire row
    wp_currents_phis_thetas = coil_currents_on_theta_phi_grid(base_wp_coils, winding_surface)
    unique_thetas = []
    for i, (_, _, theta) in enumerate(wp_currents_phis_thetas):
        # Check if this theta is close to an existing one
        found = False
        for rep_theta in unique_thetas:
            if abs(rep_theta - theta) < theta_tol: # add tolerance for numerical differences off thetas
                found = True
                break
        if not found:
            unique_thetas.append(theta)
    ntor = len(base_wp_coils) / len(unique_thetas)
    # For each unique theta group, find the max current
    for rep_theta in unique_thetas:
        # Find all coils at given theta
        indices = np.where(np.abs(wp_currents_phis_thetas[:, 2] - rep_theta) < theta_tol)[0]
        # Find the index with max current among them
        max_index = indices[np.argmax(wp_currents_phis_thetas[indices, 0])]
        # add the number of dipoles (ntor) with this current * arc length
        total += ntor * base_wp_coils[max_index].current.get_value() * np.sum(np.linalg.norm(np.diff(base_wp_coils[max_index].curve.gamma(), axis=0), axis=1))
    return total * 2 * winding_surface.nfp

def plot_coil_currents_on_theta_phi_grid(coils, VV, output_dir, label, plot_config):
    """
    Calculates and plots the current of each coil on a 2D grid with theta on the y-axis and phi on the x-axis.
    Note that the size of the dipoles are arbitrarily set and not physical in the plot. This is
    merely a visualization of the coil currents on the winding surface.
    Parameters:
    coils (list): List of coil objects that lie on the winding surface (VV).
    VV (Surface): Vacuum vessel surface.
    output_dir (str): Directory to save the plot.
    plot_config (PlotConfig): Plot formatting configuration.
    """
    currents_phis_thetas = coil_currents_on_theta_phi_grid(coils, VV)
    currents = currents_phis_thetas[:, 0] / 1000
    phis = currents_phis_thetas[:, 1]
    thetas = np.mod(currents_phis_thetas[:, 2], 2 * np.pi)
    # Stabilize branch-cut behavior: treat values numerically near 2*pi as 0.
    theta_wrap_tol = 1e-3 * (2 * np.pi)
    thetas = np.where(np.isclose(thetas, 2 * np.pi, atol=theta_wrap_tol, rtol=0.0), 0.0, thetas)
    vmax = np.max(np.abs(currents))  # Symmetric range
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.seismic  # Diverging colormap (red-negative, blue-positive)
    colors = cmap(norm(currents))
    fig, ax = plt.subplots(figsize=(8,6))
    theta_norm = thetas / (2 * np.pi)
    theta_norm = np.clip(theta_norm, 0.0, 1.0)
    theta_norm = np.where(np.isclose(theta_norm, 1.0, atol=1e-12, rtol=0.0), 0.0, theta_norm)
    scatter = ax.scatter(phis/(2*np.pi), theta_norm, edgecolors=colors, facecolors='none', s=200, linewidths=1.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel("WP Currents [kA]", fontsize=plot_config.cbarfontsize, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=plot_config.ticklabelfontsize)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=plot_config.axisfontsize, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=plot_config.axisfontsize, fontweight='bold')
    ax.set_ylim(-0.05, 0.95)
    ax.set_xlim(-0.005, 0.255) # hardcode nfp
    ax.set_title(f"Coil Currents on Winding Surface", fontsize=plot_config.titlefontsize, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'coil_currents_{label}.png'), dpi=plot_config.dpi)
    plt.close()
    return

def plot_relBfinal_norm_modB(bs, surf_plas, output_dir, label, plot_config):
    """
    Creates Bnormal and modB plots for Biot-Savart object on plasma surface.
    
    Parameters:
        bs: BiotSavart object
        surf_plas: Plasma surface
        output_dir (str): Directory to save plots
        label (str): Label for the plot title and filename
        plot_config (PlotConfig): Plot formatting configuration
    
    Returns:
        tuple: (relBfinal_norm, mean_abs_relBfinal_norm, max_relBfinal_norm)
    """
    theta = surf_plas.quadpoints_theta
    phi = surf_plas.quadpoints_phi
    n = surf_plas.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1./absn)[:,:,None]
    sqrt_area = np.sqrt(absn.reshape((-1,1))/float(absn.size))
    surf_area = sqrt_area**2
    bs.set_points(surf_plas.gamma().reshape((-1, 3)))
    Bfinal = bs.B().reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    mean_abs_relBfinal_norm = np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area)
    max_rBnorm = np.max(np.abs(relBfinal_norm))
    fig, ax = plt.subplots()
    contour = ax.contourf(phi, theta, np.squeeze(relBfinal_norm).T, levels=50, cmap='coolwarm', vmin=-max_rBnorm, vmax=max_rBnorm)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=plot_config.axisfontsize, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=plot_config.axisfontsize, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$\mathbf{B}\cdot\mathbf{n}/|\mathbf{B}|$', fontsize=plot_config.cbarfontsize, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=plot_config.ticklabelfontsize)
    ax.set_title(f'Surface-averaged \n |Bn|/|B| = {mean_abs_relBfinal_norm:.4e}', fontsize=plot_config.titlefontsize, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'relBn_{label}.png'), dpi=plot_config.dpi)
    plt.close()
    abs_modBfinal_dA = np.abs(modBfinal.reshape((-1, 1))) * surf_area
    mean_abs_modBfinal = np.sum(abs_modBfinal_dA) / np.sum(surf_area)
    fig, ax = plt.subplots()
    contour = ax.contour(phi, theta, np.squeeze(modBfinal).T, levels=25, cmap='viridis')
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=plot_config.axisfontsize, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=plot_config.axisfontsize, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$|\mathbf{B}|$', fontsize=plot_config.cbarfontsize, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=plot_config.ticklabelfontsize)
    ax.set_title(f'Surface-averaged |B| = {mean_abs_modBfinal:.3f}', fontsize=plot_config.titlefontsize, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'modB_{label}.png'), dpi=plot_config.dpi)
    plt.close()
    return relBfinal_norm, mean_abs_relBfinal_norm, np.max(relBfinal_norm)

def find_toroidal_plane_intersections(curve, phi_cut):
    """
    Find where a closed curve intersects the toroidal half-plane at angle phi_cut
    (the half-plane containing the +Z axis and the ray at angle phi_cut in the
    x-y plane).  Returns a list of (R, Z) tuples.

    Uses the signed perpendicular distance from the half-plane,
        d = y*cos(phi) - x*sin(phi),
    and linearly interpolates between adjacent quadrature points where d
    changes sign.  Also handles the case where a quadrature point sits
    exactly on the plane (common due to coil symmetry at the midplane).
    """
    gamma = curve.gamma()
    x, y, z = gamma[:, 0], gamma[:, 1], gamma[:, 2]

    cos_phi = np.cos(phi_cut)
    sin_phi = np.sin(phi_cut)
    d = y * cos_phi - x * sin_phi

    crossings = []
    n = len(d)
    eps = 1e-14
    for i in range(n):
        j = (i + 1) % n
        if abs(d[i]) < eps:
            # Quadrature point sits exactly on the toroidal plane
            R = np.sqrt(x[i]**2 + y[i]**2)
            if x[i] * cos_phi + y[i] * sin_phi > 0:
                crossings.append((R, z[i]))
        elif d[i] * d[j] < 0:
            t = d[i] / (d[i] - d[j])
            xi = x[i] + t * (x[j] - x[i])
            yi = y[i] + t * (y[j] - y[i])
            zi = z[i] + t * (z[j] - z[i])
            R = np.sqrt(xi**2 + yi**2)
            # Only keep crossings on the correct side of the Z-axis
            if xi * cos_phi + yi * sin_phi > 0:
                crossings.append((R, zi))

    return crossings


def plot_cross_section(surf, VV, output_dir, label, plot_config, base_dipole_coils=None):
    """
    Plots cross section of plasma and vacuum vessel at a few toroidal locations.
    
    Parameters:
        surf: Plasma surface
        VV: Vacuum vessel surface
        output_dir (str): Directory to save plot
        label (str): Label for the plot title and filename
        plot_config (PlotConfig): Plot formatting configuration
        base_dipole_coils (list, optional): Base dipole coils in one half period.
            If provided, the R-Z intersection of each coil with its own center's
            toroidal plane is overlaid as a line segment joining the two crossings.
    """
    plt.figure(figsize=(7,6))
    phi_array = np.linspace(0, 0.5 / surf.nfp, 6, endpoint=True) # scaled from 0 to 1
    for phi_slice in phi_array:
        cs = surf.cross_section(phi_slice * 2 * np.pi)
        rs = np.sqrt(cs[:,0]**2 + cs[:,1]**2); rs = np.append(rs, rs[0])
        zs = cs[:,2]; zs = np.append(zs, zs[0])
        cs2 = VV.cross_section(phi_slice * 2 * np.pi)
        rs2 = np.sqrt(cs2[:,0]**2 + cs2[:,1]**2); rs2 = np.append(rs2, rs2[0])
        zs2 = cs2[:,2]; zs2 = np.append(zs2, zs2[0])    
        plt.plot(rs, zs)
        plt.plot(rs2, zs2, 'k')
        plt.plot(np.mean(rs), np.mean(zs), 'kx')
    if base_dipole_coils:
        for coil in base_dipole_coils:
            curve = coil.curve
            phi_cut = np.arctan2(curve.get("Y"), curve.get("X"))
            crossings = find_toroidal_plane_intersections(curve, phi_cut)
            if len(crossings) >= 2:
                crossings.sort(key=lambda p: p[1])
                R0, Z0 = crossings[0]
                R1, Z1 = crossings[-1]
                plt.plot([R0, R1], [Z0, Z1], c='tab:orange', linewidth=1.5,
                         alpha=0.9, zorder=5)
            elif crossings:
                plt.plot(crossings[0][0], crossings[0][1], 'o', c='tab:orange',
                         markersize=4, zorder=5)
    plt.xlabel('R [m]', fontsize=plot_config.axisfontsize, fontweight='bold')
    plt.ylabel('Z [m]', fontsize=plot_config.axisfontsize, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=plot_config.ticklabelfontsize)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'x_section_{label}.png'), dpi=plot_config.dpi, bbox_inches='tight')
    plt.close()
    return
