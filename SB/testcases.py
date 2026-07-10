import jax.numpy as jnp
import numpy as np
from .resolution import propagate_ou_density

def gaussian_density_2d(mesh, center, variances):
    """Generates a 2D Gaussian density on the given mesh."""
    barycenters = np.asarray(mesh.barycenter)
    center = np.asarray(center)
    variances = np.asarray(variances)
    diff = barycenters - center
    density = np.exp(-0.5 * np.sum((diff ** 2) / variances, axis=1))
    area = np.asarray(mesh.area)
    density = density / np.sum(density * area)
    return jnp.array(density)

def get_1d_case(test_case, x, t_array, dx):
    """
    Returns the source marginal (mu0), target marginal (mu1), 
    reference trajectory (rho_ref_dummy), and reference label.
    """
    eps = 1e-7
    
    if test_case == "1d_gauss_to_ou":
        # Source is a Gaussian
        m0, sig0 = -3.0, 0.5
        mu0 = jnp.exp(-0.5 * ((x - m0)/sig0)**2) / (sig0 * jnp.sqrt(2 * jnp.pi))
        
        # Target is exactly the OU process at t=1
        theta_ou, sig_ou, mean_ou = 1.0, 1.5, 3.0
        mu1 = propagate_ou_density(mu0, x, 1.0, theta_ou, sig_ou, mean_ou, dx)
        
        # Reference is the true OU trajectory
        rho_ref_dummy = np.array([propagate_ou_density(mu0, x, t, theta_ou, sig_ou, mean_ou, dx) if t > 0 else mu0 for t in t_array])
        ref_label = "OU Reference"
        
    elif test_case == "1d_square_to_ou":
        # Source is a Square
        c0, w0 = -2.0, 1.5  
        mu0_raw = jnp.where(jnp.abs(x - c0) <= w0, 1.0, eps)
        mu0 = mu0_raw / jnp.sum(mu0_raw * dx)
        
        # Target is exactly the OU process at t=1
        theta_ou, sig_ou, mean_ou = 1.0, 2.0, 2.0
        mu1 = propagate_ou_density(mu0, x, 1.0, theta_ou, sig_ou, mean_ou, dx)
        
        # Reference is the true OU trajectory
        rho_ref_dummy = np.array([propagate_ou_density(mu0, x, t, theta_ou, sig_ou, mean_ou, dx) if t > 0 else mu0 for t in t_array])
        ref_label = "OU Reference"

    elif test_case == "1d_gauss_to_gauss":
        m0, sig0 = 0, 0.1
        m1, sig1 = 0.0, 0.5
        mu0 = jnp.exp(-0.5 * ((x - m0)/sig0)**2) / (sig0 * jnp.sqrt(2 * jnp.pi))
        mu1 = jnp.exp(-0.5 * ((x - m1)/sig1)**2) / (sig1 * jnp.sqrt(2 * jnp.pi))
        
        # OU cannot hit this arbitrarily, use linear McCann interpolation for the baseline
        rho_ref_dummy = np.array([(1 - t) * mu0 + t * mu1 for t in t_array])
        ref_label = "Linear Baseline"
        
    elif test_case == "1d_square_to_gauss":
        # Source is a Square
        c0, w0 = 0.0, 2.0  
        mu0_raw = jnp.where(jnp.abs(x - c0) <= w0, 1.0, eps)
        mu0 = mu0_raw / jnp.sum(mu0_raw * dx)
        
        # Target is a Gaussian
        m1, sig1 = 0.0, 1.0
        mu1 = jnp.exp(-0.5 * ((x - m1)/sig1)**2) / (sig1 * jnp.sqrt(2 * jnp.pi))
        
        # Linear baseline since natural OU won't perfectly hit this target at t=1
        rho_ref_dummy = np.array([(1 - t) * mu0 + t * mu1 for t in t_array])
        ref_label = "Linear Baseline"
        
    elif test_case == "1d_bimodal":
        m0, sig0 = 0.0, 0.3
        mu0 = jnp.exp(-0.5 * ((x - m0)/sig0)**2) / (sig0 * jnp.sqrt(2 * jnp.pi))
        
        m1_a, sig1_a = -3.0, 0.4
        m1_b, sig1_b = 3.0, 0.4
        gauss_a = jnp.exp(-0.5 * ((x - m1_a)/sig1_a)**2) / (sig1_a * jnp.sqrt(2 * jnp.pi))
        gauss_b = jnp.exp(-0.5 * ((x - m1_b)/sig1_b)**2) / (sig1_b * jnp.sqrt(2 * jnp.pi))
        mu1 = 0.5 * gauss_a + 0.5 * gauss_b
        
        # OU cannot form a bimodal shape, use linear interpolation for the baseline
        rho_ref_dummy = np.array([(1 - t) * mu0 + t * mu1 for t in t_array])
        ref_label = "Linear Baseline"

    else:
        raise ValueError(f"Unknown 1D test case: {test_case}")
        
    return mu0, mu1, rho_ref_dummy, ref_label

def gaussian_ot_interpolant_2d(mesh, center0, var0, center1, var1, t_array):
    """
    Exact W₂ McCann interpolant between isotropic Gaussians N(c0, var0·I) → N(c1, var1·I).

    For isotropic Gaussians the OT map is T(x) = c1 + (σ1/σ0)(x - c0) and the
    geodesic is N(c_t, σ_t²·I) with  c_t = (1-t)c0 + t·c1  and  σ_t = (1-t)σ0 + t·σ1.

    Returns array of shape (len(t_array), N_cells).
    """
    c0, c1 = np.array(center0), np.array(center1)
    s0, s1 = np.sqrt(var0), np.sqrt(var1)
    rho_ot = []
    for t in t_array:
        st = (1.0 - t) * s0 + t * s1
        ct = (1.0 - t) * c0 + t * c1
        rho_ot.append(np.asarray(gaussian_density_2d(mesh, ct, [st**2, st**2])))
    return np.stack(rho_ot)


def get_2d_case(test_case, mesh):
    """
    Returns the source marginal (mu0) and target marginal (mu1) for 2D meshes.
    """
    if test_case == "2d_gauss_to_gauss":
        mu0 = gaussian_density_2d(mesh, [0.4, -0.4], [0.1, 0.1])
        mu1 = gaussian_density_2d(mesh, [-0.4, 0.4], [0.03, 0.03])
        
    elif test_case == "2d_gauss_to_bimodal":
        mu0 = gaussian_density_2d(mesh, [0.0, -0.6], [0.05, 0.05])
        
        target_a = gaussian_density_2d(mesh, [-0.6, 0.5], [0.03, 0.03])
        target_b = gaussian_density_2d(mesh, [0.6, 0.5], [0.03, 0.03])
        mu1 = 0.5 * target_a + 0.5 * target_b
    else:
        raise ValueError(f"Unknown 2D test case: {test_case}")
        
    return mu0, mu1