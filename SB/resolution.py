import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax.experimental.sparse as sparse
from jax.scipy.sparse.linalg import cg

# ---------------------------------------------------------------------------
#  Ornstein-Uhlenbeck reference process
# ---------------------------------------------------------------------------

def ou_transition_density(x_target, x_source, t, theta, sigma, mean):
    """
    Gaussian transition density of an OU process:
        dX_t = -θ (X_t - m) dt + σ dW_t

    Returns p(x_target | x_source, t).
    """
    theta    = jnp.maximum(theta, 1e-12)
    variance = sigma**2 * (1.0 - jnp.exp(-2.0 * theta * t)) / (2.0 * theta)
    variance = jnp.maximum(variance, 1e-14)
    mean_t   = mean + (x_source - mean) * jnp.exp(-theta * t)
    log_k    = (-0.5 * (x_target - mean_t)**2 / variance
                - 0.5 * jnp.log(2.0 * jnp.pi * variance))
    return jnp.exp(log_k)


def propagate_ou_density(initial_density, x_grid, t, theta, sigma, mean, dx):
    """
    Propagate an initial probability density ρ₀ through the OU semigroup:
        ρ_t(x) = ∫ p(x | y, t) ρ₀(y) dy   (discretised with the trapezoid rule)

    The result is renormalised to preserve unit mass.
    """
    t       = jnp.maximum(t, 1e-7)          # safety floor
    x_col   = x_grid[:, None]               # (N, 1)
    y_row   = x_grid[None, :]               # (1, N)
    kernel  = ou_transition_density(x_col, y_row, t, theta, sigma, mean)
    propagated = kernel @ (initial_density * dx)
    return propagated / jnp.sum(propagated * dx)


def compare_with_ou(rho_solution, initial_density, x_grid, t, theta, sigma, mean, dx):
    """Return the OU reference density at time t and the L² error against it."""
    rho_ou  = propagate_ou_density(initial_density, x_grid, t, theta, sigma, mean, dx)
    l2_err  = jnp.sqrt(jnp.sum((rho_solution - rho_ou)**2 * dx))
    return rho_ou, l2_err


# ---------------------------------------------------------------------------
#  1D heat semigroup  (Neumann / zero-flux BCs via method of images)
# ---------------------------------------------------------------------------

def _neumann_log_kernel(x, t, gamma, x_min, interval_length, num_images=5):
    """
    Log of the Neumann-BC heat kernel on [x_min, x_min + interval_length],
    constructed via the method of images.

    Shape: (N, N)  — log K(x_i | x_j, t)
    """
    x_col  = x[:, None, None]                                     # (N, 1, 1)
    y_row  = x[None, :, None]                                     # (1, N, 1)
    offsets = (2.0 * interval_length) * jnp.arange(
        -num_images, num_images + 1, dtype=x.dtype)               # (2M+1,)

    direct_dist    =  x_col - (y_row + offsets[None, None, :])
    reflected_dist =  x_col + y_row - 2.0 * x_min + offsets[None, None, :]

    log_direct    = -direct_dist**2    / (4.0 * gamma * t) - 0.5 * jnp.log(4.0 * jnp.pi * gamma * t)
    log_reflected = -reflected_dist**2 / (4.0 * gamma * t) - 0.5 * jnp.log(4.0 * jnp.pi * gamma * t)

    log_image_kernel = jnp.logaddexp(log_direct, log_reflected)
    return logsumexp(log_image_kernel, axis=-1)                    # (N, N)


def _enforce_zero_flux(values):
    """Enforce Neumann (zero-flux) boundary conditions by repeating edge values."""
    values = values.at[0].set(values[1])
    values = values.at[-1].set(values[-2])
    return values


def apply_logPt_exp(h, x, t, gamma, dx):
    """
    Apply the 1D heat semigroup P_t to a log-potential h in log-space,
    Uses Neumann (zero-flux) boundary conditions via the method of images.
    The kernel rows are normalised so that ∫ K(x|y,t) dy = 1 exactly on
    the discrete grid, preventing mass leakage.
    """
    t              = jnp.maximum(t, 1e-7)
    x_min          = x[0]
    interval_length = x[-1] - x_min

    log_kernel = _neumann_log_kernel(x, t, gamma, x_min, interval_length)

    # Trapezoid weights to make ∫ K(·|y,t) dy ≈ 1 exactly on the grid
    weights     = jnp.ones_like(x).at[0].set(0.5).at[-1].set(0.5)
    log_weights = jnp.log(weights)

    # Row-normalise the kernel (removes any residual mass discretisation error)
    row_norm   = logsumexp(log_kernel + log_weights[None, :] + jnp.log(dx), axis=1)
    log_kernel = log_kernel - row_norm[:, None]

    return logsumexp(log_kernel + h[None, :] + log_weights[None, :] + jnp.log(dx), axis=1)


# ---------------------------------------------------------------------------
#  IPFP  (Iterative Proportional Fitting Procedure / Sinkhorn)
# ---------------------------------------------------------------------------

def apply_IPFP(log_mu0, log_mu1, x, gamma, dx, num_iter=50):
    """
    Solve the Schrödinger Bridge via IPFP (continuous Sinkhorn).

    Parameters
    ----------
    log_mu0, log_mu1 : arrays of shape (N,)
        Log of the source and target marginal densities.
    x                : array of shape (N,)
        Spatial grid.
    gamma            : float
        Diffusion coefficient of the reference Brownian motion.
    dx               : float
        Grid spacing.
    num_iter         : int
        Number of IPFP iterations (fixed; required for JAX JIT).

    Returns
    -------
    f, g : arrays of shape (N,)
        Converged Schrödinger half-bridge potentials.
    """
    def body_fn(_, val):
        f_k, g_k = val
        g_next = log_mu1 - apply_logPt_exp(f_k, x, 1.0, gamma, dx)
        g_next = _enforce_zero_flux(g_next)
        f_next = log_mu0 - apply_logPt_exp(g_next, x, 1.0, gamma, dx)
        f_next = _enforce_zero_flux(f_next)
        return f_next, g_next

    f_init = jnp.zeros_like(x)
    g_init = jnp.zeros_like(x)
    f_final, g_final = jax.lax.fori_loop(0, num_iter, body_fn, (f_init, g_init))
    f_final = _enforce_zero_flux(f_final)
    g_final = _enforce_zero_flux(g_final)
    return f_final, g_final


def apply_IPFP_debug(log_mu0, log_mu1, x, gamma, dx, num_iter=50, tol=1e-6):
    """
    Python-loop version of IPFP for debugging and convergence monitoring.
    Not JIT-compatible, but prints the residual at each iteration.

    Returns
    -------
    f, g        : converged potentials
    residuals   : list of float — ‖f_{k+1} − f_k‖ at each iteration
    """
    f = jnp.zeros_like(x)
    g = jnp.zeros_like(x)
    residuals = []
    for i in range(num_iter):
        g_new = log_mu1 - apply_logPt_exp(f, x, 1.0, gamma, dx)
        g_new = _enforce_zero_flux(g_new)
        f_new = log_mu0 - apply_logPt_exp(g_new, x, 1.0, gamma, dx)
        f_new = _enforce_zero_flux(f_new)
        res = float(jnp.max(jnp.abs(f_new - f)))
        residuals.append(res)
        f, g = f_new, g_new
        if res < tol:
            print(f"  IPFP converged at iteration {i+1}  (residual = {res:.2e})")
            break
    return f, g, residuals


# ---------------------------------------------------------------------------
#  Bridge density and drift field  (1D)
# ---------------------------------------------------------------------------

def retrieve_rho(f, g, x, t, gamma, dx):
    """
    Compute the Schrödinger Bridge density at time t ∈ [0, 1]:

        ρ_t(x) = (P_t e^f)(x) · (P_{1−t} e^g)(x)

    The result is renormalised to ensure ∫ ρ_t dx = 1 (numerical safeguard).

    Parameters
    ----------
    f, g : arrays of shape (N,)  — converged IPFP potentials
    x    : spatial grid  (N,)
    t    : float in [0, 1]
    gamma: diffusion coefficient
    dx   : grid spacing
    """
    f_t = apply_logPt_exp(f, x,       t,       gamma, dx)
    g_t = apply_logPt_exp(g, x, 1.0 - t,       gamma, dx)
    rho = jnp.exp(f_t + g_t)
    # Renormalise for numerical safety (theoretically exact at convergence)
    mass = jnp.sum(rho * dx)
    return rho / jnp.maximum(mass, 1e-14)


def retrieve_b(g, x, t, gamma, dx):
    """
    Compute the optimal drift field of the Schrödinger Bridge at time t:

        b_t(x) = 2γ ∇ g_t(x)    where  g_t = P_{1−t} g

    Zero-flux boundary conditions are enforced at the domain edges.
    """
    g_t       = apply_logPt_exp(g, x, 1.0 - t, gamma, dx)
    nabla_g_t = jnp.gradient(g_t, dx)
    nabla_g_t = nabla_g_t.at[0].set(0.0).at[-1].set(0.0)
    return 2.0 * gamma * nabla_g_t


def compute_drift_field(g, x, t_array, gamma, dx):
    """
    Compute the drift mapping b_t(x) for all times in t_array.

    Returns an array of shape (len(t_array), N).
    """
    b_matrix = []
    for t in t_array:
        t_safe = jnp.maximum(t, 1e-4)       # avoid singular gradient at t = 0
        b_matrix.append(retrieve_b(g, x, t_safe, gamma, dx))
    return jnp.array(b_matrix)


# ---------------------------------------------------------------------------
#  2D heat semigroup  (cell-centred FVM, implicit Euler)
# ---------------------------------------------------------------------------

def build_cell_centered_laplacian(mesh):
    """
    Build the sparse flux-weight matrix W and the inverse-area vector A_inv
    for a cell-centred Finite Volume discretisation of the Laplacian:

        (∇² u)_i ≈ (A_inv)_i · (W u)_i

    Neumann (zero-flux) boundary conditions are enforced by zeroing fluxes
    across faces that have no valid neighbour (neighbour index = −1).

    Parameters
    ----------
    mesh : Mesh
        Triangular mesh object (see mesh.py).

    Returns
    -------
    W_sparse : jax sparse BCOO matrix  (N_tris × N_tris)
    A_inv    : array of shape (N_tris,)  — reciprocal cell areas
    """
    N_tris  = mesh.tris.shape[0]
    A_inv   = 1.0 / mesh.area

    neighbors   = mesh.neighbors        # (N_tris, 3)   — −1 for boundary faces
    valid_mask  = neighbors >= 0
    safe_neighbors = jnp.where(valid_mask, neighbors, 0)

    # Distance between cell barycenters
    centers_i = mesh.barycenter[:, None, :]       # (N_tris, 1, 2)
    centers_j = mesh.barycenter[safe_neighbors]   # (N_tris, 3, 2)
    d_ij      = jnp.linalg.norm(centers_i - centers_j, axis=-1)  # (N_tris, 3)

    # Shared face lengths
    l_ij = mesh.surface[mesh.face_connectivity]   # (N_tris, 3)

    # Off-diagonal fluxes  (zero at boundary faces)
    off_diag_fluxes = jnp.where(valid_mask, l_ij / d_ij, 0.0)

    # --- Build BCOO sparse matrix ---
    row_indices = jnp.repeat(jnp.arange(N_tris), 3)
    col_indices = neighbors.flatten()
    data        = off_diag_fluxes.flatten()

    valid_sparse_mask = col_indices >= 0
    row_indices = row_indices[valid_sparse_mask]
    col_indices = col_indices[valid_sparse_mask]
    data        = data[valid_sparse_mask]

    # Diagonal: −∑ off-diagonal fluxes  (conservation of mass)
    diag_data    = -jnp.sum(off_diag_fluxes, axis=1)
    diag_indices = jnp.arange(N_tris)

    final_rows = jnp.concatenate([row_indices, diag_indices])
    final_cols = jnp.concatenate([col_indices, diag_indices])
    final_data = jnp.concatenate([data, diag_data])

    indices  = jnp.stack([final_rows, final_cols], axis=1)
    W_sparse = sparse.BCOO((final_data, indices), shape=(N_tris, N_tris))

    return W_sparse, A_inv


def apply_logPt_implicit(h, t, gamma, W_sparse, A_inv):
    """
    Apply the 2D heat semigroup P_t to a log-potential h in log-space,
    using an implicit Euler (backward Euler) time-step on the cell-centred mesh.

    Solves the linear system:
        (I − γ t A_inv W) v = e^{h − max(h)}

    then returns  log(v) + max(h)  to avoid overflow.

    Parameters
    ----------
    h        : array (N_tris,)  — log-potential
    t        : float            — propagation time
    gamma    : float            — diffusion coefficient
    W_sparse : BCOO sparse (N_tris × N_tris)
    A_inv    : array (N_tris,)  — reciprocal cell areas
    """
    t = jnp.maximum(t, 1e-12)

    # Global shift to prevent exp() overflow
    M_shift = jnp.max(h)
    u_safe  = jnp.exp(h - M_shift)

    def implicit_operator(v):
        flux        = W_sparse @ v
        scaled_flux = A_inv * flux
        return v - (gamma * t) * scaled_flux

    v, _ = cg(implicit_operator, u_safe, tol=1e-6)
    v    = jnp.maximum(v, 1e-30)
    return jnp.log(v) + M_shift


def retrieve_rho_2d(f, g, t, gamma, W_sparse, A_inv):
    """
    Compute the 2D Schrödinger Bridge density at time t ∈ [0, 1]:

        ρ_t = (P_t e^f) · (P_{1−t} e^g)

    Renormalised so that ∑_i ρ_i · area_i = 1.
    """
    f_t = apply_logPt_implicit(f, t,       gamma, W_sparse, A_inv)
    g_t = apply_logPt_implicit(g, 1.0 - t, gamma, W_sparse, A_inv)
    rho = jnp.exp(f_t + g_t)
    # Normalise using cell areas (A_inv is 1/area)
    mass = jnp.sum(rho / A_inv)
    return rho / jnp.maximum(mass, 1e-14)


def retrieve_b_2d(g, t, gamma, mesh, W_sparse, A_inv):
    """
    Compute the 2D optimal drift field:
        b_t(x) = 2γ ∇ g_t(x)

    The gradient is reconstructed cell-by-cell via Gauss's divergence theorem:
        ∇ g_i ≈ (1 / |T_i|) ∑_{faces j} g_face_j  n_j  |e_j|

    Neumann BCs are enforced: at boundary faces, g_face = g_center → ∇g · n = 0.

    Returns an array of shape (N_tris, 2).
    """
    N_tris = mesh.tris.shape[0]
    g_t    = apply_logPt_implicit(g, 1.0 - t, gamma, W_sparse, A_inv)

    neighbors    = mesh.neighbors
    valid_mask   = neighbors >= 0
    cell_indices = jnp.arange(N_tris)[:, None]
    # Neumann trick: substitute own cell index at boundary → zero normal gradient
    safe_neighbors = jnp.where(valid_mask, neighbors, cell_indices)

    g_t_center = g_t[:, None]                           # (N_tris, 1)
    g_t_neigh  = g_t[safe_neighbors]                    # (N_tris, 3)
    g_face     = 0.5 * (g_t_center + g_t_neigh)         # (N_tris, 3)

    face_lengths  = mesh.surface[mesh.face_connectivity] # (N_tris, 3)
    normals       = mesh.normals                         # (N_tris, 3, 2)

    # Gauss reconstruction: ∑_j g_j n_j |e_j|
    flux_vectors = g_face[..., None] * normals * face_lengths[..., None]
    grad_g       = jnp.sum(flux_vectors, axis=1) / mesh.area[:, None]

    return 2.0 * gamma * grad_g


def apply_IPFP_2d(log_mu0, log_mu1, gamma, W_sparse, A_inv, num_iter=50):
    """
    Solve the 2D Schrödinger Bridge via IPFP on a triangular mesh.

    Parameters
    ----------
    log_mu0, log_mu1 : arrays of shape (N_tris,)
    gamma            : diffusion coefficient
    W_sparse         : sparse Laplacian flux matrix
    A_inv            : inverse cell-area array
    num_iter         : number of Sinkhorn iterations

    Returns
    -------
    f, g : converged Schrödinger potentials of shape (N_tris,)
    """
    def body_fn(_, val):
        f_k, g_k = val
        g_next = log_mu1 - apply_logPt_implicit(f_k, 1.0, gamma, W_sparse, A_inv)
        f_next = log_mu0 - apply_logPt_implicit(g_next, 1.0, gamma, W_sparse, A_inv)
        return f_next, g_next

    N_tris  = A_inv.shape[0]
    f_init  = jnp.zeros(N_tris)
    g_init  = jnp.zeros(N_tris)
    f_final, g_final = jax.lax.fori_loop(0, num_iter, body_fn, (f_init, g_init))
    return f_final, g_final