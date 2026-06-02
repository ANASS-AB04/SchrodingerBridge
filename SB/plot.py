import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import tri as mtri
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
#  Dashboard  (3-panel summary)
# ---------------------------------------------------------------------------

def plot_bridge_dashboard(output, x, mu0, mu1, t_array, rho_sb, rho_ref,
                          b_field, ref_label="OU Reference"):
    """
    3-panel dashboard:
      • Panel 1 – Schrödinger Bridge density ρ_t
      • Panel 2 – Reference (OU) density ρ_t
      • Panel 3 – Drift field b_t(x) as a filled contour plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # ------------------------------------------------------------------
    # Panel 1: Schrödinger Bridge density
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(x, mu0, color='blue', linewidth=2.5, label='t = 0 (source)')
    for i, t_val in enumerate(t_array):
        ax.plot(x, rho_sb[i], color=cmap(norm(t_val)), alpha=0.8, linewidth=1.8)
    ax.plot(x, mu1, color='red', linewidth=2.5, label='t = 1 (target)')
    ax.set_title(r"Schrödinger Bridge $\rho_t(x)$", fontsize=14)
    ax.set_xlabel("Space (x)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Shared colorbar for time axis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time t", fraction=0.04, pad=0.02)

    # ------------------------------------------------------------------
    # Panel 2: Reference density
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.plot(x, mu0, color='blue', linewidth=2.5, label='t = 0 (source)')
    for i, t_val in enumerate(t_array):
        ax.plot(x, rho_ref[i], color=cmap(norm(t_val)),
                alpha=0.8, linestyle='--', linewidth=1.8)
    ax.plot(x, mu1, color='red', linewidth=2.5, label='t = 1 (target)')
    ax.set_title(rf"{ref_label} $\rho_t(x)$", fontsize=14)
    ax.set_xlabel("Space (x)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sm, ax=ax, label="Time t", fraction=0.04, pad=0.02)

    # ------------------------------------------------------------------
    # Panel 3: Drift field  (RdBu centred at 0)
    # ------------------------------------------------------------------
    ax = axes[2]
    T_mesh, X_mesh = np.meshgrid(t_array, x)

    # Symmetric colour scale centred on zero
    b_abs_max = np.abs(b_field).max()
    b_abs_max = b_abs_max if b_abs_max > 0 else 1.0
    drift_norm = mcolors.TwoSlopeNorm(vmin=-b_abs_max, vcenter=0.0, vmax=b_abs_max)

    c = ax.contourf(T_mesh, X_mesh, b_field.T, levels=30,
                    cmap='RdBu', norm=drift_norm)
    fig.colorbar(c, ax=ax, label=r"Drift velocity $b_t(x)$")
    ax.set_title("Bridge Drift Field $b_t(x)$", fontsize=14)
    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Space (x)", fontsize=12)

    plt.tight_layout()
    fname = f"{output}_dashboard.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dashboard:  {fname}")


# ---------------------------------------------------------------------------
#  3D surface evolution
# ---------------------------------------------------------------------------

def plot_3d_evolution(output, x, t_array, rho_matrix, title="3D Density Evolution"):
    """
    3D surface plot of ρ(x, t) over the spatial–temporal domain,
    with the initial and final densities highlighted on the time boundaries.
    """
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    X, T = np.meshgrid(x, t_array)
    surf = ax.plot_surface(X, T, rho_matrix, cmap='viridis',
                           edgecolor='none', alpha=0.90)

    t0 = float(t_array[0])
    t1 = float(t_array[-1])
    source_density = rho_matrix[0]
    target_density = rho_matrix[-1]

    ax.plot(x, np.full_like(x, t0), source_density,
            color='tab:blue', linewidth=3.0, label='Initial density (t=0)')
    ax.plot(x, np.full_like(x, t1), target_density,
            color='tab:red', linewidth=3.0, label='Target density (t=1)')

    ax.set_xlabel('Space (x)', fontsize=12)
    ax.set_ylabel('Time (t)', fontsize=12)
    ax.set_zlabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()    # t = 0 at the front
    ax.legend(loc='upper right', fontsize=10)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Density')

    fname = f"{output}_3d_surface.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D surface: {fname}")


def plot_3d_drift_surface(output, x, t_array, b_field, title="3D Drift Evolution"):
    """
    3D surface plot of the drift field b_t(x) over space and time.
    """
    with plt.rc_context({"text.usetex": False}):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, T = np.meshgrid(x, t_array)
        surf = ax.plot_surface(
            X, T, b_field,
            cmap='RdBu_r',
            edgecolor='none',
            alpha=0.90,
        )

        ax.set_xlabel('Space (x)', fontsize=12)
        ax.set_ylabel('Time (t)', fontsize=12)
        ax.set_zlabel(r'Drift $b_t(x)$', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.view_init(elev=20, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=r'Drift $b_t(x)$')

    fname = f"{output}_drift_3d_surface.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D drift surface: {fname}")


def integrate_drift_map(x, t_array, b_field):
    """
    Integrate the drift field forward in time to obtain a transport map.

    The map returns the position X_t(x_0) of particles initially located at x_0,
    using first-order explicit Euler integration on the interpolated drift field.
    """
    x_array = np.asarray(x)
    t_array = np.asarray(t_array)
    b_field = np.asarray(b_field)

    mapped_positions = np.zeros_like(b_field)
    mapped_positions[0] = x_array

    current_positions = x_array.copy()
    for idx in range(len(t_array) - 1):
        dt = t_array[idx + 1] - t_array[idx]
        drift_now = np.interp(current_positions, x_array, b_field[idx])
        current_positions = current_positions + dt * drift_now
        mapped_positions[idx + 1] = current_positions

    return mapped_positions


def plot_transport_map(output, x, t_array, transport_map, title="Transport map (integral of drift)"):
    """
    Plot the drift-integrated transport map as a 3D surface.
    """
    with plt.rc_context({"text.usetex": False}):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X0, T = np.meshgrid(x, t_array)
        surf = ax.plot_surface(
            X0,
            T,
            transport_map,
            cmap='plasma',
            edgecolor='none',
            alpha=0.90,
        )

        ax.plot_surface(
            X0,
            T,
            X0,
            color='white',
            alpha=0.15,
            linewidth=0,
        )

        ax.set_xlabel('Initial position x₀', fontsize=12)
        ax.set_ylabel('Time (t)', fontsize=12)
        ax.set_zlabel('Mapped position X_t(x₀)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.view_init(elev=20, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Mapped position')

    fname = f"{output}_transport_map_3d.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved transport map: {fname}")

def plot_1d_sde_trajectories(output, x, t_array, b_field, mu0, mu1, gamma, num_particles=15, seed=42):
    """
    Trace le schéma 2D des trajectoires SDE (temps vs espace) avec les distributions
    marginales, en colorant chaque trajectoire distinctement.
    """
    fig = plt.figure(figsize=(10, 6))
    # Création d'une grille à 3 panneaux : [Marginale Gauche] [Trajectoires] [Marginale Droite]
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 5, 1.5], wspace=0.05)
    
    ax_left = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2], sharey=ax_left)
    
    # ---------------------------------------------------------
    # 1. Panneau Gauche : Marginale Initiale (mu0)
    # ---------------------------------------------------------
    ax_left.plot(mu0, x, color='gray', linewidth=2)
    ax_left.fill_betweenx(x, 0, mu0, color='gray', alpha=0.2)
    ax_left.invert_xaxis()  # Retourner horizontalement
    ax_left.set_ylabel("Espace (x)", fontsize=12)
    ax_left.set_title(r"$\mu_0(x)$", fontsize=14)
    ax_left.set_xticks([]) 
    ax_left.grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # 2. Panneau Droit : Marginale Cible (mu1)
    # ---------------------------------------------------------
    ax_right.plot(mu1, x, color='gray', linewidth=2)
    ax_right.fill_betweenx(x, 0, mu1, color='gray', alpha=0.2)
    ax_right.set_title(r"$\mu_1(x)$", fontsize=14)
    ax_right.set_xticks([]) 
    ax_right.grid(True, alpha=0.3)
    
    # ---------------------------------------------------------
    # 3. Panneau Central : Trajectoires SDE (Fluctuations colorées)
    # ---------------------------------------------------------
    rng = np.random.default_rng(seed)
    
    # Échantillonnage des positions initiales depuis mu0 
    cdf = np.cumsum(mu0)
    cdf = cdf / cdf[-1]
    # Prendre 'num_particles' parfaitement espacées 
    u = np.linspace(0.2, 0.98, num_particles)
    
    X = np.zeros((num_particles, len(t_array)))
    X[:, 0] = np.interp(u, cdf, x)
    
    # Intégration d'Euler-Maruyama
    for i in range(len(t_array) - 1):
        dt = t_array[i+1] - t_array[i]
        drift = np.interp(X[:, i], x, b_field[i])
        noise = rng.normal(0, np.sqrt(2 * gamma * dt), size=num_particles)
        X[:, i+1] = X[:, i] + drift * dt + noise
        
    # --- MODIFICATION DES COULEURS ICI ---
    # Trier les indices par position initiale pour créer un beau dégradé
    sorted_indices = np.argsort(X[:, 0])
    
    # Générer une palette de couleurs (plasma va du bleu/violet au jaune)
    colors = cm.plasma(np.linspace(0.0, 1.0, num_particles))
    
    # Tracer chaque trajectoire avec sa couleur unique
    for color_idx, particle_idx in enumerate(sorted_indices):
        ax_main.plot(t_array, X[particle_idx, :], color=colors[color_idx], alpha=0.35, linewidth=1.2)
        
    # Superposer les points de départ et d'arrivée avec les mêmes couleurs
    ax_main.scatter(np.zeros(num_particles), X[sorted_indices, 0], color=colors, s=15, zorder=3, edgecolors='none')
    ax_main.scatter(np.ones(num_particles) * t_array[-1], X[sorted_indices, -1], color=colors, s=15, zorder=3, edgecolors='none')
    # -------------------------------------
    
    ax_main.set_xlabel("Temps de Diffusion, t", fontsize=12)
    ax_main.set_xlim([t_array[0], t_array[-1]])
    ax_main.set_ylim([x[0], x[-1]])
    ax_main.grid(True, alpha=0.3)
    
    # Synchroniser les axes Y et cacher les étiquettes internes
    ax_main.sharey(ax_left)
    plt.setp(ax_main.get_yticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    
    fname = f"{output}_sde_schematic.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique des trajectoires SDE sauvegardé : {fname}")
# ---------------------------------------------------------------------------
#  Difference plots  (SB − reference)
# ---------------------------------------------------------------------------

def plot_differences(output, x, t_array, rho_sb, rho_ref):
    """
    Spatial difference  ρ_SB(x, t) − ρ_ref(x, t)  at selected intermediate times.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    key_times = [0.2, 0.5, 0.8, 1.0]

    for t_val in key_times:
        idx        = int(np.argmin(np.abs(t_array - t_val)))
        difference = rho_sb[idx] - rho_ref[idx]
        ax.plot(x, difference, label=f't = {t_val:.1f}', linewidth=2)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.set_title("Spatial Mismatch: Schrödinger Bridge vs. Reference", fontsize=14)
    ax.set_xlabel("Space (x)", fontsize=12)
    ax.set_ylabel(r"Density difference $(\rho_{SB} - \rho_{ref})$", fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    fname = f"{output}_differences.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved differences: {fname}")


# ---------------------------------------------------------------------------
#  Convergence diagnostics
# ---------------------------------------------------------------------------

def plot_ipfp_convergence(output, residuals):
    """
    Plot IPFP residuals (‖f_{k+1} − f_k‖_∞) versus iteration number.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(residuals, linewidth=2, color='steelblue')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"Residual $\|f_{k+1} - f_k\|_\infty$", fontsize=12)
    ax.set_title("IPFP Convergence", fontsize=14)
    ax.grid(True, which='both', alpha=0.3)

    fname = f"{output}_ipfp_convergence.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot: {fname}")


def plot_marginal_fit(output, x, mu0, mu1, rho_sb_t0, rho_sb_t1, dx):
    """
    Check that the bridge marginals at t=0 and t=1 match μ₀ and μ₁.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ['Source marginal  (t = 0)', 'Target marginal  (t = 1)']
    refs   = [mu0, mu1]
    sbs    = [rho_sb_t0, rho_sb_t1]

    for ax, title, ref, sb in zip(axes, titles, refs, sbs):
        ax.plot(x, ref, 'b-', linewidth=2.5, label='Target marginal μ')
        ax.plot(x, sb,  'r--', linewidth=2.0, label='Bridge marginal ρ_t')
        l2 = float(np.sqrt(np.sum((sb - ref)**2 * dx))  )
        ax.set_title(f"{title}\nL² error = {l2:.2e}", fontsize=12)
        ax.set_xlabel("Space (x)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{output}_marginal_fit.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved marginal fit: {fname}")


def _figure_path(output, suffix):
    if output.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
        base, _ = os.path.splitext(output)
        return f"{base}{suffix}"
    return f"{output}{suffix}"


def plot_density_transport(mesh, rho_matrix, t_array, output, ncols=4, label=r"$\rho$"):
    """
    Visualise the transport of a 2D density over several times on the mesh.

    Parameters
    ----------
    mesh : Mesh
        Triangular mesh object.
    rho_matrix : array, shape (T, N_tris)
        Density snapshots on mesh cells.
    t_array : array, shape (T,)
        Times associated with each snapshot.
    output : str
        Prefix or image path for the saved figure.
    ncols : int
        Number of columns in the subplot grid.
    label : str
        Colorbar label.
    """
    rho_matrix = np.asarray(rho_matrix)
    t_array = np.asarray(t_array)

    triang = mtri.Triangulation(
        np.asarray(mesh.points)[:, 0],
        np.asarray(mesh.points)[:, 1],
        np.asarray(mesh.tris),
    )

    n_times = rho_matrix.shape[0]
    ncols = min(max(1, ncols), n_times)
    nrows = int(np.ceil(n_times / ncols))

    with plt.rc_context({"text.usetex": False}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows), squeeze=False)

        vmin = float(np.min(rho_matrix))
        vmax = float(np.max(rho_matrix))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        scatter_handle = None
        for idx, ax in enumerate(axes.flat):
            if idx >= n_times:
                ax.axis("off")
                continue

            scatter_handle = ax.tripcolor(
                triang,
                rho_matrix[idx],
                cmap="viridis",
                shading="flat",
                norm=norm,
            )
            ax.set_aspect("equal")
            ax.set_title(f"t = {float(t_array[idx]):.2f}", fontsize=12)
            ax.set_xlabel(r"$x$", fontsize=10)
            ax.set_ylabel(r"$y$", fontsize=10)

        if scatter_handle is not None:
            fig.colorbar(scatter_handle, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label=label)

        fig.suptitle("2D density transport", fontsize=15)
        fig.subplots_adjust(right=0.9)

    fname = _figure_path(output, "_density_transport.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved density transport: {fname}")


def plot_drift_field(mesh, drift_matrix, t_array, output, time_index=0, scale=30, label=r"$|\mathbf{b}_t|$"):
    """
    Visualise the 2D drift field on the mesh for a selected time.

    Parameters
    ----------
    mesh : Mesh
        Triangular mesh object.
    drift_matrix : array, shape (T, N_tris, 2)
        Drift snapshots on mesh cells.
    t_array : array, shape (T,)
        Times associated with each snapshot.
    output : str
        Prefix or image path for the saved figure.
    time_index : int
        Snapshot index to visualise.
    scale : float
        Quiver scale factor.
    label : str
        Colorbar label.
    """
    drift_matrix = np.asarray(drift_matrix)
    t_array = np.asarray(t_array)

    if time_index < 0:
        time_index = drift_matrix.shape[0] + time_index

    if time_index < 0 or time_index >= drift_matrix.shape[0]:
        raise IndexError("time_index is outside the drift snapshot range")

    b_t = drift_matrix[time_index]
    magnitudes = np.linalg.norm(b_t, axis=1)

    barycenters = np.asarray(mesh.barycenter)
    triang = mtri.Triangulation(
        np.asarray(mesh.points)[:, 0],
        np.asarray(mesh.points)[:, 1],
        np.asarray(mesh.tris),
    )

    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(7, 6))
        drift_plot = ax.tripcolor(
            triang,
            magnitudes,
            cmap="RdBu_r",
            shading="flat",
        )

        step = max(1, int(np.sqrt(len(barycenters)) / 8))
        sample = barycenters[::step]
        sample_drift = b_t[::step]
        ax.quiver(
            sample[:, 0],
            sample[:, 1],
            sample_drift[:, 0],
            sample_drift[:, 1],
            color="black",
            scale=scale,
            width=0.002,
            headwidth=4,
        )

        ax.set_aspect("equal")
        ax.set_title(f"Drift field at t = {float(t_array[time_index]):.2f}", fontsize=13)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$y$", fontsize=10)
        fig.colorbar(drift_plot, ax=ax, label=label)

        plt.tight_layout()
    fname = _figure_path(output, "_drift_field.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved drift field: {fname}")


def plot_3d_density_surface(mesh, density, output, title="3D density surface"):
    """
    Visualise a cell-centred density as a 3D surface over the mesh barycentres.

    Parameters
    ----------
    mesh : Mesh
        Triangular mesh object.
    density : array, shape (N_tris,)
        Cell-centred density values.
    output : str
        Prefix or image path for the saved figure.
    title : str
        Plot title.
    """
    density = np.asarray(density)
    barycenters = np.asarray(mesh.barycenter)

    triang = mtri.Triangulation(barycenters[:, 0], barycenters[:, 1])

    with plt.rc_context({"text.usetex": False}):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_trisurf(
            barycenters[:, 0],
            barycenters[:, 1],
            density,
            triangles=triang.triangles,
            cmap="viridis",
            linewidth=0.0,
            antialiased=False,
        )

        ax.set_xlabel(r"$x$", fontsize=11)
        ax.set_ylabel(r"$y$", fontsize=11)
        ax.set_zlabel(r"Density", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.view_init(elev=30, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Density")

    fname = _figure_path(output, "_3d_density.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D density surface: {fname}")


def plot_3d_density_transport(mesh, rho_matrix, t_array, output, ncols=4, label=r"Density"):
    """
    Visualise transported cell-centred densities as 3D surfaces over time.

    Parameters
    ----------
    mesh : Mesh
        Triangular mesh object.
    rho_matrix : array, shape (T, N_tris)
        Density snapshots on the mesh cells.
    t_array : array, shape (T,)
        Times associated with each snapshot.
    output : str
        Prefix or image path for the saved figure.
    ncols : int
        Number of columns in the subplot grid.
    label : str
        Colorbar label.
    """
    rho_matrix = np.asarray(rho_matrix)
    t_array = np.asarray(t_array)

    barycenters = np.asarray(mesh.barycenter)
    triang = mtri.Triangulation(barycenters[:, 0], barycenters[:, 1])

    n_times = rho_matrix.shape[0]
    ncols = min(max(1, ncols), n_times)
    nrows = int(np.ceil(n_times / ncols))

    vmin = float(np.min(rho_matrix))
    vmax = float(np.max(rho_matrix))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    with plt.rc_context({"text.usetex": False}):
        fig = plt.figure(figsize=(4 * ncols, 3.6 * nrows))
        shared_surf = None

        for idx in range(n_times):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
            surf = ax.plot_trisurf(
                barycenters[:, 0],
                barycenters[:, 1],
                rho_matrix[idx],
                triangles=triang.triangles,
                cmap="viridis",
                linewidth=0.0,
                antialiased=False,
                norm=norm,
            )
            if shared_surf is None:
                shared_surf = surf
            ax.set_title(f"t = {float(t_array[idx]):.2f}", fontsize=11)
            ax.set_xlabel(r"$x$", fontsize=9)
            ax.set_ylabel(r"$y$", fontsize=9)
            ax.set_zlabel(label, fontsize=9)
            ax.view_init(elev=25, azim=-60)

        fig.subplots_adjust(right=0.88)
        fig.colorbar(shared_surf, ax=fig.axes, fraction=0.025, pad=0.02, label=label)

    fname = _figure_path(output, "_transported_3d.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved transported 3D density evolution: {fname}")

def plot_entropy(output, t_array, entropy_array, title="Evolution of Differential Entropy"):
    """
    Plots the differential entropy H(rho) over time.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_array, entropy_array, color='tab:purple', linewidth=2.5, marker='o', markersize=4)
    
    ax.set_xlabel("Diffusion Time (t)", fontsize=12)
    ax.set_ylabel(r"Entropy $H(\rho_t) = -\int \rho_t \log \frac{\rho_t}{\rho_1} dx$", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    fname = _figure_path(output, "_entropy.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Entropy plot: {fname}")