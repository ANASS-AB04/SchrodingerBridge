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


def plot_drift_sequence(mesh, drift_matrix, t_array, output, ncols=4, scale=30,
                        label=r"$|\mathbf{b}_t|$"):
    """
    Grid of the 2D drift field b_t at EVERY interpolation time in t_array
    (magnitude tripcolor + sampled quiver per panel, shared colour scale).

    Companion to plot_drift_field (which shows a single time).  Used for the
    drifted bridges (oblique / ffd) to show how the optimal drift b_t = β + 2γ∇g_t
    evolves across the interpolation.

    drift_matrix : array, shape (T, N_tris, 2)
    """
    drift_matrix = np.asarray(drift_matrix)
    t_array = np.asarray(t_array)

    barycenters = np.asarray(mesh.barycenter)
    triang = mtri.Triangulation(
        np.asarray(mesh.points)[:, 0],
        np.asarray(mesh.points)[:, 1],
        np.asarray(mesh.tris),
    )
    mags = np.linalg.norm(drift_matrix, axis=2)          # (T, N)

    n_times = drift_matrix.shape[0]
    ncols = min(max(1, ncols), n_times)
    nrows = int(np.ceil(n_times / ncols))
    step = max(1, int(np.sqrt(len(barycenters)) / 8))    # quiver subsampling

    with plt.rc_context({"text.usetex": False}):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3.8 * nrows), squeeze=False)
        norm = mcolors.Normalize(vmin=float(mags.min()), vmax=float(mags.max()))

        handle = None
        for idx, ax in enumerate(axes.flat):
            if idx >= n_times:
                ax.axis("off")
                continue
            b_t = drift_matrix[idx]
            handle = ax.tripcolor(triang, mags[idx], cmap="RdBu_r",
                                  shading="flat", norm=norm)
            ax.quiver(barycenters[::step, 0], barycenters[::step, 1],
                      b_t[::step, 0], b_t[::step, 1],
                      color="black", scale=scale, width=0.002, headwidth=4)
            ax.set_aspect("equal")
            ax.set_title(f"t = {float(t_array[idx]):.2f}", fontsize=12)
            ax.set_xlabel(r"$x$", fontsize=10)
            ax.set_ylabel(r"$y$", fontsize=10)

        if handle is not None:
            fig.colorbar(handle, ax=axes.ravel().tolist(),
                         fraction=0.025, pad=0.02, label=label)
        fig.suptitle(r"SB drift field $b_t$ over interpolation time", fontsize=15)
        fig.subplots_adjust(right=0.9)

    fname = _figure_path(output, "_drift_sequence.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved drift sequence: {fname}")


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


def plot_abs_error_grid(mesh, err_matrix, t_ref, output, method_name, ncols=3):
    """
    One figure per method showing |M_interp − M_ref| on the mesh for every
    reference timestep, laid out like plot_density_transport.

    err_matrix : (len(t_ref), N_cells)  absolute error fields
    t_ref      : 1-D array of reference times (e.g. [0.1, …, 0.9])
    output     : path prefix  →  saves  <output>_abs_error.png
    """
    err_matrix = np.asarray(err_matrix)
    t_ref      = np.asarray(t_ref)
    n_times    = err_matrix.shape[0]
    ncols      = min(max(1, ncols), n_times)
    nrows      = int(np.ceil(n_times / ncols))

    triang = mtri.Triangulation(
        np.asarray(mesh.points)[:, 0],
        np.asarray(mesh.points)[:, 1],
        np.asarray(mesh.tris),
    )

    vmin = 0.0
    vmax = float(err_matrix.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    with plt.rc_context({"text.usetex": False}):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)
        im = None
        for idx, ax in enumerate(axes.flat):
            if idx >= n_times:
                ax.axis("off")
                continue
            im = ax.tripcolor(triang, err_matrix[idx],
                              cmap="hot", shading="flat", norm=norm)
            ax.set_aspect("equal")
            ax.set_title(f"t = {float(t_ref[idx]):.2f}", fontsize=11)
            ax.axis("off")

        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(),
                         fraction=0.02, pad=0.02,
                         label=r"$|M_\mathrm{interp} - M_\mathrm{ref}|$")
        fig.suptitle(f"|{method_name} − ref|", fontsize=14)

    fname = _figure_path(output, "_abs_error.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved abs error grid: {fname}")


def plot_l2_linf_errors(output, t_array, errors_dict, title="Validation error vs t",
                        l2_formula=None, linf_formula=None):
    """
    Plot per-time L2 and L∞ errors for one or several methods, side by side.

    Parameters
    ----------
    errors_dict : dict {label: {"l2": 1-D array, "linf": 1-D array}}, each
        array of length len(t_array).
    l2_formula, linf_formula : str, optional
        LaTeX strings (without surrounding ``$``) describing exactly how each
        error is computed; rendered as an annotation on the corresponding
        subplot. Defaults are the Mach-interpolation definitions (relative L2,
        absolute L∞).
    """
    if l2_formula is None:
        l2_formula = (r"\|M-M_{\mathrm{ref}}\|_{L^2}/\|M_{\mathrm{ref}}\|_{L^2}"
                      r"=\sqrt{\sum_i (M_i-M_i^{\mathrm{ref}})^2 A_i}\,/\,"
                      r"\sqrt{\sum_i (M_i^{\mathrm{ref}})^2 A_i}")
    if linf_formula is None:
        linf_formula = r"\|M-M_{\mathrm{ref}}\|_{L^\infty}=\max_i |M_i-M_i^{\mathrm{ref}}|"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (label, errs), color in zip(errors_dict.items(), colors):
        axes[0].plot(t_array, errs["l2"],   "o-", label=label, color=color, linewidth=2, markersize=4)
        axes[1].plot(t_array, errs["linf"], "o-", label=label, color=color, linewidth=2, markersize=4)
    axes[0].set_title("L2 error vs t", fontsize=13)
    axes[1].set_title("L∞ error vs t", fontsize=13)
    for ax, ylabel, formula in zip(axes, ["L2 error", "L∞ error"],
                                   [l2_formula, linf_formula]):
        ax.set_xlabel("t", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.98, rf"${formula}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"))
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fname = _figure_path(output, "_l2_linf_errors.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved L2/L∞ error plot: {fname}")


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