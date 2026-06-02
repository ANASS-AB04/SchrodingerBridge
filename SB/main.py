import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt  

# 1. Get the directory where main.py lives (the 'SB' folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the 'GenerativePDE' root folder
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 3. Get the path to the Euler folder specifically
euler_dir = os.path.join(parent_dir, 'Euler')

# 4. Add BOTH folders to Python's search path
sys.path.insert(0, euler_dir)  # This lets it find 'jax_fvm' directly
sys.path.insert(0, parent_dir)

# 1D functions
from resolution import apply_IPFP, retrieve_rho, compute_drift_field
from plot import plot_bridge_dashboard, plot_3d_evolution, plot_differences, plot_1d_sde_trajectories

# 2D functions
from Euler.jax_fvm.src.mesh import Mesh
from resolution import apply_IPFP_2d, build_cell_centered_laplacian, retrieve_b_2d, retrieve_rho_2d
from plot import (
    plot_3d_density_surface, 
    plot_3d_density_transport, 
    plot_density_transport, 
    plot_drift_field,
    plot_entropy,
)
from Euler.jax_fvm.src.plot import plot_solution

# Test case factories
from testcases import get_1d_case, get_2d_case

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_relative_differential_entropy(rho_1, rho_sequence, measure):
    """
    Computes H(rho) = - integral(rho * log(rho) * measure) for each time step.
    measure: dx for 1D, mesh.area for 2D.
    """
    entropies = []
    for rho in rho_sequence:
        # Normalize to prevent mass drift inflation in the entropy calculation
        mass = jnp.sum(rho * measure)
        rho_norm = rho / mass
        
        rho_safe = jnp.maximum(rho_norm / rho_1, 1e-14)
        h = -jnp.sum(rho_safe * jnp.log(rho_safe) * measure)
        entropies.append(float(h))
    return np.array(entropies)

# ==============================================================================
# 1D Runner
# ==============================================================================

def run_1d_case(test_case="1d_gauss_to_ou"):
    print(f"\n--- Running 1D Case: {test_case} ---")
    
    # Grid Config
    N = 400
    L = 6.0
    x = jnp.linspace(-L, L, N)
    dx = x[1] - x[0]
    
    # SDE Parameters
    gamma = 0.5 
    num_iter = 500
    output_prefix = os.path.join(OUTPUT_DIR, f"SB_{test_case}_gamma{gamma}")

    # Time steps
    t_array = np.linspace(0.0, 1.0, 101)

    # 1. Load Marginals and Baselines from testcases.py
    mu0, mu1, rho_ref_dummy, ref_label = get_1d_case(test_case, x, t_array, dx)

    eps = 1e-7
    log_mu0 = jnp.log(jnp.maximum(mu0, eps))
    log_mu1 = jnp.log(jnp.maximum(mu1, eps))

    # 2. Run IPFP
    print("Running 1D IPFP...")
    ipfp_jit = jax.jit(apply_IPFP, static_argnames=['num_iter'])
    f, g = ipfp_jit(log_mu0, log_mu1, x, gamma, dx, num_iter=num_iter)
    
    # 3. Reconstruct Physics
    rho_sequence = np.array([retrieve_rho(f, g, x, t, gamma, dx) for t in t_array])
    
    b_field = compute_drift_field(g, x, t_array, gamma, dx)
    
    # 4. Compute Entropy
    entropies = compute_relative_differential_entropy(mu1, rho_sequence, dx)
    
    # 5. Generate Plots
    print("Generating 1D plots...")

    plot_bridge_dashboard(
        output_prefix, np.array(x), np.array(mu0), np.array(mu1), 
        t_array, rho_sequence, rho_ref_dummy, np.array(b_field), ref_label=ref_label
    )
    plot_3d_evolution(output_prefix, np.array(x), t_array, rho_sequence, title=f"SB Evolution: {test_case}")
    plot_entropy(output_prefix, t_array, entropies, title=f"Entropy Evolution ({test_case})")
    
    # Paper-style 1D SDE trajectories
    plot_1d_sde_trajectories(
        output_prefix, np.array(x), t_array, np.array(b_field), 
        np.array(mu0), np.array(mu1), gamma=gamma, num_particles=20
    )
    print("Done with 1D case.\n")

# ==============================================================================
# 2D Runner
# ==============================================================================

def run_2d_case(test_case="2d_gauss_to_gauss"):
    print(f"\n--- Running 2D Case: {test_case} ---")
    
    # Build Mesh
    mesh = Mesh()
    mesh.mesh_generator(maxV=1e-3, marker_boundary=1, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0)
    mesh.save_mesh(os.path.join(OUTPUT_DIR, f"{test_case}_mesh.vtk"))
    
    gamma = 0.05
    num_iter = 100
    output_prefix = os.path.join(OUTPUT_DIR, f"SB_{test_case}_gamma{gamma}")

    # 1. Load Marginals from testcases.py
    mu0, mu1 = get_2d_case(test_case, mesh)

    # 2. Build Laplacian
    W_sparse, A_inv = build_cell_centered_laplacian(mesh)

    # 3. Run IPFP
    print("Running 2D IPFP on Mesh...")
    f, g = apply_IPFP_2d(jnp.log(mu0), jnp.log(mu1), gamma, W_sparse, A_inv, num_iter=num_iter)

    # 4. Time steps
    t_array = np.linspace(0.0, 1.0, 10) 
    
    # 5. Reconstruct Physics
    rho_sequence = np.stack([np.asarray(retrieve_rho_2d(f, g, float(t), gamma, W_sparse, A_inv)) for t in t_array], axis=0)
    
    # Normalize 2D density
    area_array = np.asarray(mesh.area)
    masses = np.sum(rho_sequence * area_array, axis=1, keepdims=True)
    rho_sequence = rho_sequence / masses
    
    drift_sequence = np.stack([np.asarray(retrieve_b_2d(g, float(t), gamma, mesh, W_sparse, A_inv)) for t in t_array], axis=0)
    
    # 6. Compute Entropy
    entropies = compute_relative_differential_entropy(mu1, rho_sequence, np.asarray(mesh.area))

    # 7. Generate Plots
    print("Generating 2D plots...")
    plot_3d_density_surface(mesh, mu0, output_prefix + "_source", title="Source Density (t=0)")
    plot_3d_density_surface(mesh, mu1, output_prefix + "_target", title="Target Density (t=1)")
    
    plot_density_transport(mesh, rho_sequence, t_array, output_prefix)
    plot_drift_field(mesh, drift_sequence, t_array, output_prefix, time_index=5)
    
    plot_entropy(output_prefix, t_array, entropies, title=f"Entropy Evolution ({test_case})")
    
    print("Done with 2D case.\n")


# ==============================================================================
# Field data & Mach Interpolation Case
# ==============================================================================

def transforme_field_data_to_density_2d(mesh, field):
    """
    Transforms a field defined on the mesh cells into a true probability density 
    by normalizing with respect to the total spatial mass.
    """
    if np.any(field < 0):
        print("Warning: Field contains negative values. Setting them to zero for density transformation.")
        field = np.where(field < 0, 0, field)
    field = np.asarray(field)
    area = np.asarray(mesh.area)
    
    # Calculate the total physical mass (spatial integral)
    Mass = np.sum(field * area)
    
    # Mathematical probability density: integrating density * area will equal 1.0
    density = field / Mass 
    return jnp.array(density), Mass


def retrieve_field_from_density_2d(t, density, Mass0, Mass1):
    """
    Transforms a probability density back into the physical Mach field 
    by scaling with the linearly interpolated physical mass.
    """
    current_mass = (1.0 - t) * Mass0 + t * Mass1
    field = density * current_mass
    return jnp.array(field)


def run_mach_interpolation_case():
    print("\n--- Running 2D Case: Mach Field Interpolation ---")
    
    # Results path
    bundle_path0 = "/home/anass/GenerativePDE/results/bump/h0.02/M1.20_HLLC_MUSCL_SRK2_t4.00.npz"
    bundle_path1 = "/home/anass/GenerativePDE/results/bump/h0.02/M1.30_HLLC_MUSCL_SRK2_t4.00.npz"
    
    # load mesh
    mesh_path = "/home/anass/GenerativePDE/meshes/bump/bump_h0.02.npy"
    mesh = Mesh()
    mesh.load_mesh(mesh_path)

    # load Mach fields
    print("Loading Mach bundles...")
    data0 = np.load(bundle_path0)
    data1 = np.load(bundle_path1)

    mach0 = data0["mach"]
    mach1 = data1["mach"]

    data0.close()
    data1.close()
    
    print("Plotting the original Mach fields...")
    plot_solution(mesh, mach0, labels=r'$M$', title="Initial Mach Field (M=1.2)", 
                  filename=os.path.join(OUTPUT_DIR, "Mach_Field_M1.20.png"), cmap="viridis")
    plot_solution(mesh, mach1, labels=r'$M$', title="Target Mach Field (M=1.3)", 
                  filename=os.path.join(OUTPUT_DIR, "Mach_Field_M1.30.png"), cmap="viridis")

    print("Mach fields loaded. Transforming to densities...")
    mu0, Mass0 = transforme_field_data_to_density_2d(mesh, mach0)
    mu1, Mass1 = transforme_field_data_to_density_2d(mesh, mach1)
    
    if abs(Mass0 - Mass1) > 1e-6:
        print(f"Warning: Masses of the two densities differ significantly (Mass0={Mass0:.4f}, Mass1={Mass1:.4f}). This will be accounted for during field reconstruction.")
    
    # Run Schrodinger Bridge on these densities
    print("Running Schrodinger Bridge on Mach-derived densities...")
    gamma = 0.004        
    num_iter = 1000     # Sufficient for strong convergence under safe gamma
    eps = 1e-12         # Robust threshold preventing log(0) -> -inf
    W_sparse, A_inv = build_cell_centered_laplacian(mesh)
    
    # Safe log-transforms to ensure matrix stability
    log_mu0 = jnp.log(jnp.maximum(mu0, eps))
    log_mu1 = jnp.log(jnp.maximum(mu1, eps))
    
    f, g = apply_IPFP_2d(log_mu0, log_mu1, gamma, W_sparse, A_inv, num_iter=num_iter)
    t_array = np.linspace(0.0, 1.0, 10)
    
    rho_sequence = np.stack([np.asarray(retrieve_rho_2d(f, g, float(ti), gamma, W_sparse, A_inv)) for ti in t_array], axis=0)
    
    # Standardize density integration across intermediate steps
    area_array = np.asarray(mesh.area)
    masses = np.sum(rho_sequence * area_array, axis=1, keepdims=True)
    rho_sequence = rho_sequence / masses
    
    drift_sequence = np.stack([np.asarray(retrieve_b_2d(g, float(ti), gamma, mesh, W_sparse, A_inv)) for ti in t_array], axis=0)
    
    # SUCCESSFUL MAPPING: Reconstruct physical Mach variables from probability states
    retrieve_field_sequence = np.stack([
        retrieve_field_from_density_2d(float(ti), rho, Mass0, Mass1) 
        for ti, rho in zip(t_array, rho_sequence)
    ], axis=0)
    
    # Plotting
    print("Generating plots for Mach-derived fields...")
    # Plots the dynamic evolution mapping using the actual physical Mach ranges
    plot_density_transport(mesh, retrieve_field_sequence, t_array, os.path.join(OUTPUT_DIR, "SB_Mach_density_transport"))
    plot_drift_field(mesh, drift_sequence, t_array, os.path.join(OUTPUT_DIR, "SB_Mach_drift_field"), time_index=5)
    
    # Output distinct frames reflecting true Mach 1.2 -> 1.3 fluid transformation
    for i, t_val in enumerate(t_array):
        fname = os.path.join(OUTPUT_DIR, f"SB_Mach_Interpolation_t{t_val:.2f}.png")
        plot_solution(
            mesh, 
            retrieve_field_sequence[i], 
            labels=r'$M$', 
            title=f"Schrödinger Bridge Mach Field (t = {t_val:.2f})", 
            filename=fname, 
            cmap="viridis"
        )
    print("Done with Mach-derived density case.\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("Available Test Cases:")
    print(" - 1d_gauss_to_ou")
    print(" - 1d_square_to_ou")
    print(" - 1d_gauss_to_gauss")
    print(" - 1d_square_to_gauss")
    print(" - 1d_bimodal")
    print(" - 2d_gauss_to_gauss")
    print(" - 2d_gauss_to_bimodal")
    print(" - 2d_mach_interpolation\n") 
    
    test_case = input("Enter test case: ").strip()
    
    if test_case.startswith("1d"):
        run_1d_case(test_case)
    elif test_case.startswith("2d_mach"):
        run_mach_interpolation_case()
    elif test_case.startswith("2d"):
        run_2d_case(test_case)
    else:
        print("Invalid test case. Please run again and enter a valid test case.")