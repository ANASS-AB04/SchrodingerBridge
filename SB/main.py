import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

if __name__ == "__main__" and __package__ is None:
    import sys
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    __package__ = "SB"

import re
import tomllib

from .utils import run_1d_case, run_2d_case, run_mach_interpolation_case

CASES_1D       = ["1d_gauss_to_ou", "1d_square_to_ou",
                  "1d_gauss_to_gauss", "1d_square_to_gauss", "1d_bimodal"]
CASES_2D_SYNTH = ["2d_gauss_to_gauss", "2d_gauss_to_bimodal"]
CASE_MACH      = "2d_mach_interpolation"
ALL_CASES      = CASES_1D + CASES_2D_SYNTH + [CASE_MACH]

# Config path: env var SB_CONFIG overrides the default (used by the sweep
# submitter to point each SLURM job at its own generated Config copy).
_CFG_PATH = os.environ.get(
    "SB_CONFIG", os.path.join(os.path.dirname(__file__), "Config.toml"))

_MACH_MODE_FOLDER = {
    "mask":     "ducros",
    "isocurve": "isocurve",
}


def _load_config() -> dict:
    with open(_CFG_PATH, "rb") as fh:
        cfg = tomllib.load(fh)
    # The FFD drift block is read as cmach["drift"]["ffd"], i.e. it must be
    # declared as [run.mach.drift.ffd].  A top-level [drift.ffd] parses fine but
    # never reaches build_ffd_beta, which then silently uses DEFAULT_CFG — so any
    # grid_n / n_cp tuning would be a no-op.  Fail loudly instead.
    if "drift" in cfg:
        raise SystemExit(
            f"ERROR: {_CFG_PATH} declares a top-level [drift] section; it is never read.\n"
            f"       Rename [drift.ffd] → [run.mach.drift.ffd].")
    return cfg


def _mesh_h_tag(mesh_path: str) -> str:
    """'meshes/diamond/diamond_h0.0125.npy' → 'h0.0125'.

    The mesh resolution is a first-class axis of the output tree: without it two
    runs that differ only by mesh land in the same directory and overwrite each
    other's metrics.json and figures.
    """
    stem = os.path.splitext(os.path.basename(mesh_path or ""))[0]
    m = re.search(r"_(h[0-9.]+)$", stem)
    return m.group(1) if m else (stem or "hunknown")


def _mach_output_dir(base_output_dir: str, cmach: dict, case_cfg: dict) -> str:
    case = cmach.get("case", "diamond")          # diamond | bump
    mode = cmach.get("density_mode", "mask")
    if mode in ("screened", "tv"):
        sub = os.path.join("denoising", mode)
    elif mode == "iso":
        sub = os.path.join("iso", "contours")
    elif mode == "iso_hessian":
        hmode = cmach.get("hessian_mode", "det")  # det | eig
        sub = os.path.join("iso", "hessian", hmode)
    elif mode == "field":
        fsrc = cmach.get("field_source", "grad_mach")
        sub = os.path.join("field", fsrc)
    elif mode in _MACH_MODE_FOLDER:
        sub = _MACH_MODE_FOLDER[mode]
    else:
        raise ValueError(f"unknown density_mode {mode!r}")

    # Keep each run's outputs separate: split by mesh resolution, then by drift
    # kind, then tag the smallest annealing γ used (so a new run never
    # overwrites a previous one).
    htag     = _mesh_h_tag(case_cfg.get("mesh", ""))
    drift    = cmach.get("reference_drift", "null") or "null"
    schedule = cmach.get("gamma_sb_schedule") or [cmach.get("gamma_sb", 0.002)]
    gamma_min = min(schedule)
    gtag = f"gmin{gamma_min:g}"
    return os.path.join(base_output_dir, "Mach_interpolation", case, sub,
                        htag, f"drift_{drift}", gtag)


def main():
    import jax
    devs = jax.devices()
    if not any(d.platform == "gpu" for d in devs):
        print(f"ERROR: no GPU found — JAX is running on {devs[0].platform.upper()}. "
              "Re-run without interrupting JAX startup (first ~3 s).")
        raise SystemExit(1)

    cfg  = _load_config()
    run  = cfg.get("run", {})
    case = run.get("case", CASE_MACH)
    base_output_dir = run.get("output_dir", "outputs")

    if case in CASES_1D:
        c1d = run.get("1d", {})
        run_1d_case(
            case,
            output_dir=os.path.join(base_output_dir, "1d_test_case"),
            gamma=c1d.get("gamma", 0.05),
            num_iter=c1d.get("num_iter", 1000),
        )

    elif case in CASES_2D_SYNTH:
        c2d = run.get("2d", {})
        run_2d_case(
            case,
            output_dir=os.path.join(base_output_dir, "2d_test_case"),
            gamma=c2d.get("gamma", 0.05),
            num_iter=c2d.get("num_iter", 1000),
        )

    elif case == CASE_MACH:
        cmach     = run.get("mach", {})
        mach_case = cmach.get("case", "diamond")
        case_cfg  = cmach.get(mach_case, {})
        output_dir = _mach_output_dir(base_output_dir, cmach, case_cfg)

        if cmach.get("debug_ipfp", False):
            _run_debug_ipfp(cmach, case_cfg, output_dir)
        else:
            run_mach_interpolation_case(
                bundle_path0=case_cfg["bundle0"],
                bundle_path1=case_cfg["bundle1"],
                mesh_path=case_cfg["mesh"],
                output_dir=output_dir,
                gamma_sb=cmach.get("gamma_sb", 0.002),
                gamma_sb_schedule=cmach.get("gamma_sb_schedule") or None,
                num_iter=cmach.get("num_iter", 4000),
                sensor_p=cmach.get("sensor_p", 2),
                n_frames=cmach.get("n_frames", 11),
                anderson_m=cmach.get("anderson_m", 5),
                ref_bundle_paths=case_cfg.get("ref_bundles") or None,
                density_mode=cmach.get("density_mode", "mask"),
                density_pct=cmach.get("density_pct", 92.0),
                density_decay=cmach.get("density_decay", 0.05),
                tv_weight=cmach.get("tv_weight", 1.0),
                density_target=cmach.get("density_target", "unit"),
                iso_n_contours=cmach.get("iso_n_contours", 6),
                iso_ducros_pct=cmach.get("iso_ducros_pct", 50.0),
                hessian_lp=cmach.get("hessian_lp", 2.0),
                hessian_mode=cmach.get("hessian_mode", "det"),
                field_source=cmach.get("field_source", "pert_mach"),
                field_floor_pct=cmach.get("field_floor_pct", 50.0),
                isocurve_order=cmach.get("isocurve_order", 1),
                isocurve_sigma=cmach.get("isocurve_sigma", 0.0),
                smooth_heat=cmach.get("smooth_heat", True),
                smooth_gamma=cmach.get("smooth_gamma", 0.1),
                smooth_t=cmach.get("smooth_t", 0.005),
                ipfp_cfl=cmach.get("ipfp_cfl", 0.8),
                ipfp_tol=cmach.get("ipfp_tol", 1e-7),
                mach0_inlet=case_cfg.get("mach0_inlet"),
                mach1_inlet=case_cfg.get("mach1_inlet"),
                compute_w2=cmach.get("compute_w2", True),
                wass_gamma=cmach.get("wass_gamma", 0.005),
                wass_iter=cmach.get("wass_iter", 50),
                reference_drift=cmach.get("reference_drift", "null"),
                drift_sigma_w=cmach.get("drift_sigma_w", 0.2),
                drift_gamma_gas=cmach.get("drift_gamma_gas", 1.4),
                drift_cfl_adv=cmach.get("drift_cfl_adv", 0.5),
                drift_ffd_cfg = cmach.get("drift", {}).get("ffd", {})
            )

    else:
        raise ValueError(f"Unknown case {case!r}. Must be one of {ALL_CASES}.")


def _run_debug_ipfp(cmach: dict, case_cfg: dict, output_dir: str):
    """Plain Sinkhorn with per-iteration residuals — use to calibrate num_iter."""
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from Euler.jax_fvm.src.mesh import Mesh
    from . import heat_solver
    from .resolution import apply_IPFP_2d_debug
    from .utils import transform_to_shock_density

    gamma_sb = cmach.get("gamma_sb", 0.002)
    sensor_p = cmach.get("sensor_p", 2)
    num_iter = cmach.get("num_iter", 4000)

    mesh = Mesh()
    mesh.load_mesh(case_cfg["mesh"])

    data0  = np.load(case_cfg["bundle0"]);  data1 = np.load(case_cfg["bundle1"])
    prims0 = data0["primitives"].astype(float)
    prims1 = data1["primitives"].astype(float)
    data0.close();  data1.close()

    n_steps = heat_solver.compute_n_steps(mesh, gamma_sb, CFL=0.8)
    # Fixed smoothing: σ=0.032 < |d|=0.104 so densities are distinct
    smooth_gamma, smooth_t = 0.1, 0.015
    n_steps_smooth = heat_solver.compute_n_steps(mesh, smooth_gamma, t_target=smooth_t)
    print(f"FVM steps / IPFP solve:    {n_steps}")
    print(f"FVM steps / mask smoothing: {n_steps_smooth}  (σ={0.032:.3f})")

    mu0, _, _ = transform_to_shock_density(
        mesh, prims0, n_steps_smooth, p=sensor_p,
        smooth_gamma=smooth_gamma, smooth_t=smooth_t)
    mu1, _, _ = transform_to_shock_density(
        mesh, prims1, n_steps_smooth, p=sensor_p,
        smooth_gamma=smooth_gamma, smooth_t=smooth_t)

    EPS = 1e-12
    f, g, residuals = apply_IPFP_2d_debug(
        jnp.log(jnp.maximum(mu0, EPS)),
        jnp.log(jnp.maximum(mu1, EPS)),
        gamma_sb, mesh, n_steps=n_steps,
        num_iter=num_iter, tol=1e-5, print_every=25)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(residuals, label="Sinkhorn residual")
    ax.axhline(1e-4, color="red",    ls="--", label="CDI target (1e-4)")
    ax.axhline(1e-3, color="orange", ls="--", label="minimum (1e-3)")
    ax.set_xlabel("iteration");  ax.set_ylabel("‖Δf‖∞")
    ax.set_title(f"Plain Sinkhorn convergence  (γ={gamma_sb})")
    ax.legend();  ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "ipfp_convergence_plain.png")
    plt.savefig(out, dpi=200, bbox_inches="tight");  plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()