from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")
repo_root = Path(__file__).resolve().parents[1]


def load_default_config(config_path: str | Path | None = None):
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["export"] = deepcopy(config.get("export", {}))
    return config


def build_config_from_cli(default_cfg):
    parser = ArgumentParser(description="Simulation supersonique avec Euler 2D")
    parser.add_argument("--case", choices=("bump", "diamond", "naca0012", "naca2412", "rae2822", "oneraD", "oa209"), default=default_cfg["case"], help="Cas test : bump, diamond, naca0012, naca2412, rae2822, oneraD ou oa209")
    parser.add_argument("--mach", type=float, default=default_cfg["Mach"], help="Nombre de Mach du flux entrant")
    parser.add_argument("--mesh-path", type=str, default=default_cfg["mesh_path"], help="Chemin vers le fichier de maillage (.npy)")
    parser.add_argument("--time-scheme", choices=("EE", "RK2", "RK4", "SRK2", "SSP_RK2"), default=default_cfg["time_scheme"], help="Schéma temporel")
    parser.add_argument("--flux", choices=("Rusanov", "Tadmor", "AUSM", "Roe", "HLLC"), default=default_cfg["flux"], help="Solveur de flux numérique")
    parser.add_argument("--reconstruction", choices=("constant", "MUSCL", "muscl"), default=default_cfg["reconstruction"], help="Type de reconstruction")
    parser.add_argument("--aoa", type=float, default=default_cfg.get("aoa", 0.0), help="Angle d'attaque en degrés (utilisé pour le cas diamond)")
    parser.add_argument("--stationarity-threshold", type=float, default=default_cfg.get("stationarity_threshold", 1e-6), help="Seuil d'arrêt sur le résidu relatif")
    parser.add_argument("--sponge-width-frac", type=float,
                        default=default_cfg.get("sponge_width_frac"),
                        help="Largeur de la sponge en fraction de Ly (defaut config.toml)")
    parser.add_argument("--sponge-strength-frac", type=float,
                        default=default_cfg.get("sponge_strength_frac"),
                        help="Force de la sponge en fraction de (U_inf+a_inf)/width")
    parser.add_argument("--stationarity-check-every", type=int, default=default_cfg.get("stationarity_check_every", 25), help="Nombre de pas entre deux tests de stationnarité")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=default_cfg.get("verbose", True), help="Affiche les logs détaillés de la simulation")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Mode concis : uniquement les infos principales au début et le temps de simulation à la fin")
    # ── Warm start: seed the solver from an interpolated snapshot ────────────
    # All default to None, so omitting them reproduces today's behaviour exactly.
    parser.add_argument("--init-state", type=str, default=None,
                        help="Fichier .npz d'état initial (warmstart.npz de SB, ou un bundle Euler) au lieu de l'écoulement uniforme")
    parser.add_argument("--init-method", type=str, default=None,
                        help="Méthode d'interpolation à extraire d'un warmstart.npz (BaryCDI | FFD | Linear)")
    parser.add_argument("--init-t", type=float, default=None,
                        help="Instant d'interpolation à extraire d'un warmstart.npz")
    parser.add_argument("--init-allow-repair", action="store_true", default=False,
                        help="Autorise un état initial non admissible (rho<=0 ou p<=0) après clamp — le chronométrage obtenu n'est alors pas propre")
    parser.add_argument("--dt", type=float, default=None,
                        help="Impose le pas de temps au lieu de le déduire de l'état initial (comparaison équitable entre plusieurs états initiaux)")
    parser.add_argument("--out-tag", type=str, default=None,
                        help="Sous-répertoire + suffixe de nom de fichier, pour ne pas écrire dans le corpus results/<case>/<h>/")
    parser.add_argument("--save-convergence", action="store_true", default=False,
                        help="Exporte la courbe de stationnarité (rel vs pas) : permet de relire 'pas pour atteindre la tolérance X' a posteriori, pour tout X")
    parser.add_argument("--summary", dest="summary", action="store_true", default=None,
                        help="Force l'export du résumé JSON (C_D, C_L, wall time, stopping_step) même si export.summary=false dans config.toml")
    args = parser.parse_args()

    cfg = deepcopy(default_cfg)
    cfg["case"] = args.case
    cfg["Mach"] = args.mach
    cfg["mesh_path"] = args.mesh_path
    cfg["time_scheme"] = "SRK2" if args.time_scheme == "SSP_RK2" else args.time_scheme
    cfg["flux"] = args.flux
    cfg["reconstruction"] = "MUSCL" if str(args.reconstruction).lower() == "muscl" else args.reconstruction
    cfg["aoa"] = float(args.aoa)
    cfg["stationarity_threshold"] = float(args.stationarity_threshold)
    # None => fall through to helper.get_sponge_source's own defaults.
    cfg["sponge_width_frac"] = (None if args.sponge_width_frac is None
                                else float(args.sponge_width_frac))
    cfg["sponge_strength_frac"] = (None if args.sponge_strength_frac is None
                                   else float(args.sponge_strength_frac))
    cfg["stationarity_check_every"] = max(1, int(args.stationarity_check_every))
    cfg["verbose"] = bool(args.verbose)
    cfg["init_state"] = args.init_state
    cfg["init_method"] = args.init_method
    cfg["init_t"] = args.init_t
    cfg["init_allow_repair"] = bool(args.init_allow_repair)
    cfg["dt"] = None if args.dt is None else float(args.dt)
    cfg["out_tag"] = args.out_tag
    cfg["save_convergence"] = bool(args.save_convergence)
    # --summary overrides config.toml's [export].summary, which is false by
    # default.  Without the summary there is no C_D/C_L on disk, so the
    # "did the warm start land on the SAME steady state?" gate cannot run and a
    # speedup number has nothing validating it.
    if args.summary:
        cfg.setdefault("export", {})["summary"] = True
    return cfg


def load_mesh(cfg):
    if cfg["mesh_path"] is None:
        raise ValueError("Le chemin du maillage doit être spécifié via --mesh-path ou dans la config par défaut.")

    try:
        from jax_fvm.src.mesh import Mesh
    except ModuleNotFoundError:
        import sys

        euler_root = Path(__file__).resolve().parent
        if str(euler_root) not in sys.path:
            sys.path.insert(0, str(euler_root))
        from jax_fvm.src.mesh import Mesh

    mesh = Mesh()
    mesh.load_mesh(cfg["mesh_path"])
    return mesh


def load_solver_mesh(mesh, cfg):
    """The mesh to SOLVE on, and the child->parent map back onto ``mesh``.

    A nose-refined tag (e.g. h0.025le7, see meshing/mesh_utils.py) ships two files:
    <case>_<tag>.npy -- the standard mesh the bundles are written on and SB runs
    on -- and <case>_<tag>_solver.npy, a nested refinement of it that resolves the
    leading-edge bow shock.  Returns (mesh, None) for every ordinary mesh.
    """
    name = mesh.metadata.get("solver_mesh")
    if not name:
        return mesh, None
    solver = type(mesh)()
    solver.load_mesh(str(Path(cfg["mesh_path"]).parent / name))
    parent = np.asarray(solver.metadata["parent"])
    n_coarse = int(np.asarray(mesh.tris).shape[0])
    if int(solver.metadata.get("coarse_n_cells", -1)) != n_coarse or parent.max() != n_coarse - 1:
        raise ValueError(f"{name} was not refined from this {n_coarse}-cell mesh")
    return solver, parent


def format_h(mesh):
    # Variant meshes (nose-refined) carry their own tag, so their snapshots land in
    # results/<case>/<tag>/ and can never be mistaken for the plain-h corpus.
    if mesh.metadata.get("h_tag"):
        return str(mesh.metadata["h_tag"])
    h = mesh.metadata.get("h", None)
    if h is None:
        return "h"
    return f"h{float(h):.4f}".rstrip("0").rstrip(".")


def format_condition_tag(cfg):
    mach = f"M{float(cfg.get('Mach')):.2f}"
    if str(cfg.get("case", "")).lower() in ("diamond", "naca0012", "naca2412", "rae2822", "onerad", "oa209"):
        aoa = float(cfg.get("aoa", 0.0))
        return f"AOA{aoa:.2f}_{mach}"
    return mach


def format_snapshot_name(cfg, t):
    condition_tag = format_condition_tag(cfg)
    flux = str(cfg["flux"]).upper()
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    prefix = ""
    name = f"{prefix}{condition_tag}_{flux}_{reconstruction}_{time_scheme}_t{t:.2f}"
    # A warm-start study runs the SAME (case, Mach, AoA, scheme) several times with
    # different seeds.  Nothing above distinguishes them, so without a tag the runs
    # overwrite each other -- and a cold run reaching tf would overwrite the corpus
    # bundle of that Mach, which SB/Config.toml lists as a reference.
    out_tag = cfg.get("out_tag")
    return f"{name}_{out_tag}" if out_tag else name


def format_sample_id(cfg):
    return format_condition_tag(cfg)


def format_subtitle(cfg, mesh, t):
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    if str(cfg.get("case", "")).lower() in ("diamond", "naca0012", "naca2412", "rae2822", "onerad", "oa209"):
        return (
            f"t={t:.2f}s, M={cfg['Mach']}, AoA={float(cfg.get('aoa', 0.0)):.1f}°, "
            f"h={mesh.metadata.get('h', 'n/a')} | Solver : {cfg['flux']}, {reconstruction}, {time_scheme}"
        )
    return (
        f"t={t:.2f}s, M={cfg['Mach']}, h={mesh.metadata.get('h', 'n/a')} "
        f"| Solver : {cfg['flux']}, {reconstruction}, {time_scheme}"
    )


def setup_dirs(cfg, mesh):
    h_dir = format_h(mesh)
    case = str(cfg.get("case", ""))
    # LAYOUT PINNED to results/<case>/<h>/ with the AoA in the FILENAME.
    #
    # EulerSR writes data/raw/<case>/<h>/AOA<a>/ for AoA-carrying cases and
    # euler/results/<case>/<h>/ for the rest.  Adopting that here would:
    #   * orphan the ~700 diamond and bump bundles already on disk,
    #   * break the resume-skip in run_euler_grid_big.sbatch, which globs
    #     results/<case>/<h>/<stem>_*.npz -- so every resubmit re-solves
    #     everything while reporting COMPLETED,
    #   * break SB/make_config_for_mesh.py:find_bundle, which globs the same.
    # The AoA is already in the filename via format_condition_tag, so a
    # per-AoA directory would duplicate it.  Keep one convention.
    # An --out-tag run writes one level DOWN, into warmstart/<tag>/.  The corpus
    # globs are non-recursive (run_euler_grid_big.sbatch's `compgen -G
    # results/<case>/<h>/<stem>_*.npz` and SB/make_config_for_mesh.py:find_bundle),
    # so a subdirectory is invisible to them: benchmark runs cannot be mistaken
    # for corpus bundles, and the stale-check cannot quarantine them.
    out_tag = cfg.get("out_tag")
    sub = Path("warmstart") / str(out_tag) if out_tag else Path(".")
    dirs = {
        "fig": repo_root / "figures" / case / h_dir / sub,
        "res": repo_root / "results" / case / h_dir / sub,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    if cfg.get("verbose", True):
        print(f"Output : {dirs['res']}")
    return dirs


def print_config(cfg, mesh, out_dirs):
    reconstruction = str(cfg["reconstruction"]).upper()
    time_scheme = str(cfg["time_scheme"]).upper()
    if not cfg.get("verbose", True):
        # Mode concis : une seule ligne avec les infos principales
        tag = format_condition_tag(cfg)
        print(
            f"[{cfg['case']}] {tag} | h={mesh.metadata.get('h', 'n/a')} "
            f"| {cfg['flux']}/{reconstruction}/{time_scheme} -> {out_dirs['res']}"
        )
        return
    print("\n" + "-" * 78)
    print("CONFIGURATION SIMULATION")
    print("-" * 78)
    print(f"Cas : {cfg['case']}")
    print(f"Mesh path : {cfg['mesh_path']} (h={mesh.metadata.get('h', 'n/a')})")
    print("-" * 78)
    print("Physique")
    if str(cfg.get("case", "")).lower() in ("diamond", "naca0012", "naca2412", "rae2822", "onerad", "oa209"):
        print(f"  Mach : {cfg['Mach']}, AoA : {float(cfg.get('aoa', 0.0)):.1f}°, gamma={cfg['gamma']}, p_inf={cfg['p_inf']}, rho_inf={cfg['rho_inf']}")
    else:
        print(f"  Mach : {cfg['Mach']}, gamma={cfg['gamma']}, p_inf={cfg['p_inf']}, rho_inf={cfg['rho_inf']}")
    print("-" * 78)
    print("Solveur")
    print(f"  scheme FVM : {cfg['flux']}, {reconstruction}, {time_scheme}")
    print(f"  CFL : {cfg['CFL']}, tf={cfg['tf']}")
    print(f"  stationarity_threshold={cfg.get('stationarity_threshold', 'n/a')} | check_every={cfg.get('stationarity_check_every', 'n/a')}")
    try:
        import jax
        print(f"  Parallélisation JAX : {jax.default_backend()} (device count: {jax.device_count()})")
    except Exception:
        pass
    print("-" * 78)
