"""Génération centralisée des maillages Euler avec JAX-FVM.
Usage :
    uv run python generate_mesh.py                       # tous les cas, résolutions par défaut
    uv run python generate_mesh.py --geom oa209          # un cas, résolutions par défaut
    uv run python generate_mesh.py --geom naca2412 -h 0.1 0.05
    uv run python generate_mesh.py --geom naca0012 -h 0.025 --nose-factor 7
                                     # -> naca0012_h0.025le7.npy (+ _solver.npy)
"""
import argparse
import sys
from math import radians, tan
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))              # jax_fvm
sys.path.insert(0, str(_HERE / "meshing"))  # modules de maillage

import airfoils, diamond, bump
from mesh_utils import MeshSizeParams  # noqa: E402

_AIRFOIL_PARAMS = dict(Lx=4.0, Ly=4.0, chord=1.0, cx=1.5, cy=2.0)
_DEFAULT_H = [0.3, 0.2, 0.15, 0.1, 0.05, 0.025]
_FIG_DIR = _HERE.parent / "data" / "figures" / "meshes"


def _airfoil_builder(name):
    return lambda h, out_dir, nose=0.0: airfoils.build_mesh(
        name, h=h, out_dir=out_dir, size_params=MeshSizeParams(nose_factor=nose), **_AIRFOIL_PARAMS)


def _no_nose(build):
    # diamond has a sharp LE (attached shock) and the bump no LE at all: nothing to refine
    def wrapped(h, out_dir, nose=0.0):
        if nose > 0.0:
            raise SystemExit("--nose-factor only applies to the rounded-LE airfoils")
        return build(h, out_dir)
    return wrapped


GEOMS = {name: (_airfoil_builder(name), _DEFAULT_H) for name in airfoils.AIRFOILS}
GEOMS["diamond"] = (
    _no_nose(lambda h, out_dir: diamond.build_mesh(Lx=4.0, Ly=4.0, h=h, chord=1.0,
                                                   height=tan(radians(5)), cx=1.5, cy=2.0, out_dir=out_dir)),
    _DEFAULT_H,
)
GEOMS["bump"] = (
    _no_nose(lambda h, out_dir: bump.build_mesh(Lx=3.0, Ly=1.0, h=h, thickness=0.04,
                                                chord=1.0, center=1.5, out_dir=out_dir)),
    [0.05, 0.02],
)


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--geom", nargs="*", choices=list(GEOMS), default=list(GEOMS))
    parser.add_argument("-h", "--mesh-size", dest="h", type=float, nargs="+", default=None)
    parser.add_argument("--nose-factor", type=float, default=0.0,
                        help="airfoils: also write a nested solver mesh refined to h/N at the LE "
                             "(tag h<h>leN); 0 = standard mesh only")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fig-dir", default=str(_FIG_DIR))
    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for geom in args.geom:
        builder, default_h = GEOMS[geom]
        for h in (args.h if args.h is not None else default_h):
            mesh, path = builder(h, args.out_dir, args.nose_factor)
            mesh.plot_mesh(filename=str(fig_dir / f"{path.stem}.png"), dpi=500)


if __name__ == "__main__":
    main()
