#!/usr/bin/env bash
# Does the current Euler/ still provide everything SB/ and the sbatch scripts need?
# Run after replacing Euler/ with another repo's version, BEFORE committing.
cd "$(dirname "$0")"
fail=0
chk() {  # $1=description  $2=file  $3=symbol
    printf '  %-52s ' "$1"
    if grep -q -- "$3" "$2" 2>/dev/null; then echo "OK"; else echo "MISSING"; fail=1; fi
}
echo "── API surface SB depends on ─────────────────────────────"
chk "helper.get_force_coefficients_from_pressure (C_D/C_L)" Euler/jax_fvm/src/helper.py get_force_coefficients_from_pressure
chk "helper.get_wall_face_data"                             Euler/jax_fvm/src/helper.py get_wall_face_data
chk "config.format_condition_tag (bump omits AoA)"          Euler/config.py             format_condition_tag
chk "config --stationarity-threshold (sbatch passes 0)"     Euler/config.py             stationarity-threshold
# Mesh generation moved into meshing/ behind generate_mesh.py --geom <name> -h ...
chk "generate_mesh.py --geom (replaces diamond.py/bump.py)" Euler/generate_mesh.py     '"--geom"'
chk "meshing/diamond.py build_mesh"                         Euler/meshing/diamond.py   'def build_mesh'
chk "meshing/bump.py build_mesh"                            Euler/meshing/bump.py      'def build_mesh'
chk "Mesh.load_mesh"                                        Euler/jax_fvm/src/mesh.py   'def load_mesh'
chk "Mesh.save_mesh"                                        Euler/jax_fvm/src/mesh.py   'def save_mesh'

chk "config.toml pins sponge_width_frac"                    Euler/config.toml          sponge_width_frac
chk "config.toml pins sponge_strength_frac"                 Euler/config.toml          sponge_strength_frac
chk "main.py forwards sponge_width kwarg"                   Euler/main.py              'kw\["sponge_width"\]'

echo "── output layout matches where the corpus lives ──────────"
uv run python - <<'PY' || fail=1
import sys
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
import config
class _M: metadata = {"h": 0.025}
# The ~700 existing bundles live in results/<case>/<h>/ with the AoA in the
# FILENAME.  EulerSR writes data/raw/<case>/<h>/AOA<a>/ instead, which orphans
# them, breaks the sbatch resume-skip (every resubmit re-solves everything while
# reporting COMPLETED) and breaks find_bundle.  Assert the layout explicitly.
for case in ("diamond", "bump", "naca0012"):
    d = config.setup_dirs({"case": case, "aoa": 0.0, "verbose": False}, _M())
    rel = d["res"].relative_to(config.repo_root).as_posix()
    assert rel == f"results/{case}/h0.025", f"{case}: results dir is {rel}"
print("  results/<case>/<h>/ for every case (AoA stays in the filename)")
PY

echo "── sponge matches what the existing bundles used ─────────"
uv run python - <<'PY' || fail=1
import sys, tomllib
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
c = tomllib.load(open("Euler/config.toml", "rb"))
w, s = c.get("sponge_width_frac"), c.get("sponge_strength_frac")
# Every snapshot on disk was computed with 0.15 / 0.5.  helper.py's defaults are
# now 0.20 / 1.0, so if the pins go missing the whole corpus silently splits into
# two incompatible halves.
assert w == 0.15, f"sponge_width_frac is {w}, existing bundles used 0.15"
assert s == 0.5,  f"sponge_strength_frac is {s}, existing bundles used 0.5"
print(f"  sponge pinned at width={w}*Ly, strength={s}*(U+a)/w  (matches the corpus)")
PY

echo "── imports actually resolve ──────────────────────────────"
uv run python - <<'PY' || fail=1
import sys
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from Euler.jax_fvm.src.mesh import Mesh
from Euler.jax_fvm.src import plot                                    # noqa: F401
from Euler.jax_fvm.src.helper import get_force_coefficients_from_pressure  # noqa: F401
from Euler.config import format_condition_tag
# bump must NOT carry an AoA field, diamond must:  the bundle globs depend on it
assert format_condition_tag({"Mach": 2.0, "case": "bump"}) == "M2.00", "bump tag changed"
assert format_condition_tag({"Mach": 2.0, "case": "diamond", "aoa": 0.0}) == "AOA0.00_M2.00", "diamond tag changed"
print("  imports OK; snapshot naming unchanged for both cases")
PY

echo "── mesh format still loadable (old bundles must still read) ──"
uv run python - <<'PY' || fail=1
import sys, glob
sys.path.insert(0, "."); sys.path.insert(0, "Euler")
from Euler.jax_fvm.src.mesh import Mesh
for p in ("meshes/diamond/diamond_h0.025.npy", "meshes/bump/bump_h0.025.npy"):
    m = Mesh(); m.load_mesh(p)
    print(f"  {p}: {len(m.tris)} cells, metadata keys {len(m.metadata)}")
PY

echo
[ "$fail" -eq 0 ] && echo "PASS — SB should still run" || echo "FAIL — fix before committing"
exit $fail
