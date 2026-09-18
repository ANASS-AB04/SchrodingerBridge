"""Utilitaires communs aux modules de maillage (airfoils, diamond, bump)."""
from dataclasses import dataclass
import numpy as np
import meshpy.triangle as triangle
from scipy.spatial import cKDTree
from pathlib import Path
import sys

# Ajout du répertoire euler/ pour permettre l'import de jax_fvm
_euler_dir = str(Path(__file__).resolve().parents[1])
if _euler_dir not in sys.path:
    sys.path.insert(0, _euler_dir)

from jax_fvm.src.mesh import Mesh

WALL, INLET, OUTLET = 2, 3, 4


@dataclass(frozen=True)
class MeshSizeParams:
    # Surface refinement: the airfoil contour is sampled down to h/h_min_factor
    # where curvature demands it (the leading edge).  At the default 5.0 this
    # gives naca0012 a 4.5x cell-size SPREAD, and SB's explicit diffusion kernel
    # must then size its timestep on the smallest cell (cfl_pct=0) or go unstable
    # -- which costs ~20x in kernel steps for both the bridge and W2.  Raising
    # this trades boundary-layer resolution for a far cheaper SB solve.
    h_min_factor:       float = 5.0
    growth_rate:        float = 0.10
    left_growth_rate:   float = 0.25
    obstacle_factor:    float = 0.0
    max_size_factor:    float = 6.0
    left_margin_factor: float = 5.0
    # Nose refinement (0 = off: the meshes every existing bundle was solved on).
    # A rounded LE in supersonic flow carries a DETACHED bow shock whose standoff is
    # only 0.01-0.04 chord (Billig).  The curvature sizing above floors volume cells
    # at h/5 and grows back to h within one h of the wall, so at h0.025 the whole
    # standoff spans 1.3-5 cells: the captured shock snaps between cell rows as Mach
    # varies and the reference C_D/C_L jump by up to 18% (naca0012, 2026-09).
    # nose_factor > 0 additionally writes a nested SOLVER mesh (see
    # build_nose_solver_mesh): cells within nose_radius (chord units) of the LE are
    # bisected down to a longest edge of h/nose_factor, relaxing at nose_growth per
    # unit distance beyond it.  The standard mesh is unchanged.
    nose_factor:        float = 0.0
    nose_radius:        float = 0.06
    nose_growth:        float = 0.15


def sample_segment(p0, p1, target_size_func, min_step) -> np.ndarray:
    """Points adaptatifs le long d'un segment selon target_size_func"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length == 0.0:
        return np.empty((0, 2), dtype=float)

    direction = delta / length
    samples = [p0]
    s = 0.0
    while s < length:
        midpoint = p0 + direction * min(s + 0.5 * min_step, length)
        step = max(min_step, float(target_size_func(midpoint)))
        next_s = min(length, s + step)
        if next_s >= length or (length - next_s) < 0.5 * min_step:
            break
        samples.append(p0 + direction * next_s)
        s = next_s
    return np.asarray(samples, dtype=float)


def triangle_area_from_h(h: float) -> float:
    return np.sqrt(3.0) * h**2 / 4.0


def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def _menger_curvature(pts: np.ndarray) -> np.ndarray:
    """Courbure de Menger discrète pour raffinement adaptatif au bord d'attaque."""
    n = len(pts)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        cross = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
        ab, bc, ca = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)
        denom = ab * bc * ca
        kappa[i] = 2.0 * cross / denom if denom > 1e-20 else 0.0
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def _sample_half_contour(half_pts: np.ndarray, h_body: float, h_min: float) -> np.ndarray:
    """Échantillonnage adaptatif d'une demi-surface de profil selon la courbure."""
    diffs = np.diff(half_pts, axis=0)
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    total = arclen[-1]

    # Courbure évaluée sur un ré-échantillonnage à pas d'arc uniforme
    n_uni = max(int(total / (0.5 * h_min)) + 1, 8)
    s_uni = np.linspace(0.0, total, n_uni)
    xy_uni = np.column_stack([np.interp(s_uni, arclen, half_pts[:, 0]),
                              np.interp(s_uni, arclen, half_pts[:, 1])])
    kappa = _menger_curvature(xy_uni)

    pts, s = [], 0.0
    while s < total:
        pts.append([float(np.interp(s, arclen, half_pts[:, 0])),
                    float(np.interp(s, arclen, half_pts[:, 1]))])
        k = max(float(np.interp(s, s_uni, kappa)), 1e-10)
        s += float(np.clip(h_body / k * 2, h_min, h_body))
    return np.array(pts)


def sample_airfoil_boundary(dense: np.ndarray, n_half: int, chord: float, x0: float, y0: float, h: float,
                            h_min_factor: float = 5.0):
    """Frontière discrète d'un profil (extrados + BF + intrados) selon la courbure et la taille h"""
    h_body = h * 2.0 / 3.0
    h_min = h / h_min_factor
    upper = dense[:n_half]
    lower = np.concatenate([dense[:1], dense[n_half:][::-1], dense[n_half - 1:n_half]])
    upper_pts = _sample_half_contour(upper, h_body, h_min)
    lower_pts = _sample_half_contour(lower, h_body, h_min)
    te_pt = np.array([[x0 + chord, y0]])
    # Unique point de jonction TE si le dernier point de chaque demi-surface est proche du TE
    if len(upper_pts) > 1 and np.linalg.norm(upper_pts[-1] - te_pt[0]) < 0.5 * h_body:
        upper_pts = upper_pts[:-1]
    if len(lower_pts) > 1 and np.linalg.norm(lower_pts[-1] - te_pt[0]) < 0.5 * h_body:
        lower_pts = lower_pts[:-1]
    pts = np.concatenate([upper_pts, te_pt, lower_pts[::-1][:-1]])
    return pts, [WALL] * len(pts)


def local_size_kdtree(point, tree, cx: float, chord: float, max_thickness: float, 
                    h: float, size_params: MeshSizeParams, kappa_array=None) -> float:
    x_le = cx - chord / 2.0
    px = float(point[0])
    p_arr = np.array(point, dtype=np.float64)
    dist, idx = tree.query(p_arr)
    dist = float(dist)

    if dist <= size_params.obstacle_factor * max_thickness:
        return float(h)

    growth_length = max(max_thickness, h)

    if kappa_array is not None:
        k = max(float(kappa_array[idx]), 1e-10)
        h_surf = float(np.clip(h / k, h / 5.0, h * 2.0 / 3.0))
        h_base = h_surf + min(1.0, dist / h) * (h - h_surf)
    else:
        h_base = h

    left_gap = x_le - px
    left_transition = max(size_params.left_margin_factor * max_thickness, 4.0 * h)
    left_blend = smoothstep(left_gap / left_transition)
    growth_rate = size_params.growth_rate + left_blend * (size_params.left_growth_rate - size_params.growth_rate)
    max_size_factor = size_params.max_size_factor * (1.0 + 3.0 * left_blend)

    target = h_base * (1.0 + growth_rate * dist / growth_length)
    return float(np.clip(target, h_base, max_size_factor * h))


def make_airfoil_refinement(tree, cx: float, cy: float, chord: float, max_thickness: float, h: float,
       size_params: MeshSizeParams, kappa_array=None):
    """Retourne la fonction de raffinement pour meshpy (profils naca0012/rae2822)"""
    def refinement_func(vertices, area):
        centroid = np.mean([(v.x, v.y) for v in vertices], axis=0)
        target_h = local_size_kdtree(centroid, tree, cx, chord, max_thickness, h, size_params, kappa_array)
        return bool(area > triangle_area_from_h(target_h))
    return refinement_func

def build_outer_boundary(Lx: float, Ly: float, size_func, h: float):
    """Points et marqueurs sur le contour rectangulaire du domaine"""
    outer = [(0.0, 0.0), (Lx, 0.0), (Lx, Ly), (0.0, Ly)]
    pts, ms, last = [], [], None
    for p0, p1, marker in [
        (outer[0], outer[1], OUTLET),
        (outer[1], outer[2], OUTLET),
        (outer[2], outer[3], OUTLET),
        (outer[3], outer[0], INLET),
    ]:
        seg = sample_segment(p0, p1, size_func, min_step=h)
        if last is not None and len(seg) > 0 and np.allclose(seg[0], last):
            seg = seg[1:]
        pts.append(seg)
        ms.extend([marker] * len(seg))
        if len(seg) > 0:
            last = seg[-1]
    return np.concatenate(pts), ms


def populate_mesh_from_triangle(mesh, raw_mesh):
    mesh.mesh_generator_from_points(
        raw_mesh.points, raw_mesh.elements,
        np.roll(np.asarray(raw_mesh.neighbors), 1, axis=-1),
        raw_mesh.faces, raw_mesh.face_markers,
    )
    return mesh


def triangulate_with_hole(outer_pts, outer_ms, obstacle_pts, obstacle_ms, hole_pt, refinement_func) -> Mesh:
    """Assemble la triangulation Delaunay avec obstacle creux"""
    all_pts = np.concatenate([outer_pts, obstacle_pts])
    off = len(outer_pts)
    mesh = Mesh()
    facets = (mesh.round_trip_connect(0, off - 1) + mesh.round_trip_connect(off, off + len(obstacle_pts) - 1))

    info = triangle.MeshInfo()
    info.set_points([tuple(pt) for pt in all_pts])
    info.set_facets(facets, facet_markers=list(outer_ms) + list(obstacle_ms))
    info.set_holes([hole_pt])

    raw_mesh = triangle.build(
        info, refinement_func=refinement_func,
        min_angle=30, generate_faces=True, generate_neighbor_lists=True,
    )
    return populate_mesh_from_triangle(Mesh(), raw_mesh)


DEFAULT_MESH_DIR = Path(__file__).resolve().parents[2] / "data" / "meshes"


def build_airfoil_mesh(case, dense, t_max, hole_pt, *, Lx, Ly, h, chord, cx, cy,
                       symmetric, size_params, extra_metadata,
                       out_dir=None, export_vtk=False):
    x_le = cx - chord / 2.0
    x_te = cx + chord / 2.0
    tree = cKDTree(dense)
    kappa = _menger_curvature(dense)
    size_func = lambda pt: local_size_kdtree(pt, tree, cx, chord, t_max, h, size_params, kappa)
    refinement = make_airfoil_refinement(tree, cx, cy, chord, t_max, h, size_params, kappa)

    if symmetric:
        airfoil_upper = sample_airfoil_upper(dense, 2000, chord, x_le, cy, h,
                                             size_params.h_min_factor)
        mesh = triangulate_symmetric_airfoil(Lx, Ly, cy, x_le, x_te, airfoil_upper,
                                             size_func, refinement, h)
    else:
        outer_pts, outer_ms = build_outer_boundary(Lx, Ly, size_func, h)
        airfoil_pts, airfoil_ms = sample_airfoil_boundary(dense, 2000, chord, x_le, cy, h,
                                                          size_params.h_min_factor)
        mesh = triangulate_with_hole(outer_pts, outer_ms, airfoil_pts, airfoil_ms, hole_pt, refinement)

    mesh.set_metadata(case=case, h=h, domain={"Lx": Lx, "Ly": Ly},
                      obstacle_length=chord, chord=chord, **extra_metadata)
    mesh.print_statistics()

    out_dir = DEFAULT_MESH_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"h{h}"
    if size_params.nose_factor > 0.0:
        tag = f"h{h}le{size_params.nose_factor:g}"
        solver, parent = build_nose_solver_mesh(mesh, dense, chord, h, size_params, Lx, Ly)
        solver_name = f"{case}_{tag}_solver.npy"
        # The standard mesh keeps its cells; it only learns its tag (so results go
        # to results/<case>/<tag>/, never into the h0.025 corpus) and where its
        # solver mesh lives.  The solver mesh carries the child->parent map.
        mesh.set_metadata(h_tag=tag, solver_mesh=solver_name)
        solver.set_metadata(**{k: v for k, v in mesh.metadata.items() if k != "solver_mesh"},
                            parent=parent, coarse_n_cells=int(len(mesh.tris)),
                            nose_factor=size_params.nose_factor,
                            nose_radius=size_params.nose_radius,
                            nose_growth=size_params.nose_growth)
        solver.save_mesh(str(out_dir / solver_name))
        print(f"Solver mesh : {out_dir / solver_name}  "
              f"({len(mesh.tris)} -> {len(solver.tris)} cells)")
    path = out_dir / f"{case}_{tag}.npy"
    mesh.save_mesh(str(path))
    if export_vtk:
        mesh.export_vtk(str(path.with_suffix(".vtk")))
    print(f"Mesh saved : {path}")
    return mesh, path


# Maillage symétrique : maillage supérieur puis miroir sur l'axe y=cy

def sample_airfoil_upper(dense: np.ndarray, n_half: int, chord: float, x0: float, y0: float, h: float,
                         h_min_factor: float = 5.0):
    """Surface supérieure LE→BF seule (extrémités sur l'axe y0), pour maillage symétrique."""
    h_body = h * 2.0 / 3.0
    h_min = h / h_min_factor
    upper_pts = _sample_half_contour(dense[:n_half], h_body, h_min)
    te_pt = np.array([[x0 + chord, y0]])
    if len(upper_pts) > 1 and np.linalg.norm(upper_pts[-1] - te_pt[0]) < 0.5 * h_body:
        upper_pts = upper_pts[:-1]
    return np.concatenate([upper_pts, te_pt])


def _mirror_upper_half(pts_u: np.ndarray, tris_u: np.ndarray, cy: float, tol: float = 1e-9):
    """Reflète le demi-maillage supérieur sous l'axe y=cy"""
    on_axis = np.abs(pts_u[:, 1] - cy) < tol
    n_u = len(pts_u)
    mir = np.arange(n_u)
    lower = []
    nxt = n_u
    for i in range(n_u):
        if not on_axis[i]:
            mir[i] = nxt
            nxt += 1
            lower.append((pts_u[i, 0], 2.0 * cy - pts_u[i, 1]))
    points = np.vstack([pts_u, np.array(lower, dtype=float).reshape(-1, 2)])
    tris_l = mir[tris_u[:, ::-1]]
    tris = np.vstack([tris_u, tris_l]).astype(np.int32)
    return points, tris


def _faces_neighbors_from_tris(tris: np.ndarray):
    """Reconstruit faces (arêtes uniques), voisins (convention arête i=(v_i,v_i+1)) et masque de bord."""
    from collections import defaultdict
    edge_map = defaultdict(list)
    for t, tri in enumerate(tris):
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge_map[(a, b) if a < b else (b, a)].append((t, i))
    n_tris = len(tris)
    neighbors = -np.ones((n_tris, 3), dtype=np.int32)
    faces = np.empty((len(edge_map), 2), dtype=np.int32)
    boundary = np.zeros(len(edge_map), dtype=bool)
    for fidx, (key, owners) in enumerate(edge_map.items()):
        faces[fidx] = key
        if len(owners) == 2:
            (t0, i0), (t1, i1) = owners
            neighbors[t0, i0] = t1
            neighbors[t1, i1] = t0
        else:
            boundary[fidx] = True
    return faces, neighbors, boundary


def _classify_face_markers(points, faces, boundary, Lx, Ly, tol=1e-6):
    """Marqueurs de bord par géométrie : gauche=INLET, autres bords rect.=OUTLET, sinon WALL."""
    markers = np.zeros(len(faces), dtype=np.int32)
    for f in np.where(boundary)[0]:
        (xa, ya), (xb, yb) = points[faces[f, 0]], points[faces[f, 1]]
        if xa < tol and xb < tol:
            markers[f] = INLET
        elif ((xa > Lx - tol and xb > Lx - tol) or (ya < tol and yb < tol)
              or (ya > Ly - tol and yb > Ly - tol)):
            markers[f] = OUTLET
        else:
            markers[f] = WALL
    return markers


def triangulate_symmetric_airfoil(Lx, Ly, cy, x_le, x_te, airfoil_upper,size_func, refinement_func, h) -> Mesh:
    """Maille la moitié supérieure puis reflete"""
    seg_l = sample_segment((0.0, cy), (x_le, cy), size_func, min_step=h)
    seg_r = sample_segment((x_te, cy), (Lx, cy), size_func, min_step=h)
    right = sample_segment((Lx, cy), (Lx, Ly), size_func, min_step=h)
    top = sample_segment((Lx, Ly), (0.0, Ly), size_func, min_step=h)
    left = sample_segment((0.0, Ly), (0.0, cy), size_func, min_step=h)
    loop = np.concatenate([seg_l, airfoil_upper, seg_r[1:], right, top, left])

    mesh = Mesh()
    facets = mesh.round_trip_connect(0, len(loop) - 1)
    info = triangle.MeshInfo()
    info.set_points([tuple(pt) for pt in loop])
    info.set_facets(facets)
    raw = triangle.build(
        info, refinement_func=refinement_func,
        min_angle=30, generate_faces=True, generate_neighbor_lists=True,
    )

    points, tris = _mirror_upper_half(np.asarray(raw.points), np.asarray(raw.elements), cy)
    faces, neighbors, boundary = _faces_neighbors_from_tris(tris)
    markers = _classify_face_markers(points, faces, boundary, Lx, Ly)

    out = Mesh()
    out.mesh_generator_from_points(points, tris, neighbors, faces, markers)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Nose refinement: a NESTED solver mesh for rounded leading edges
# ─────────────────────────────────────────────────────────────────────────────
#
# The Euler solve needs the bow-shock standoff resolved, but SB must never see the
# small cells: its diffusion kernel is explicit and sizes its timestep on the
# smallest cell, so a nose-refined mesh would cost it ~21x per run at nose_factor=10.
# Hence the refined mesh is only ever a SOLVER mesh.  It is built by bisecting cells
# of the standard mesh, so every fine cell has exactly one standard-mesh ancestor,
# and Euler/main.py exports the area-weighted average over children onto the
# standard mesh.  Cells outside the nose zone are never split, so there the bundle
# is exactly what a solve on those cells produces.

def _project_to_polyline(pt, poly, tree):
    """Nearest point to pt on the closed polyline poly (a dense airfoil contour)."""
    i = int(tree.query(pt)[1])
    n = len(poly)
    best, best_d2 = poly[i], float(np.sum((poly[i] - pt) ** 2))
    for j in ((i - 1) % n, (i + 1) % n):
        a, b = poly[i], poly[j]
        ab = b - a
        t = float(np.clip(np.dot(pt - a, ab) / max(float(np.dot(ab, ab)), 1e-300), 0.0, 1.0))
        q = a + t * ab
        d2 = float(np.sum((q - pt) ** 2))
        if d2 < best_d2:
            best, best_d2 = q, d2
    return best


def refine_longest_edge(points, tris, target, wall_edges=(), wall_proj=None, max_passes=60):
    """Conforming longest-edge bisection (Rivara) until every triangle's longest
    edge is <= target(centroid).

    ``target`` maps an (K,2) array of centroids to (K,) sizes.  Wall edges that get
    split have their midpoint moved by ``wall_proj`` onto the true contour.
    Returns (points, tris, parent) with parent[i] the input triangle that fine
    triangle i descends from.  "Longest" is decided on (length, vertex ids), a
    strict total order, so the neighbour recursion always terminates -- even on
    exact length ties, which mirrored meshes produce.
    """
    P = [(float(x), float(y)) for x, y in np.asarray(points, dtype=float)]
    T = [[int(v) for v in t] for t in np.asarray(tris)]
    parent = list(range(len(T)))

    def key(u, v):
        return (u, v) if u < v else (v, u)

    edge_tris = {}
    def attach(t):
        a, b, c = T[t]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault(key(u, v), []).append(t)
    def detach(t):
        a, b, c = T[t]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_tris[key(u, v)].remove(t)
    for t in range(len(T)):
        attach(t)
    wall = {key(int(u), int(v)) for u, v in wall_edges}

    def length(u, v):
        return float(np.hypot(P[u][0] - P[v][0], P[u][1] - P[v][1]))
    def longest(t):
        a, b, c = T[t]
        return max((length(u, v), key(u, v)) for u, v in ((a, b), (b, c), (c, a)))[1]

    def bisect(t):
        e = longest(t)
        while True:
            nb = [s for s in edge_tris[e] if s != t]
            if not nb or longest(nb[0]) == e:
                break
            # the neighbour's longest edge is strictly longer than e, so refining it
            # first never touches t; its child that inherits e is re-checked
            bisect(nb[0])
        u, v = e
        m = len(P)
        mid = np.array([(P[u][0] + P[v][0]) / 2, (P[u][1] + P[v][1]) / 2])
        if e in wall:
            if wall_proj is not None:
                mid = wall_proj(mid)
            wall.discard(e)
            wall.update((key(u, m), key(v, m)))
        P.append((float(mid[0]), float(mid[1])))
        for s in [t] + nb:
            detach(s)
            old = T[s]
            T[s] = [m if x == u else x for x in old]        # replacing one end of the
            T.append([m if x == v else x for x in old])     # split edge keeps CCW order
            parent.append(parent[s])
            attach(s)
            attach(len(T) - 1)

    for _ in range(max_passes):
        V = np.asarray(P)[np.asarray(T)]
        L = np.hypot(*(V - np.roll(V, -1, axis=1)).transpose(2, 0, 1)).max(axis=1)
        todo = np.flatnonzero(L > target(V.mean(axis=1)))
        if todo.size == 0:
            return np.asarray(P), np.asarray(T, dtype=np.int32), np.asarray(parent)
        for t in todo:
            a, b, c = T[t]                 # may already have been split this pass
            cen = np.array([[(P[a][0] + P[b][0] + P[c][0]) / 3, (P[a][1] + P[b][1] + P[c][1]) / 3]])
            if max(length(a, b), length(b, c), length(c, a)) > float(target(cen)[0]):
                bisect(int(t))
    raise RuntimeError(f"longest-edge refinement did not converge in {max_passes} passes")


def build_nose_solver_mesh(mesh, dense, chord, h, size_params, Lx, Ly):
    """Nested nose refinement of ``mesh``; returns (solver_mesh, parent)."""
    le = dense[int(np.argmin(dense[:, 0]))]
    h_nose = h / size_params.nose_factor
    radius = size_params.nose_radius * chord
    def target(xy):
        d = np.hypot(xy[:, 0] - le[0], xy[:, 1] - le[1])
        return h_nose + size_params.nose_growth * np.maximum(0.0, d - radius)
    faces, markers = np.asarray(mesh.faces), np.asarray(mesh.face_markers)
    tree = cKDTree(dense)
    points, tris, parent = refine_longest_edge(
        np.asarray(mesh.points), np.asarray(mesh.tris), target,
        wall_edges=[tuple(f) for f in faces[markers == WALL]],
        wall_proj=lambda q: _project_to_polyline(q, dense, tree))
    faces_f, neighbors, boundary = _faces_neighbors_from_tris(tris)
    solver = Mesh()
    solver.mesh_generator_from_points(points, tris, neighbors, faces_f,
                                      _classify_face_markers(points, faces_f, boundary, Lx, Ly))
    return solver, parent
