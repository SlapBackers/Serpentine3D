"""Points to snap to on a mesh: its corners and the centres of its holes.

An imported STL has no CAD edges, only triangles, so it used to offer no
snap points at all: nothing to draw a line from, no three points to put a
plane through. But a mesh of a made object has the edges that were drawn
on it, as folds where neighbouring triangles meet at an angle, and those
are already found for drawing its outline (MeshShape.feature_edges).

A corner is where three or more such edges meet, or where an outline turns
sharply. A hole or a round boss is a closed loop of short, evenly turning
edges lying in one plane and on one circle; its centre is a point the mesh
never had. Every command gets its points through the snaps, so all of them
can use these.
"""

from __future__ import annotations

import numpy as np

from .mesh import MeshShape

_CORNER_DEG = 30.0           # an outline turning more than this is a corner
_MIN_LOOP_POINTS = 8
_ROUND = 1e-3                # how far off its circle a loop's point may be,
                             # as a share of the radius


def mesh_snap_points(mesh: MeshShape) -> list:
    """[(point, kind)]: corners as "end", hole and boss centres as
    "center"."""
    segments = mesh.feature_edges()
    if not len(segments):
        return []
    ends = np.asarray(segments, float).reshape(-1, 3)
    points, ids = np.unique(ends, axis=0, return_inverse=True)
    ids = np.asarray(ids).reshape(-1, 2)
    ids = ids[ids[:, 0] != ids[:, 1]]
    degree = np.bincount(ids.ravel(), minlength=len(points))
    out = [(tuple(map(float, points[i])), "end")
           for i in np.flatnonzero(_corners(points, ids, degree))]
    out += [(tuple(map(float, c)), "center")
            for c in _round_loop_centres(points, ids, degree)]
    return out


def _corners(points, ids, degree) -> np.ndarray:
    corner = (degree != 2) & (degree > 0)
    # where exactly two edges meet, a corner if they turn sharply
    order = np.argsort(ids.ravel(), kind="stable")
    vertex = ids.ravel()[order]
    other = ids[:, ::-1].ravel()[order]
    two = np.flatnonzero(degree == 2)
    first = np.searchsorted(vertex, two)
    a = points[other[first]] - points[two]
    b = points[other[first + 1]] - points[two]
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
    # straight on is cos = -1; a turn of more than the limit is a corner
    corner[two] = cos > -np.cos(np.radians(_CORNER_DEG))
    return corner


def _round_loop_centres(points, ids, degree) -> list:
    """Centres of closed loops of edges that lie on one circle."""
    neighbours = {}
    for a, b in ids.tolist():
        if degree[a] == 2 and degree[b] == 2:
            neighbours.setdefault(a, []).append(b)
            neighbours.setdefault(b, []).append(a)
    seen = set()
    centres = []
    for start, nbrs in neighbours.items():
        if start in seen or len(nbrs) != 2:
            continue
        loop = [start]
        seen.add(start)
        previous, at = start, nbrs[0]
        closed = False
        while True:
            if at == start:
                closed = True
                break
            if at in seen or len(neighbours.get(at, ())) != 2:
                break
            seen.add(at)
            loop.append(at)
            nxt = neighbours[at]
            previous, at = at, nxt[0] if nxt[1] == previous else nxt[1]
        if closed and len(loop) >= _MIN_LOOP_POINTS:
            centre = _circle_centre(points[loop])
            if centre is not None:
                centres.append(centre)
    return centres


def _circle_centre(pts):
    mean = pts.mean(axis=0)
    q = pts - mean
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    size = float(np.linalg.norm(q, axis=1).max())
    # flat: its points stray from their plane, on average, by no more than
    # the share of its size a loop may stray from its circle
    if size == 0 or s[2] / len(pts) ** 0.5 > _ROUND * size:
        return None
    u, v = vt[0], vt[1]
    x, y = q @ u, q @ v
    a = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    (cx, cy, c), *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    r = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    if r == 0:
        return None
    miss = np.abs(np.hypot(x - cx, y - cy) - r).max()
    if miss > _ROUND * r:
        return None
    return mean + cx * u + cy * v
