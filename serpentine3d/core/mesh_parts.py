"""The parts of an imported mesh: what an STL of an assembly is made of.

An STL keeps no parts, only triangles. Its pieces (triangles that reach
one another through shared corners) are a start, but a real file has far
more pieces than parts: a product model of 733,000 triangles measured
1,916 pieces, of which 434 were closed solids and the rest open surfaces
and fragments of a few triangles, the leftovers of an export. Splitting it
piece by piece gives 1,916 objects nobody can work with.

A part here is a closed piece. Every other piece, an open surface or a
fragment, belongs to the smallest part whose box holds it, or failing that
the smallest one it touches; what belongs nowhere is kept apart as loose.
A file of nothing but open surfaces has its large pieces as parts. All
of it is done on arrays, so the 733,000 triangles take seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import MeshShape, _weld

# in a file with no closed pieces at all, a piece is a part when its box is
# at least this share of the whole model's
_SURFACE_PART_SHARE = 0.05
# how far a box may stick out of another's and still count as inside it
_PAD = 0.01
# rows of pieces compared against all parts at once
_CHUNK = 2048


@dataclass
class MeshParts:
    parts: list             # [MeshShape], largest first
    loose: MeshShape | None  # what belongs to no part, in one mesh
    pieces: int             # separate pieces found
    closed: int             # of which closed
    folded: int             # pieces folded into a part


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For n nodes joined in pairs (a[i], b[i]), each node's component,
    named by its smallest node. Labels flow to the smallest along every
    pair and are chased to their root, until nothing moves: a few dozen
    array passes where a Python union-find walks every pair."""
    label = np.arange(n)
    if not len(a):
        return label
    while True:
        lowest = np.minimum(label[a], label[b])
        moved = label.copy()
        np.minimum.at(moved, a, lowest)
        np.minimum.at(moved, b, lowest)
        while True:
            chased = moved[moved]
            if np.array_equal(chased, moved):
                break
            moved = chased
        if np.array_equal(moved, label):
            return label
        label = moved


def smart_parts(mesh: MeshShape, smallest: float = 0.0) -> MeshParts:
    """Split a mesh into its parts. `smallest` is a size (the diagonal of
    a part's box): a closed piece smaller than that is folded into the part
    around it rather than kept as a part of its own; 0 keeps every one."""
    points, canonical = _weld(np.asarray(mesh.vertices, np.float64))
    tris = canonical[np.asarray(mesh.triangles, np.int64)]
    real = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2]))
    tris = tris[real]
    if not len(tris):
        return MeshParts([mesh], None, 1, 0, 0)
    label = components(len(points), np.concatenate((tris[:, 0], tris[:, 1])),
                       np.concatenate((tris[:, 1], tris[:, 2])))
    _, piece, counts = np.unique(label[tris[:, 0]], return_inverse=True,
                                 return_counts=True)
    piece = np.asarray(piece).ravel()
    n_pieces = len(counts)
    closed = _closed_pieces(tris, piece, n_pieces, len(points))
    lo, hi = _boxes(points, tris, piece, n_pieces)
    diag = np.linalg.norm(hi - lo, axis=1)
    model = float(np.linalg.norm(points.max(0) - points.min(0)))

    is_part = closed & (diag >= smallest)
    if not is_part.any():
        is_part = diag >= _SURFACE_PART_SHARE * model
    if not is_part.any():
        is_part[int(np.argmax(counts))] = True
    owner = _owners(lo, hi, is_part, _PAD * model)

    group = np.where(is_part, np.arange(n_pieces), owner)   # -1: loose
    tri_group = group[piece]
    parts = []
    for g in sorted(set(np.flatnonzero(is_part).tolist()),
                    key=lambda g: -int((tri_group == g).sum())):
        parts.append(_mesh_of(points, tris[tri_group == g]))
    loose_tris = tris[tri_group < 0]
    loose = _mesh_of(points, loose_tris) if len(loose_tris) else None
    folded = int((~is_part & (owner >= 0)).sum())
    return MeshParts(parts, loose, n_pieces, int(closed.sum()), folded)


def _closed_pieces(tris, piece, n_pieces, n_points) -> np.ndarray:
    """Which pieces have every edge shared by exactly two triangles."""
    start = tris.ravel()
    end = tris[:, [1, 2, 0]].ravel()
    key = np.minimum(start, end) * n_points + np.maximum(start, end)
    owner = np.repeat(piece, 3)
    keys, first, uses = np.unique(key, return_index=True, return_counts=True)
    open_pieces = np.unique(owner[first[uses != 2]])
    closed = np.ones(n_pieces, bool)
    closed[open_pieces] = False
    return closed


def _boxes(points, tris, piece, n_pieces):
    corners = points[tris]                       # (T, 3, 3)
    tlo, thi = corners.min(axis=1), corners.max(axis=1)
    lo = np.full((n_pieces, 3), np.inf)
    hi = np.full((n_pieces, 3), -np.inf)
    np.minimum.at(lo, piece, tlo)
    np.maximum.at(hi, piece, thi)
    return lo, hi


def _owners(lo, hi, is_part, pad) -> np.ndarray:
    """For each piece that is not a part, the smallest part whose box holds
    its box, else the smallest whose box touches it; -1 for neither."""
    parts = np.flatnonzero(is_part)
    owner = np.full(len(lo), -1)
    others = np.flatnonzero(~is_part)
    if not len(parts) or not len(others):
        return owner
    plo, phi = lo[parts], hi[parts]
    volume = np.prod(np.maximum(phi - plo, 1e-12), axis=1)
    for start in range(0, len(others), _CHUNK):
        rows = others[start:start + _CHUNK]
        olo, ohi = lo[rows][:, None, :], hi[rows][:, None, :]
        inside = np.all((plo[None] - pad <= olo) & (ohi <= phi[None] + pad),
                        axis=2)
        touching = np.all((olo <= phi[None] + pad) & (plo[None] - pad <= ohi),
                          axis=2)
        for k, row in enumerate(rows):
            for fit in (inside[k], touching[k]):
                if fit.any():
                    candidates = np.flatnonzero(fit)
                    owner[row] = parts[candidates[np.argmin(
                        volume[candidates])]]
                    break
    return owner


def _mesh_of(points, tris) -> MeshShape:
    used, local = np.unique(tris, return_inverse=True)
    return MeshShape(points[used], np.asarray(local).reshape(-1, 3))
