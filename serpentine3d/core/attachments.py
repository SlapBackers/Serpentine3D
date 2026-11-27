"""Objects attached to a face, following it when it moves.

A cylinder stands on a cube. The cube is made taller, wider, moved or
turned: the cylinder stays on its top face, at the place it was given
there. Nothing links by itself; `attach` makes each link (and `unattach`
takes one away):

- the object sits on the face by one of its own faces (its contact), and
  rides the face up and down;
- across the face it is measured from two edges of the face, each one
  keeping either its distance from that edge, or its share of the face
  (a cylinder in the middle stays in the middle);
- when the face's object turns, it turns with it;
- stacks move as stacks: a ball attached to that cylinder follows the
  cylinder however the cylinder came to move.

The link is the user's own placement, so it gives way to the user: moving
the object by hand, or deleting what it is attached to, ends the link, and
says so.

A link lives in the follower's user text, as JSON, with what it last saw:
its parent's flat faces, the face, and the follower's contact. Saving,
undo and redo therefore carry it with the object, and a change is found by
comparing the shapes with what the link last saw rather than by being told
what changed: the scene's listeners are told only that something did.
"""

from __future__ import annotations

import json

import numpy as np

from . import user_text

KEY = "attachment"        # the user text key a link is kept under
LABEL = "attached to"     # a line anyone can read in Properties
VERSION = 2
_PARALLEL = 1 - 1e-6
_SAMPLES = 24


class AttachError(Exception):
    """The link cannot be made."""


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-15 else v


# --- reading shapes -------------------------------------------------------------

def _planar_faces(shape):
    """[(face, outward normal)] for every flat face of a shape, in order."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from .geometry import face_normal, faces_of
    out = []
    for face in faces_of(shape):
        if BRepAdaptor_Surface(face).GetType() != \
                GeomAbs_SurfaceType.GeomAbs_Plane:
            continue
        out.append((face, _unit(face_normal(face))))
    return out


def is_flat(face) -> bool:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    return BRepAdaptor_Surface(face).GetType() == \
        GeomAbs_SurfaceType.GeomAbs_Plane


def outward_normal(face):
    from .geometry import face_normal
    return _unit(face_normal(face))


def _points(face):
    """Points around a face's boundary: its vertices, and enough along each
    edge that a circle counts."""
    from .geometry import edges_of, sample_curve
    pts = []
    for edge in edges_of(face):
        try:
            pts.extend(sample_curve(edge, _SAMPLES))
        except Exception:                                   # noqa: BLE001
            continue
    return np.asarray(pts, float).reshape(-1, 3)


def _centre(face):
    from . import occ
    from .geometry import pnt_tuple
    return np.asarray(pnt_tuple(
        occ.surface_properties(face).CentreOfMass()), float)


def signature(shape) -> list:
    """The outward normals of a shape's flat faces, in order: what a turn of
    the whole shape is read from."""
    return [n.tolist() for _f, n in _planar_faces(shape)][:400]


def rotation_between(old, new):
    """The turn taking the normals `old` to `new`, matched in order, or
    None when they are not the same faces turned (a face added, a mirror,
    a change of shape). The identity when nothing turned."""
    a, b = np.asarray(old, float), np.asarray(new, float)
    if a.shape != b.shape or len(a) == 0:
        return None
    if np.allclose(a, b, atol=1e-9):
        return np.eye(3)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    if np.linalg.det(r) < 0 or not np.allclose(a @ r.T, b, atol=1e-6):
        return None
    return r


def find_face(shape, normal):
    """The flat face of `shape` facing `normal` and furthest along it: a
    cube's top, for a normal pointing up."""
    n = _unit(normal)
    best = None
    for face, fn in _planar_faces(shape):
        if float(fn @ n) < _PARALLEL:
            continue
        h = float(_centre(face) @ n)
        if best is None or h > best[0]:
            best = (h, face)
    return None if best is None else best[1]


def straight_edges(face):
    """[(start, end)] of a face's straight edges."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_CurveType
    from .geometry import curve_endpoints, edges_of
    out = []
    for edge in edges_of(face):
        if BRepAdaptor_Curve(edge).GetType() != GeomAbs_CurveType.GeomAbs_Line:
            continue
        p, q = curve_endpoints(edge)
        out.append((np.asarray(p, float), np.asarray(q, float)))
    return out


def _square(normal):
    """Four directions across a face, square to each other: along the
    world's X where they can be."""
    n = _unit(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else \
        np.array([0.0, 1.0, 0.0])
    u = _unit(ref - (ref @ n) * n)
    v = np.cross(n, u)
    return [u, -u, v, -v]


def edge_direction(face, normal, edge):
    """The direction across the face away from an edge, out of the face: the
    way its distance from that edge is measured, turned round."""
    p, q = edge
    m = _unit(np.cross(normal, _unit(q - p)))
    if float(m @ (p - _centre(face))) < 0:
        m = -m
    return m


def _segment_distance(point, edge):
    p, q = edge
    t = q - p
    s = float(np.clip((point - p) @ t / max(float(t @ t), 1e-30), 0, 1))
    return float(np.linalg.norm(point - (p + s * t)))


def nearest_edge(face, normal, point, across=None):
    """(direction, edge) of the face's straight edge nearest `point`; with
    `across`, only an edge that is not parallel to that direction."""
    point = np.asarray(point, float)
    best = None
    for edge in straight_edges(face):
        m = edge_direction(face, normal, edge)
        if across is not None and abs(float(m @ across)) > 0.999:
            continue
        d = _segment_distance(point, edge)
        if best is None or d < best[0]:
            best = (d, m, edge)
    if best is None and not straight_edges(face):
        # a round face has no straight edge: its sides are where it
        # reaches furthest along two square directions across it
        pts = _points(face)
        for m in _square(normal):
            if across is not None and abs(float(m @ across)) > 0.999:
                continue
            d = float((pts @ m).max() - point @ m)
            if best is None or d < best[0]:
                best = (d, m, None)
    if best is None:
        raise AttachError("that face has no straight edge to measure from"
                          if across is None else
                          "that face has no straight edge across the first")
    return best[1], best[2]


def face_frame(face, normal, dirs) -> dict:
    """Where the face is: its height along `normal`, and for each direction
    the place of its outermost edge that way (`e`) and its width that way
    (`w`)."""
    pts = _points(face)
    n = _unit(normal)
    e, w = [], []
    for m in dirs:
        along = pts @ np.asarray(m, float)
        e.append(float(along.max()))
        w.append(float(along.max() - along.min()))
    return {"h": float(_centre(face) @ n), "e": e, "w": w}


def contact(shape, c):
    """(point, face) the shape sits by, facing `c`: the middle of its flat
    face facing `c` that lies furthest that way, or, with no such face (a
    ball), its furthest point that way."""
    c = _unit(c)
    best = None
    for face, fn in _planar_faces(shape):
        if float(fn @ c) < _PARALLEL:
            continue
        h = float(_centre(face) @ c)
        if best is None or h > best[0]:
            best = (h, face)
    if best is not None:
        return _centre(best[1]), best[1]
    from . import occ
    from .geometry import edges_of, sample_curve
    pts = []
    exp = occ.TopExp_Explorer(shape, occ.VERTEX)
    while exp.More():
        v = occ.point_of_vertex(occ.to_vertex(exp.Current()))
        pts.append((v.X(), v.Y(), v.Z()))
        exp.Next()
    for edge in edges_of(shape):
        try:
            pts.extend(sample_curve(edge, _SAMPLES))
        except Exception:                                   # noqa: BLE001
            continue
    pts = np.asarray(pts, float).reshape(-1, 3)
    if not len(pts):
        raise AttachError("that object has nothing to sit on")
    along = pts @ c
    tol = 1e-6 * max(1.0, float(np.abs(pts).max()))
    return pts[along >= along.max() - tol].mean(axis=0), None


def lowest_side(shape, normal):
    """The way the side an object would sit on faces: its flat face that
    looks most against `normal`, or straight against it."""
    down = -_unit(normal)
    best = None
    for _face, fn in _planar_faces(shape):
        d = float(fn @ down)
        if best is None or d > best[0] + 1e-9:
            best = (d, fn)
    if best is None or best[0] <= 0:
        return down
    return best[1]


def turn_to(a, b):
    """The smallest turn taking direction `a` to direction `b`."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    s, c = float(np.linalg.norm(v)), float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # half a turn about any axis square to a
        axis = _unit(np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9
                              else [0, 1, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s ** 2)


def moved(shape, turn, about, to):
    """`shape` turned by `turn` about the point `about`, which then goes to
    `to`."""
    from .geometry import apply_matrix
    m = np.eye(4)
    m[:3, :3] = turn
    m[:3, 3] = np.asarray(to, float) - turn @ np.asarray(about, float)
    if np.allclose(m, np.eye(4), atol=1e-15):
        return shape
    return apply_matrix(shape, m)


# --- placing ------------------------------------------------------------------

def target(normal, frame, dirs):
    """The point on the face where the contact goes: on its plane, at each
    direction's kept distance (or share) from its edge."""
    n = _unit(normal)
    rows, rhs = [n], [frame["h"]]
    for k, d in enumerate(dirs):
        m = np.asarray(d["m"], float)
        dist = d["d"] if d["keep"] == "edge" else d["s"] * frame["w"][k]
        rows.append(m)
        rhs.append(frame["e"][k] - dist)
    return np.linalg.solve(np.asarray(rows), np.asarray(rhs))


def seat(parent_shape, face, follower_shape, side):
    """The follower turned so its side `side` lies against `face`, and put
    down on it along the face's normal: (shape, contact point)."""
    n = outward_normal(face)
    turn = turn_to(side, -n)
    p, _f = contact(follower_shape, side)
    h = float(_centre(face) @ n)
    to = p + (h - float(p @ n)) * n
    return moved(follower_shape, turn, p, to), to


def place(follower_shape, normal, frame, dirs, p):
    """The follower slid on the face so its contact at `p` goes to the
    place `dirs` give: (shape, new contact)."""
    to = target(normal, frame, dirs)
    return moved(follower_shape, np.eye(3), p, to), to


def make_link(parent, face, follower, side, dirs) -> dict:
    """The link that keeps `follower` where it now sits on `face`.

    `dirs` are two {"m": direction away from an edge, "keep": "edge" or
    "share"}; their distances are read from where the follower is."""
    if parent.id == follower.id:
        raise AttachError("an object cannot be attached to itself")
    n = outward_normal(face)
    ms = [np.asarray(d["m"], float) for d in dirs]
    frame = face_frame(face, n, ms)
    p, _f = contact(follower.shape, side)
    out = []
    for k, d in enumerate(dirs):
        dist = frame["e"][k] - float(ms[k] @ p)
        w = frame["w"][k]
        out.append({"m": ms[k].tolist(), "keep": d["keep"], "d": dist,
                    "s": dist / w if w > 1e-12 else 0.0})
    return {"v": VERSION, "parent": parent.id, "normal": n.tolist(),
            "dirs": out,
            "seen": {"sig": signature(parent.shape), "face": frame,
                     "c": _unit(side).tolist(), "p": p.tolist()},
            "hangs": False}


def describe(link, parent_name) -> str:
    parts = []
    for d in link["dirs"]:
        parts.append(f"{d['d']:g} from an edge" if d["keep"] == "edge"
                     else f"{d['s'] * 100:.4g}% across")
    return f"{parent_name}: " + ", ".join(parts)


def link_of(obj) -> dict | None:
    text = user_text.value_of(obj.user_text, KEY)
    if not text:
        return None
    try:
        link = json.loads(text)
    except ValueError:
        return None
    return link if isinstance(link, dict) and link.get("v") == VERSION \
        else None


def with_link(text, link, parent_name=None):
    """User text holding `link` (None takes it away) and its readable line."""
    text = user_text.with_value(text, KEY, json.dumps(link) if link else "")
    return user_text.with_value(
        text, LABEL, describe(link, parent_name) if link and parent_name
        else "")


# --- following ----------------------------------------------------------------

def _close(a, b, tol) -> bool:
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


def _same_frame(a, b, tol) -> bool:
    return (abs(a["h"] - b["h"]) <= tol and _close(a["e"], b["e"], tol)
            and _close(a["w"], b["w"], tol))


def hangs(face, normal, link, follower_shape) -> bool:
    """Whether the follower's contact reaches past the face's edges."""
    c = link["seen"]["c"]
    p, cface = contact(follower_shape, c)
    pts = _points(cface) if cface is not None else p.reshape(1, 3)
    frame = face_frame(face, normal, [d["m"] for d in link["dirs"]])
    tol = 1e-6 * max(1.0, float(np.abs(pts).max()))
    for k, d in enumerate(link["dirs"]):
        along = pts @ np.asarray(d["m"], float)
        if along.max() > frame["e"][k] + tol or \
                along.min() < frame["e"][k] - frame["w"][k] - tol:
            return True
    return False


def follow(scene, echo=None) -> int:
    """Move every follower whose face has moved or turned since its link
    last looked. Returns how many moved."""
    say = echo or (lambda _text: None)
    count = 0
    for _round in range(len(scene.all()) + 1):
        progress = False
        for obj in list(scene.all()):
            link = link_of(obj)
            if link is None:
                continue
            obj = scene.get(obj.id)
            parent = scene.get(link["parent"])
            if parent is None:
                scene.update(obj.id, user_text=with_link(obj.user_text, None))
                say(f"{obj.name} is no longer attached: what it was "
                    "attached to is gone.")
                progress = True
                continue
            seen = link["seen"]
            sig = signature(parent.shape)
            turn = rotation_between(seen["sig"], sig)
            turned = turn is not None and not np.allclose(turn, np.eye(3))
            if turn is None:
                turn = np.eye(3)
            n = turn @ np.asarray(link["normal"], float)
            ms = [turn @ np.asarray(d["m"], float) for d in link["dirs"]]
            face = find_face(parent.shape, n)
            if face is None:
                continue
            frame = face_frame(face, n, ms)
            scale = max(1.0, abs(frame["h"]), *map(abs, frame["e"]))
            tol = 1e-7 * scale
            parent_same = not turned and _same_frame(frame, seen["face"], tol)
            try:
                p_now, _f = contact(obj.shape, seen["c"])
            except AttachError:
                continue
            follower_same = _close(p_now, seen["p"], tol)
            if parent_same and follower_same:
                continue
            dirs = [dict(d, m=m.tolist()) for d, m in zip(link["dirs"], ms)]
            to = target(n, frame, dirs)
            c = turn @ np.asarray(seen["c"], float)
            new = None
            if not follower_same:
                # moved along with its parent (both picked and moved)?
                try:
                    p_rot, _f = contact(obj.shape, c)
                except AttachError:
                    p_rot = None
                if parent_same or p_rot is None or not _close(p_rot, to, tol):
                    scene.update(obj.id,
                                 user_text=with_link(obj.user_text, None))
                    say(f"{obj.name} was moved, so it is no longer attached "
                        f"to {parent.name}.")
                    progress = True
                    continue
            else:
                new = moved(obj.shape, turn, np.asarray(seen["p"]), to)
            link.update(normal=n.tolist(), dirs=dirs)
            link["seen"] = {"sig": sig, "face": frame, "c": c.tolist(),
                            "p": to.tolist()}
            if new is not None:
                scene.replace_shape(obj.id, new)
                count += 1
            fresh = scene.get(obj.id)
            over = hangs(face, n, link, fresh.shape)
            if over and not link.get("hangs"):
                say(f"{obj.name} hangs over the edge of {parent.name}'s "
                    "face.")
            link["hangs"] = over
            scene.update(obj.id, user_text=with_link(fresh.user_text, link,
                                                     parent.name))
            progress = True
        if not progress:
            break
    return count


class _Watcher:
    """Calls follow() after every change to a scene's objects, once."""

    def __init__(self, scene, echo=None):
        self.scene = scene
        self.echo = echo
        self.busy = False
        scene.add_listener(self, kinds=("objects",))

    def __call__(self):
        if self.busy:
            return
        self.busy = True
        try:
            follow(self.scene, self.echo)
        finally:
            self.busy = False


def watch(scene, echo=None):
    """Make the scene's attachments follow their faces from now on."""
    watcher = getattr(scene, "_attachments_watched", None)
    if watcher is None:
        watcher = scene._attachments_watched = _Watcher(scene, echo)
    elif echo is not None:
        watcher.echo = echo
    return watcher


def paused(scene):
    """A `with` block in which the scene's attachments don't follow: a
    command placing an object it is about to attach."""
    class _Pause:
        def __enter__(self):
            self.watcher = getattr(scene, "_attachments_watched", None)
            self.was = self.watcher.busy if self.watcher else None
            if self.watcher:
                self.watcher.busy = True
            return self

        def __exit__(self, *exc):
            if self.watcher:
                self.watcher.busy = self.was
    return _Pause()


# --- what is drawn ----------------------------------------------------------

def outline(face):
    """The face's outer boundary as an ordered loop of points."""
    from OCP.BRepTools import BRepTools
    pts = []
    from .geometry import sample_curve
    from OCP.BRepTools import BRepTools_WireExplorer
    wire = BRepTools.OuterWire_s(face)
    exp = BRepTools_WireExplorer(wire, face)
    while exp.More():
        edge = exp.Current()
        seg = sample_curve(edge, _SAMPLES)
        from OCP.TopAbs import TopAbs_Orientation
        if edge.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
            seg = seg[::-1]
        pts.extend(seg[:-1])
        exp.Next()
    return np.asarray(pts, float).reshape(-1, 3)


def marks(scene, selected) -> dict:
    """What to draw for the selected objects, in world coordinates.

    For a selected follower: its parent's face (`faces`), the edges it is
    measured from (`edges`), a dimension from each edge to it (`dims`:
    from, to, text) and a badge naming its parent (`badges`). For a
    selected parent: the outlines of what follows it (`followers`)."""
    out = {"faces": [], "edges": [], "dims": [], "badges": [],
           "followers": []}
    chosen = set(selected)
    from .geometry import bbox, edges_of, sample_curve
    for obj in scene.all():
        link = link_of(obj)
        if link is None:
            continue
        parent = scene.get(link["parent"])
        if parent is None:
            continue
        if obj.id in chosen:
            n = np.asarray(link["normal"], float)
            face = find_face(parent.shape, n)
            if face is None:
                continue
            out["faces"].append(outline(face))
            p = np.asarray(link["seen"]["p"], float)
            frame = face_frame(face, n, [d["m"] for d in link["dirs"]])
            for k, d in enumerate(link["dirs"]):
                m = np.asarray(d["m"], float)
                foot = p + (frame["e"][k] - float(m @ p)) * m
                for edge in straight_edges(face):
                    if float(edge_direction(face, n, edge) @ m) > _PARALLEL \
                            and abs(float(edge[0] @ m) - frame["e"][k]) \
                            < 1e-6 * max(1.0, abs(frame["e"][k])):
                        out["edges"].append(edge)
                text = f"{d['d']:g}" if d["keep"] == "edge" \
                    else f"{d['s'] * 100:.3g}%"
                out["dims"].append((foot, p, text))
            _lo, hi = bbox(obj.shape)
            out["badges"].append((np.asarray(hi, float),
                                  f"attached to {parent.name}"))
        if parent.id in chosen and obj.id not in chosen:
            lines = []
            for edge in edges_of(obj.shape):
                try:
                    lines.append(np.asarray(sample_curve(edge, _SAMPLES)))
                except Exception:                           # noqa: BLE001
                    continue
            out["followers"].append(lines)
    return out
