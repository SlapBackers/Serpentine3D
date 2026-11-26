"""A solid keeps its numbers: pick an edge, retype it (#182).

The workflow from the discussion: a box 40 x 30 x 20, its four upright
edges filleted at 3; the box turns out too small, so an edge is picked and
its length typed again, and the fillets stay 3; then a fillet is picked and
made 5. Then the same for chamfers, holes and cylinders. A length always
asks first for the point that stays put; Enter takes the one offered.
"""

import math

import numpy as np
import pytest

from serpentine3d.core import geometry as g
from serpentine3d.core import solid_dimensions as sd
from serpentine3d.scripting import Document


# --- helpers ----------------------------------------------------------------

def _curve(edge):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    return BRepAdaptor_Curve(edge)


def _surface(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    return BRepAdaptor_Surface(face)


def _type(name):
    from OCP.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType
    return {"line": GeomAbs_CurveType.GeomAbs_Line,
            "circle": GeomAbs_CurveType.GeomAbs_Circle,
            "plane": GeomAbs_SurfaceType.GeomAbs_Plane,
            "cylinder": GeomAbs_SurfaceType.GeomAbs_Cylinder}[name]


def _straight_edge_along(shape, axis):
    for i, e in enumerate(g.edges_of(shape)):
        c = _curve(e)
        if c.GetType() == _type("line"):
            d = c.Line().Direction()
            if abs((d.X(), d.Y(), d.Z())[axis]) > 0.999:
                return i
    raise AssertionError("no straight edge along that axis")


def _faces(shape, kind):
    return [i for i, f in enumerate(g.faces_of(shape))
            if _surface(f).GetType() == _type(kind)]


def _radii(shape):
    return sorted(round(_surface(g.faces_of(shape)[i]).Cylinder().Radius(), 6)
                  for i in _faces(shape, "cylinder"))


def _upright_edges(shape):
    out = []
    for e in g.edges_of(shape):
        c = _curve(e)
        if c.GetType() == _type("line") and abs(c.Line().Direction().Z()) \
                > 0.999:
            out.append(e)
    return out


def _filleted_box():
    box = g.make_box((0, 0, 0), 40, 30, 20)
    return g.fillet_edges(box, 3.0, edges=_upright_edges(box))


def _corner_loss(r, height):
    return 4 * (r * r - math.pi * r * r / 4) * height


def _doc_with(shape, name="Box"):
    doc = Document()
    obj = doc.add(shape, name=name)
    return doc, obj


def _edit(doc, obj, kind, index, inputs):
    doc.selection.set_subobjects([(obj.id, kind, index)])
    return doc.run("editdimension", inputs)


def _bounds(doc, name="Box"):
    return g.bbox(doc.get(name).shape)


def _normal(shape, index):
    d = _surface(g.faces_of(shape)[index]).Plane().Axis().Direction()
    return (d.X(), d.Y(), d.Z())


# --- boxes and fillets ------------------------------------------------------

def test_the_sizes_and_fillets_are_read_back():
    model = sd.read(_filleted_box())
    assert model.size == pytest.approx((40, 30, 20))
    assert len(model.fillets) == 4
    assert all(r == pytest.approx(3) for r in model.fillets.values())


def test_rebuilt_from_what_was_read_it_is_the_same_solid():
    shape = _filleted_box()
    again = sd.build(sd.read(shape))
    assert g.volume(again) == pytest.approx(g.volume(shape), rel=1e-9)
    assert len(g.faces_of(again)) == len(g.faces_of(shape))


def test_retyping_an_edge_grows_the_box_and_keeps_the_fillets():
    doc, obj = _doc_with(_filleted_box())
    said = _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 0),
                 ["", "50"])
    shape = doc.get("Box").shape
    lo, hi = g.bbox(shape)
    assert hi[0] - lo[0] == pytest.approx(50)
    assert lo[0] == pytest.approx(0, abs=1e-6)        # held at the left
    assert _radii(shape) == pytest.approx([3, 3, 3, 3])
    assert g.volume(shape) == pytest.approx(
        50 * 30 * 20 - _corner_loss(3, 20), rel=1e-6)
    assert any("Size along X" in s and "50" in s for s in said)


def test_the_point_that_stays_put_is_always_asked_for():
    doc, obj = _doc_with(_filleted_box())
    doc.selection.set_subobjects(
        [(obj.id, "edge", _straight_edge_along(obj.shape, 2))])
    with pytest.raises(RuntimeError, match="Point that stays put "
                                           r"\(Enter for the bottom\)"):
        doc.run("editdimension", [])


def test_a_box_on_the_construction_plane_stays_on_it():
    doc, obj = _doc_with(_filleted_box())
    said = _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 2),
                 ["", "50"])
    lo, hi = _bounds(doc)
    assert lo[2] == pytest.approx(0, abs=1e-6)
    assert hi[2] == pytest.approx(50)
    assert any("held at the bottom" in s for s in said)


def test_a_box_hanging_under_the_plane_keeps_its_top_on_it():
    doc, obj = _doc_with(g.translate(_filleted_box(), (0, 0, -20)))
    _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 2), ["", "50"])
    lo, hi = _bounds(doc)
    assert hi[2] == pytest.approx(0, abs=1e-6)
    assert lo[2] == pytest.approx(-50)


def test_middle_grows_both_ways():
    doc, obj = _doc_with(_filleted_box())
    _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 0),
          ["Middle", "50"])
    lo, hi = _bounds(doc)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(20)


def test_right_keeps_the_right_side_where_it_was():
    doc, obj = _doc_with(_filleted_box())
    _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 0),
          ["Right", "50"])
    lo, hi = _bounds(doc)
    assert hi[0] == pytest.approx(40, abs=1e-6)
    assert lo[0] == pytest.approx(-10)


def test_a_picked_point_stays_where_it_is():
    # held at x=30 of 0..40: that point stays, the box scales about it
    doc, obj = _doc_with(_filleted_box())
    said = _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 0),
                 ["30,0,0", "80"])
    lo, hi = _bounds(doc)
    assert lo[0] == pytest.approx(-30)
    assert hi[0] == pytest.approx(50)
    assert any("held at the picked point" in s for s in said)


def test_a_picked_face_moves_and_the_face_across_holds():
    doc, obj = _doc_with(g.make_box((0, 0, 0), 40, 30, 20))
    shape = obj.shape
    right = next(i for i in _faces(shape, "plane")
                 if g.bbox(g.faces_of(shape)[i])[0][0] == pytest.approx(40))
    _edit(doc, obj, "face", right, ["", "60"])
    lo, hi = _bounds(doc)
    assert lo[0] == pytest.approx(0, abs=1e-6)
    assert hi[0] == pytest.approx(60)


def test_retyping_a_fillet_changes_its_radius():
    doc, obj = _doc_with(_filleted_box())
    _edit(doc, obj, "face", _faces(obj.shape, "cylinder")[0], ["5"])
    shape = doc.get("Box").shape
    assert _radii(shape) == pytest.approx([3, 3, 3, 5])
    lo, hi = g.bbox(shape)
    assert hi[0] - lo[0] == pytest.approx(40)


def test_a_turned_and_moved_box_still_reads_its_sizes():
    shape = g.rotate(_filleted_box(), (0, 0, 0), (0, 0, 1), 30)
    shape = g.translate(shape, (100, 50, 7))
    model = sd.read(shape)
    assert sorted(model.size) == pytest.approx([20, 30, 40])
    assert len(model.fillets) == 4


def test_a_fillet_too_big_for_the_new_size_changes_nothing():
    doc, obj = _doc_with(_filleted_box())
    before = g.volume(obj.shape)
    said = _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 1),
                 ["", "5"])
    assert g.volume(doc.get("Box").shape) == pytest.approx(before)
    assert any("Nothing changed" in s for s in said)


def test_nothing_picked_says_what_to_do():
    doc, _obj = _doc_with(_filleted_box())
    said = doc.run("editdimension", [])
    assert any("Ctrl+Shift-click" in s for s in said)


def test_sizes_follow_the_world_axes_for_a_box_drawn_on_it():
    model = sd.read(g.make_box((5, 6, 7), 10, 20, 30))
    assert np.allclose(model.axes, np.eye(3))
    assert model.size == pytest.approx((10, 20, 30))
    assert model.origin == pytest.approx((5, 6, 7))


# --- chamfers ---------------------------------------------------------------

def _chamfered_box():
    box = g.make_box((0, 0, 0), 40, 30, 20)
    return g.fillet_edges(box, 2.0, edges=_upright_edges(box), chamfer=True)


def test_chamfers_are_read_back():
    model = sd.read(_chamfered_box())
    assert len(model.chamfers) == 4
    assert all(d == pytest.approx(2) for d in model.chamfers.values())


def test_a_box_grows_and_its_chamfers_keep_their_distance():
    doc, obj = _doc_with(_chamfered_box())
    _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 0), ["", "50"])
    shape = doc.get("Box").shape
    assert g.volume(shape) == pytest.approx(50 * 30 * 20 - 4 * 2 * 20,
                                            rel=1e-6)
    assert len(sd.read(shape).chamfers) == 4


def test_retyping_a_chamfer_changes_its_distance():
    doc, obj = _doc_with(_chamfered_box())
    slanted = next(i for i in _faces(obj.shape, "plane")
                   if all(abs(c) < 0.999 for c in _normal(obj.shape, i)))
    _edit(doc, obj, "face", slanted, ["4"])
    distances = sorted(sd.read(doc.get("Box").shape).chamfers.values())
    assert distances == pytest.approx([2, 2, 2, 4])


def test_fillets_and_chamfers_on_one_box():
    box = g.make_box((0, 0, 0), 40, 30, 20)
    box = g.fillet_edges(box, 2.0, edges=_upright_edges(box)[:2],
                         chamfer=True)
    box = g.fillet_edges(box, 3.0, edges=_upright_edges(box)[-2:])
    model = sd.read(box)
    assert len(model.chamfers) == 2
    assert len(model.fillets) == 2


# --- holes ------------------------------------------------------------------

def _plate_with_holes():
    plate = g.make_box((0, 0, 0), 40, 30, 10)
    through = g.make_cylinder((10, 15, -1), 3, 12)
    blind = g.make_cylinder((30, 15, 5), 2, 6)          # 5 deep from the top
    return g.boolean_difference(g.boolean_difference(plate, through), blind)


def test_holes_are_read_back():
    model = sd.read(_plate_with_holes())
    holes = sorted(model.holes, key=lambda h: h.radius)
    assert [h.radius for h in holes] == pytest.approx([2, 3])
    assert not holes[0].through and holes[0].side == 1
    assert holes[0].depth == pytest.approx(5)
    assert holes[1].through


def _hole_wall(shape, radius):
    return next(i for i in _faces(shape, "cylinder")
                if abs(_surface(g.faces_of(shape)[i]).Cylinder().Radius()
                       - radius) < 1e-6)


def test_retyping_a_hole_changes_its_diameter():
    doc, obj = _doc_with(_plate_with_holes(), "Plate")
    said = _edit(doc, obj, "face", _hole_wall(obj.shape, 3), ["10"])
    assert _radii(doc.get("Plate").shape) == pytest.approx([2, 5])
    assert any("Hole diameter: 6" in s for s in said)


def test_a_hole_given_0_is_filled():
    doc, obj = _doc_with(_plate_with_holes(), "Plate")
    _edit(doc, obj, "face", _hole_wall(obj.shape, 3), ["0"])
    assert _radii(doc.get("Plate").shape) == pytest.approx([2])


def test_a_blind_hole_floor_gives_its_depth():
    doc, obj = _doc_with(_plate_with_holes(), "Plate")
    floor = next(i for i in _faces(obj.shape, "plane")
                 if g.bbox(g.faces_of(obj.shape)[i])[0][2]
                 == pytest.approx(5))
    _edit(doc, obj, "face", floor, ["8"])
    hole = [h for h in sd.read(doc.get("Plate").shape).holes
            if not h.through][0]
    assert hole.depth == pytest.approx(8)


def test_holes_stay_put_and_a_blind_hole_follows_its_face():
    # made taller from the bottom: the top rises, the blind hole drilled
    # from the top rises with it and keeps its depth; the through hole
    # still goes through
    doc, obj = _doc_with(_plate_with_holes(), "Plate")
    _edit(doc, obj, "edge", _straight_edge_along(obj.shape, 2), ["", "20"])
    model = sd.read(doc.get("Plate").shape)
    blind = [h for h in model.holes if not h.through][0]
    assert blind.depth == pytest.approx(5)
    assert model.size[2] == pytest.approx(20)
    centres = sorted(round(h.point[0], 6) for h in model.holes)
    assert centres == pytest.approx([10, 30])


# --- cylinders --------------------------------------------------------------

def _cylinder_doc():
    return _doc_with(g.make_cylinder((0, 0, 0), 10, 30), "Rod")


def test_a_cylinder_is_read_back():
    model = sd.read(g.make_cylinder((5, 5, 0), 10, 30))
    assert model.radius == pytest.approx(10)
    assert model.height == pytest.approx(30)
    assert model.base == pytest.approx((5, 5, 0))


def test_a_cylinder_takes_a_new_diameter_about_its_axis():
    doc, obj = _cylinder_doc()
    _edit(doc, obj, "face", _faces(obj.shape, "cylinder")[0], ["30"])
    lo, hi = _bounds(doc, "Rod")
    assert hi[0] - lo[0] == pytest.approx(30)
    assert (hi[0] + lo[0]) / 2 == pytest.approx(0, abs=1e-6)


def test_a_cylinder_on_the_plane_grows_upwards():
    doc, obj = _cylinder_doc()
    seam = next(i for i, e in enumerate(g.edges_of(obj.shape))
                if _curve(e).GetType() == _type("line"))
    said = _edit(doc, obj, "edge", seam, ["", "50"])
    lo, hi = _bounds(doc, "Rod")
    assert lo[2] == pytest.approx(0, abs=1e-6)
    assert hi[2] == pytest.approx(50)
    assert any("Height, held at the bottom" in s for s in said)


def test_a_cylinder_end_moves_when_picked():
    doc, obj = _cylinder_doc()
    bottom = next(i for i in _faces(obj.shape, "plane")
                  if g.bbox(g.faces_of(obj.shape)[i])[1][2]
                  == pytest.approx(0, abs=1e-6))
    _edit(doc, obj, "face", bottom, ["", "50"])
    lo, hi = _bounds(doc, "Rod")
    assert hi[2] == pytest.approx(30)
    assert lo[2] == pytest.approx(-20)


# --- what is left alone -----------------------------------------------------

def test_a_solid_that_is_none_of_these_is_left_alone():
    doc, obj = _doc_with(g.make_sphere((0, 0, 0), 5), "Ball")
    said = _edit(doc, obj, "face", 0, [])
    assert any("no dimensions to edit" in s for s in said)
