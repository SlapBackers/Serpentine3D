"""A box and its fillets keep their numbers: pick an edge, retype it (#182).

The workflow from the discussion: a box 40 x 30 x 20, its four upright
edges filleted at 3; the box turns out too small, so an edge is picked and
its length typed again, and the fillets stay 3; then a fillet is picked and
made 5.
"""

import math

import numpy as np
import pytest

from serpentine3d.core import box_dimensions as bd
from serpentine3d.core import geometry as g
from serpentine3d.scripting import Document


def _curve(edge):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    return BRepAdaptor_Curve(edge)


def _surface(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    return BRepAdaptor_Surface(face)


def _straight_edge_along(shape, axis):
    from OCP.GeomAbs import GeomAbs_CurveType
    for i, e in enumerate(g.edges_of(shape)):
        c = _curve(e)
        if c.GetType() == GeomAbs_CurveType.GeomAbs_Line:
            d = c.Line().Direction()
            if abs((d.X(), d.Y(), d.Z())[axis]) > 0.999:
                return i
    raise AssertionError("no straight edge along that axis")


def _face(shape, kind):
    from OCP.GeomAbs import GeomAbs_SurfaceType
    want = {"cylinder": GeomAbs_SurfaceType.GeomAbs_Cylinder,
            "plane": GeomAbs_SurfaceType.GeomAbs_Plane}[kind]
    return [i for i, f in enumerate(g.faces_of(shape))
            if _surface(f).GetType() == want]


def _radii(shape):
    return sorted(round(_surface(g.faces_of(shape)[i]).Cylinder().Radius(), 6)
                  for i in _face(shape, "cylinder"))


def _upright_edges(shape):
    from OCP.GeomAbs import GeomAbs_CurveType
    out = []
    for e in g.edges_of(shape):
        c = _curve(e)
        if c.GetType() == GeomAbs_CurveType.GeomAbs_Line \
                and abs(c.Line().Direction().Z()) > 0.999:
            out.append(e)
    return out


def _filleted_box():
    box = g.make_box((0, 0, 0), 40, 30, 20)
    return g.fillet_edges(box, 3.0, edges=_upright_edges(box))


def _corner_loss(r, height):
    return 4 * (r * r - math.pi * r * r / 4) * height


def test_the_sizes_and_fillets_are_read_back():
    model = bd.read(_filleted_box())
    assert model.size == pytest.approx((40, 30, 20))
    assert len(model.fillets) == 4
    assert all(r == pytest.approx(3) for r in model.fillets.values())


def test_rebuilt_from_what_was_read_it_is_the_same_solid():
    shape = _filleted_box()
    again = bd.build(bd.read(shape))
    assert g.volume(again) == pytest.approx(g.volume(shape), rel=1e-9)
    assert len(g.faces_of(again)) == len(g.faces_of(shape))


def _doc_with(shape, name="Box"):
    doc = Document()
    obj = doc.add(shape, name=name)
    return doc, obj


def test_retyping_an_edge_grows_the_box_and_keeps_the_fillets():
    doc, obj = _doc_with(_filleted_box())
    edge = _straight_edge_along(obj.shape, 0)
    doc.selection.set_subobjects([(obj.id, "edge", edge)])
    said = doc.run("editdimension", ["50"])
    shape = doc.get("Box").shape
    lo, hi = g.bbox(shape)
    assert hi[0] - lo[0] == pytest.approx(50)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(20)    # grew both ways
    assert _radii(shape) == pytest.approx([3, 3, 3, 3])
    assert g.volume(shape) == pytest.approx(
        50 * 30 * 20 - _corner_loss(3, 20), rel=1e-6)
    assert any("Size along X" in s and "50" in s for s in said)


def test_grow_end_keeps_the_start_where_it_was():
    doc, obj = _doc_with(_filleted_box())
    edge = _straight_edge_along(obj.shape, 0)
    doc.selection.set_subobjects([(obj.id, "edge", edge)])
    doc.run("editdimension", ["Grow=End", "50"])
    lo, hi = g.bbox(doc.get("Box").shape)
    assert lo[0] == pytest.approx(0)
    assert hi[0] == pytest.approx(50)


def test_a_picked_face_is_the_one_that_moves():
    doc, obj = _doc_with(g.make_box((0, 0, 0), 40, 30, 20))
    shape = obj.shape
    right = next(i for i in _face(shape, "plane")
                 if g.bbox(g.faces_of(shape)[i])[0][0] == pytest.approx(40))
    doc.selection.set_subobjects([(obj.id, "face", right)])
    doc.run("editdimension", ["60"])
    lo, hi = g.bbox(doc.get("Box").shape)
    assert lo[0] == pytest.approx(0)
    assert hi[0] == pytest.approx(60)


def test_retyping_a_fillet_changes_its_radius():
    doc, obj = _doc_with(_filleted_box())
    fillet = _face(obj.shape, "cylinder")[0]
    doc.selection.set_subobjects([(obj.id, "face", fillet)])
    doc.run("editdimension", ["5"])
    shape = doc.get("Box").shape
    assert _radii(shape) == pytest.approx([3, 3, 3, 5])
    lo, hi = g.bbox(shape)
    assert hi[0] - lo[0] == pytest.approx(40)


def test_a_turned_and_moved_box_still_reads_its_sizes():
    shape = g.rotate(_filleted_box(), (0, 0, 0), (0, 0, 1), 30)
    shape = g.translate(shape, (100, 50, 7))
    model = bd.read(shape)
    assert sorted(model.size) == pytest.approx([20, 30, 40])
    assert len(model.fillets) == 4


def test_a_solid_that_is_no_box_is_left_alone():
    doc, obj = _doc_with(g.make_cylinder((0, 0, 0), 5, 10), "Tube")
    doc.selection.set_subobjects([(obj.id, "face", 0)])
    said = doc.run("editdimension", [])
    assert any("no dimensions to edit" in s for s in said)


def test_a_fillet_too_big_for_the_new_size_changes_nothing():
    doc, obj = _doc_with(_filleted_box())
    before = g.volume(obj.shape)
    edge = _straight_edge_along(obj.shape, 1)
    doc.selection.set_subobjects([(obj.id, "edge", edge)])
    said = doc.run("editdimension", ["5"])
    assert g.volume(doc.get("Box").shape) == pytest.approx(before)
    assert any("Nothing changed" in s for s in said)


def test_nothing_picked_says_what_to_do():
    doc, _obj = _doc_with(_filleted_box())
    said = doc.run("editdimension", [])
    assert any("Ctrl+Shift-click" in s for s in said)


def test_sizes_follow_the_world_axes_for_a_box_drawn_on_it():
    model = bd.read(g.make_box((5, 6, 7), 10, 20, 30))
    assert np.allclose(model.axes, np.eye(3))
    assert model.size == pytest.approx((10, 20, 30))
    assert model.origin == pytest.approx((5, 6, 7))
