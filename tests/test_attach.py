"""An object attached to a face follows it (placement and anchoring).

A cylinder stands on a cube. The cube changes: the cylinder rides its top
face, and keeps its place across it either as a distance from the point it
was measured from or as a share of the face. Links are made only by
attach; a link follows a chain only when asked to. Undo carries the link
with the object, so nothing moves twice.
"""

import pytest

from serpentine3d.core import attachments as at
from serpentine3d.core import geometry as g
from serpentine3d.scripting import Document


def _top_face(shape):
    return next(i for i, f in enumerate(g.faces_of(shape))
                if g.bbox(f)[0][2] == pytest.approx(g.bbox(shape)[1][2]))


def _scene(cylinder_at=(10, 10)):
    doc = Document()
    cube = doc.add(g.make_box((0, 0, 0), 40, 30, 20), name="Cube")
    x, y = cylinder_at
    doc.add(g.make_cylinder((x, y, 20), 3, 10), name="Cyl")
    return doc, cube


def _attach(doc, cube, anchor="Centre", mode="Distance", chain="No",
            follower="Cyl"):
    doc.selection.set_subobjects(
        [(doc.get(cube.name).id, "face", _top_face(doc.get(cube.name).shape))])
    return doc.run("attach", [follower, anchor, mode, chain])


def _centre(doc, name):
    lo, hi = g.bbox(doc.get(name).shape)
    return [(a + b) / 2 for a, b in zip(lo, hi)], lo, hi


def _resize(doc, name, *box):
    doc.scene.replace_shape(doc.get(name).id, g.make_box(*box))


def test_it_rides_the_face_when_the_cube_gets_taller():
    doc, cube = _scene()
    said = _attach(doc, cube)
    _resize(doc, "Cube", (0, 0, 0), 40, 30, 35)
    (cx, cy, _), lo, _hi = _centre(doc, "Cyl")
    assert lo[2] == pytest.approx(35, abs=1e-6)
    assert (cx, cy) == pytest.approx((10, 10), abs=1e-6)
    assert any("Cyl attached to Cube" in s for s in said)


def test_distance_from_a_corner_stays_the_same():
    doc, cube = _scene()
    _attach(doc, cube, anchor="40,0,20")              # the right front corner
    _resize(doc, "Cube", (0, 0, 0), 60, 30, 20)       # wider to the right
    (cx, cy, _), _lo, _hi = _centre(doc, "Cyl")
    assert (cx, cy) == pytest.approx((30, 10), abs=1e-6)  # still 30 from it


def test_distance_from_the_left_corner_keeps_it_in_place():
    doc, cube = _scene()
    _attach(doc, cube, anchor="0,0,20")
    _resize(doc, "Cube", (0, 0, 0), 60, 30, 20)
    (cx, cy, _), _lo, _hi = _centre(doc, "Cyl")
    assert (cx, cy) == pytest.approx((10, 10), abs=1e-6)


def test_proportion_keeps_its_share_of_the_face():
    doc, cube = _scene(cylinder_at=(20, 15))          # in the middle
    _attach(doc, cube, mode="Proportion")
    _resize(doc, "Cube", (0, 0, 0), 60, 50, 20)
    (cx, cy, _), _lo, _hi = _centre(doc, "Cyl")
    assert (cx, cy) == pytest.approx((30, 25), abs=1e-6)


def test_moving_the_cube_carries_the_cylinder():
    doc, cube = _scene()
    _attach(doc, cube)
    doc.scene.replace_shape(cube.id, g.translate(doc.get("Cube").shape,
                                                 (100, 0, 5)))
    (cx, cy, _), lo, _hi = _centre(doc, "Cyl")
    assert (cx, cy) == pytest.approx((110, 10), abs=1e-6)
    assert lo[2] == pytest.approx(25, abs=1e-6)


def _stack(chain):
    doc, cube = _scene()
    _attach(doc, cube)
    doc.add(g.make_sphere((10, 10, 32), 2), name="Ball")
    cyl = doc.get("Cyl")
    top = _top_face(cyl.shape)
    doc.selection.set_subobjects([(cyl.id, "face", top)])
    doc.run("attach", ["Ball", "Centre", "Distance", chain])
    _resize(doc, "Cube", (0, 0, 0), 40, 30, 30)
    return doc


def test_a_link_follows_one_level_unless_asked():
    (bx, by, bz), _lo, _hi = _centre(_stack("No"), "Ball")
    assert bz == pytest.approx(32, abs=1e-6)          # the ball stayed


def test_a_chained_link_follows_down_the_stack():
    (bx, by, bz), _lo, _hi = _centre(_stack("Yes"), "Ball")
    assert bz == pytest.approx(42, abs=1e-6)          # rode up with both


def test_moved_by_hand_it_keeps_its_new_place():
    doc, cube = _scene()
    _attach(doc, cube, anchor="0,0,20")
    doc.scene.replace_shape(doc.get("Cyl").id,
                            g.translate(doc.get("Cyl").shape, (5, 0, 0)))
    _resize(doc, "Cube", (0, 0, 0), 60, 30, 20)
    (cx, cy, _), _lo, _hi = _centre(doc, "Cyl")
    assert (cx, cy) == pytest.approx((15, 10), abs=1e-6)


def test_undo_puts_both_back_and_nothing_moves_twice():
    doc, cube = _scene()
    _attach(doc, cube)
    doc.history.checkpoint("taller")
    _resize(doc, "Cube", (0, 0, 0), 40, 30, 35)
    doc.history.undo()
    _c, lo, _hi = _centre(doc, "Cyl")
    assert lo[2] == pytest.approx(20, abs=1e-6)
    assert g.bbox(doc.get("Cube").shape)[1][2] == pytest.approx(20, abs=1e-6)


def test_detach_stops_it_following():
    doc, cube = _scene()
    _attach(doc, cube)
    doc.run("detach", ["Cyl", ""])
    _resize(doc, "Cube", (0, 0, 0), 40, 30, 35)
    _c, lo, _hi = _centre(doc, "Cyl")
    assert lo[2] == pytest.approx(20, abs=1e-6)


def test_the_link_is_kept_in_user_text():
    doc, cube = _scene()
    _attach(doc, cube)
    link = at.link_of(doc.get("Cyl"))
    assert link["parent"] == cube.id and link["mode"] == "distance"


def test_without_a_face_picked_it_says_what_to_do():
    doc, _cube = _scene()
    said = doc.run("attach", [])
    assert any("Ctrl+Shift-click the one face" in s for s in said)
