"""An object attached to a face follows it (placement and anchoring).

A cylinder is attached to the top of a cube: it is put down on the face,
measured from two of the face's edges, and from then on rides the face,
keeps its distance from those edges (or its share of the face), turns when
the cube turns, and carries what is attached to it. Moving it by hand, or
deleting the cube, ends the link and says so.
"""

import numpy as np
import pytest

from serpentine3d.core import attachments as at
from serpentine3d.core import geometry as g
from serpentine3d.scripting import Document

TOP = (20, 15, 20)          # a point on the cube's top face
LEFT = (0, 15, 20)          # on its left edge (x = 0)
FRONT = (20, 0, 20)         # on its front edge (y = 0)


def _scene(cylinder_base=(10, 10, 40)):
    doc = Document()
    doc.add(g.make_box((0, 0, 0), 40, 30, 20), name="Cube")
    doc.add(g.make_cylinder(cylinder_base, 3, 10), name="Cyl")
    return doc


def _attach(doc, follower="Cyl", parent="Cube", face_at=TOP, side_at=None,
            edges=(LEFT, FRONT), distances=(12, 8), keep=("", "")):
    """Run attach as the window would: clicks carry where they landed, an
    empty answer is Enter."""
    proc = doc._proc
    assert proc.run("attach")
    proc.click_object(doc.get(follower).id)
    proc.click_object(doc.get(parent).id, at=face_at)
    if side_at is None:
        proc.finish_selection()
    else:
        proc.click_object(doc.get(follower).id, at=side_at)
    for k in range(2):
        if edges[k] is None:
            proc.finish_selection()
        else:
            proc.click_object(doc.get(parent).id, at=edges[k])
        if keep[k]:
            proc.provide_text(f"Keep={keep[k]}")
        proc.provide_text("" if distances[k] is None else str(distances[k]))
    assert not proc.busy, proc.prompt_text()
    return doc.messages


def _box(doc, name, *box):
    doc.scene.replace_shape(doc.get(name).id, g.make_box(*box))


def _bottom(doc, name="Cyl"):
    """The middle of the object's bottom, and its box."""
    lo, hi = g.bbox(doc.get(name).shape)
    return ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]), lo, hi


def test_it_is_put_down_on_the_face_at_the_typed_distances():
    doc = _scene()
    said = _attach(doc)
    assert _bottom(doc)[0] == pytest.approx((12, 8, 20), abs=1e-6)
    assert any("Cyl attached to Cube: 12 from an edge, 8 from an edge" in s
               for s in said)


def test_enter_keeps_where_it_landed():
    doc = _scene(cylinder_base=(10, 10, 40))
    _attach(doc, edges=(None, None), distances=(None, None))
    assert _bottom(doc)[0] == pytest.approx((10, 10, 20), abs=1e-6)


def test_it_rides_the_face_when_the_cube_gets_taller():
    doc = _scene()
    _attach(doc)
    _box(doc, "Cube", (0, 0, 0), 40, 30, 35)
    assert _bottom(doc)[0] == pytest.approx((12, 8, 35), abs=1e-6)


def test_its_distance_from_the_edge_holds_whichever_way_the_cube_grows():
    doc = _scene()
    _attach(doc)
    _box(doc, "Cube", (0, 0, 0), 60, 30, 20)            # wider to the right
    assert _bottom(doc)[0] == pytest.approx((12, 8, 20), abs=1e-6)
    _box(doc, "Cube", (-20, 0, 0), 80, 30, 20)          # and to the left
    assert _bottom(doc)[0] == pytest.approx((-8, 8, 20), abs=1e-6)


def test_share_keeps_its_share_of_the_face():
    doc = _scene()
    _attach(doc, keep=("Share", "Edge"))
    _box(doc, "Cube", (0, 0, 0), 80, 30, 20)
    assert _bottom(doc)[0] == pytest.approx((24, 8, 20), abs=1e-6)


def test_it_follows_the_cube_moved():
    doc = _scene()
    _attach(doc)
    cube = doc.get("Cube")
    doc.scene.replace_shape(cube.id, g.translate(cube.shape, (5, -3, 7)))
    assert _bottom(doc)[0] == pytest.approx((17, 5, 27), abs=1e-6)


def test_it_turns_with_the_cube():
    doc = _scene()
    doc.add(g.make_box((30, 20, 40), 6, 2, 3), name="Bar")
    _attach(doc, follower="Bar", edges=(LEFT, FRONT), distances=(12, 8))
    cube = doc.get("Cube")
    doc.scene.replace_shape(cube.id,
                            g.rotate(cube.shape, (0, 0, 0), (0, 0, 1), 90))
    (x, y, z), lo, hi = _bottom(doc, "Bar")
    assert (x, y, z) == pytest.approx((-8, 12, 20), abs=1e-6)
    # it turned as well: 6 along X before, 6 along Y now
    assert hi[0] - lo[0] == pytest.approx(2, abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(6, abs=1e-6)


def test_it_tips_over_with_the_cube():
    doc = _scene()
    _attach(doc)
    cube = doc.get("Cube")
    doc.scene.replace_shape(cube.id,
                            g.rotate(cube.shape, (0, 0, 0), (1, 0, 0), 90))
    lo, hi = g.bbox(doc.get("Cyl").shape)
    # the top now faces -Y at y = -20, and the cylinder stands out of it
    assert hi[1] == pytest.approx(-20, abs=1e-6)
    assert lo[1] == pytest.approx(-30, abs=1e-6)
    assert ((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2) == \
        pytest.approx((12, 8), abs=1e-6)


def test_a_stack_moves_as_a_stack():
    doc = _scene()
    _attach(doc)
    doc.add(g.make_box((0, 0, 50), 2, 2, 2), name="Cap")
    _attach(doc, follower="Cap", parent="Cyl", face_at=(12, 8, 30),
            edges=(None, None), distances=(None, None))
    before = _bottom(doc, "Cap")[0]
    assert before[2] == pytest.approx(30, abs=1e-6)
    _box(doc, "Cube", (0, 0, 0), 40, 30, 35)
    after = _bottom(doc, "Cap")[0]
    assert after == pytest.approx((before[0], before[1], 45), abs=1e-6)


def test_the_side_picked_is_the_one_it_sits_on():
    doc = _scene()
    doc.add(g.make_box((0, 40, 0), 10, 2, 2), name="Rod")      # lying down
    _attach(doc, follower="Rod", side_at=(10, 41, 1))           # its end
    lo, hi = g.bbox(doc.get("Rod").shape)
    assert hi[2] - lo[2] == pytest.approx(10, abs=1e-6)        # stands up
    assert lo[2] == pytest.approx(20, abs=1e-6)


def test_moving_it_by_hand_ends_the_link():
    doc = _scene()
    _attach(doc)
    cyl = doc.get("Cyl")
    doc.scene.replace_shape(cyl.id, g.translate(cyl.shape, (3, 0, 0)))
    assert at.link_of(doc.get("Cyl")) is None
    assert any("Cyl was moved, so it is no longer attached to Cube" in s
               for s in doc.messages)
    _box(doc, "Cube", (0, 0, 0), 40, 30, 35)
    assert _bottom(doc)[0] == pytest.approx((15, 8, 20), abs=1e-6)


def test_moved_together_it_stays_attached():
    doc = _scene()
    _attach(doc)
    with doc.scene.batched():
        for name in ("Cyl", "Cube"):
            o = doc.get(name)
            doc.scene.replace_shape(o.id, g.translate(o.shape, (0, 0, 5)))
    assert at.link_of(doc.get("Cyl")) is not None
    assert _bottom(doc)[0] == pytest.approx((12, 8, 25), abs=1e-6)
    _box(doc, "Cube", (0, 0, 5), 40, 30, 30)
    assert _bottom(doc)[0] == pytest.approx((12, 8, 35), abs=1e-6)


def test_deleting_the_cube_ends_the_link_and_undo_brings_it_back():
    doc = _scene()
    _attach(doc)
    doc.selection.set([doc.get("Cube").id])
    doc.run("delete")
    assert at.link_of(doc.get("Cyl")) is None
    assert any("no longer attached" in s for s in doc.messages)
    doc.history.undo()
    assert at.link_of(doc.get("Cyl")) is not None
    _box(doc, "Cube", (0, 0, 0), 40, 30, 35)
    assert _bottom(doc)[0] == pytest.approx((12, 8, 35), abs=1e-6)


def test_hanging_over_the_edge_is_said_once():
    doc = _scene()
    _attach(doc)
    _box(doc, "Cube", (0, 0, 0), 13, 30, 20)
    _box(doc, "Cube", (0, 0, 0), 13, 30, 22)
    said = [s for s in doc.messages if "hangs over the edge" in s]
    assert len(said) == 1
    assert _bottom(doc)[0] == pytest.approx((12, 8, 22), abs=1e-6)


def test_a_curved_face_is_refused():
    doc = _scene()
    doc.add(g.make_box((50, 0, 0), 2, 2, 2), name="Pin")
    proc = doc._proc
    assert proc.run("attach")
    proc.click_object(doc.get("Pin").id)
    proc.click_object(doc.get("Cyl").id, at=(13, 10, 45))      # its side
    assert not proc.busy
    assert any("that face is curved" in s for s in doc.messages)
    assert at.link_of(doc.get("Pin")) is None


def test_attaching_again_replaces_the_link_without_ending_it():
    doc = _scene()
    _attach(doc)
    _attach(doc, distances=(20, 15))
    assert _bottom(doc)[0] == pytest.approx((20, 15, 20), abs=1e-6)
    assert not any("no longer attached" in s for s in doc.messages)


def test_what_is_drawn():
    doc = _scene()
    _attach(doc, keep=("Edge", "Share"))
    cyl, cube = doc.get("Cyl"), doc.get("Cube")
    marks = at.marks(doc.scene, [cyl.id])
    assert len(marks["faces"]) == 1
    assert sorted(t for _a, _b, t in marks["dims"]) == ["12", "26.7%"]
    assert len(marks["edges"]) == 2
    assert marks["badges"][0][1] == "attached to Cube"
    for foot, p, _t in marks["dims"]:
        assert np.asarray(p) == pytest.approx((12, 8, 20), abs=1e-6)
    marks = at.marks(doc.scene, [cube.id])
    assert len(marks["followers"]) == 1 and not marks["dims"]


def test_properties_read_what_it_is_attached_to_and_unattach_clears_it():
    doc = _scene()
    _attach(doc)
    from serpentine3d.core import user_text
    text = doc.get("Cyl").user_text
    assert user_text.value_of(text, at.LABEL) == \
        "Cube: 12 from an edge, 8 from an edge"
    doc.run("unattach", ["Cyl", ""])
    text = doc.get("Cyl").user_text
    assert not user_text.value_of(text, at.KEY)
    assert not user_text.value_of(text, at.LABEL)


def test_undo_carries_the_link():
    doc = _scene()
    _attach(doc)
    doc.selection.set([doc.get("Cube").id])
    doc.run("move", ["0,0,0", "0,0,10"])
    assert _bottom(doc)[0] == pytest.approx((12, 8, 30), abs=1e-6)
    doc.history.undo()
    assert _bottom(doc)[0] == pytest.approx((12, 8, 20), abs=1e-6)
    _box(doc, "Cube", (0, 0, 0), 40, 30, 25)
    assert _bottom(doc)[0] == pytest.approx((12, 8, 25), abs=1e-6)
