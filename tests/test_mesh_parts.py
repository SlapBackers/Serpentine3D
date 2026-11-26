"""An STL of an assembly splits into its parts, not into every piece.

A real file had 1,916 pieces for 434 closed parts: open surfaces and
fragments left by the export. Each closed piece is a part; every other
piece joins the smallest part around it, or what it touches, or else the
loose rest. And a mesh now offers its corners and hole centres as snaps.
"""

import numpy as np
import pytest

from serpentine3d.core import geometry as g
from serpentine3d.core.mesh import MeshShape, merge
from serpentine3d.core.mesh_parts import components, smart_parts
from serpentine3d.core.mesh_snaps import mesh_snap_points
from serpentine3d.fileio import stl
from serpentine3d.scripting import Document


def _stl(shape, tmp_path, name):
    path = str(tmp_path / f"{name}.stl")
    stl.export_stl([(name, shape)], path, quality="fine")
    return stl.import_stl(path)[0][1]


def _patch(x, y, z, size=2.0):
    """An open square: two triangles, the kind of surface an export leaves."""
    return MeshShape([[x, y, z], [x + size, y, z], [x + size, y + size, z],
                      [x, y + size, z]], [[0, 1, 2], [0, 2, 3]])


def _sliver(x, y, z):
    return MeshShape([[x, y, z], [x + 0.5, y, z], [x, y + 0.5, z]],
                     [[0, 1, 2]])


def test_components_join_what_is_linked():
    label = components(6, np.array([0, 1, 4]), np.array([1, 2, 5]))
    assert label.tolist() == [0, 0, 0, 3, 4, 4]


def test_open_bits_join_the_part_around_them(tmp_path):
    case = _stl(g.make_box((0, 0, 0), 50, 40, 30), tmp_path, "case")
    pin = _stl(g.make_box((100, 0, 0), 5, 5, 20), tmp_path, "pin")
    mesh = merge([case, pin, _patch(10, 10, 10), _patch(20, 20, 15),
                  _sliver(101, 1, 5)])
    found = smart_parts(mesh)
    assert found.pieces == 5
    assert found.closed == 2
    assert len(found.parts) == 2
    assert found.folded == 3
    assert found.loose is None
    sizes = sorted(len(p.triangles) for p in found.parts)
    assert sizes == [len(pin.triangles) + 1, len(case.triangles) + 4]


def test_what_belongs_to_no_part_stays_loose_together(tmp_path):
    case = _stl(g.make_box((0, 0, 0), 50, 40, 30), tmp_path, "case")
    mesh = merge([case, _patch(500, 500, 0), _sliver(600, 0, 0)])
    found = smart_parts(mesh)
    assert len(found.parts) == 1
    assert found.loose is not None and len(found.loose.triangles) == 3


def test_a_part_smaller_than_asked_joins_the_one_around_it(tmp_path):
    case = _stl(g.make_box((0, 0, 0), 50, 40, 30), tmp_path, "case")
    nub = _stl(g.make_box((10, 10, 10), 2, 2, 2), tmp_path, "nub")
    mesh = merge([case, nub])
    assert len(smart_parts(mesh).parts) == 2
    assert len(smart_parts(mesh, smallest=5.0).parts) == 1


def test_a_file_of_open_surfaces_keeps_its_large_pieces(tmp_path):
    sheet = MeshShape([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]],
                      [[0, 1, 2], [0, 2, 3]])
    found = smart_parts(merge([sheet, _sliver(10, 10, 0)]))
    assert len(found.parts) == 1


def test_meshparts_makes_one_object_per_part(tmp_path):
    case = _stl(g.make_box((0, 0, 0), 50, 40, 30), tmp_path, "case")
    pin = _stl(g.make_box((100, 0, 0), 5, 5, 20), tmp_path, "pin")
    doc = Document()
    doc.add(merge([case, pin, _patch(10, 10, 10)]), name="Model")
    said = doc.run("meshparts", ["Model", "", ""])
    assert doc.get("Model") is None
    assert doc.get("Model part 1").kind == "mesh"
    assert doc.get("Model part 2").kind == "mesh"
    assert any("3 pieces make 2 parts" in s for s in said)


def test_a_mesh_snaps_to_its_corners(tmp_path):
    mesh = _stl(g.make_box((0, 0, 0), 10, 20, 30), tmp_path, "box")
    ends = sorted(p for p, kind in mesh_snap_points(mesh) if kind == "end")
    expected = sorted((x, y, z) for x in (0, 10) for y in (0, 20)
                      for z in (0, 30))
    assert ends == pytest.approx(expected, abs=1e-4)


def test_a_hole_in_a_mesh_snaps_at_its_centre(tmp_path):
    plate = g.boolean_difference(g.make_box((0, 0, 0), 40, 30, 8),
                                 g.make_cylinder((12, 15, -1), 4, 10))
    mesh = _stl(plate, tmp_path, "plate")
    centres = [p for p, kind in mesh_snap_points(mesh) if kind == "center"]
    assert any(np.allclose(c, (12, 15, 0), atol=1e-3) for c in centres)
    assert any(np.allclose(c, (12, 15, 8), atol=1e-3) for c in centres)
