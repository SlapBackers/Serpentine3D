"""meshtobrep finds the flat faces and holes an STL was made of (#184).

Each part is modelled, written to STL and read back, so the mesh is what a
real import gives: float32 points and the triangles a CAD export makes.
"""

import math

import numpy as np
import pytest

from serpentine3d.core import geometry as g
from serpentine3d.core.mesh import MeshShape
from serpentine3d.core.mesh_recognise import RecogniseError, recognise
from serpentine3d.fileio import stl
from serpentine3d.scripting import Document


def _as_stl(shape, tmp_path, name="part"):
    path = str(tmp_path / f"{name}.stl")
    stl.export_stl([(name, shape)], path, quality="fine")
    return stl.import_stl(path)[0][1]


def _cylinder_radii(shape):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    radii = []
    for face in g.faces_of(shape):
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            radii.append(surface.Cylinder().Radius())
    return sorted(radii)


def _plate_with_holes():
    plate = g.make_box((0, 0, 0), 60, 40, 8)
    for x in (15, 45):
        plate = g.boolean_difference(
            plate, g.make_cylinder((x, 20, -1), 3, 10))
    return plate


def test_a_plate_with_holes_comes_back_as_an_extrusion(tmp_path):
    mesh = _as_stl(_plate_with_holes(), tmp_path)
    found = recognise(mesh)
    assert found.kind == "extrusion"
    # top, bottom, four sides and one wall per hole
    assert found.faces == 8
    assert found.triangles == len(mesh.triangles)
    assert g.shape_kind(found.shape) == "solid"
    assert g.is_valid(found.shape)


def test_its_holes_are_true_cylinders_of_the_drawn_size(tmp_path):
    found = recognise(_as_stl(_plate_with_holes(), tmp_path))
    assert _cylinder_radii(found.shape) == pytest.approx([3, 3], abs=1e-3)
    exact = 60 * 40 * 8 - 2 * math.pi * 3 ** 2 * 8
    assert g.volume(found.shape) == pytest.approx(exact, rel=1e-4)


def test_a_slot_comes_back_as_two_half_cylinders_and_two_flats(tmp_path):
    plate = g.make_box((0, 0, 0), 60, 30, 6)
    slot = g.boolean_union(g.make_cylinder((20, 15, -1), 4, 8),
                           g.make_cylinder((40, 15, -1), 4, 8))
    slot = g.boolean_union(slot, g.make_box((20, 11, -1), 20, 8, 8))
    found = recognise(_as_stl(g.boolean_difference(plate, slot), tmp_path))
    assert found.kind == "extrusion"
    radii = _cylinder_radii(found.shape)
    assert radii and all(r == pytest.approx(4, abs=1e-3) for r in radii)
    exact = 60 * 30 * 6 - (math.pi * 4 ** 2 + 20 * 8) * 6
    assert g.volume(found.shape) == pytest.approx(exact, rel=1e-4)


def test_an_extrusion_along_another_axis_is_found_too(tmp_path):
    # an L seen from the front, pushed back along y
    step = g.boolean_union(g.make_box((0, 0, 0), 40, 20, 10),
                           g.make_box((0, 0, 10), 15, 20, 10))
    found = recognise(_as_stl(step, tmp_path))
    assert found.kind == "extrusion"
    assert found.faces == 8
    assert g.volume(found.shape) == pytest.approx(40 * 20 * 10 + 15 * 20 * 10,
                                                  rel=1e-6)


def test_a_part_that_is_no_extrusion_keeps_its_flat_faces(tmp_path):
    # a pedestal: the shoulder rules out every axis
    pedestal = g.boolean_union(g.make_box((0, 0, 0), 40, 40, 10),
                               g.make_box((10, 10, 10), 20, 20, 10))
    mesh = _as_stl(pedestal, tmp_path)
    found = recognise(mesh)
    assert found.kind == "faces"
    assert found.faces == 11
    assert g.shape_kind(found.shape) == "solid"
    assert g.volume(found.shape) == pytest.approx(40 * 40 * 10 + 20 * 20 * 10,
                                                  rel=1e-6)


def test_a_mesh_with_nothing_flat_is_left_to_the_facets(tmp_path):
    with pytest.raises(RecogniseError):
        recognise(_as_stl(g.make_sphere((0, 0, 0), 10), tmp_path))


def test_an_open_mesh_is_refused():
    triangle = MeshShape(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
                         np.array([[0, 1, 2]], np.uint32))
    with pytest.raises(RecogniseError, match="not closed"):
        recognise(triangle)


def test_meshtobrep_says_what_it_found(tmp_path):
    doc = Document()
    doc.add(_as_stl(_plate_with_holes(), tmp_path), name="Plate")
    said = doc.run("meshtobrep", ["Plate", ""])
    obj = doc.get("Plate")
    assert obj.kind == "solid"
    assert len(g.faces_of(obj.shape)) == 8
    assert any("became 8 faces (a straight extrusion)" in s for s in said)


def test_meshtobrep_keeps_the_facets_when_nothing_matches(tmp_path):
    doc = Document()
    doc.add(_as_stl(g.make_sphere((0, 0, 0), 10), tmp_path), name="Ball")
    said = doc.run("meshtobrep", ["Ball", ""])
    assert any("kept as triangles" in s for s in said)
    assert doc.get("Ball").kind in ("surface", "solid")


def test_method_faceted_still_gives_one_face_per_triangle(tmp_path):
    doc = Document()
    mesh = _as_stl(g.make_box((0, 0, 0), 10, 10, 10), tmp_path)
    doc.add(mesh, name="Box")
    doc.run("meshtobrep", ["Method=Faceted", "Box", ""])
    assert len(g.faces_of(doc.get("Box").shape)) == len(mesh.triangles)
