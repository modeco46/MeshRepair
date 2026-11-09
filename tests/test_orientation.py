import hashlib
import time
from pathlib import Path
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.orientation import orient_consistently, prepare_orientation
from meshrepair.engine import run, Settings


def damaged_cube(path, many=False):
    # Fixed subdivision yields 12,288 faces independent of SDK edge-length heuristics.
    mesh = mm.makeCube(mm.Vector3f(10, 10, 10), mm.Vector3f(-5,-5,-5))
    vertices, faces = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    for _ in range(5):
        edges = np.sort(np.concatenate((faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]])), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        mid = inverse.reshape(3,-1) + len(vertices)
        vertices = np.concatenate((vertices, vertices[unique].mean(axis=1)))
        a,b,c = faces.T; ab,bc,ca = mid
        faces = np.concatenate((np.stack((a,ab,ca),axis=1), np.stack((ab,b,bc),axis=1),
                                np.stack((ca,bc,c),axis=1), np.stack((ab,bc,ca),axis=1))).astype(np.int32)
    mesh = mn.meshFromFacesVerts(faces, vertices)
    mesh.pack()
    vertices, faces = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology).copy()
    top = np.flatnonzero(np.all(vertices[faces, 2] == 5, axis=1))
    if many:
        centers = vertices[faces].mean(axis=1)
        mask = np.zeros(len(faces), bool)
        grid = [(a,b) for a in (-3,-1,1,3) for b in (-3,0,3)][:10]
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            for side in (-5,5):
                on_side = np.all(vertices[faces, axis] == side, axis=1)
                for a,b in grid:
                    mask |= on_side & (np.abs(centers[:,other[0]]-a)<.38) & (np.abs(centers[:,other[1]]-b)<.38)
        selected = np.flatnonzero(mask)
    else:
        centers = vertices[faces[top]].mean(axis=1)
        selected = top[np.argsort(np.sum(centers[:, :2]**2, axis=1))[:180]]
    faces[selected] = faces[selected][:, [0, 2, 1]]
    path.write_text(''.join('v %.9g %.9g %.9g\n' % tuple(v) for v in vertices) +
                    ''.join('f %d %d %d\n' % tuple(f+1) for f in faces), encoding='utf-8')
    return mesh


def test_reversed_patch_preserves_geometry_and_volume(tmp_path):
    path = tmp_path/'patch.obj'
    pristine = damaged_cube(path)
    original = hashlib.sha256(path.read_bytes()).digest()
    report = run(path)
    assert report['orientation']['components_before'] > 1
    assert report['orientation']['components_after'] == 1
    assert report['orientation']['disoriented_edges'] > 0
    assert report['algorithm'] == 'surface_repair'
    assert report['after']['closed']
    assert report['after']['signed_volume_mm3'] == pytest.approx(1000, abs=1e-5)
    assert report['source_deviation']['max_deviation_mm'] == pytest.approx(0, abs=1e-6)
    assert report['after']['triangles'] == pristine.topology.numValidFaces() == 12288
    assert hashlib.sha256(path.read_bytes()).digest() == original


def test_many_reversed_patches_avoid_voxels_and_reduce(tmp_path):
    path = tmp_path/'patches.obj'
    damaged_cube(path, True)
    report = run(path, settings=Settings(.02, True))
    assert report['orientation']['components_before'] > 32
    assert report['prepared']['components'] == 1
    assert report['algorithm'] == 'surface_repair'
    assert report['after']['signed_volume_mm3'] == pytest.approx(1000, abs=1e-4)
    assert report['simplification']['accepted']
    assert report['simplification']['after'] < report['simplification']['before']*.5


def test_40_solids_with_13_inverted_keep_volume_and_shells(tmp_path):
    mesh = mm.Mesh()
    for i in range(40):
        cube = mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(i*2, 0, 0))
        if i < 13:
            cube.topology.flipOrientation()
        mesh.addMesh(cube)
    source = tmp_path/'solids.stl'
    mm.saveMesh(mesh, source)
    report = run(source)
    assert report['after']['components'] == 40
    assert report['after']['signed_volume_mm3'] == pytest.approx(40)
    assert report['source_deviation']['max_deviation_mm'] == 0


def test_open_components_do_not_use_volume_sign():
    vertices = np.array([[0,0,5], [1,0,5], [0,1,5]], np.float32)
    face = np.array([[0,2,1]], np.int32)
    oriented, report = orient_consistently(vertices, face)
    np.testing.assert_array_equal(oriented, face)
    assert report['open_components'] == 1
    assert report['negative_closed_components'] == 0


def test_orientation_720k_native_graph_benchmark():
    params = mm.SphereParams()
    params.numMeshVertices = 360002
    mesh = mm.makeSphere(params)
    vertices = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology).copy()
    faces[::17] = faces[::17][:, [0,2,1]]
    start = time.perf_counter()
    oriented, report = orient_consistently(vertices, faces)
    elapsed = time.perf_counter()-start
    print(f'orientation: {len(faces)} triangles, {elapsed:.3f} s')
    assert report['components'] == 1
    assert report['closed_components'] == 1
    assert elapsed < 5, f'Native graph orientation took {elapsed:.3f}s'
    # Check consistency without constructing a costly Python BFS.
    from meshrepair.orientation import edge_pairs
    assert not edge_pairs(oriented, len(vertices))[2].any()


def test_nan_coordinates_clear_error(tmp_path):
    from meshrepair.inspection import inspect_mesh, export_preview
    mesh = mm.makeCube()
    vertices, faces = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    vertices[0,0] = np.nan
    path = tmp_path/'nan.obj'
    path.write_text(''.join('v %g %g %g\n' % tuple(v) for v in vertices)+
                    ''.join('f %d %d %d\n' % tuple(f+1) for f in faces))
    for func in (run, inspect_mesh, lambda p: export_preview(p, tmp_path/'preview.npz')):
        with pytest.raises((ValueError, RuntimeError), match='NaN/inf'):
            func(path)
