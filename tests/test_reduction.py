import json
import subprocess
import sys
import textwrap
from pathlib import Path
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.engine import Settings, Progress, run
from meshrepair.reduction import propose_reduction, verify_surface_bound, triangle_upper_bounds, quad_upper_bounds


def dense_cube():
    mesh = mm.makeCube()
    settings = mm.SubdivideSettings()
    settings.maxEdgeLen = .15
    settings.maxAngleChangeAfterFlip = 0
    mm.subdivideMesh(mesh, settings)
    return mesh


@pytest.mark.parametrize('angle', [-.1, 45.1, float('nan'), float('inf')])
def test_invalid_angle(angle):
    with pytest.raises(ValueError):
        Settings(collapse_coplanar=True, coplanar_angle_deg=angle).validate()


def test_planar_collapse_reduces_and_keeps_cube_corners():
    mesh = dense_cube()
    result, _ = propose_reduction(mesh, mm, Settings(.02, True, 1, False), Progress(lambda _: None))
    assert result.topology.numValidFaces() < mesh.topology.numValidFaces() / 2
    verts = mn.getNumpyVerts(result)
    for x in [-.5, .5]:
        for y in [-.5, .5]:
            for z in [-.5, .5]:
                assert np.any(np.all(verts == [x, y, z], axis=1))
    bound = verify_surface_bound(mesh, result, .02)
    assert bound['verified']
    assert bound['upper_bound_mm'] <= .02


def test_angle_is_a_crease_threshold_not_edge_flip_setting():
    params = mm.SphereParams()
    params.numMeshVertices = 1000
    mesh = mm.makeSphere(params)
    narrow, _ = propose_reduction(mesh, mm, Settings(.1, True, 0, False), Progress(lambda _: None))
    wider, _ = propose_reduction(mesh, mm, Settings(.1, True, 20, False), Progress(lambda _: None))
    assert narrow.topology.numValidFaces() == mesh.topology.numValidFaces()
    assert wider.topology.numValidFaces() < narrow.topology.numValidFaces()


def test_curved_reduction_bound():
    params = mm.SphereParams()
    params.radius = 1
    params.numMeshVertices = 1000
    mesh = mm.makeSphere(params)
    result, _ = propose_reduction(mesh, mm, Settings(.03, False, 1, True), Progress(lambda _: None))
    assert result.topology.numValidFaces() < mesh.topology.numValidFaces()
    assert verify_surface_bound(mesh, result, .03)['verified']


def test_parallel_decimation_completes_without_callback_deadlock():
    # A subprocess timeout makes this regression fail instead of hanging CI.
    script = textwrap.dedent('''
        from meshlib import mrmeshpy as mm
        from meshrepair.engine import Settings, Progress
        from meshrepair import reduction
        reduction.os.cpu_count = lambda: 2
        params = mm.SphereParams()
        params.radius = 10
        params.numMeshVertices = 30000
        mesh = mm.makeSphere(params)
        events = []
        result, _ = reduction.propose_reduction(mesh, mm, Settings(.02, False, 1, True), Progress(events.append))
        assert result.topology.numValidFaces() < mesh.topology.numValidFaces()
        assert result.topology.isClosed()
        assert events and events[0]['percent'] == 78
    ''')
    process = subprocess.run([sys.executable, '-c', script], capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr


def test_interior_gap_rejected_despite_small_vertex_distances():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    source = mn.meshFromFacesVerts(np.array([[0, 1, 2]], dtype=np.int32), points)
    verts = []
    faces = []
    # All source vertices AND its centroid lie on target patches. Large interior gaps remain.
    for p in [*points, np.array([1/3, 1/3, 0])]:
        i = len(verts)
        verts += [p, p + [.005, 0, 0], p + [0, .005, 0]]
        faces.append([i, i + 1, i + 2])
    target = mn.meshFromFacesVerts(np.array(faces, dtype=np.int32), np.array(verts, dtype=np.float32))
    assert mm.findMaxDistanceSq(mm.MeshPart(source), mm.MeshPart(target)) < .02 ** 2
    assert not verify_surface_bound(source, target, .02)['verified']


def test_uncertain_verification_is_not_success():
    mesh = dense_cube()
    proposal, _ = propose_reduction(mesh, mm, Settings(.02, True), Progress(lambda _: None))
    report = verify_surface_bound(mesh, proposal, .02, max_checks=1)
    assert not report['verified']
    assert report['reason'] == 'verification_work_limit'


def test_triangle_bounds_parallel_plane():
    a = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    b = a + [0, 0, .1]
    assert triangle_upper_bounds(a, b)[0] == pytest.approx(.1)


def test_convex_pair_bound_includes_fold_height():
    source = np.array([[[0., 0., 0.], [1., 1., 0.], [1., 0., 0.]]])
    quad = np.array([[[0., 0., 0.], [1., 1., 0.], [0., 1., 0.], [1., 0., .1]]])
    assert quad_upper_bounds(source, quad)[0] == pytest.approx(.1)
    quad[:, 3, 2] = 0
    assert quad_upper_bounds(source, quad)[0] == pytest.approx(0)


def test_concave_pair_cannot_certify_convex_hull():
    source = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    quad = np.array([[[0., 0., 0.], [.1, .1, 0.], [0., 1., 0.], [1., 0., 0.]]])
    assert np.isinf(quad_upper_bounds(source, quad)[0])


def test_planar_20mm_mesh_verified_at_micron_tolerance():
    mesh = dense_cube()
    mesh.transform(mm.AffineXf3f.linear(mm.Matrix3f.scale(20)))
    result, _ = propose_reduction(mesh, mm, Settings(.001, True), Progress(lambda _: None))
    assert result.topology.numValidFaces() < mesh.topology.numValidFaces() / 2
    assert verify_surface_bound(mesh, result, .001, max_checks=100000)['verified']


def test_exported_simplification_verified_against_repaired_surface(tmp_path):
    source = tmp_path / 'cube.stl'
    mesh = dense_cube()
    mm.saveMesh(mesh, source)
    result = run(source, source.with_name('result.stl'), settings=Settings(.02, True, 1, True, True))
    reduction = result['simplification']
    assert reduction['accepted']
    assert reduction['after'] < reduction['before']
    assert reduction['verification']['verified']
    assert reduction['verification']['upper_bound_mm'] <= .02
    assert result['after']['closed']
    assert result['after']['self_intersecting_faces'] == 0
    assert not result['tolerance_certified']  # original repair remains uncertified
    assert json.loads(Path(result['output'] + '.json').read_text(encoding='utf-8'))['simplification']['accepted']


def test_verification_failure_rolls_back_without_losing_repair(tmp_path, monkeypatch):
    from meshrepair import reduction
    source = tmp_path / 'cube.stl'
    mesh = dense_cube()
    mm.saveMesh(mesh, source)
    monkeypatch.setattr(reduction, 'verify_surface_bound', lambda *a, **kw: {'verified': False, 'reason': 'test_uncertainty'})
    result = run(source, settings=Settings(.02, True, strict_tolerance=True))
    assert not result['simplification']['accepted']
    assert result['after']['triangles'] == mesh.topology.numValidFaces()
    assert Path(result['output']).is_file()
    assert result['after']['closed']
