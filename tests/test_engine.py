import json
import math
import subprocess
import sys
import pytest
from meshlib import mrmeshpy as mm
from meshrepair.engine import Settings, grid_estimate, run


def cube(size=.1, base=0):
    return mm.makeCube(mm.Vector3f(size, size, size), mm.Vector3f(base, base, base))


def save(tmp_path, mesh):
    path = tmp_path / 'input.stl'
    mm.saveMesh(mesh, path)
    return path


@pytest.mark.parametrize('tolerance', [0, .0009, .021, math.inf, math.nan])
def test_reject_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        Settings(tolerance_mm=tolerance).validate()


def test_budget_rejects_before_allocation(tmp_path):
    source = save(tmp_path, cube(40))
    out = tmp_path / 'result.stl'
    with pytest.raises(ValueError, match='ячеек'):
        run(source, out, Settings(device='cpu', method='voxel'))
    assert not out.exists()
    assert grid_estimate([40, 40, 40], .001)['cells'] > 64_000_000_000_000


def test_analysis_does_not_create_result(tmp_path):
    report = run(save(tmp_path, cube()))
    assert report['before']['closed']
    assert report['before']['components'] == 1
    assert report['status'] == 'analysis_only'
    assert not report['tolerance_certified']


def test_rebuild_and_report(tmp_path):
    source = save(tmp_path, cube())
    original = source.read_bytes()
    target = tmp_path / 'result.stl'
    report = run(source, target, Settings(device='cpu'))
    assert target.is_file()
    assert report['after']['closed']
    assert report['after']['self_intersecting_faces'] == 0
    assert report['after']['components'] == 1
    assert report['after']['signed_volume_mm3'] > 0
    assert report['vertex_surface_deviation_mm'] >= 0
    assert not report['tolerance_certified']
    assert source.read_bytes() == original
    assert json.loads((tmp_path / 'result.stl.json').read_text(encoding='utf-8'))['output'] == str(target)
    reloaded = mm.loadMesh(target)
    assert reloaded.topology.isClosed()


def test_no_overwrite(tmp_path):
    source = save(tmp_path, cube())
    with pytest.raises(ValueError, match='перезапись'):
        run(source, source)
    target = tmp_path / 'existing.stl'
    target.write_bytes(b'keep me')
    with pytest.raises(ValueError):
        run(source, target)
    assert target.read_bytes() == b'keep me'


def test_inverted_closed_shell(tmp_path):
    mesh = cube()
    mesh.topology.flipOrientation()
    report = run(save(tmp_path, mesh), tmp_path / 'fixed.stl', Settings(device='cpu'))
    assert report['after']['signed_volume_mm3'] > 0
    assert report['after']['closed']


def test_intersecting_shells(tmp_path):
    mesh = cube()
    mesh.addMesh(cube(base=.04))
    report = run(save(tmp_path, mesh), tmp_path / 'union.stl', Settings(device='cpu', method='union'))
    assert report['before']['self_intersecting_faces'] > 0
    assert report['after']['self_intersecting_faces'] == 0
    assert report['after']['components'] == 1


def test_hole(tmp_path):
    mesh = cube()
    mesh.topology.deleteFace(mm.FaceId(0))
    report = run(save(tmp_path, mesh), tmp_path / 'closed.stl', Settings(device='cpu'))
    assert report['before']['boundary_loops'] > 0
    assert report['after']['closed']


def test_separate_parts_are_not_silently_deleted(tmp_path):
    mesh = cube()
    mesh.addMesh(cube(base=.3))
    source = save(tmp_path, mesh)
    target = tmp_path / 'single.stl'
    with pytest.raises(ValueError, match='отдельных замкнутых'):
        run(source, target, Settings(device='cpu'))
    assert not target.exists()
    report = run(source, target, Settings(device='cpu', require_single_component=False))
    assert report['after']['components'] == 2


def test_cli_protocol(tmp_path):
    source = save(tmp_path, cube())
    p = subprocess.run([sys.executable, '-m', 'meshrepair.worker', str(source)],
                       capture_output=True, encoding='utf-8')
    assert p.returncode == 0, p.stderr
    events = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
    assert events[-1]['type'] == 'result'


def test_large_volume_no_longer_requires_voxels(tmp_path):
    source = save(tmp_path, cube(40))
    report = run(source, tmp_path / 'surface.stl', Settings(tolerance_mm=.001))
    assert report['after']['triangles'] == 12
    assert report['geometry_unchanged']
    assert report['vertex_surface_deviation_mm'] == 0
    assert report['grid_estimate']['cells'] > 100_000_000


def test_nested_shell_union_is_explicit(tmp_path):
    mesh = cube(1)
    inner = cube(.2, .2)
    inner.topology.flipOrientation()  # a genuine hollow cavity
    mesh.addMesh(inner)
    source = save(tmp_path, mesh)
    kept = run(source, tmp_path / 'hollow.stl', Settings(require_single_component=False))
    assert kept['after']['components'] == 2
    assert kept['after']['signed_volume_mm3'] == pytest.approx(.992, abs=1e-6)
    union = run(source, tmp_path / 'solid.stl', Settings(method='union'))
    assert union['after']['components'] == 1
    assert union['after']['signed_volume_mm3'] == pytest.approx(1, abs=1e-6)
    assert not union['sampled_tolerance_pass']  # deleted interior is intentionally reported


def test_hole_size_limit_is_separate(tmp_path):
    mesh = cube(4)
    mesh.topology.deleteFace(mm.FaceId(0))
    source = save(tmp_path, mesh)
    with pytest.raises(ValueError, match='Периметр дыры'):
        run(source, tmp_path / 'no.stl')
    report = run(source, tmp_path / 'yes.stl', Settings(max_hole_perimeter_mm=20))
    assert report['after']['closed']


@pytest.mark.parametrize('inverted', [False, True])
def test_voxel_method_still_available(tmp_path, inverted):
    mesh = cube()
    if inverted:
        mesh.topology.flipOrientation()
    report = run(save(tmp_path, mesh), tmp_path / 'voxel.stl', Settings(method='voxel', device='cpu'))
    assert report['after']['closed']
    assert report['after']['triangles'] > 12


def test_cli_errors_are_readable(tmp_path):
    p = subprocess.run([sys.executable, '-m', 'meshrepair.worker', str(tmp_path / 'missing.stl')],
                       capture_output=True, encoding='utf-8')
    assert p.returncode == 1
    assert 'Traceback' not in p.stderr + p.stdout
    assert json.loads(p.stdout)['type'] == 'error'
