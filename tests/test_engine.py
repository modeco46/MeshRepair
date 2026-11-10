import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
from meshlib import mrmeshpy as mm
from meshrepair.engine import Settings, Progress, output_path, run


def cube(size=.1, base=0):
    return mm.makeCube(mm.Vector3f(size, size, size), mm.Vector3f(base, base, base))


def save(tmp_path, mesh):
    path = tmp_path / 'input.stl'
    mm.saveMesh(mesh, path)
    return path


@pytest.mark.parametrize('quality', [0, .0009, .501, math.inf, math.nan])
def test_quality_validation(quality):
    with pytest.raises(ValueError):
        Settings(quality).validate()


@pytest.mark.parametrize('quality', [.001, .02, .1, .5])
def test_valid_quality(quality):
    Settings(quality).validate()


def test_valid_large_mesh_no_voxel_budget(tmp_path):
    source = save(tmp_path, cube(40))
    original = source.read_bytes()
    report = run(source, settings=Settings(.001))
    assert report['after']['triangles'] == 12
    assert report['after']['closed']
    assert report['algorithm'] == 'surface_repair'
    assert source.read_bytes() == original
    assert not report['tolerance_certified']
    assert Path(report['output']).name == 'input_repaired.stl'
    assert json.loads(Path(report['output'] + '.json').read_text(encoding='utf-8'))['status'] == 'repaired'


def test_no_overwrite_and_auto_names(tmp_path):
    source = save(tmp_path, cube())
    with pytest.raises(ValueError):
        run(source, source)
    first = run(source)
    data = Path(first['output']).read_bytes()
    assert output_path(source).name == 'input_repaired_1.stl'
    second = run(source)
    assert second['output'] != first['output']
    assert Path(first['output']).read_bytes() == data


def test_reverse_normals(tmp_path):
    mesh = cube()
    mesh.topology.flipOrientation()
    report = run(save(tmp_path, mesh))
    assert report['after']['signed_volume_mm3'] > 0


def test_overlapping_bodies_union(tmp_path):
    mesh = cube()
    mesh.addMesh(cube(base=.04))
    report = run(save(tmp_path, mesh))
    assert report['before']['self_intersecting_faces'] > 0
    assert report['after']['self_intersecting_faces'] == 0
    assert report['after']['components'] == 1


def test_nested_bodies_union(tmp_path):
    mesh = cube(1)
    mesh.addMesh(cube(.2, .2))
    report = run(save(tmp_path, mesh))
    assert report['after']['components'] == 1
    assert report['after']['signed_volume_mm3'] == pytest.approx(1, abs=1e-6)


def test_disjoint_bodies_are_saved_without_invented_bridges(tmp_path):
    mesh = cube()
    mesh.addMesh(cube(base=.3))
    report = run(save(tmp_path, mesh))
    assert report['after']['components'] == 2
    assert report['status'] == 'repaired_multiple_shells'
    assert any('перемычки' in x for x in report['warnings'])
    assert Path(report['output']).is_file()


def test_hole_not_blocked_by_perimeter(tmp_path):
    mesh = cube(4)
    mesh.topology.deleteFace(mm.FaceId(0))
    report = run(save(tmp_path, mesh))
    assert report['before']['boundary_loops'] == 1
    assert report['after']['closed']


def test_1150_open_components_fill_before_reconstruction(tmp_path):
    mesh = mm.Mesh()
    for index in range(1150):
        part = cube(.04)
        part.topology.deleteFace(mm.FaceId(0))
        part.transform(mm.AffineXf3f.translation(mm.Vector3f(
            (index % 10) * .08, ((index // 10) % 10) * .08, (index // 100) * .08)))
        mesh.addMesh(part)
    events = []
    report = run(save(tmp_path, mesh), settings=Settings(.005), progress=events.append)
    assert report['before']['components'] == 1150
    assert report['before']['boundary_loops'] == 1150
    assert report['algorithm'] == 'closed_components_preserved'
    assert report['hole_filling']['filled'] == 1150
    assert report['after']['closed']
    assert report['after']['self_intersecting_faces'] == 0
    assert report['after']['components'] == 1150
    assert events[-1]['percent'] == 100
    assert [e['percent'] for e in events] == sorted(e['percent'] for e in events)
    assert any(42 < e['percent'] < 78 for e in events)


def test_progress_does_not_go_backwards():
    events = []
    p = Progress(events.append)
    p.update(60, 'GPU')
    p.update(42, 'CPU retry')
    p.update(100, 'Done')
    assert [e['percent'] for e in events] == [60, 60, 100]


def test_cli_utf8_protocol(tmp_path):
    source = save(tmp_path, cube())
    p = subprocess.run([sys.executable, '-m', 'meshrepair.worker', str(source), '--quality', '.5'],
                       capture_output=True, encoding='utf-8')
    assert p.returncode == 0, p.stderr
    events = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
    assert events[-1]['type'] == 'result'
    assert any(e['type'] == 'progress' and e['value']['percent'] == 100 for e in events)


def test_cli_errors_no_traceback(tmp_path):
    p = subprocess.run([sys.executable, '-m', 'meshrepair.worker', str(tmp_path / 'missing.stl')],
                       capture_output=True, encoding='utf-8')
    assert p.returncode == 1
    assert 'Traceback' not in p.stderr + p.stdout
    assert json.loads(p.stdout)['type'] == 'error'
