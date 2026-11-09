import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from meshlib import mrmeshpy as mm
from meshrepair.inspection import inspect_mesh, export_preview


def sample(tmp_path):
    mesh = mm.makeCube()
    opened = mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(3, 0, 0))
    opened.topology.deleteFace(mm.FaceId(0))
    mesh.addMesh(opened)
    source = tmp_path / 'mesh with space.stl'
    mm.saveMesh(mesh, source)
    return source


def test_check_reports_open_and_closed_shells_without_writing(tmp_path):
    source = sample(tmp_path)
    original = hashlib.sha256(source.read_bytes()).digest()
    report = inspect_mesh(source)
    info = report['info']
    assert info['triangles'] == 23
    assert info['vertices'] == 16
    assert info['components'] == 2
    assert info['closed_components'] == info['open_components'] == 1
    assert info['boundary_loops'] == 1
    assert info['boundary_edges'] == 3
    assert not info['closed']
    assert [row['closed'] for row in report['shells']] == [True, False]
    assert list(tmp_path.iterdir()) == [source]
    assert hashlib.sha256(source.read_bytes()).digest() == original


def test_preview_keeps_all_faces_and_does_not_repair(tmp_path):
    source = sample(tmp_path)
    target = tmp_path / 'view.npz'
    report = export_preview(source, target)
    assert not report['checked']
    assert not report['info']['closed']
    with np.load(target, allow_pickle=False) as data:
        assert data['faces'].shape == (23, 3)
        assert data['vertices'].shape == (16, 3)
        assert np.isfinite(data['vertices']).all()
        assert data['faces'].max() < len(data['vertices'])


def test_check_cli_reports_json(tmp_path):
    source = sample(tmp_path)
    p = subprocess.run([sys.executable, '-m', 'meshrepair.worker', str(source), '--check'],
                       capture_output=True, encoding='utf-8', timeout=30)
    assert p.returncode == 0, p.stderr
    events = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
    assert events[-1]['type'] == 'result'
    assert events[-1]['value']['operation'] == 'check'
    assert events[-1]['value']['info']['open_components'] == 1
