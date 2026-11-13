"""End-to-end regressions for conditional patches and full Sharpening rollback."""
import ast
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair import engine
from meshrepair.engine import Settings, SharpeningRetryRequired


def uv_sphere_with_38_holes():
    mesh = mm.makeUVSphere(5, 200, 200)
    mesh.pack()
    verts, faces = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    centers = verts[faces].mean(axis=1)
    keep = np.ones(len(faces), dtype=bool)
    for center, radius in [((0, 0, 5), 1.5), ((5, 0, 0), .8)]:
        keep &= np.linalg.norm(centers-np.array(center), axis=1) > radius
    for index in np.random.default_rng(2).choice(len(faces), 40, replace=False):
        keep &= np.linalg.norm(centers-centers[index], axis=1) > .1
    return mn.meshFromFacesVerts(faces[keep], verts)


def test_uv_sphere_local_patches_surface_and_cpu_time(tmp_path, monkeypatch):
    source = tmp_path/'sphere.stl'
    mm.saveMesh(uv_sphere_with_38_holes(), source)
    def forbidden(*args, **kwargs):
        pytest.fail('The sphere must be repaired locally without CUDA or voxel reconstruction')
    monkeypatch.setattr(engine, 'sparse_repair', forbidden)
    monkeypatch.setattr(engine, 'cuda_module', forbidden)
    started = time.monotonic()
    report = engine.run(source, settings=Settings(.05, disable_cuda=True), work_dir=tmp_path/'job')
    elapsed = time.monotonic()-started
    result = mm.loadMesh(report['output'])
    assert engine.valid(engine.diagnostics(result, mm))
    assert report['after']['components'] == 1
    assert report['algorithm'] == 'local_cut_and_fill'
    holes = report['hole_filling']
    assert (holes['holes_before'], holes['holes_after'], holes['filled']) == (38, 0, 38)
    # MeshLib's Windows build produces one additional colliding tiny patch.
    # The invariant is local repair of every provisional patch, not identical
    # floating-point triangulation on two native SDK builds.
    assert holes['accepted_with_local_fix'] >= 4
    assert 0 < holes['intersections_before_local_fix'] <= report['local_repair']['threshold_faces']
    assert holes['intersections_after_local_fix'] == 0
    assert holes['pending_local_fix'] == 0
    assert sum(bool(x.get('accepted_with_local_fix')) for x in holes['details']) == holes['accepted_with_local_fix']
    # For every point on every triangle: the maximum radius is at a vertex;
    # the minimum is the closest point of the complete surface to the origin.
    # This bounds the entire surface, including triangle interiors, not samples.
    maximum = np.linalg.norm(mn.getNumpyVerts(result).astype(float), axis=1).max()
    minimum = math.sqrt(mm.findProjection(mm.Vector3f(), mm.MeshPart(result)).distSq)
    deviation = max(abs(maximum-5), abs(minimum-5))
    print(f'UV sphere: {elapsed:.3f} CPU seconds, surface error {deviation:.6f} mm')
    assert deviation <= .05
    assert elapsed <= 30


def overlapping_cubes_source(tmp_path):
    mesh = mm.makeCube()
    mesh.addMesh(mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0)))
    source = tmp_path/'overlap.stl'
    mm.saveMesh(mesh, source)
    return source


def mock_full_sharpening_failure(monkeypatch, calls):
    def sharp(part, value, params):
        calls.append(('sharp', params.voxelSize, params.fwn))
        result = mm.makeCube()
        if params.voxelSize < .2:
            result.addMesh(mm.makeCube(mm.Vector3f(1, 1, 1), mm.Vector3f(0, 0, 0)))
        return result
    native = mm.offsetMesh
    def plain(part, value, params):
        calls.append(('plain', params.voxelSize, params.fwn))
        return native(part, value, params)
    def forbidden(*args):
        pytest.fail('CutAndFill must not repair the full Sharpening result')
    monkeypatch.setattr(mm, 'generalOffsetMesh', sharp)
    monkeypatch.setattr(mm, 'offsetMesh', plain)
    monkeypatch.setattr(mm.SelfIntersections, 'fix', forbidden)


@pytest.mark.parametrize('cuda', [False, True])
def test_full_sharpening_failure_runs_plain_offset_end_to_end(tmp_path, monkeypatch, cuda):
    source = overlapping_cubes_source(tmp_path)
    calls, initializations = [], []
    mock_full_sharpening_failure(monkeypatch, calls)
    def winding(mesh):
        initializations.append(True)
        # The CPU implementation of the same native interface lets CI verify
        # ownership/reuse without requiring a physical CUDA device.
        return mm.FastWindingNumber(mesh)
    monkeypatch.setattr(engine, 'cuda_module', lambda: SimpleNamespace(FastWindingNumber=winding))
    settings = Settings(.1, local_fix_max_faces=0, local_fix_fraction=0, disable_cuda=not cuda, union_max_seconds=0)
    report = engine.run(source, settings=settings, work_dir=tmp_path/'job')
    assert [x[0] for x in calls] == ['sharp', 'sharp', 'plain']
    assert [x[1] for x in calls] == pytest.approx([.4, .1, .1])
    assert engine.valid(report['after']) and report['after']['components'] == 1
    assert engine.valid(engine.diagnostics(mm.loadMesh(report['output']), mm))
    sharp = report['sharpening']
    assert sharp['probe']['closed'] and sharp['full_validation']['closed']
    assert sharp['full_validation']['self_intersecting_faces'] > 0
    assert not sharp['accepted']
    fallback = sharp['fallback']
    assert fallback['reason'] == 'full_sharpening_invalid'
    assert fallback['automatic'] and fallback['attempted'] and fallback['accepted']
    assert 0 < fallback['elapsed_seconds'] and fallback['estimated_seconds'] <= 120
    assert len(initializations) == int(cuda)
    assert all(call[2] is calls[0][2] for call in calls)


def test_expensive_sharpening_retry_is_deferred_end_to_end(tmp_path, monkeypatch):
    source = overlapping_cubes_source(tmp_path)
    calls = []
    mock_full_sharpening_failure(monkeypatch, calls)
    settings = Settings(.1, local_fix_max_faces=0, local_fix_fraction=0, disable_cuda=True,
                        sharpening_fallback_max_seconds=0, union_max_seconds=0)
    with pytest.raises(SharpeningRetryRequired, match='Повторить без Sharpening') as error:
        engine.run(source, settings=settings, work_dir=tmp_path/'job')
    assert [x[0] for x in calls] == ['sharp', 'sharp']
    assert not (tmp_path/'job'/'result.stl').exists()
    report = error.value.report
    assert report['retry_without_sharpening']
    assert not report['sharpening']['fallback']['attempted']
    assert not report['sharpening']['fallback']['automatic']
    json.dumps(report, allow_nan=False)
    # The settings snapshot used by the GUI succeeds unchanged except Sharpening.
    result = engine.run(source, settings=replace(settings, restore_sharp_edges=False), work_dir=tmp_path/'retry')
    assert engine.valid(result['after'])
    assert [x[0] for x in calls] == ['sharp', 'sharp', 'plain']


@pytest.mark.parametrize('threshold', [0, 200])
def test_conditional_patch_rollback_restores_safe_holes(tmp_path, monkeypatch, threshold):
    source = tmp_path/'sphere.stl'
    mm.saveMesh(uv_sphere_with_38_holes(), source)
    def invalid_fix(mesh, params):
        mesh.topology.deleteFace(mm.FaceId(100))
        mesh.invalidateCaches()
    monkeypatch.setattr(mm.SelfIntersections, 'fix', invalid_fix)
    inputs = []
    def plain(part, value, params):
        inputs.append(engine.diagnostics(part.mesh, mm))
        return mm.makeUVSphere(5, 200, 200)
    monkeypatch.setattr(mm, 'offsetMesh', plain)
    settings = Settings(.05, restore_sharp_edges=False, disable_cuda=True,
                        local_fix_max_faces=threshold, local_fix_fraction=0)
    report = engine.run(source, settings=settings, work_dir=tmp_path/'job')
    holes = report['hole_filling']
    assert holes['rolled_back_local_fix'] >= 4
    assert holes['accepted_with_local_fix'] == 0 and holes['pending_local_fix'] == 0
    assert holes['holes_after'] >= 4 and holes['filled'] + holes['holes_after'] == 38
    assert holes['conditional_attempt']['pending_local_fix'] == holes['rolled_back_local_fix']
    assert len(inputs) == 1 and inputs[0]['boundary_loops'] == holes['holes_after']
    assert inputs[0]['self_intersecting_faces'] == 0
    assert engine.valid(report['after'])


def test_bundle_harness_does_not_import_application():
    path = Path(__file__).with_name('check_bundle.py')
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        modules = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        assert not any(name.split('.')[0] == 'meshrepair' for name in modules), 'Standalone bundle harness cannot import the source application'


def test_internal_pr_jobs_are_not_skipped():
    workflow = (Path(__file__).parents[1]/'.github/workflows/check.yml').read_text()
    assert 'pull_request:' in workflow and 'branches: [main]' in workflow
    assert 'github.event_name' not in workflow
    assert 'head.repo' not in workflow


def test_worker_persists_retry_report_in_terminal_file(tmp_path, monkeypatch):
    import sys
    from meshrepair import worker
    report = {'retry_without_sharpening': True, 'sharpening': {
        'fallback': {'estimated_seconds': 121, 'attempted': False}}}
    def fail(*args, **kwargs):
        raise SharpeningRetryRequired(report)
    monkeypatch.setattr(worker, 'run', fail)
    terminal = tmp_path/'terminal.json'
    monkeypatch.setattr(sys, 'argv', ['worker', 'unused.stl', '--result-file', str(terminal)])
    assert worker.main() == 1
    event = json.loads(terminal.read_text(encoding='utf-8'))
    assert event['type'] == 'retry_without_sharpening'
    assert event['value']['report'] == report
    assert 'Повторить без Sharpening' in event['value']['message']


def test_cuda_failure_stays_on_cpu_for_sharpening_retry(tmp_path, monkeypatch):
    source = overlapping_cubes_source(tmp_path)
    calls, initializations, failures = [], [], []
    mock_full_sharpening_failure(monkeypatch, calls)
    sharp = mm.generalOffsetMesh
    def fail_gpu(part, value, params):
        if params.fwn is not None:
            failures.append(True)
            raise RuntimeError('simulated CUDA failure')
        return sharp(part, value, params)
    def winding(mesh):
        initializations.append(True)
        return mm.FastWindingNumber(mesh)
    monkeypatch.setattr(engine, 'cuda_module', lambda: SimpleNamespace(FastWindingNumber=winding))
    monkeypatch.setattr(mm, 'generalOffsetMesh', fail_gpu)
    report = engine.run(source, settings=Settings(.1, local_fix_max_faces=0, local_fix_fraction=0, union_max_seconds=0),
                        work_dir=tmp_path/'job')
    assert initializations == failures == [True]
    assert [x[0] for x in calls] == ['sharp', 'sharp', 'plain']
    assert all(call[2] is None for call in calls)
    assert report['compute_device'] == 'cpu_native_parallel'
    assert report['sharpening']['fallback']['accepted']
    assert engine.valid(report['after'])
