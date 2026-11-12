"""Keep usable passes and isolate drifting thin shells on complex repairs."""
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair import engine, reduction
from meshrepair.engine import Settings, Progress, diagnostics


def simplify(mesh, tmp_path, settings=None):
    report = {'after': diagnostics(mesh, mm), 'algorithm': 'sparse_openvdb_winding',
              'steps': [], 'warnings': []}
    result = engine.simplify_attempts(mesh, None, mm, settings or Settings(.04, simplify_curved=True),
        Progress(lambda _: None), report, lambda _: None, tmp_path)
    return result, report['simplification']


@pytest.mark.parametrize('failure', ['distance', 'intersection', 'native'])
def test_earlier_valid_pass_survives_later_failure(tmp_path, monkeypatch, failure):
    mesh = mm.makeUVSphere(2, 60, 60)
    native = reduction.propose_reduction
    selected_counts = []
    def proposal(*args, checkpoint, **kwargs):
        calls = 0
        def checked(candidate, stages):
            nonlocal calls
            calls += 1
            if calls == 1:
                selected_counts.append(candidate.topology.numValidFaces())
            elif failure == 'native':
                raise RuntimeError('later native pass failed')
            return checkpoint(candidate, stages)
        return native(*args, checkpoint=checked, **kwargs)
    monkeypatch.setattr(reduction, 'propose_reduction', proposal)
    original_diagnostics = engine.diagnostics
    call = 0
    def diagnose(candidate, *args):
        nonlocal call
        info = original_diagnostics(candidate, *args)
        # The initial reference is larger than every decimated proposal.
        if info['triangles'] < mesh.topology.numValidFaces():
            call += 1
            if call % 2 == 0 and failure == 'intersection':
                info['self_intersecting_faces'] = 2
        return info
    monkeypatch.setattr(engine, 'diagnostics', diagnose)
    measurements = 0
    def measure(*args):
        nonlocal measurements
        measurements += 1
        return .01 if failure != 'distance' or measurements % 2 else .1
    monkeypatch.setattr(reduction, 'measured_distance', measure)
    result, summary = simplify(mesh, tmp_path)
    assert summary['accepted']
    assert result.topology.numValidFaces() in selected_counts
    assert summary['after'] == result.topology.numValidFaces()
    assert len(summary['stages']) == 1
    assert all(a.get('stopped_reason') for a in summary['attempts'])
    assert diagnostics(result, mm)['self_intersecting_faces'] == 0


def test_search_can_go_below_previous_floor_and_prepares_once(tmp_path, monkeypatch):
    mesh = mm.makeUVSphere(2, 40, 40)
    native_propose, native_prepare = reduction.propose_reduction, reduction.prepare_reduction
    scales, preparations = [], []
    def prepare(*args):
        preparations.append(1)
        return native_prepare(*args)
    def propose(*args, error_scale, **kwargs):
        scales.append(error_scale)
        return native_propose(*args, error_scale=error_scale, **kwargs)
    monkeypatch.setattr(reduction, 'prepare_reduction', prepare)
    monkeypatch.setattr(reduction, 'propose_reduction', propose)
    monkeypatch.setattr(reduction, 'measured_distance', lambda *args: .01 if scales[-1] < .1 else .1)
    result, summary = simplify(mesh, tmp_path)
    assert summary['accepted'] and summary['error_scale'] < .1
    assert len(preparations) == 1
    assert result.topology.numValidFaces() < mesh.topology.numValidFaces()


def test_thin_satellite_rolls_back_without_blocking_main_body():
    mesh = mm.makeUVSphere(2, 100, 100)
    satellite = mm.makeUVSphere(1, 12, 8)
    satellite.transform(mm.AffineXf3f.linear(mm.Matrix3f.scale(mm.Vector3f(.1, .1, .008))))
    satellite.transform(mm.AffineXf3f.translation(mm.Vector3f(4, 0, 0)))
    mesh.addMesh(satellite)
    result, stages = reduction.propose_reduction(mesh, mm, Settings(.04, simplify_curved=True),
        Progress(lambda _: None), error_scale=.475, component_tolerance=.038)
    assert any(d.get('reverted_reason') == 'component_exceeds_tolerance'
               for stage in stages for d in stage['component_passes'])
    info = diagnostics(result, mm)
    assert engine.valid(info) and info['components'] == 2
    assert info['triangles'] < mesh.topology.numValidFaces()/10
    assert reduction.measured_distance(mesh, result, mm) < .038


def intersecting_surface():
    mesh = mm.makeUVSphere(1, 24, 24)
    vertices = mn.getNumpyVerts(mesh).copy()
    vertices[np.argmax(vertices[:, 0])] += [0, .6, 0]
    return mn.meshFromFacesVerts(mn.getNumpyFaces(mesh.topology), vertices*.01)


def test_small_reconstruction_collision_gets_bounded_local_repair():
    mesh = intersecting_surface()
    info = diagnostics(mesh, mm)
    assert info['self_intersecting_faces'] > 0
    report = {}
    result, after = engine.repair_reconstruction_intersections(mesh, info, mm, Settings(.02),
        Progress(lambda _: None), report, lambda _: None)
    item = report['reconstruction_local_repair']
    assert item['accepted'] and item['verification']['verified']
    assert item['verification']['upper_bound_mm'] <= .005
    assert engine.valid(after) and after['components'] == info['components']
    assert diagnostics(mesh, mm)['self_intersecting_faces'] == info['self_intersecting_faces']
    assert result is not mesh


@pytest.mark.parametrize('open_input, cut_band', [(False, False), (True, False), (False, True)])
def test_cpu_surface_band_never_opens_mesh(monkeypatch, open_input, cut_band):
    source = mm.makeCube()
    if open_input:
        source.topology.deleteFace(mm.FaceId(0))
    monkeypatch.setattr(engine, 'DENSE_WINDING_VOXEL_LIMIT', 1)
    calls = []
    def reconstruct(part, offset, params):
        calls.append(params.closeHolesInHoleWindingNumber)
        result = mm.makeCube()
        if cut_band and len(calls) == 1:
            result.topology.deleteFace(mm.FaceId(0))
        return result
    monkeypatch.setattr(mm, 'mcOffsetMesh', reconstruct)
    report = {'warnings': []}
    result = engine.sparse_repair(source, mm, Settings(.02, restore_sharp_edges=False, disable_cuda=True),
        Progress(lambda _: None), report, lambda _: None)
    assert engine.valid(diagnostics(result, mm))
    assert calls == ([False, True] if cut_band else [open_input])
    assert report['voxel_passes'][0]['cpu_closed_surface_band'] == (not open_input)


@pytest.mark.parametrize('failure', ['proof', 'distance', 'shells'])
def test_unsafe_reconstruction_fix_rolls_back(monkeypatch, failure):
    mesh = intersecting_surface()
    info = diagnostics(mesh, mm)
    if failure == 'proof':
        monkeypatch.setattr(reduction, 'verify_surface_bound', lambda *a, **k:
                            {'verified': False, 'reason': 'verification_work_limit'})
    elif failure == 'distance':
        monkeypatch.setattr(reduction, 'measured_distance', lambda *a: 1.)
    else:
        native = engine.diagnostics
        def changed(*args):
            after = native(*args)
            after['components'] += 1
            return after
        monkeypatch.setattr(engine, 'diagnostics', changed)
    report = {}
    result, after = engine.repair_reconstruction_intersections(mesh, info, mm, Settings(.02),
        Progress(lambda _: None), report, lambda _: None)
    assert result is mesh and after is info
    assert not report['reconstruction_local_repair']['accepted']
