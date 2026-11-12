"""0.9 regressions: real geometry, native stage ownership and no hidden IO."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair import engine, holes, reduction, storage
from meshrepair.engine import Settings, Progress, diagnostics

DATA = Path(__file__).parent/'data'/'test.stl'


@pytest.fixture(scope='module')
def real_mesh():
    assert DATA.stat().st_size > 9_000_000, 'Missing or truncated tests/data/test.stl; retrieve the complete repository'
    from meshrepair.orientation import prepare_orientation
    mesh, orientation = prepare_orientation(mm.loadMesh(DATA), mm)
    mesh, filled = holes.fill_simple_holes(mesh, mm, .02)
    assert (orientation['components_before'], orientation['components_after']) == (2147, 1)
    assert (filled['holes_before'], filled['holes_after']) == (5, 0)
    assert diagnostics(mesh, mm)['self_intersecting_faces'] == 42
    return mesh


def test_real_cpu_repair_under_one_minute_without_voxels(tmp_path, monkeypatch):
    import shutil
    source = tmp_path/'input'/'test.stl'; source.parent.mkdir()
    shutil.copyfile(DATA, source)
    def forbidden(*a, **k): pytest.fail('Local repair must not voxelize or prove in measured mode')
    monkeypatch.setattr(engine, 'sparse_repair', forbidden)
    monkeypatch.setattr(reduction, 'verify_surface_bound', forbidden)
    reads, writes = [], []
    load, save = mm.loadMesh, mm.saveMesh
    def read(path, *args):
        reads.append(Path(path).name); return load(path, *args)
    def write(mesh, path, *args):
        writes.append(Path(path).name); return save(mesh, path, *args)
    monkeypatch.setattr(mm, 'loadMesh', read); monkeypatch.setattr(mm, 'saveMesh', write)
    r = engine.run(source, settings=Settings(.02, True, 5, True, disable_cuda=True), work_dir=tmp_path/'job')
    assert r['algorithm'] == 'local_cut_and_fill'
    assert r['local_repair']['accepted']
    assert r['after']['closed'] and r['after']['topology_valid'] and r['after']['self_intersecting_faces'] == 0
    assert 20_000 < r['after']['triangles'] < 30_000
    assert r['simplification']['accepted']
    assert r['simplification']['verification']['mode'] == 'measured_not_proven'
    assert r['simplification']['verification']['max_deviation_mm'] <= .02
    assert r['elapsed_seconds'] < 60
    assert r['source_deviation']['max_deviation_mm'] == pytest.approx(.091, abs=.002)
    np.testing.assert_allclose(r['source_deviation']['top10'][0]['coordinates_mm'], [3.15,3.2,3.2], atol=.15)
    assert reads == ['test.stl', 'result.stl'] and writes == ['result.stl']
    assert sorted(p.name for p in (tmp_path/'job').iterdir()) == ['deviation.npz', 'result.stl']
    assert list(source.parent.iterdir()) == [source]
    small = [s for s in r['local_repair']['shells'] if s['triangles'] < 50]
    assert len(small) == 1 and small[0]['triangles'] == 10
    assert r['acceleration']['cuda_disabled']
    assert r['stage_timings'] and all('compute_device' in t for t in r['stage_timings'])


def test_real_sharpening_probe_rejects_before_full_calculation(real_mesh, monkeypatch):
    native = mm.generalOffsetMesh
    steps = []
    def sharp(part, offset, params):
        steps.append(('sharp', params.voxelSize))
        assert params.voxelSize == pytest.approx(.08)
        return native(part, offset, params)
    def plain(part, offset, params):
        steps.append(('plain', params.voxelSize))
        return mm.makeCube()
    monkeypatch.setattr(mm, 'generalOffsetMesh', sharp)
    monkeypatch.setattr(mm, 'offsetMesh', plain)
    r = {'warnings': []}
    engine.sparse_repair(mm.Mesh(real_mesh), mm, Settings(disable_cuda=True), Progress(lambda _: None), r, lambda _: None)
    assert [kind for kind, _ in steps] == ['sharp', 'plain']
    assert steps[-1][1] == pytest.approx(.02)
    assert not r['sharpening']['accepted']
    assert r['sharpening']['probe']['self_intersecting_faces'] > 0


def test_local_failure_uses_one_full_voxel_pass(tmp_path, monkeypatch):
    mesh = mm.makeCube(); mesh.addMesh(mm.makeCube(mm.Vector3f(1,1,1), mm.Vector3f(0,0,0)))
    source = tmp_path/'overlap.stl'; mm.saveMesh(mesh, source)
    def fail(*a, **k): raise RuntimeError('local repair unavailable')
    monkeypatch.setattr(mm.SelfIntersections, 'fix', fail)
    calls = []
    native = mm.offsetMesh
    def offset(part, value, params):
        calls.append(params.voxelSize); return native(part, value, params)
    monkeypatch.setattr(mm, 'offsetMesh', offset)
    r = engine.run(source, settings=Settings(.1, restore_sharp_edges=False, disable_cuda=True), work_dir=tmp_path/'work')
    assert r['local_repair']['attempted'] and not r['local_repair']['accepted']
    assert r['algorithm'] == 'sparse_openvdb_winding'
    assert calls == pytest.approx([.1])
    assert r['after']['closed'] and not r['after']['self_intersecting_faces']


def test_adaptive_retry_uses_measured_violation(tmp_path, monkeypatch):
    m = mm.makeCube(); p = mm.SubdivideSettings(); p.maxEdgeLen = .1; mm.subdivideMesh(m,p)
    path = tmp_path/'dense.stl'; mm.saveMesh(m,path)
    distances = iter([.04])
    # Every later candidate/pass is within tolerance; the number of distance
    # checks is allowed to grow when intermediate passes are validated.
    monkeypatch.setattr(reduction,'measured_distance',lambda *args: next(distances, .001))
    r = engine.run(path, settings=Settings(.02, True), work_dir=tmp_path/'work')
    attempts = r['simplification']['attempts']
    assert len(attempts) <= 6
    assert attempts[0]['error_scale'] == pytest.approx(.475)
    assert attempts[1]['error_scale'] < attempts[0]['error_scale']
    assert attempts[0]['verification']['max_deviation_mm'] == .04
    assert r['simplification']['accepted']


def test_roundtrip_failure_rolls_back_from_memory(tmp_path, monkeypatch):
    m = mm.makeCube(); p=mm.SubdivideSettings(); p.maxEdgeLen=.1; mm.subdivideMesh(m,p)
    source=tmp_path/'source.stl'; mm.saveMesh(m,source)
    native=mm.loadMesh; count=[]
    def load(path,*args):
        result=native(path,*args)
        if Path(path).name == 'result.stl':
            count.append(True)
            if len(count)==1:
                result.topology.deleteFace(mm.FaceId(0)); result.invalidateCaches()
        return result
    monkeypatch.setattr(mm,'loadMesh',load)
    r=engine.run(source,settings=Settings(.02,True),work_dir=tmp_path/'job')
    assert r['final_roundtrips']==2
    assert not r['simplification']['accepted']
    assert r['simplification']['reason']=='export_validation_failed'
    assert r['after']['triangles']==m.topology.numValidFaces()
    assert r['after']['closed']


def test_remove_small_shells_is_explicit(tmp_path):
    m=mm.makeCube(); p=mm.SubdivideSettings();p.maxEdgeLen=.15;mm.subdivideMesh(m,p)
    m.addMesh(mm.makeCube(mm.Vector3f(.01,.01,.01),mm.Vector3f(2,0,0)))
    source=tmp_path/'source.stl';mm.saveMesh(m,source)
    r=engine.run(source,settings=Settings(remove_small_shells=True),work_dir=tmp_path/'job')
    assert r['after']['components']==1
    assert r['removed_shells'][0]['triangles']==12
    assert r['removed_shells'][0]['center_mm'][0]>2


def test_no_cuda_skips_native_cuda_initialization(monkeypatch):
    from meshrepair import compute
    def forbidden(): pytest.fail('CUDA must not be initialized')
    monkeypatch.setattr(compute, 'cuda_module', forbidden)
    report=compute.capabilities(True)
    projector=compute.Projector(mm.makeCube(),mm,report)
    assert not projector.gpu and not report['cuda_available']
