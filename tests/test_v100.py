"""Regression tests for 1.0.0: large multi-shell models that used to end without a result."""
import os
from pathlib import Path

import numpy as np
import pytest
from meshlib import mrmeshpy as mm

from meshrepair import deviation, engine, reduction
from meshrepair.engine import Progress, Settings


def subdivided(mesh, edge):
    s = mm.SubdivideSettings()
    s.maxEdgeLen, s.maxAngleChangeAfterFlip, s.maxEdgeSplits = edge, 0, 50_000_000
    mm.subdivideMesh(mesh, s)
    mesh.pack()
    return mesh


def plate_with_bail(cavities):
    """A flat plate, a bail ring crossing its wall, and optional flat internal voids (inverted closed shells)."""
    plate = subdivided(mm.makeCube(mm.Vector3f(6, 4, 1), mm.Vector3f(0, 0, 0)), .1)
    ring = mm.makeTorus(.9, .25, 64, 20)
    ring.transform(mm.AffineXf3f.translation(mm.Vector3f(3, 4.3, .5)))
    plate.addMesh(ring)
    for i in range(cavities):
        void = mm.makeCube(mm.Vector3f(.4, .4, .12), mm.Vector3f(.8 + i*.6, 1.8, .44))
        void.topology.flipOrientation()
        plate.addMesh(void)
    return plate


def run_plate(tmp_path, cavities=0, **kwargs):
    source = tmp_path/'plate.stl'
    mm.saveMesh(plate_with_bail(cavities), source)
    return engine.run(source, settings=Settings(.05, True, 2.0, True, disable_cuda=True, **kwargs),
                      work_dir=tmp_path/'job')


def test_shells_that_only_cross_each_other_are_merged_by_boolean_union(tmp_path, monkeypatch):
    voxel_calls = []
    native = mm.offsetMesh
    monkeypatch.setattr(mm, 'offsetMesh', lambda *a, **k: voxel_calls.append(1) or native(*a, **k))
    r = run_plate(tmp_path, cavities=0)
    union = r['union_attempt']
    assert r['algorithm'] == 'solid_union' and union['accepted'] and union['pairs_between_shells'] > 0
    assert union['pairs_inside_shell'] == 0
    assert not voxel_calls and 'voxel_estimate' not in r
    assert r['after']['components'] == 1 and r['after']['closed'] and r['after']['self_intersecting_faces'] == 0
    assert r['simplification']['accepted']
    # wall vertices that end up inside the bail's tube are internal, not a shape deviation (concave edges too)
    deviation_report = r['source_deviation']
    assert deviation_report['interior_vertices'] > 0
    assert deviation_report['max_deviation_mm'] <= 2*.05 and deviation_report['exceeding_vertices'] == 0
    assert deviation_report['max_deviation_all_mm'] > .1


def test_union_failure_falls_back_to_voxels_and_removes_cavities(tmp_path):
    # union disabled by a zero time budget: the voxel result contains zero-thickness internal sheets
    r = run_plate(tmp_path, cavities=6, union_max_seconds=0.0)
    assert r['union_attempt']['reason'] == 'union_failed'
    assert r['algorithm'] == 'sparse_openvdb_winding'
    removed = r['removed_cavities']
    assert removed and all(item['volume_mm3'] <= 1e-3*.05**3 for item in removed)
    assert r['after']['components'] == 1 and r['after']['self_intersecting_faces'] == 0
    assert r['simplification']['accepted'] and r['simplification']['after'] < 2000
    assert r['simplification']['attempts'][0].get('reason') is None


def test_cavity_removal_keeps_the_only_real_shell(tmp_path):
    mesh = mm.makeCube(mm.Vector3f(2, 2, 2), mm.Vector3f(0, 0, 0))
    mesh.topology.flipOrientation()                       # the only shell is inverted: nothing may be dropped
    cleaned, after, original = engine.drop_cavities(mesh, {}, mm, {}, lambda _: None, .05)
    assert original is None and cleaned is mesh
    solid = mm.makeCube(mm.Vector3f(2, 2, 2), mm.Vector3f(0, 0, 0))
    void = mm.makeCube(mm.Vector3f(.5, .5, .5), mm.Vector3f(.7, .7, .7))
    void.topology.flipOrientation()
    solid.addMesh(void)
    report = {}
    cleaned, after, original = engine.drop_cavities(solid, {}, mm, report, lambda _: None, .05)
    assert original is solid and after['components'] == 1 and len(report['removed_cavities']) == 1


def test_deviation_ignores_internal_surfaces_but_reports_them(tmp_path):
    outer = mm.makeCube(mm.Vector3f(10, 10, 10), mm.Vector3f(0, 0, 0))
    source = mm.Mesh(outer)
    inner = mm.makeCube(mm.Vector3f(2, 2, 2), mm.Vector3f(4, 4, 4))      # internal sheet far from the result surface
    source.addMesh(inner)
    stats = deviation.source_deviation(source, outer, mm, .05, tmp_path/'map.npz')
    assert stats['interior_classified'] and stats['interior_vertices'] == 8
    assert stats['max_deviation_mm'] < 1e-3 and stats['exceeding_vertices'] == 0
    assert stats['max_deviation_all_mm'] == pytest.approx(4.0, abs=.01)
    outside = mm.Mesh(outer)
    outside.addMesh(mm.makeCube(mm.Vector3f(2, 2, 2), mm.Vector3f(12, 4, 4)))   # a real part the result lacks
    stats = deviation.source_deviation(outside, outer, mm, .05, tmp_path/'map2.npz')
    assert stats['interior_vertices'] == 0 and stats['max_deviation_mm'] > 1.5 and stats['exceeding_vertices'] > 0


def multi_shell_mesh():
    mesh = subdivided(mm.makeCube(mm.Vector3f(8, 6, 2), mm.Vector3f(0, 0, 0)), .12)
    for index in range(4):
        ball = mm.makeUVSphere(.02 + .004*index, 24, 24)                 # smaller than the tolerance
        ball.transform(mm.AffineXf3f.translation(mm.Vector3f(9 + index*.5, 3, 1)))
        mesh.addMesh(ball)
    big = mm.makeUVSphere(1, 90, 90)
    big.transform(mm.AffineXf3f.translation(mm.Vector3f(12, 3, 1)))
    mesh.addMesh(big)
    return mesh


def test_shellwise_reduction_keeps_every_shell():
    mesh = multi_shell_mesh()
    shells = mm.MeshComponents.getNumComponents(mm.MeshPart(mesh))
    whole, _ = reduction.propose_reduction(mesh, mm, Settings(.05, True, 2.0, True), Progress(lambda _: None), .6)
    assert mm.MeshComponents.getNumComponents(mm.MeshPart(whole)) < shells      # the whole-mesh pass loses shells
    by_shell, stages = reduction.propose_reduction_by_shell(mesh, mm, Settings(.05, True, 2.0, True),
                                                            Progress(lambda _: None), .6)
    assert mm.MeshComponents.getNumComponents(mm.MeshPart(by_shell)) == shells
    assert by_shell.topology.numValidFaces() < mesh.topology.numValidFaces()*.01
    assert not mm.findSelfCollidingTrianglesBS(mm.MeshPart(by_shell)).count() and by_shell.volume() > 0
    assert stages and stages[0]['shells'] == shells


def test_engine_switches_to_shellwise_after_one_whole_mesh_failure(tmp_path, monkeypatch):
    source = tmp_path/'multi.stl'
    mm.saveMesh(multi_shell_mesh(), source)
    real = reduction.propose_reduction
    calls = []

    def flaky(mesh, mm_, settings, progress, error_scale=1.0, **kw):
        candidate, stages = real(mesh, mm_, settings, progress, error_scale, **kw)
        calls.append(mesh.topology.numValidFaces())
        if len(calls) == 1:                                              # emulate a pass that collapses a shell
            regions = sorted(mm_.MeshComponents.getAllComponents(mm_.MeshPart(candidate)), key=lambda x: x.count())
            candidate = candidate.cloneRegion(candidate.topology.getValidFaces() - regions[0])
        return candidate, stages
    monkeypatch.setattr(reduction, 'propose_reduction', flaky)
    r = engine.run(source, settings=Settings(.05, True, 2.0, True, disable_cuda=True), work_dir=tmp_path/'job')
    attempts = r['simplification']['attempts']
    assert attempts[0]['reason'] in {'topology_changed', 'kept_shell_count_changed'}
    assert r['simplification']['shellwise'] and attempts[1]['shellwise'] and r['simplification']['accepted']
    assert r['after']['components'] == r['before']['components']


def test_planar_only_pass_is_not_partitioned(monkeypatch):
    cube = subdivided(mm.makeCube(mm.Vector3f(10, 10, 10), mm.Vector3f(0, 0, 0)), .1)
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    monkeypatch.setattr(reduction, 'cpu_count', lambda: 8)
    parts = []
    native = mm.decimateMesh
    monkeypatch.setattr(mm, 'decimateMesh', lambda m, params: parts.append(params.subdivideParts) or native(m, params))
    result, _ = reduction.propose_reduction(cube, mm, Settings(.02, collapse_coplanar=True, coplanar_angle_deg=2.0),
                                           Progress(lambda _: None), .45)
    assert parts == [1] and result.topology.numValidFaces() == 12


def test_voxel_estimate_uses_grid_cells_and_a_timed_probe(monkeypatch):
    mesh = mm.makeCube(mm.Vector3f(5, 5, 5), mm.Vector3f(0, 0, 0))
    steps = []
    operation = lambda part, offset, params: steps.append(params.voxelSize) or mm.makeCube()
    monkeypatch.setattr(mm, 'offsetMesh', operation)
    monkeypatch.setattr(mm, 'mcOffsetMesh', operation)
    report = {'warnings': []}
    settings = Settings(.01, restore_sharp_edges=False, disable_cuda=True, voxel_warn_seconds=0.0)
    engine.sparse_repair(mm.Mesh(mesh), mm, settings, Progress(lambda _: None), report, lambda _: None)
    estimate = report['voxel_estimate']
    assert estimate['grid'] == [502, 502, 502] and estimate['cells'] == pytest.approx(502**3)
    assert estimate['estimated_seconds'] is not None and estimate['probe_seconds'] >= 0
    assert steps[0] == pytest.approx(.04) and steps[-1] == pytest.approx(.01)
    assert report['voxel_passes'][0]['timing'] and any('Допуск' in w for w in report['warnings'])


def test_absurd_voxel_grid_fails_early_with_a_usable_tolerance(monkeypatch):
    mesh = mm.makeCube(mm.Vector3f(100, 100, 100), mm.Vector3f(0, 0, 0))
    monkeypatch.setattr(mm, 'offsetMesh', lambda *a, **k: pytest.fail('voxel pass must not start'))
    with pytest.raises(RuntimeError, match='Увеличьте допуск до'):
        engine.sparse_repair(mm.Mesh(mesh), mm, Settings(.02, disable_cuda=True), Progress(lambda _: None),
                             {'warnings': []}, lambda _: None)
