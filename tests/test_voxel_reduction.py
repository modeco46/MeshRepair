"""Regression coverage for dense voxel seams and tiny detached shells."""
import numpy as np
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.engine import Settings, Progress, diagnostics
from meshrepair import reduction


def test_straight_crease_can_shrink_without_moving_corner():
    mesh = mm.makeCube()
    params = mm.SubdivideSettings()
    params.maxEdgeLen = .1
    params.maxAngleChangeAfterFlip = 0
    mm.subdivideMesh(mesh, params)
    result, _ = reduction.propose_reduction(mesh, mm, Settings(.001, True), Progress(lambda _: None))
    # Old behavior pinned every vertex of each cube edge, leaving hundreds of
    # unnecessary triangles even though there are only six flat faces.
    assert result.topology.numValidFaces() <= 24
    assert diagnostics(result, mm)['self_intersecting_faces'] == 0
    assert reduction.verify_surface_bound(mesh, result, .001)['verified']


def test_voxel_slanted_planes_reduce_and_keep_roundtrip_bound(tmp_path):
    mesh = mm.makeCube(mm.Vector3f(2, 2, 2))
    mesh.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f(1, 2, 3), .37)))
    params = mm.OffsetParameters()
    params.voxelSize = .04
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    dense = mm.offsetMesh(mm.MeshPart(mesh), 0, params)
    candidate, stages = reduction.propose_reduction(dense, mm, Settings(.04, True, 25, True), Progress(lambda _: None))
    # Both checkboxes share one budget against the original repaired surface.
    assert 1 <= len(stages) <= 3 and all(s['kind'] == 'curved' for s in stages)
    path = tmp_path/'roundtrip.stl'
    mm.saveMesh(candidate, path)
    candidate = mm.loadMesh(path)
    info = diagnostics(candidate, mm)
    assert info['closed'] and info['components'] == 1
    assert info['self_intersecting_faces'] == 0
    assert info['triangles'] < dense.topology.numValidFaces() / 10
    assert reduction.verify_surface_bound(dense, candidate, .04)['verified']


def test_parallel_decimation_keeps_tiny_shell_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(reduction, 'cpu_count', lambda: 2)
    monkeypatch.setattr(reduction.os, 'cpu_count', lambda: 2)
    params = mm.SphereParams()
    params.numMeshVertices = 30000
    params.radius = 2
    mesh = mm.makeSphere(params)
    islands = mm.Mesh()
    for x in (4., 5.):
        islands.addMesh(mm.makeCube(mm.Vector3f(.001, .001, .001), mm.Vector3f(x, 0, 0)))
    mesh.addMesh(islands)
    candidate, stages = reduction.propose_reduction(mesh, mm, Settings(.02, True, 25, True), Progress(lambda _: None))
    assert stages[0]['protected_components'] == 2
    path = tmp_path/'islands.stl'
    mm.saveMesh(candidate, path)
    candidate = mm.loadMesh(path)
    components = mm.MeshComponents.getAllComponents(mm.MeshPart(candidate))
    assert sorted(r.count() for r in components)[:2] == [12, 12]
    assert len(components) == 3
    preserved = mm.Mesh()
    for region in components:
        if region.count() < 50:
            preserved.addMesh(candidate.cloneRegion(region))
    # A zero bidirectional distance alone could miss topology changes. Also
    # compare exact triangle coordinates after STL roundtrip.
    def triangle_rows(m):
        verts = mn.getNumpyVerts(m)
        points = verts[mn.getNumpyFaces(m.topology)]
        return sorted(tuple(sorted(map(tuple, tri))) for tri in points)
    assert triangle_rows(preserved) == triangle_rows(islands)
    assert diagnostics(candidate, mm)['self_intersecting_faces'] == 0
