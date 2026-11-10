"""Synthetic geometry regressions; not accuracy promises for real scans."""
from pathlib import Path
import math
import time
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.engine import run, Settings, diagnostics
from meshrepair.holes import fill_simple_holes
from meshrepair.orientation import prepare_orientation
from meshrepair.reduction import projection_arrays, ProjectionLayoutError


def subdivide(mesh, length):
    p = mm.SubdivideSettings()
    p.maxEdgeLen = length
    p.maxEdgeSplits = 2000000
    p.maxAngleChangeAfterFlip = 0
    mm.subdivideMesh(mesh, p)
    mesh.pack()
    return mesh


def separated_small_holes(mesh, mask, count, allowed=None):
    faces = mn.getNumpyFaces(mesh.topology)
    used = set(faces[mask].ravel().tolist())
    rng = np.random.default_rng(31)
    pool = np.flatnonzero(~mask if allowed is None else ~mask & allowed)
    chosen = 0
    for i in rng.permutation(pool):
        if not any(int(v) in used for v in faces[i]):
            mask[i] = True
            used.update(map(int, faces[i]))
            chosen += 1
            if chosen == count:
                break
    assert chosen == count
    mesh.topology.deleteFaces(mn.faceBitSetFromBools(mask))
    return mesh


def cube_with_holes():
    mesh = subdivide(mm.makeCube(mm.Vector3f(10, 10, 10), mm.Vector3f(-5, -5, -5)), .15)
    v, f = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    centers = v[f].mean(axis=1)
    top = np.all(v[f, 2] == 5, axis=1)
    mask = np.zeros(len(f), bool)
    for x, y, radius in [(-2, -1, 2), (2, -1, 1), (3, 2, .4)]:
        mask |= top & ((centers[:, 0]-x)**2+(centers[:, 1]-y)**2 < radius**2)
    return separated_small_holes(mesh, mask, 40, np.all(v[f, 2] == -5, axis=1))


def sphere_with_holes():
    p = mm.SphereParams(); p.radius = 5; p.numMeshVertices = 20000
    mesh = mm.makeSphere(p)
    v, f = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    c = v[f].mean(axis=1)
    mask = ((c[:, 2] > 0) & (np.linalg.norm(c[:, :2], axis=1) < 1.5))
    mask |= (c[:, 0] > 0) & (np.linalg.norm(c[:, 1:], axis=1) < .8)
    return separated_small_holes(mesh, mask, 40, (c[:, 2] < 0) & (c[:, 0] < 0))


def sample_surface(mesh):
    v, f = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    tri = v[f].astype(float)
    return np.concatenate((v, tri.mean(axis=1), (tri[:,0]+tri[:,1])*.5,
                           (tri[:,1]+tri[:,2])*.5, (tri[:,2]+tri[:,0])*.5))


def max_distance_to(mesh, points):
    native = mm.PointsToMeshProjector(); native.updateMeshData(mesh)
    maximum = 0.
    for i in range(0, len(points), 131072):
        projections = mm.std_vector_MeshProjectionResult()
        native.findProjections(projections, mn.fromNumpyArray(np.asarray(points[i:i+131072], np.float32)),
                               None, None, float(np.finfo(np.float32).max), 0)
        _, sq = projection_arrays(projections, mm)
        maximum = max(maximum, math.sqrt(float(sq.max())))
    return maximum


def test_cube_43_holes_stays_planar(tmp_path):
    mesh = cube_with_holes()
    assert mesh.topology.findNumHoles() >= 43
    source = tmp_path/'cube.stl'; mm.saveMesh(mesh, source)
    start = time.monotonic()
    r = run(source, settings=Settings(.05))
    result = mm.loadMesh(r['output'])
    assert r['algorithm'] == 'surface_repair'
    assert r['hole_filling']['holes_after'] == 0
    assert r['after']['closed'] and r['after']['self_intersecting_faces'] == 0
    assert max_distance_to(mm.makeCube(mm.Vector3f(10,10,10), mm.Vector3f(-5,-5,-5)), sample_surface(result)) <= .001
    assert time.monotonic()-start < 30


def test_sphere_42_holes_surface_samples(tmp_path):
    mesh = sphere_with_holes()
    assert mesh.topology.findNumHoles() >= 42
    source = tmp_path/'sphere.stl'; mm.saveMesh(mesh, source)
    r = run(source, settings=Settings(.05))
    result = mm.loadMesh(r['output'])
    radial_error = float(np.max(np.abs(np.linalg.norm(sample_surface(result), axis=1)-5)))
    print('sphere radial error', radial_error)
    assert r['hole_filling']['holes_after'] == 0
    assert r['algorithm'] == 'surface_repair'
    assert radial_error <= .07


def wedge_with_holes():
    first = mm.makeCube(mm.Vector3f(12, 8, 3), mm.Vector3f(0,-4,-1.5))
    second = mm.Mesh(first)
    second.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f(0,1,0), math.radians(120))))
    ideal = mm.boolean(first, second, mm.BooleanOperation.Union).mesh
    raw = mm.Mesh(ideal)  # One wedge solid, not two overlapping internal volumes.
    subdivide(raw, .35)
    v, f = mn.getNumpyVerts(raw), mn.getNumpyFaces(raw.topology)
    centers = v[f].mean(axis=1)
    # Isolated triangle holes on the broad planar sides, away from the joint.
    allowed = np.all(np.isclose(v[f, 1], -4), axis=1) & (np.linalg.norm(centers[:, [0,2]], axis=1) > 3)
    separated_small_holes(raw, np.zeros(len(f), bool), 45, allowed)
    rotation = mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f(1,1,.3), math.radians(23)))
    raw.transform(rotation); ideal.transform(rotation)
    return raw, ideal


def test_sharpened_120_degree_wedge(tmp_path):
    mesh, ideal = wedge_with_holes()
    assert mesh.topology.findNumHoles() == 45
    source = tmp_path/'wedge.stl'; mm.saveMesh(mesh, source)
    r = run(source, settings=Settings(.05, True, 25, True, True))
    result = mm.loadMesh(r['output'])
    error = max(max_distance_to(ideal, sample_surface(result)), max_distance_to(result, sample_surface(subdivide(ideal,.35))))
    print('wedge', r['after']['triangles'], error, r.get('sharpening'))
    assert r['algorithm'] == 'surface_repair'  # All 45 holes now close before voxels.
    assert r['hole_filling']['holes_after'] == 0
    assert r['after']['triangles'] <= 1000
    assert error <= .002


def mobius():
    v=[]; faces=[]
    for i in range(12):
        t=2*math.pi*i/12
        for w in (-.3,.3):
            v.append([20+(2+w*math.cos(t/2))*math.cos(t), (2+w*math.cos(t/2))*math.sin(t), w*math.sin(t/2)])
    for i in range(12):
        a,b=2*i,2*i+1
        c,d=(2*(i+1),2*(i+1)+1) if i<11 else (1,0)
        faces.extend([[a,c,d],[a,d,b]])
    return np.array(v,np.float32), np.array(faces,np.int32)


def test_mobius_does_not_block_cube_patch(tmp_path):
    from test_orientation import damaged_cube
    source = tmp_path/'mixed.obj'; pristine=damaged_cube(source)
    v,f=mobius()
    offset=len(mn.getNumpyVerts(pristine))
    with source.open('a',encoding='utf-8') as stream:
        stream.write(''.join('v %.9g %.9g %.9g\n' % tuple(p) for p in v))
        stream.write(''.join('f %d %d %d\n' % tuple(tri+offset+1) for tri in f))
    mesh=mm.loadMesh(source)
    oriented,r=prepare_orientation(mesh,mm)
    assert r['accepted'] and r['partial']
    assert r['conflicting_faces']==24
    assert r['unoriented_components']==1
    largest=max(mm.MeshComponents.getAllComponents(mm.MeshPart(oriented)),key=lambda x:x.count())
    cube=oriented.cloneRegion(largest)
    assert cube.topology.isClosed()
    assert cube.topology.numValidFaces()==12288
    assert cube.volume()==pytest.approx(1000,abs=1e-5)


def test_projection_layout_mismatch_is_explicit():
    from types import SimpleNamespace
    wrong = SimpleNamespace(MeshProjectionResult=SimpleNamespace(_offsetof_proj=0,_offsetof_mtp=16,_offsetof_distSq=24),
                            PointOnFace=SimpleNamespace(_offsetof_face=0))
    with pytest.raises(ProjectionLayoutError, match='projection_layout_mismatch'):
        projection_arrays([], wrong)


def test_large_hole_skipped_and_time_budget_is_reported():
    mesh=mm.makeCube(mm.Vector3f(100,100,100)); mesh.topology.deleteFace(mm.FaceId(0))
    result,r=fill_simple_holes(mesh,mm,.01)
    assert r['skipped']==1 and r['details'][0]['reason']=='hole_size_limit'
    assert result.topology.findNumHoles()==1
    _,r=fill_simple_holes(mesh,mm,.01,time_budget_seconds=0)
    assert r['skipped']==1 and r['details'][0]['reason']=='hole_fill_time_budget'


def test_patch_collision_rolls_back_original_geometry(monkeypatch):
    mesh = mm.makeCube()
    mesh.topology.deleteFace(mm.FaceId(0))
    before_v = mn.getNumpyVerts(mesh).copy()
    before_f = mn.getNumpyFaces(mesh.topology).copy()
    from meshrepair import holes
    def collide(*args):
        result = mm.FaceBitSet()
        result.autoResizeSet(mm.FaceId(0))
        return result
    monkeypatch.setattr(holes, 'patch_collisions', collide)
    result,r = fill_simple_holes(mesh,mm,.02)
    assert r['rejected'] == 1 and r['holes_after'] == 1
    np.testing.assert_array_equal(mn.getNumpyVerts(result),before_v)
    np.testing.assert_array_equal(mn.getNumpyFaces(result.topology),before_f)


def test_sharpening_failure_uses_standard_same_step(monkeypatch):
    from meshrepair.engine import sparse_repair, Progress
    mesh=mm.makeCube()
    def fail(*args): raise RuntimeError('synthetic sharp failure')
    monkeypatch.setattr(mm,'generalOffsetMesh',fail)
    standard=mm.offsetMesh
    steps=[]
    def record(meshpart,offset,params):
        steps.append(params.voxelSize)
        return standard(meshpart,offset,params)
    monkeypatch.setattr(mm,'offsetMesh',record)
    report={'warnings':[]}
    result=sparse_repair(mesh,mm,Settings(.05),Progress(lambda _:None),report,lambda _:None)
    assert result.topology.isClosed()
    assert steps == pytest.approx([.05])
    assert not report['sharpening']['accepted']
    assert 'synthetic sharp failure' in report['warnings'][0]


def test_heatmap_all_vertices_hotspots_and_scene_alignment(tmp_path):
    from meshrepair.deviation import source_deviation
    from meshrepair.viewer import heatmap_polydata
    from vtkmodules.util.numpy_support import vtk_to_numpy
    source = mm.makeCube(mm.Vector3f(2,2,2), mm.Vector3f(10,20,30))
    target = mm.Mesh(source)
    target.transform(mm.AffineXf3f.translation(mm.Vector3f(.2,0,0)))
    path=tmp_path/'heat.npz'
    report=source_deviation(source,target,mm,.02,path)
    assert report['max_deviation_mm']==pytest.approx(.2,abs=2e-6)
    assert report['vertex_count']==8 and report['exceeding_vertices']>0
    assert report['top10'][0]['deviation_mm']==report['max_deviation_mm']
    assert min(report['top10'][0]['coordinates_mm']) >= 10
    data=heatmap_polydata(path,[10,20,30])
    np.testing.assert_allclose(vtk_to_numpy(data.GetPoints().GetData()).min(axis=0),[0,0,0])
    colors=vtk_to_numpy(data.GetPointData().GetScalars())
    assert np.any(np.all(colors==[255,45,45],axis=1))


def test_sharpening_preserves_closed_wedge_crease():
    from meshrepair.engine import sparse_repair, Progress
    from meshrepair.reduction import propose_reduction
    _, ideal=wedge_with_holes()
    report={'warnings':[]}
    dense=sparse_repair(ideal,mm,Settings(.05),Progress(lambda _:None),report,lambda _:None)
    assert report['sharpening']['accepted']
    candidate,_=propose_reduction(dense,mm,Settings(.05,True,25,True),Progress(lambda _:None))
    info=diagnostics(candidate,mm)
    assert info['closed'] and info['self_intersecting_faces']==0
    assert info['triangles']<=1000
    assert max_distance_to(ideal,sample_surface(candidate))<=.002
    assert max_distance_to(candidate,sample_surface(subdivide(ideal,.35)))<=.002


def test_degenerate_cleanup_cannot_break_valid_stl(tmp_path, monkeypatch):
    from meshrepair.engine import cleanup_exported, Progress
    mesh = mm.makeCube()
    path = tmp_path/'result.stl'
    mm.saveMesh(mesh, path)
    original = path.read_bytes()
    report = {'after': diagnostics(mesh, mm), 'warnings': [], 'steps': []}
    def broken_cleanup(candidate, params):
        candidate.topology.deleteFace(mm.FaceId(0))
    monkeypatch.setattr(mm, 'fixMeshDegeneracies', broken_cleanup)
    result = cleanup_exported(mesh, path, mm, Settings(.05), Progress(lambda _: None), report, tmp_path, lambda _: None)
    assert result.topology.isClosed()
    assert path.read_bytes() == original
    assert not report['degenerate_cleanup']['accepted']
    assert report['degenerate_cleanup']['reason'] == 'cleanup_topology_changed'
