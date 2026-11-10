import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.engine import run, Settings, clean_json
from meshrepair import reduction
from meshrepair.reporting import report_html


def dense(path):
    mesh = mm.makeCube()
    s = mm.SubdivideSettings(); s.maxEdgeLen=.15; s.maxAngleChangeAfterFlip=0
    mm.subdivideMesh(mesh, s)
    mm.saveMesh(mesh, path)
    return mesh


@pytest.mark.parametrize('reason', ['verification_work_limit','distance_not_confirmed','projection_failed'])
def test_uncertain_proof_measured_fallback(tmp_path, monkeypatch, reason):
    path=tmp_path/'cube.stl';dense(path)
    monkeypatch.setattr(reduction, 'verify_surface_bound', lambda *a, **k: pytest.fail('Measured mode must not run proof'))
    report=run(path,settings=Settings(.02,True))
    result=report['simplification']
    assert result['accepted']
    assert result['verification']['mode']=='measured_not_proven'
    assert not result['verification']['verified']
    assert result['verification']['max_deviation_mm'] <= .02
    assert 'не доказано' in report_html(report)


def test_strict_rejection_attempts_and_visible_reason(tmp_path,monkeypatch):
    path=tmp_path/'cube.stl';dense(path)
    monkeypatch.setattr(reduction,'verify_surface_bound', lambda *a,**k: {'verified':False,'reason':'verification_work_limit'})
    report=run(path,settings=Settings(.02,True,1,False,True))
    result=report['simplification']
    assert not result['accepted']
    assert 'Исходник → после ремонта → после упрощения' in report_html(report)
    assert result['input_triangles'] == report['before']['triangles']
    assert [x['max_error_mm'] for x in result['attempts']]==[.005,.0025,.00125]
    assert 'verification_work_limit' in report_html(report)
    assert Path(report['output']).exists()


def test_actual_violation_never_uses_measured_fallback(tmp_path,monkeypatch):
    path=tmp_path/'cube.stl';dense(path)
    monkeypatch.setattr(reduction,'verify_surface_bound', lambda *a,**k: {'verified':False,'reason':'sample_exceeds_tolerance'})
    monkeypatch.setattr(reduction,'measured_distance', lambda *a: pytest.fail('Actual violation must not be ignored'))
    report=run(path,settings=Settings(.02,True,strict_tolerance=True))
    assert not report['simplification']['accepted']


def test_small_island_protected(tmp_path):
    path=tmp_path/'combined.stl'; mesh=dense(path)
    mesh.addMesh(mm.makeCube(mm.Vector3f(.001,.001,.001),mm.Vector3f(2,0,0)))
    mm.saveMesh(mesh,path)
    report=run(path,settings=Settings(.02,True,1,True))
    assert report['after']['components']==2
    assert report['after']['small_components'] >= 1


def test_result_file_and_json_sanitization(tmp_path):
    source=tmp_path/'cube.stl';dense(source)
    output=tmp_path/'channel.json'
    p=subprocess.run([sys.executable,'-m','meshrepair.worker',str(source),'--check','--result-file',str(output)],capture_output=True,encoding='utf-8',timeout=30)
    assert p.returncode==0,p.stdout+p.stderr
    event=json.loads(output.read_text(encoding='utf-8'))
    assert event['type']=='result'
    assert event['value']['orientation']['disoriented_edges']==0
    assert clean_json({'a':float('nan'),'b':[np.float32('inf')]})=={'a':None,'b':[None]}


def test_projection_bulk_and_numpy_adjacency_match_sdk():
    mesh=mm.makeCube();faces=mn.getNumpyFaces(mesh.topology)
    neighbors=reduction.face_neighbors(faces)
    for i,f in enumerate(faces):
        for edge in mm.leftRing(mesh.topology, mm.FaceId(i)):
            u,v=int(mesh.topology.org(edge)),int(mesh.topology.dest(edge))
            slot=int(np.flatnonzero(f==u)[0])
            opposite=int(mesh.topology.right(edge))
            expect=next(int(x) for x in faces[opposite] if x not in (u,v))
            assert neighbors[i,slot]==expect
    projector=mm.PointsToMeshProjector();projector.updateMeshData(mesh)
    points=mn.fromNumpyArray(np.array([[2,0,0],[0,0,2],[.1,.2,.3]],np.float32))
    projections=mm.std_vector_MeshProjectionResult()
    projector.findProjections(projections,points,None,None,float(np.finfo(np.float32).max),0)
    ids,distances=reduction.projection_arrays(projections,mm)
    np.testing.assert_array_equal(ids,[int(p.proj.face) for p in projections])
    np.testing.assert_array_equal(distances,[p.distSq for p in projections])


def test_sidecar_failure_keeps_valid_stl(tmp_path,monkeypatch):
    from meshrepair import engine
    source=tmp_path/'cube.stl';dense(source)
    def fail(*args, **kwargs): raise OSError('read-only report')
    monkeypatch.setattr(engine,'atomic_json',fail)
    result=run(source,source.with_name('result.stl'))
    assert Path(result['output']).exists()
    assert any('JSON' in w for w in result['warnings'])
