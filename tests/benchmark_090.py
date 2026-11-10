"""Reproducible before/after costs of the checks removed in 0.9 (CPU by default).

python tests/benchmark_090.py --json benchmark.json [--real tests/data/test.stl]
The baseline is the 0.8 sequence of operations, not an emulated faster SDK.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair import engine, holes, reduction
from meshrepair.engine import Settings, Progress, diagnostics, topology_info


def timed(function):
    start=time.perf_counter();value=function();return value,time.perf_counter()-start


def holes_mesh():
    m=mm.makeCube(mm.Vector3f(10,10,10))
    s=mm.SubdivideSettings();s.maxEdgeLen=.3;s.maxAngleChangeAfterFlip=0;s.maxEdgeSplits=2_000_000
    mm.subdivideMesh(m,s);m.pack()
    f=mn.getNumpyFaces(m.topology);used=set();chosen=[]
    for i in np.random.default_rng(31).permutation(len(f)):
        if not used.intersection(map(int,f[i])):
            chosen.append(int(i));used.update(map(int,f[i]))
            if len(chosen)==200:break
    mask=np.zeros(len(f),bool);mask[chosen]=True
    m.topology.deleteFaces(mn.faceBitSetFromBools(mask));m.invalidateCaches()
    assert m.topology.findNumHoles()==200
    return m


def benchmark_holes():
    m=holes_mesh()
    def full(before,candidate,region,mm):
        return mm.findSelfCollidingTrianglesBS(mm.MeshPart(candidate))
    with patch.object(holes,'patch_collisions',full):
        (old,old_r),before=timed(lambda:holes.fill_simple_holes(m,mm,.05,time_budget_seconds=300))
    (new,new_r),after=timed(lambda:holes.fill_simple_holes(m,mm,.05,time_budget_seconds=300))
    assert old_r['filled']==new_r['filled']==200
    assert diagnostics(new,mm)['self_intersecting_faces']==0
    np.testing.assert_array_equal(mn.getNumpyVerts(old),mn.getNumpyVerts(new))
    np.testing.assert_array_equal(mn.getNumpyFaces(old.topology),mn.getNumpyFaces(new.topology))
    return {'before_seconds':before,'after_seconds':after,'saved_seconds':before-after,
            'input_triangles':m.topology.numValidFaces(),'holes':200}


def benchmark(real=None):
    result={'compute_device':'cpu_native_parallel','checks':{}}
    def pair(name,before,after):
        _,a=timed(before);_,b=timed(after)
        result['checks'][name]={'before_seconds':a,'after_seconds':b,'saved_seconds':a-b}
    result['checks']['200_holes_global_vs_local']=benchmark_holes()
    q=.05
    if real:
        from meshrepair.orientation import prepare_orientation
        mesh,_=prepare_orientation(mm.loadMesh(Path(real)),mm)
        mesh,_=holes.fill_simple_holes(mesh,mm,q)
    else:
        mesh=mm.makeCube(mm.Vector3f(2,2,2))
        mesh.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f(1,2,3),.37)))
    p=mm.OffsetParameters();p.voxelSize=q;p.signDetectionMode=mm.SignDetectionMode.HoleWindingRule
    dense,seconds=timed(lambda:mm.offsetMesh(mm.MeshPart(mesh),0,p));dense.pack()
    result.update(dense_triangles=dense.topology.numValidFaces(),voxel_seconds=seconds)
    with tempfile.TemporaryDirectory(prefix='meshrepair-benchmark-') as folder:
        path=Path(folder)/'candidate.stl'
        def roundtrip(mesh):
            mm.saveMesh(mesh,path);return diagnostics(mm.loadMesh(path),mm)
        def clean():
            c=mm.Mesh(dense);p=mm.FixMeshDegeneraciesParams();p.mode=mm.FixMeshDegeneraciesParams.Mode.Decimate
            p.maxDeviation=q*.01;p.tinyEdgeLength=q*.001;mm.fixMeshDegeneracies(c,p);return c
        pair('prepared_diagnostics_vs_topology',lambda:diagnostics(dense,mm),lambda:topology_info(dense,mm))
        cached=diagnostics(dense,mm)
        pair('duplicate_repair_validation_vs_cached',lambda:diagnostics(dense,mm),lambda:cached)
        pair('cleanup_before_simplify_vs_skip',lambda:roundtrip(clean()),lambda:None)
        pair('cleanup_roundtrip_vs_memory',lambda:roundtrip(clean()),lambda:diagnostics(clean(),mm))
        small,_=reduction.propose_reduction(dense,mm,Settings(q,simplify_curved=True),Progress(lambda _:None),error_scale=.25)
        pair('candidate_roundtrip_vs_memory',lambda:roundtrip(small),lambda:diagnostics(small,mm))
        pair('strict_proof_vs_measured',lambda:reduction.verify_surface_bound(dense,small,q),lambda:reduction.measured_distance(dense,small,mm))
        pair('extra_final_load_vs_reuse',lambda:mm.loadMesh(path),lambda:small)
        pair('full_sharp_validation_vs_common_final_only',lambda:diagnostics(dense,mm),lambda:None)
        if real:
            def sharp(step):
                p=mm.GeneralOffsetParameters();p.voxelSize=step;p.signDetectionMode=mm.SignDetectionMode.HoleWindingRule
                p.mode=mm.GeneralOffsetParametersMode.Sharpening
                m=mm.generalOffsetMesh(mm.MeshPart(mesh),0,p);return diagnostics(m,mm)
            full,a=timed(lambda:sharp(q));probe,b=timed(lambda:sharp(q*4))
            result['checks']['rejected_full_sharp_vs_probe']={'before_seconds':a,'after_seconds':b,'saved_seconds':a-b,
                'full_intersections':full['self_intersecting_faces'],'probe_intersections':probe['self_intersecting_faces']}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real');parser.add_argument('--json');parser.add_argument('--one-cpu',action='store_true')
    args=parser.parse_args()
    if args.one_cpu:
        if hasattr(os,'sched_setaffinity'):
            os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        elif os.name=='nt':
            import ctypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            mask=ctypes.c_size_t();system=ctypes.c_size_t()
            kernel.GetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.byref(mask),ctypes.byref(system))
            if not kernel.SetProcessAffinityMask(ctypes.c_void_p(-1),ctypes.c_size_t(mask.value & -mask.value)):
                raise OSError(ctypes.get_last_error(),'SetProcessAffinityMask')
    result=benchmark(args.real)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.json:Path(args.json).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
