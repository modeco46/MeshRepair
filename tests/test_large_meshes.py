"""Large-mesh regressions, including shells dense enough to bypass <50 protection."""
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair import engine, reduction
from meshrepair.engine import Settings, Progress


@pytest.mark.parametrize('threads', [1, 3])
def test_dense_small_shell_is_not_deleted_between_passes(monkeypatch, threads):
    monkeypatch.setattr(reduction, 'cpu_count', lambda: threads)
    monkeypatch.setattr(reduction.os, 'cpu_count', lambda: threads)
    mesh = mm.makeUVSphere(2, 180, 180)
    island = mm.makeUVSphere(.001, 20, 20)  # 800 faces, far above the old threshold
    island.transform(mm.AffineXf3f.translation(mm.Vector3f(4, 0, 0)))
    mesh.addMesh(island)
    original_faces = mesh.topology.numValidFaces()
    result, stages = reduction.propose_reduction(mesh, mm, Settings(.02, simplify_curved=True),
        Progress(lambda _: None), error_scale=.475)
    assert mesh.topology.numValidFaces() == original_faces
    assert engine.valid(engine.diagnostics(result, mm))
    regions = mm.MeshComponents.getAllComponents(mm.MeshPart(result))
    assert len(regions) == 2
    assert min(r.count() for r in regions) >= 4
    assert result.topology.numValidFaces() < original_faces / 10
    assert all(s['component_count'] == 2 for s in stages)
    assert reduction.measured_distance(mesh, result, mm) <= .02


def test_large_multishell_pipeline_completes_and_exports_all_shells(tmp_path):
    # A process deadline catches native callback deadlocks without hanging CI.
    script = textwrap.dedent('''
        import json, sys, time
        from pathlib import Path
        from meshlib import mrmeshpy as mm
        from meshrepair.engine import run, Settings, diagnostics, valid
        from meshrepair import reduction
        reduction.cpu_count = lambda: 3
        reduction.os.cpu_count = lambda: 3
        folder = Path(sys.argv[1])
        mesh = mm.makeTorus(10, 3, 1000, 400)  # 800,000 faces
        for x in range(8):
            island = mm.makeUVSphere(.001, 20, 20)
            island.transform(mm.AffineXf3f.translation(mm.Vector3f(20+x, 0, 0)))
            mesh.addMesh(island)
        source = folder/'nine-shells.stl'
        mm.saveMesh(mesh, source)
        started = time.monotonic()
        r = run(source, settings=Settings(.02, True, 2, True,
            remove_small_shells=True, disable_cuda=True), work_dir=folder/'job')
        s = r['simplification']
        assert r['before']['triangles'] == 806400
        assert s['accepted'], s
        assert s['after'] < s['before']/10
        assert r['after']['components'] == 9
        assert valid(diagnostics(mm.loadMesh(r['output']), mm))
        assert s['verification']['max_deviation_mm'] <= .019
        print(json.dumps({'before':s['before'], 'after':s['after'], 'components':r['after']['components'],
            'deviation_mm':s['verification']['max_deviation_mm'], 'seconds':time.monotonic()-started}))
    ''')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


def test_planar_786k_mesh_keeps_corners_without_partition_seams():
    script = textwrap.dedent('''
        import numpy as np
        from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
        from meshrepair import reduction
        from meshrepair.engine import Settings, Progress, diagnostics, valid
        reduction.cpu_count = lambda: 16
        reduction.os.cpu_count = lambda: 16
        mesh = mm.makeCube()
        p = mm.SubdivideSettings()
        p.maxEdgeLen = .007
        p.maxEdgeSplits = 1000000
        p.maxAngleChangeAfterFlip = 0
        mm.subdivideMesh(mesh, p)
        assert mesh.topology.numValidFaces() == 786432
        result, stages = reduction.propose_reduction(mesh, mm, Settings(.02, True), Progress(lambda _:None))
        assert result.topology.numValidFaces() <= 48
        assert valid(diagnostics(result, mm))
        assert reduction.measured_distance(mesh, result, mm) < 1e-6
        vertices = mn.getNumpyVerts(result)
        for x in [-.5, .5]:
            for y in [-.5, .5]:
                for z in [-.5, .5]:
                    assert np.any(np.all(vertices == [x, y, z], axis=1))
        assert all(d['subdivide_parts'] == 1 for s in stages for d in s['component_passes'])
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('curved', [False, True])
def test_large_open_mesh_preserves_every_boundary_vertex(monkeypatch, curved):
    monkeypatch.setattr(reduction, 'cpu_count', lambda: 3)
    monkeypatch.setattr(reduction.os, 'cpu_count', lambda: 3)
    n = 180
    x, y = np.meshgrid(np.arange(n+1, dtype=np.float32), np.arange(n+1, dtype=np.float32))
    vertices = np.column_stack([x.ravel()/n, y.ravel()/n, np.zeros(x.size, np.float32)])
    a = (np.arange(n)[:, None]*(n+1)+np.arange(n)[None, :]).ravel()
    faces = np.concatenate([np.stack([a, a+1, a+n+2], axis=1),
                            np.stack([a, a+n+2, a+n+1], axis=1)]).astype(np.int32)
    mesh = mn.meshFromFacesVerts(faces, vertices)
    def boundary(m):
        vertices = mn.getNumpyVerts(m)
        return {tuple(vertices[int(v)]) for v in m.topology.findBdVerts()}
    result, _ = reduction.propose_reduction(mesh, mm, Settings(.02, True, simplify_curved=curved), Progress(lambda _:None))
    assert len(boundary(mesh)) == 720
    assert boundary(result) == boundary(mesh)
    assert result.topology.numValidFaces() < len(faces)/10


def test_streaming_reconstruction_uses_same_step_and_valid_surface(tmp_path):
    script = textwrap.dedent('''
        import sys
        from pathlib import Path
        from meshlib import mrmeshpy as mm
        from meshrepair import engine, reduction
        from meshrepair.engine import Settings
        folder = Path(sys.argv[1])
        mesh = mm.makeCube()
        mesh.addMesh(mm.makeCube(mm.Vector3f(1,1,1), mm.Vector3f(0,0,0)))
        source = folder/'overlap.stl'
        mm.saveMesh(mesh, source)
        settings = Settings(.1, restore_sharp_edges=False, disable_cuda=True,
            local_fix_max_faces=0, local_fix_fraction=0)
        normal = engine.run(source, settings=settings, work_dir=folder/'normal')
        engine.DENSE_WINDING_VOXEL_LIMIT = 1  # Exercise the real native large-volume path cheaply.
        streamed = engine.run(source, settings=settings, work_dir=folder/'streamed')
        assert engine.valid(streamed['after'])
        assert streamed['after']['components'] == normal['after']['components'] == 1
        assert all(p['streaming'] and p['backend']=='marching_cubes_by_parts'
            and p['voxel_size_mm']==.1 for p in streamed['voxel_passes'])
        assert reduction.measured_distance(mm.loadMesh(normal['output']), mm.loadMesh(streamed['output']), mm) <= .1
    ''')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_large_volume_never_supplies_non_chunked_winding(monkeypatch):
    from types import SimpleNamespace
    mesh = mm.makeCube(mm.Vector3f(20, 20, 20))
    monkeypatch.setattr(engine, 'cuda_module', lambda: SimpleNamespace(FastWindingNumber=mm.FastWindingNumber))
    calls = []
    def streaming(part, offset, params):
        calls.append(params.voxelSize)
        assert params.memoryEfficient and params.fwn is None
        return mm.makeCube()
    monkeypatch.setattr(mm, 'mcOffsetMesh', streaming)
    monkeypatch.setattr(mm, 'offsetMesh', lambda *args: pytest.fail('Must avoid dense OpenVDB winding'))
    report = {'warnings': []}
    engine.sparse_repair(mesh, mm, Settings(.02, restore_sharp_edges=False), Progress(lambda _:None), report, lambda _:None)
    assert calls == pytest.approx([.02])
    assert report['voxel_estimate']['grid_voxels'] > engine.DENSE_WINDING_VOXEL_LIMIT
    assert report['voxel_passes'][0]['compute_device'] == 'cpu_native_parallel'
