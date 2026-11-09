"""Exercise bundled extension loading and repair without launching a GUI."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from meshlib import mrmeshpy as mm

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    executable = str(Path(sys.argv[1]).resolve())
    ui = subprocess.run([executable, '--smoke-ui'], capture_output=True, timeout=30)
    assert ui.returncode == 0, ui.stderr
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / 'cube.stl'
        target = Path(folder) / 'result.stl'
        mm.saveMesh(mm.makeCube(mm.Vector3f(.1, .1, .1)), source)
        p = subprocess.run([executable, '--worker', str(source), '--output', str(target), '--quality', '.02'],
                           capture_output=True, encoding='utf-8', errors='replace', timeout=180)
        print(p.stdout)
        print(p.stderr)
        assert p.returncode == 0
        assert target.is_file()
        report = json.loads(Path(str(target) + '.json').read_text(encoding='utf-8'))
        assert report['after']['closed']
        assert report['after']['components'] == 1

        checked = subprocess.run([executable, '--worker', str(source), '--check'],
                                 capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        events = [json.loads(line) for line in checked.stdout.splitlines() if line.startswith('{')]
        assert events[-1]['value']['info']['closed_components'] == 1
        preview = Path(folder) / 'view.npz'
        viewed = subprocess.run([executable, '--worker', str(source), '--preview-output', str(preview)],
                                capture_output=True, encoding='utf-8', errors='replace', timeout=60)
        assert viewed.returncode == 0, viewed.stdout + viewed.stderr
        assert preview.is_file()
        loaded = subprocess.run([executable, '--smoke-preview', str(preview)], capture_output=True, timeout=60)
        assert loaded.returncode == 0, loaded.stderr

        # Exercise reduction, NumPy and the whole-surface verifier in the EXE.
        dense = mm.makeCube()
        subdivision = mm.SubdivideSettings()
        subdivision.maxEdgeLen = .15
        subdivision.maxAngleChangeAfterFlip = 0
        mm.subdivideMesh(dense, subdivision)
        mm.saveMesh(dense, source)
        target = Path(folder) / 'simplified.stl'
        p = subprocess.run([executable, '--worker', str(source), '--output', str(target),
                            '--quality', '.02', '--collapse-coplanar', '--coplanar-angle', '1',
                            '--simplify-curved'], capture_output=True, encoding='utf-8',
                           errors='replace', timeout=180)
        print(p.stdout)
        print(p.stderr)
        assert p.returncode == 0
        report = json.loads(Path(str(target) + '.json').read_text(encoding='utf-8'))
        assert report['simplification']['accepted']
        assert len(report['simplification']['stages']) == 1
        assert report['simplification']['stages'][0]['angle_weighted_metric']
        assert report['simplification']['after'] < report['simplification']['before']
        assert report['simplification']['verification']['upper_bound_mm'] <= .02
        assert report['after']['closed']
        assert report['after']['self_intersecting_faces'] == 0

        # Exercise bundled SciPy sparse graph, local reversed patches and the
        # separate terminal-result channel inside the actual Windows EXE.
        from meshlib import mrmeshnumpy as mn
        import numpy as np
        vertices, faces = mn.getNumpyVerts(dense), mn.getNumpyFaces(dense.topology).copy()
        faces[::7] = faces[::7][:, [0,2,1]]
        reversed_source = Path(folder)/'reversed.obj'
        reversed_source.write_text(''.join('v %.9g %.9g %.9g\n' % tuple(v) for v in vertices)+
                                   ''.join('f %d %d %d\n' % tuple(f+1) for f in faces))
        channel = Path(folder)/'terminal.json'
        p = subprocess.run([executable,'--worker',str(reversed_source),'--collapse-coplanar',
                            '--result-file',str(channel)],capture_output=True,encoding='utf-8',errors='replace',timeout=180)
        assert p.returncode == 0, p.stdout+p.stderr
        report = json.loads(channel.read_text(encoding='utf-8'))['value']
        assert report['orientation']['components_before'] > 32
        assert report['prepared']['components'] == 1
        assert report['after']['closed']
        assert report['algorithm'] == 'surface_repair'
        assert report['simplification']['accepted']

        damaged = mm.Mesh()
        for i in range(40):
            part = mm.makeCube(mm.Vector3f(.1, .1, .1), mm.Vector3f(i * .2, 0, 0))
            part.topology.deleteFace(mm.FaceId(0))
            damaged.addMesh(part)
        mm.saveMesh(damaged, source)
        target = Path(folder) / 'sparse.stl'
        p = subprocess.run([executable, '--worker', str(source), '--output', str(target), '--quality', '.02'],
                           capture_output=True, encoding='utf-8', errors='replace', timeout=180)
        print(p.stdout)
        print(p.stderr)
        assert p.returncode == 0
        report = json.loads(Path(str(target) + '.json').read_text(encoding='utf-8'))
        assert report['algorithm'] == 'sparse_openvdb_winding'
        assert report['after']['closed']
        assert report['after']['self_intersecting_faces'] == 0
