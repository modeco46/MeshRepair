"""Exercise bundled extension loading and repair without launching a GUI."""
import json
import os
import struct
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from meshlib import mrmeshpy as mm

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    executable = str(Path(sys.argv[1]).resolve())
    if os.name == 'nt':
        payload = Path(executable).read_bytes()
        pe = struct.unpack_from('<I', payload, 0x3c)[0]
        assert payload[pe:pe+4] == b'PE\0\0'
        subsystem = struct.unpack_from('<H', payload, pe+24+68)[0]
        assert subsystem == 2, f'Expected GUI PE subsystem, got {subsystem}'
    with tempfile.TemporaryDirectory() as folder:
        console_state = Path(folder)/'console.json'
        ui = subprocess.run([executable, '--smoke-ui', '--console-state', str(console_state)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        assert ui.returncode == 0, ui.stderr
        if os.name == 'nt':
            assert json.loads(console_state.read_text())['console_window'] == 0
        def worker(*arguments):
            # Unique files ensure every operation supplies its own terminal result.
            token = len(list(Path(folder).glob('terminal-*.json')))
            terminal = Path(folder)/f'terminal-{token}.json'
            progress = Path(folder)/f'progress-{token}.jsonl'
            p = subprocess.run([executable, '--worker', *map(str, arguments),
                                '--result-file', str(terminal), '--progress-file', str(progress)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
            assert p.returncode == 0, p.stderr
            event = json.loads(terminal.read_text(encoding='utf-8'))
            assert event['type'] == 'result', event
            events = [json.loads(line) for line in progress.read_text(encoding='utf-8').splitlines()]
            assert any(event['type'] == 'progress' for event in events)
            assert events[-1]['type'] == 'result_file'
            return event['value']
        source = Path(folder)/'cube.stl'
        target = Path(folder)/'result.stl'
        mm.saveMesh(mm.makeCube(mm.Vector3f(.1, .1, .1)), source)
        report = worker(source, '--output', target, '--quality', '.02')
        assert target.is_file() and report['after']['closed']
        assert report['after']['components'] == 1
        checked = worker(source, '--check')
        assert checked['info']['closed_components'] == 1
        preview = Path(folder)/'view.npz'
        worker(source, '--preview-output', preview)
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
        report = worker(source, '--output', target, '--quality', '.02', '--collapse-coplanar',
                        '--coplanar-angle', '1', '--simplify-curved', '--strict-tolerance')
        assert report['simplification']['accepted']
        assert 1 <= len(report['simplification']['stages']) <= 3
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
        report = worker(reversed_source, '--collapse-coplanar')
        assert report['orientation']['components_before'] > 32
        assert report['prepared']['components'] == 1
        assert report['after']['closed']
        assert report['algorithm'] == 'surface_repair'
        assert report['simplification']['accepted']

        # This standalone harness must not import the source application. The
        # executable has its own bundled package, while sys.path here starts at
        # tests/. Clean up the worker's artifacts with the standard library.
        shutil.rmtree(report['work_directory'])

        damaged = mm.Mesh()
        for i in range(40):
            part = mm.makeCube(mm.Vector3f(.1, .1, .1), mm.Vector3f(i * .2, 0, 0))
            part.topology.deleteFace(mm.FaceId(0))
            damaged.addMesh(part)
        mm.saveMesh(damaged, source)
        target = Path(folder) / 'sparse.stl'
        report = worker(source, '--output', target, '--quality', '.02')
        assert report['algorithm'] == 'closed_components_preserved'
        assert report['hole_filling']['filled'] == 40
        assert report['source_deviation']['vertex_count'] > 0
        assert Path(report['source_deviation']['heatmap_file']).is_file()
        assert report['after']['closed']
        assert report['after']['self_intersecting_faces'] == 0
