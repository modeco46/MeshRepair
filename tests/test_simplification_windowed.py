"""Repeated QEM reference ownership, real CPU budget and windowed protocol."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import numpy as np
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.engine import Settings, Progress, run
from meshrepair import reduction
from meshrepair.protocol import ProgressTail


def test_real_strict_multipass_one_cpu(tmp_path):
    # A fresh process sets affinity before native libraries create thread pools.
    code = r'''
import os, sys, time, json
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
elif sys.platform == 'win32':
    import ctypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system))
    assert k.SetProcessAffinityMask(k.GetCurrentProcess(), process.value & -process.value)
from meshrepair.engine import run, Settings
r = run('tests/data/test.stl', settings=Settings(.02, True, 5, True, True, disable_cuda=True), work_dir=sys.argv[1])
s = r['simplification']
assert s['before'] == 209466
assert s['accepted'] and s['after'] <= 29000, s
assert s['verification']['verified'] and s['verification']['upper_bound_mm'] <= .02
assert s['verification']['reference'] == 'repaired_mesh_before_simplification'
assert 2 <= len(s['stages']) <= 3
assert s['elapsed_seconds'] < 15, s['elapsed_seconds']
assert r['after']['closed'] and r['after']['self_intersecting_faces'] == 0
print(json.dumps({'triangles':s['after'], 'simplification_seconds':s['elapsed_seconds'], 'upper_bound_mm':s['verification']['upper_bound_mm']}))
'''
    p = subprocess.run([sys.executable, '-c', code, str(tmp_path/'job')], capture_output=True, text=True, timeout=100)
    assert p.returncode == 0, p.stdout+p.stderr
    print(p.stdout)


def test_passes_accumulate_but_search_checks_original(tmp_path, monkeypatch):
    mesh = mm.makeUVSphere(2, 80, 80)
    source = tmp_path/'sphere.stl'; mm.saveMesh(mesh, source)
    native = reduction.measured_distance
    references = []
    def measure(reference, candidate, mm):
        references.append(reference.topology.numValidFaces())
        return native(reference, candidate, mm)
    monkeypatch.setattr(reduction, 'measured_distance', measure)
    r = run(source, settings=Settings(.02, simplify_curved=True, disable_cuda=True), work_dir=tmp_path/'job')
    s = r['simplification']
    assert references and set(references) == {s['before']}
    assert s['accepted'] and len(s['stages']) >= 2
    stages = s['stages']
    assert all(b['before'] == a['after'] for a,b in zip(stages, stages[1:]))
    assert s['verification']['max_deviation_mm'] <= .019


def test_normal_clusters_and_original_noise():
    from meshrepair.planes import triangle_distribution
    result = mm.makeCube(mm.Vector3f(4, 4, 4))
    original = mm.Mesh(result)
    params = mm.SubdivideSettings(); params.maxEdgeLen = .2
    mm.subdivideMesh(original, params)
    v, f = mn.getNumpyVerts(original), mn.getNumpyFaces(original.topology)
    # Deliberate normal noise on all large faces, without touching the result.
    rng = np.random.default_rng(41)
    noisy = v.copy()
    axis = np.argmax(np.abs(v), axis=1)
    noisy[np.arange(len(v)), axis] += rng.normal(0, .06, len(v))
    original = mn.meshFromFacesVerts(f, noisy)
    r = triangle_distribution(original, result, mm, .02)
    assert r['total_triangles'] == r['large_planes_triangles'] + r['other_triangles']
    assert len(r['planes']) == 6
    assert r['large_planes_triangles'] == 12
    assert all(p['rougher_than_tolerance'] and p['source_residual_p90_mm'] > .02 for p in r['planes'])
    assert r['tolerance_hint']['kind'] == 'heuristic_not_prediction'


def test_progress_tail_handles_split_utf8_and_duplicate_polls(tmp_path):
    path = tmp_path/'progress.jsonl'; tail = ProgressTail(path)
    assert tail.read() == []
    line = json.dumps({'type':'log','value':'Русский прогресс'}, ensure_ascii=False).encode()+b'\n'
    split = line.index('Р'.encode())+1
    path.write_bytes(line[:split]); assert tail.read() == []
    with path.open('ab') as f: f.write(line[split:])
    assert json.loads(tail.read()[0])['value'] == 'Русский прогресс'
    assert tail.read() == []
    path.write_bytes(b'{}\n'); assert tail.read() == ['{}']


def test_worker_runs_without_stdout_stderr_and_flushes_files(tmp_path):
    source = tmp_path/'cube.stl'; mm.saveMesh(mm.makeCube(), source)
    result, progress = tmp_path/'result.json', tmp_path/'progress.jsonl'
    code = "import sys,runpy; sys.stdout=sys.stderr=None; sys.argv=['launch.py',*sys.argv[1:]]; runpy.run_path('launch.py',run_name='__main__')"
    env = dict(os.environ, LOCALAPPDATA=str(tmp_path/'local'))
    p = subprocess.run([sys.executable, '-c', code, '--worker', str(source), '--work-dir', str(tmp_path/'job'),
                        '--result-file', str(result), '--progress-file', str(progress)],
                       capture_output=True, timeout=45, env=env)
    assert p.returncode == 0, p.stderr
    report = json.loads(result.read_text(encoding='utf-8'))
    assert report['type'] == 'result' and report['value']['after']['closed']
    events = [json.loads(x) for x in progress.read_text(encoding='utf-8').splitlines()]
    assert any(e['type']=='progress' and e['value']['percent']==100 for e in events)
    assert events[-1]['type'] == 'result_file'
    assert (tmp_path/'local'/'MeshRepair'/'logs'/'meshrepair.log').exists()


def test_windowed_logging_rotation_and_fatal_dialog(tmp_path):
    logs = tmp_path/'MeshRepair'/'logs'; logs.mkdir(parents=True)
    (logs/'meshrepair.log').write_bytes(b'x'*(5*1024*1024))
    code = textwrap.dedent('''
        import sys
        sys.stdout = sys.stderr = None
        from meshrepair.runtime import configure
        path = configure()
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication([])
        QMessageBox.critical = lambda *args: path.with_name('dialog.txt').write_text(args[-1], encoding='utf-8')
        raise RuntimeError('Проверка аварии')
    ''')
    env = dict(os.environ, LOCALAPPDATA=str(tmp_path), QT_QPA_PLATFORM='offscreen')
    p = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, timeout=30)
    assert p.returncode != 0
    assert (logs/'meshrepair.log.1').stat().st_size == 5*1024*1024
    assert 'Проверка аварии' in (logs/'meshrepair.log').read_text(encoding='utf-8')
    assert str(logs/'meshrepair.log') in (logs/'dialog.txt').read_text(encoding='utf-8')
