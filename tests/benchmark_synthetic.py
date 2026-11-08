"""Reproducible synthetic benchmark, not a proxy for arbitrary damaged meshes.
Run from repository root: python tests/benchmark_synthetic.py.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from meshlib import mrmeshpy as mm


def main():
    with tempfile.TemporaryDirectory() as folder:
        params = mm.SphereParams()
        params.radius = 10
        params.numMeshVertices = 84502
        a = mm.makeSphere(params)
        b = mm.makeSphere(params)
        b.transform(mm.AffineXf3f.translation(mm.Vector3f(7, 0, 0)))
        a.addMesh(b)
        a.addMesh(mm.makeCube())
        source = Path(folder) / 'overlap.stl'
        output = Path(folder) / 'result.stl'
        mm.saveMesh(a, source)
        p = subprocess.run([sys.executable, '-m', 'meshrepair.worker', str(source), '--output', str(output),
                            '--quality', '.001'],
                           capture_output=True, encoding='utf-8', timeout=180)
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
        report = json.loads(Path(str(output) + '.json').read_text(encoding='utf-8'))
        print(json.dumps({key: report[key] for key in ['before', 'after', 'elapsed_seconds',
                         'algorithm', 'compute_device']}, indent=2))


if __name__ == '__main__':
    main()
