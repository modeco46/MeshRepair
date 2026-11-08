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
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / 'cube.stl'
        target = Path(folder) / 'result.stl'
        mm.saveMesh(mm.makeCube(mm.Vector3f(.1, .1, .1)), source)
        p = subprocess.run([executable, '--worker', str(source), '--output', str(target), '--device', 'cpu'],
                           capture_output=True, encoding='utf-8', errors='replace', timeout=180)
        print(p.stdout)
        print(p.stderr)
        assert p.returncode == 0
        assert target.is_file()
        report = json.loads(Path(str(target) + '.json').read_text(encoding='utf-8'))
        assert report['after']['closed']
        assert report['after']['components'] == 1
