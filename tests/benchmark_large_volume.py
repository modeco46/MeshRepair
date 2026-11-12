"""Manual CPU memory comparison; run each backend in a fresh process."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshlib import mrmeshpy as mm
from meshrepair import engine
from meshrepair.engine import Settings, Progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['auto', 'dense'], default='auto')
    args = parser.parse_args()
    if args.backend == 'dense':
        engine.DENSE_WINDING_VOXEL_LIMIT = 10**12
    mesh = mm.makeCube(mm.Vector3f(8, 8, 8), mm.Vector3f(-4, -4, -4))
    mesh.addMesh(mm.makeCube(mm.Vector3f(8, 8, 8), mm.Vector3f(-3.8, -3.8, -3.8)))
    report = {'warnings': []}
    started = time.monotonic()
    result = engine.sparse_repair(mesh, mm, Settings(.02, restore_sharp_edges=False, disable_cuda=True),
        Progress(lambda _: None), report, lambda message: print(message, flush=True))
    summary = {'seconds': time.monotonic()-started, 'faces': result.topology.numValidFaces(),
        'valid': engine.valid(report['reconstruction_validation']), 'volume_mm3': result.volume(),
        'estimate': report['voxel_estimate'], 'passes': report['voxel_passes']}
    if sys.platform.startswith('linux'):
        import resource
        summary['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    assert summary['valid']
