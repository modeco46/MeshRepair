from types import SimpleNamespace
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
import numpy as np
from meshrepair import compute


def test_cuda_projection_failure_falls_back_without_losing_batch(monkeypatch):
    class BrokenProjector:
        def updateMeshData(self, mesh):
            pass
        def findProjections(self, *args):
            raise RuntimeError('simulated CUDA memory exhaustion')
    monkeypatch.setattr(compute, 'cuda_module', lambda: SimpleNamespace(PointsToMeshProjector=BrokenProjector))
    report = {}
    projector = compute.Projector(mm.makeCube(), mm, report)
    assert projector.gpu
    result = mm.std_vector_MeshProjectionResult()
    points = mn.fromNumpyArray(np.array([[0, 0, 2]], dtype=np.float32))
    projector.findProjections(result, points, None, None, 1e30, 0)
    assert not projector.gpu
    assert len(result) == 1 and int(result[0].proj.face) >= 0
    assert report['projection_backends'] == ['cuda', 'cpu_native_parallel']


def test_cuda_setup_failure_falls_back(monkeypatch):
    class BrokenProjector:
        def updateMeshData(self, mesh):
            raise RuntimeError('simulated unsupported GPU')
    monkeypatch.setattr(compute, 'cuda_module', lambda: SimpleNamespace(PointsToMeshProjector=BrokenProjector))
    projector = compute.Projector(mm.makeCube(), mm)
    assert not projector.gpu
