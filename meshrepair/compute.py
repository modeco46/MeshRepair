"""Use supported CUDA kernels with a native CPU fallback; report actual choices."""
from functools import lru_cache
import os


def cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


@lru_cache(maxsize=1)
def cuda_module():
    try:
        from meshlib import mrcudapy as cuda
        info = cuda.getDeviceInfo()
        if info.fitForComputations():
            return cuda
    except Exception:
        pass
    return None


def capabilities():
    result = {'cpu_threads': cpu_count(), 'cuda_available': False,
              'topology_backend': 'cpu_native_parallel', 'decimation_backend': 'cpu_native_parallel'}
    cuda = cuda_module()
    if cuda is not None:
        info = cuda.getDeviceInfo()
        result.update(cuda_available=True, gpu_name=info.name,
                      gpu_memory_bytes=int(info.totalGlobalMem))
    return result


class Projector:
    """Own the target mesh for the lifetime of either native projector."""
    def __init__(self, mesh, mm, report=None, emit=lambda message: None):
        self.mesh, self.mm, self.report, self.emit = mesh, mm, report, emit
        self.native = None
        self.gpu = False
        cuda = cuda_module()
        if cuda is not None:
            try:
                self.native = cuda.PointsToMeshProjector()
                self.native.updateMeshData(mesh)
                self.gpu = True
                self._record('cuda')
            except (RuntimeError, MemoryError, TypeError, AttributeError) as exc:
                self._cpu(str(exc))
        else:
            self._cpu()

    def _record(self, backend):
        if self.report is not None:
            devices = self.report.setdefault('projection_backends', [])
            if backend not in devices:
                devices.append(backend)
        self.emit('Поиск проекций для проверки допуска: ' + ('CUDA GPU.' if self.gpu else 'многопоточный CPU.'))

    def _cpu(self, reason=None):
        # Release GPU buffers before allocating a fallback.
        self.native = None
        self.gpu = False
        self.native = self.mm.PointsToMeshProjector()
        self.native.updateMeshData(self.mesh)
        if reason:
            self.emit('CUDA-проекции недоступны; продолжаю на CPU: ' + reason)
        self._record('cpu_native_parallel')

    def findProjections(self, *args):
        try:
            self.native.findProjections(*args)
        except (RuntimeError, MemoryError, TypeError) as exc:
            if not self.gpu:
                raise
            self._cpu(str(exc))
            self.native.findProjections(*args)
