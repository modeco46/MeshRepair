"""Owned temporary jobs and explicit, atomic publication of selected artifacts."""
import atexit
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

_owned = set()


def pid_alive(pid):
    if pid <= 0:
        return True  # Unknown owner: conservative, never delete another live job.
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER = dead PID.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True


def cleanup_stale(root=None, *, legacy=False, now=None):
    root = Path(root or tempfile.gettempdir())
    now = time.time() if now is None else now
    removed = []
    for path in root.glob('.meshrepair-*' if legacy else 'meshrepair-job-*'):
        try:
            if path.is_symlink() or not path.is_dir() or now-path.stat().st_mtime <= 86400:
                continue
            owner = path/'owner.pid'
            if owner.exists():
                if pid_alive(int(owner.read_text(encoding='ascii').strip())):
                    continue
            elif not legacy:
                continue
            shutil.rmtree(path)
            removed.append(str(path))
        except (OSError, ValueError):
            continue
    return removed


def check_space(folder, triangles):
    required = max(0, int(triangles))*50*2
    free = shutil.disk_usage(folder).free
    if free < required:
        raise OSError(f'Недостаточно места во временной папке: нужно примерно {required:,} байт, свободно {free:,}.')
    return {'estimated_bytes': required, 'free_bytes': free}


def create_job(root=None, triangles=0):
    root = Path(root or tempfile.gettempdir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cleanup_stale(root)
    check_space(root, triangles)
    path = root / ('meshrepair-job-'+uuid.uuid4().hex)
    path.mkdir()
    try:
        (path/'owner.pid').write_text(str(os.getpid()), encoding='ascii')
    except OSError:
        path.rmdir()
        raise
    _owned.add(path)
    return path


def remove_job(path):
    if path is not None:
        path = Path(path)
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            _owned.discard(path)


def cleanup_owned():
    for path in list(_owned):
        remove_job(path)


def release_job(path):
    path = Path(path)
    (path/'owner.pid').write_text(str(os.getppid()), encoding='ascii')
    _owned.discard(path)


atexit.register(cleanup_owned)


def save_paths(target, selected):
    target = Path(target).expanduser().resolve()
    suffixes = {'stl': '', 'json': '.json', 'deviation': '.deviation.npz', 'log': '.log.txt'}
    return {key: Path(str(target)+suffix) for key, suffix in suffixes.items() if selected.get(key)}


def same_file(a, b):
    a, b = Path(a), Path(b)
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and os.path.samefile(a, b))


def save_artifacts(report, target, selected, log='', *, overwrite=False, no_replace=False):
    """Stage every selected file beside its destination before replacing any.

    Each rename is atomic. A filesystem error midway through several renames can
    leave a partial set; the caller reports the error and keeps the temporary job.
    """
    from .engine import clean_json
    target = Path(target).expanduser().resolve()
    paths = save_paths(target, selected)
    if not paths:
        raise ValueError('Отметьте хотя бы один файл для сохранения.')
    if target.suffix.lower() != '.stl' or not target.parent.is_dir():
        raise ValueError('Выберите существующую папку и имя с расширением .stl.')
    protected = [report['source'], *report.get('artifacts', {}).values()]
    if any(same_file(p, source) for p in [target, *paths.values()] for source in protected):
        raise ValueError('Нельзя перезаписать исходник или рабочие артефакты.')
    if not overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError('Выбранные файлы уже существуют. Требуется подтверждение перезаписи.')
    saved = deepcopy(report)
    saved.update(output=str(target), saved_at=datetime.now(timezone.utc).isoformat(),
                 saved_files={k: str(p) for k, p in paths.items()})
    if 'deviation' in paths:
        saved['source_deviation']['heatmap_file'] = str(paths['deviation'])
    staged = []
    try:
        for kind, path in paths.items():
            part = Path(str(path)+'.part')
            with part.open('xb') as stream:
                staged.append((part, path))
                if kind in ('stl', 'deviation'):
                    with open(report['artifacts'][kind], 'rb') as source:
                        shutil.copyfileobj(source, stream)
                else:
                    text = json.dumps(clean_json(saved), ensure_ascii=False, indent=2, allow_nan=False) if kind == 'json' else log
                    stream.write(text.encode('utf-8'))
                stream.flush()
                os.fsync(stream.fileno())
        for part, path in staged:
            if not overwrite and path.exists():
                raise FileExistsError(f'Файл появился во время сохранения: {path}')
            if no_replace:
                from .engine import publish_stl
                publish_stl(part, path)
            else:
                os.replace(part, path)
    finally:
        for part, _ in staged:
            part.unlink(missing_ok=True)
    return saved, {str(p): p.stat().st_size for p in paths.values()}
