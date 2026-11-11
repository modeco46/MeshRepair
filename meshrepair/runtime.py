"""Early diagnostics for pythonw / PyInstaller windowed entry points."""
import faulthandler
import os
from pathlib import Path
import sys
import tempfile
import traceback

_log_stream = None


def configure(worker=False):
    global _log_stream
    worker = worker or '--worker' in sys.argv
    root = Path(os.environ.get('LOCALAPPDATA') or tempfile.gettempdir())/'MeshRepair'/'logs'
    root.mkdir(parents=True, exist_ok=True)
    path = root/'meshrepair.log'
    # Rotate at GUI startup only: workers may still have the append log open.
    # Keeping a real file object also gives faulthandler a native descriptor.
    if not worker and path.exists() and path.stat().st_size >= 5*1024*1024:
        try:
            path.with_suffix('.log.3').unlink(missing_ok=True)
            for index in (2, 1):
                old = path.with_suffix(f'.log.{index}')
                if old.exists():
                    old.replace(path.with_suffix(f'.log.{index+1}'))
            path.replace(path.with_suffix('.log.1'))
        except OSError:
            pass  # Another app instance may own the file on Windows.
    _log_stream = path.open('a', encoding='utf-8', buffering=1)
    sys._meshrepair_redirected_stdout = sys.stdout is None
    if sys.stdout is None:
        sys.stdout = _log_stream
    if sys.stderr is None:
        sys.stderr = _log_stream
    faulthandler.enable(file=_log_stream, all_threads=True)

    def fatal(kind, value, tb):
        traceback.print_exception(kind, value, tb, file=_log_stream)
        _log_stream.flush()
        if worker or '--smoke-ui' in sys.argv or '--smoke-preview' in sys.argv:
            return
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(sys.argv[:1])
            QMessageBox.critical(None, 'MeshRepair — ошибка',
                                 f'Не удалось продолжить работу: {value}\n\nЖурнал: {path}')
        except Exception:
            pass
    sys.excepthook = fatal
    return path


def worker_program():
    """pythonw avoids a console even where PySide6 exposes no process modifier."""
    if sys.platform == 'win32' and not getattr(sys, 'frozen', False):
        windowed = Path(sys.executable).with_name('pythonw.exe')
        if windowed.is_file():
            return str(windowed)
    return sys.executable
