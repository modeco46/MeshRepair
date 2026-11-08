"""Windows integration: actual GUI worker completion and user-controlled stop."""
import subprocess
import sys
import time
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows desktop integration')


def test_gui_repairs_and_updates_progress(tmp_path):
    import tkinter as tk
    from meshlib import mrmeshpy as mm
    from meshrepair.gui import App
    source = tmp_path / 'sample.stl'
    mm.saveMesh(mm.makeCube(), source)
    root = tk.Tk()
    root.withdraw()
    app = App(root)
    try:
        app.source.set(str(source))
        app.quality.set('0,5')
        app.start()
        deadline = time.monotonic() + 30
        while app.process and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert app.process is None
        assert app.finished
        assert float(app.progress['value']) == 100
        assert (tmp_path / 'sample_repaired.stl').is_file()
    finally:
        app.close()


def test_stop_button_terminates_worker():
    import tkinter as tk
    from meshrepair.gui import App
    root = tk.Tk()
    root.withdraw()
    app = App(root)
    try:
        app.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
        app.cancel()
        app.process.wait(timeout=5)
        assert app.stopping
        assert app.process.poll() is not None
    finally:
        app.close()
