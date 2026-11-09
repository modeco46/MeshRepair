"""Windows integration: actual GUI worker completion and user-controlled stop."""
import subprocess
import json
import sys
import time
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows desktop integration')


def test_gui_repairs_and_updates_progress(tmp_path):
    import tkinter as tk
    from meshlib import mrmeshpy as mm
    from meshrepair.gui import App
    source = tmp_path / 'sample.stl'
    mesh = mm.makeCube()
    subdivision = mm.SubdivideSettings()
    subdivision.maxEdgeLen = .15
    subdivision.maxAngleChangeAfterFlip = 0
    mm.subdivideMesh(mesh, subdivision)
    mm.saveMesh(mesh, source)
    root = tk.Tk()
    root.withdraw()
    app = App(root)
    try:
        app.source.set(str(source))
        app.quality.set('0,02')
        assert str(app.angle_entry['state']) == 'disabled'
        app.planar_check.invoke()
        app.angle.set('0,5')
        app.curved_check.invoke()
        assert str(app.angle_entry['state']) == 'normal'
        app.start()
        assert str(app.angle_entry['state']) == 'disabled'
        deadline = time.monotonic() + 30
        while app.process and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert app.process is None
        assert app.finished
        assert float(app.progress['value']) == 100
        assert (tmp_path / 'sample_repaired.stl').is_file()
        report = json.loads((tmp_path / 'sample_repaired.stl.json').read_text(encoding='utf-8'))
        assert report['settings']['coplanar_angle_deg'] == .5
        assert report['settings']['simplify_curved']
        assert report['simplification']['accepted']
        assert str(app.angle_entry['state']) == 'normal'
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
