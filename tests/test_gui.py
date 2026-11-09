"""Windows integration: actual GUI worker completion and user-controlled stop."""
import subprocess
import json
from pathlib import Path
import sys
import time
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows desktop integration')


def run_gui_case(name, *args):
    # Each application launch owns one Tcl/Tk interpreter, just like production.
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), name, *map(str, args)],
                             capture_output=True, timeout=45)
    assert process.returncode == 0, process.stderr


def test_gui_repairs_and_updates_progress(tmp_path):
    run_gui_case('repair', tmp_path)


def test_stop_button_terminates_worker():
    run_gui_case('stop')


def exercise_repair(tmp_path):
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


def exercise_stop():
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


if __name__ == '__main__':
    # Running the file directly otherwise puts only tests/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if sys.argv[1] == 'repair':
        exercise_repair(Path(sys.argv[2]))
    else:
        exercise_stop()
