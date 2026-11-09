"""Real Qt controls, QProcess protocol and preview buffers, in isolated launches."""
import os
from pathlib import Path
import subprocess
import sys
import time


def run_case(name, tmp_path):
    env = dict(os.environ)
    if sys.platform != 'win32':
        env['QT_QPA_PLATFORM'] = 'offscreen'
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), name, str(tmp_path)],
                             capture_output=True, env=env, timeout=90)
    assert process.returncode == 0, process.stderr.decode('utf-8', errors='replace')


def test_gui_preview_check_repair_and_progress(tmp_path):
    run_case('repair', tmp_path)


def test_drag_drop_clipboard_and_clear(tmp_path):
    run_case('drop', tmp_path)


def test_stop_button_terminates_native_worker(tmp_path):
    run_case('stop', tmp_path)


def pump(app, window, predicate=None):
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        app.processEvents()
        done = predicate() if predicate else not window.is_busy
        if done:
            return
        time.sleep(.01)
    raise AssertionError(window.log.toPlainText() + '\n' + window.status.text())


def exercise(name, tmp_path):
    from PySide6.QtCore import Qt, QMimeData, QUrl, QPoint, QPointF, QProcess
    from PySide6.QtGui import QDragEnterEvent, QDropEvent, QTextCursor
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from meshlib import mrmeshpy as mm
    from meshrepair.gui import App
    app = QApplication([])
    app.setStyle('Fusion')
    window = App(preview_enabled=False, persist=False)
    window.show()
    source = tmp_path / 'тест mesh.stl'
    mesh = mm.makeCube()
    settings = mm.SubdivideSettings()
    settings.maxEdgeLen = .15
    settings.maxAngleChangeAfterFlip = 0
    mm.subdivideMesh(mesh, settings)
    mm.saveMesh(mesh, source)
    try:
        if name == 'repair':
            window.set_source(source)
            pump(app, window)
            assert window.view.data.GetNumberOfCells() == mesh.topology.numValidFaces()
            assert not window.reports[str(source)]['checked']
            window.check_button.click()
            pump(app, window)
            assert window.last_check_report['info']['closed_components'] == 1
            assert window.shell_model.rowCount() == 1
            window.planar.click()
            window.curved.click()
            assert window.quality.valueFromText('0.02') == .02
            assert window.quality.valueFromText('0,02') == .02
            window.angle.setValue(.5)
            assert window.angle.isEnabled()
            window.repair.click()
            assert not window.angle.isEnabled()
            pump(app, window, lambda: window.last_repair_report is not None and not window.is_busy
                 and window.view_path == window.result_path and window.view.data is not None)
            assert window.last_repair_report['simplification']['accepted']
            assert window.progress.value() == 100
            assert window.result_path.is_file()
            assert window.angle.isEnabled()
            window.edge_button.click()
            assert window.view.edges
            window.surface_button.click()
            assert not window.view.surface
            window.source_button.click()
            pump(app, window)
            assert window.view_path == source
            assert window.view.data.GetNumberOfCells() == mesh.topology.numValidFaces()
            assert window.reports[str(source)]['checked']  # Preview must retain Check results.
        elif name == 'drop':
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(source))])
            enter = QDragEnterEvent(QPoint(40, 40), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            app.sendEvent(window, enter)
            assert enter.isAccepted()
            event = QDropEvent(QPointF(40, 40), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            app.sendEvent(window, event)
            assert event.isAccepted()
            pump(app, window)
            assert window.source_path == source
            window.clear_button.click()
            window.write('Первая строка\nВторая строка')
            window.copy_button.click()
            assert app.clipboard().text() == 'Первая строка\nВторая строка'
            cursor = window.log.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 6)
            window.log.setTextCursor(cursor)
            window.log.setFocus()
            QTest.keyClick(window.log, Qt.Key_C, Qt.ControlModifier)
            assert app.clipboard().text() == 'Первая'
            window.copy_button.click()
            assert app.clipboard().text() == 'Первая'
            window.clear_button.click()
            assert not window.log.toPlainText()
        else:
            window.process = QProcess(window)
            window.process.finished.connect(window.worker_finished)
            window.process.start(sys.executable, ['-c', 'import time;time.sleep(300)'])
            assert window.process.waitForStarted(5000)
            window.set_busy(True)
            window.stop_button.click()
            pump(app, window)
            assert window.process is None and window.stopping
            assert 'Остановлено' in window.status.text()
    finally:
        window.close()
        app.processEvents()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    exercise(sys.argv[1], Path(sys.argv[2]))
