"""Qt offscreen save flow, overwrite/close questions and job ownership."""
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from meshlib import mrmeshpy as mm
from meshrepair.gui import App
from meshrepair.engine import run
from meshrepair.storage import create_job
from meshrepair.saving import SaveDialog


@pytest.fixture
def window(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    app=QApplication.instance() or QApplication([])
    w=App(preview_enabled=False,persist=False)
    w.source_path=tmp_path/'source.stl';mm.saveMesh(mm.makeCube(),w.source_path)
    w.result_directory=create_job(tmp_path/'temp')
    w.last_repair_report=run(w.source_path,work_dir=w.result_directory)
    w.result_path=Path(w.last_repair_report['output'])
    w.unsaved_result=True;w.set_busy(False)
    yield w,app
    w.unsaved_result=False;w.close();app.processEvents()


def test_dialog_browse_selected_set_and_repeated_save(window,tmp_path,monkeypatch):
    w,app=window;target=tmp_path/'saved.stl'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(target),'STL'))
    def dialog(d):
        assert d.selection()==dict(stl=True,json=False,deviation=False,log=False)
        d.browse();d.choices['json'].setChecked(True);return QDialog.Accepted
    monkeypatch.setattr(SaveDialog,'exec',dialog)
    assert w.save_result()
    assert target.exists() and Path(str(target)+'.json').exists()
    assert not Path(str(target)+'.deviation.npz').exists()
    assert not w.unsaved_result and w.result_path.exists()
    assert w.save_to(tmp_path/'second.stl',{'log':True})
    assert not (tmp_path/'second.stl').exists()
    assert (tmp_path/'second.stl.log.txt').is_file()


def test_decline_overwrite_and_forbid_source(window,tmp_path,monkeypatch):
    w,_=window;target=tmp_path/'saved.stl';target.write_bytes(b'old')
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:QMessageBox.No)
    assert not w.save_to(target,{'stl':True}) and target.read_bytes()==b'old'
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:QMessageBox.Yes)
    assert w.save_to(target,{'stl':True}) and target.read_bytes()!=b'old'
    warnings=[];monkeypatch.setattr(QMessageBox,'warning',lambda *a: warnings.append(a))
    before=w.source_path.read_bytes()
    assert not w.save_to(w.source_path,{'stl':True})
    assert before==w.source_path.read_bytes() and warnings
    assert not list(tmp_path.glob('*.part'))


def test_close_unsaved_cancel_then_discard_cleans_temp(window,monkeypatch):
    w,app=window;w.show();folder=w.result_directory
    monkeypatch.setattr(QMessageBox,'exec',lambda *a:QMessageBox.Cancel)
    assert not w.close();app.processEvents()
    assert folder.exists() and not w.closing
    monkeypatch.setattr(QMessageBox,'exec',lambda *a:QMessageBox.Discard)
    assert w.close();app.processEvents();assert not folder.exists()


def test_saving_sidecars_alone_does_not_discard_unsaved_geometry(window,tmp_path):
    w,_=window
    assert w.save_to(tmp_path/'report.stl',{'json':True})
    assert w.unsaved_result


def test_new_source_removes_old_result_job(window,tmp_path,monkeypatch):
    w,_=window;folder=w.result_directory
    monkeypatch.setattr(QMessageBox,'exec',lambda *a:QMessageBox.Discard)
    monkeypatch.setattr(w,'start',lambda *a:None)
    new=tmp_path/'new.stl';mm.saveMesh(mm.makeCube(),new)
    w.set_source(new)
    assert not folder.exists() and w.result_path is None


def test_progress_heartbeat(window,monkeypatch):
    w,_=window
    w.started=w.last_progress_at=w.stage_started=10.;w.last_heartbeat=0.;w.last_stage='Восстановление поверхности'
    # Only update_clock uses this fake; restore before close's process lifecycle.
    w.process=object()
    monkeypatch.setattr('meshrepair.gui.time.monotonic',lambda:26.)
    w.update_clock()
    assert 'идёт 16 с, этап без процентов' in w.status.text()
    w.process=None


def test_retry_without_sharpening_keeps_original_settings(window, monkeypatch):
    import json
    from meshrepair.engine import Settings
    w, _ = window
    original = Settings(.05, True, 5, True, disable_cuda=True, temp_directory='/tmp/original')
    w.job_settings = original
    w.operation = 'repair'
    event = {'type': 'retry_without_sharpening', 'value': {
        'message': 'Нажмите «Повторить без Sharpening».',
        'report': {'before': {'triangles': 12}}}}
    w.set_busy(True)
    w.handle_line(json.dumps(event))
    assert not w.retry_button.isHidden() and not w.retry_button.isEnabled()
    w.set_busy(False)
    assert w.retry_button.isEnabled()
    # Editing controls afterwards must not alter the failed job's snapshot.
    w.quality.setValue(.2)
    w.planar.setChecked(False)
    calls = []
    monkeypatch.setattr(w, 'start', lambda *args, **kwargs: calls.append((args, kwargs)))
    w.retry_button.click()
    assert len(calls) == 1 and calls[0][0] == ('repair',)
    settings = calls[0][1]['settings_override']
    assert settings.quality_mm == .05 and settings.collapse_coplanar
    assert settings.simplify_curved and settings.coplanar_angle_deg == 5
    assert settings.disable_cuda and settings.temp_directory == original.temp_directory
    assert not settings.restore_sharp_edges
    w.unsaved_result = False
    w.set_source(w.source_path)
    assert w.retry_settings is None and w.retry_button.isHidden()
