"""Qt application controller; MeshLib operations stay in a killable process."""
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import time
from PySide6.QtCore import QTimer, QProcess, QProcessEnvironment, QSettings
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from .engine import Settings
from .theme import STYLE
from .viewer import PreviewLoader
from .widgets import LayoutMixin
from .reporting import report_html


class App(QMainWindow, LayoutMixin):
    def __init__(self, *, preview_enabled=True, persist=True):
        super().__init__()
        self.setWindowTitle('MeshRepair 0.8.1')
        self.setMinimumSize(980, 680)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, screen.width() - 60), min(880, screen.height() - 70))
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)
        self.preferences = QSettings('MeshRepair', 'MeshRepair') if persist else None
        self.source_path = self.view_path = self.result_path = None
        self.process = self.loader = None
        self.is_busy = self.stopping = self.closing = False
        self.operation = None
        self.reports = {}
        self.last_repair_report = self.last_check_report = None
        self.temp = tempfile.TemporaryDirectory(prefix='meshrepair-preview-')
        self.raw_output = b''
        self.job_result = None
        self.job_directory = self.job_result_file = None
        self.job_error = False
        self.setup_ui(preview_enabled and os.environ.get('MESHREPAIR_NO_3D') != '1')
        self.clock = QTimer(self)
        self.clock.setInterval(250)
        self.clock.timeout.connect(self.update_clock)
        QShortcut(QKeySequence.Open, self, activated=self.pick_source)
        if self.preferences:
            self.quality.setValue(float(self.preferences.value('quality', .02)))
            self.angle.setValue(float(self.preferences.value('angle', 1)))
            self.planar.setChecked(self.preferences.value('planar', False, type=bool))
            self.curved.setChecked(self.preferences.value('curved', False, type=bool))
            self.strict.setChecked(self.preferences.value('strict', False, type=bool))
            self.weld_close.setChecked(self.preferences.value('weld_close', False, type=bool))
            self.sharp.setChecked(self.preferences.value('sharp', True, type=bool))
        self.set_busy(False)

    def update_reduction_controls(self):
        self.angle.setEnabled(not self.is_busy and self.planar.isChecked() and not self.curved.isChecked())

    def write(self, text):
        self.log.appendPlainText(str(text))

    def clear_logs(self):
        self.log.clear()

    def copy_logs(self):
        text = self.log.textCursor().selectedText().replace('\u2029', '\n')
        QApplication.clipboard().setText(text or self.log.toPlainText())

    def show_renderer(self, name):
        self.render_label.setToolTip('Устройство отображения OpenGL: ' + name)
        self.write('3D-отображение: ' + name)

    def update_heatmap(self):
        self.view.set_heatmap(self.heatmap_button.isChecked())
        self.heatmap_legend.setVisible(self.heatmap_button.isChecked() and self.view.heatmap_data is not None)

    def update_view(self):
        self.view.set_mode(self.surface_button.isChecked(), self.edge_button.isChecked())

    @staticmethod
    def drop_path(mime):
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.is_file() and path.suffix.lower() in {'.stl', '.obj', '.ply'}:
                        return path
        return None

    def dragEnterEvent(self, event):
        if not self.is_busy and self.drop_path(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self.drop_path(event.mimeData())
        if path and not self.is_busy:
            self.set_source(path)
            event.acceptProposedAction()

    def pick_source(self):
        if self.is_busy:
            return
        filename, _ = QFileDialog.getOpenFileName(self, 'Открыть сетку', '', 'Сетки (*.stl *.obj *.ply)')
        if filename:
            self.set_source(filename)

    def set_source(self, path):
        if self.is_busy:
            return
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() not in {'.stl', '.obj', '.ply'}:
            self.write('Поддерживаются существующие STL, OBJ и PLY.')
            return
        self.source_path = self.view_path = path
        self.result_path = None
        self.reports.clear()
        self.last_repair_report = self.last_check_report = None
        self.heatmap_button.setChecked(False)
        self.update_heatmap()
        self.report_view.setHtml(report_html({}))
        self.filename.setText(path.name)
        self.filename.setToolTip(str(path))
        self.view.clear()
        self.update_info({})
        self.source_button.setChecked(True)
        self.result_button.setChecked(False)
        self.write('\nОткрыт файл: ' + str(path))
        self.start('preview')

    def show_source(self, result):
        if self.is_busy:
            return
        path = self.result_path if result else self.source_path
        if path is None:
            return
        self.source_button.setChecked(not result)
        self.result_button.setChecked(result)
        if self.view_path == path and self.view.data is not None:
            return
        self.view.clear()
        self.view_path = path
        self.update_info(self.reports.get(str(path), {}))
        self.start('preview')

    def set_busy(self, busy):
        self.is_busy = busy
        for w in (self.browse, self.quality, self.planar, self.curved, self.strict, self.weld_close, self.sharp):
            w.setEnabled(not busy)
        self.update_reduction_controls()
        self.repair.setEnabled(not busy and self.source_path is not None)
        self.check_button.setEnabled(not busy and self.view_path is not None)
        self.source_button.setEnabled(not busy and self.source_path is not None)
        self.result_button.setEnabled(not busy and self.result_path is not None)
        self.stop_button.setEnabled(busy)
        self.heatmap_button.setEnabled(not busy and self.view.heatmap_data is not None)

    def start(self, operation):
        if self.is_busy or self.source_path is None:
            return
        path = self.source_path if operation == 'repair' else self.view_path
        settings = Settings(self.quality.value(), self.planar.isChecked(), self.angle.value(), self.curved.isChecked(), self.strict.isChecked(), self.weld_close.isChecked(), self.sharp.isChecked())
        settings.validate()
        command = ['--worker'] if getattr(sys, 'frozen', False) else ['-u', '-m', 'meshrepair.worker']
        self.job_directory = Path(tempfile.mkdtemp(prefix='.meshrepair-', dir=path.parent if operation == 'repair' else self.temp.name))
        self.job_result_file = self.job_directory / 'worker-result.json'
        command += [str(path), '--result-file', str(self.job_result_file), '--work-dir', str(self.job_directory)]
        if operation == 'check':
            command += ['--check']
        elif operation == 'preview':
            command += ['--preview-output', str(Path(self.temp.name) / 'view.npz')]
        else:
            command += ['--quality', str(settings.quality_mm)]
            if settings.collapse_coplanar:
                command += ['--collapse-coplanar', '--coplanar-angle', str(settings.coplanar_angle_deg)]
            if settings.simplify_curved:
                command += ['--simplify-curved']
            if settings.strict_tolerance:
                command += ['--strict-tolerance']
            if settings.weld_close_vertices:
                command += ['--weld-close-vertices']
            if not settings.restore_sharp_edges:
                command += ['--no-restore-sharp-edges']
        self.operation = operation
        self.job_path = path
        self.raw_output = b''
        self.job_result = None
        self.job_error = self.stopping = False
        self.started = time.monotonic()
        self.progress.setValue(0)
        self.status.setText({'preview': 'Подготовка предпросмотра…', 'check': 'Проверка сетки…', 'repair': 'Исправление сетки…'}[operation])
        if operation != 'preview':
            self.write('\n' + self.status.text() + '\n' + str(path))
            self.tabs.setCurrentIndex(0)
        self.set_busy(True)
        self.clock.start()
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardError.connect(lambda: self.write(bytes(process.readAllStandardError()).decode("utf-8", errors="replace").rstrip()))
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONIOENCODING', 'utf-8')
        env.insert('PYTHONUNBUFFERED', '1')
        process.setProcessEnvironment(env)
        if not getattr(sys, 'frozen', False):
            process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.readyReadStandardOutput.connect(self.read_worker)
        process.finished.connect(self.worker_finished)
        process.errorOccurred.connect(self.process_error)
        self.process = process
        process.start(sys.executable, command)

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.write('Не удалось запустить рабочий процесс: ' + self.process.errorString())
            self.job_error = True
            self.worker_finished(-1, QProcess.CrashExit)

    def read_worker(self):
        if self.process is None:
            return
        self.raw_output += bytes(self.process.readAllStandardOutput())
        while b'\n' in self.raw_output:
            line, self.raw_output = self.raw_output.split(b'\n', 1)
            self.handle_line(line.decode('utf-8', errors='replace').rstrip())

    def handle_line(self, line):
        try:
            event = json.loads(line)
        except ValueError:
            if line:
                self.write(line)
            return
        if not isinstance(event, dict) or 'type' not in event:
            self.write(line)
            return
        kind, value = event['type'], event.get('value')
        if kind == 'progress' and not self.stopping:
            percent = int(value['percent'] * (.8 if self.operation == 'preview' else 1))
            self.progress.setValue(percent)
            self.status.setText(f"{percent}% · {value['stage']}")
        elif kind == 'log':
            self.write(value)
        elif kind == 'error':
            self.job_error = True
            self.write(value)
        elif kind == 'result':
            self.job_result = value

    def worker_finished(self, code, exit_status):
        if self.process is None:
            return
        self.read_worker()
        if self.raw_output:
            self.handle_line(self.raw_output.decode('utf-8', errors='replace'))
            self.raw_output = b''
        if not self.stopping and self.job_result_file and self.job_result_file.is_file():
            try:
                event = json.loads(self.job_result_file.read_text(encoding='utf-8'))
                if event.get('type') == 'result':
                    self.job_result = event['value']
                    self.job_error = False
                elif event.get('type') == 'error':
                    self.job_error = True
                    self.write(event['value'])
            except (ValueError, OSError, KeyError) as exc:
                self.job_error = True
                self.write('Не удалось прочитать отчёт рабочего процесса: '+str(exc))
        self.cleanup_job()
        self.process.deleteLater()
        self.process = None
        if self.closing:
            return
        if self.stopping:
            self.finish_ui('Остановлено пользователем.')
            self.write('Остановлено. Исходный файл не изменён.')
            return
        if code or self.job_error or self.job_result is None:
            self.finish_ui('Операция не завершена. Подробности в журнале.')
            if not self.job_error:
                self.write(f'Рабочий процесс завершился без результата (код {code}).')
            return
        report = self.job_result
        capability = report.get('acceleration', {})
        cpu = capability.get('cpu_threads', 'авто')
        gpu = capability.get('gpu_name', 'CUDA недоступна')
        self.hardware.setText(f'CPU × {cpu} · ' + ('CUDA' if capability.get('cuda_available') else 'без CUDA'))
        self.hardware.setToolTip(gpu + '\nCUDA: winding number и проекции. Топология и decimation: CPU.')
        if self.operation == 'preview':
            previous = self.reports.get(str(self.job_path), {})
            if previous.get('checked'):
                report = dict(report, info=dict(previous['info'], **report['info']), checked=True,
                              shells=previous.get('shells', []))
            self.reports[str(self.job_path)] = report
            self.update_info(report)
            skipped = report.get('loader', {}).get('skipped_faces', 0)
            if skipped:
                self.write(f'Загрузчик пропустил непригодные грани: {skipped}.')
            duplicated = report.get('loader', {}).get('duplicated_vertices', 0)
            if duplicated:
                self.write(f'При загрузке разделены вершины: {duplicated}. Число связных частей может включать перевёрнутые фрагменты. Подробности — кнопка «Проверить».')
            self.status.setText('85% · Загрузка геометрии в 3D-окно…')
            self.progress.setValue(85)
            heatmap = (self.last_repair_report or {}).get('source_deviation', {}).get('heatmap_file')
            self.loader = PreviewLoader(report['preview'], self, heatmap=heatmap, center=report.get('center_mm', [0, 0, 0]))
            self.loader.loaded.connect(self.preview_loaded)
            self.loader.failed.connect(self.preview_failed)
            self.loader.finished.connect(self.loader_finished)
            self.loader.start()
            return
        self.progress.setValue(100)
        self.finish_ui('Готово.')
        if self.operation == 'check':
            self.last_check_report = report
            self.report_view.setHtml(report_html(report))
            self.tabs.setCurrentWidget(self.report_view)
            self.reports[str(self.job_path)] = report
            self.update_info(report)
            self.write_inspection(report)
            self.status.setText('Проверка завершена. Файл не изменён.')
        else:
            self.last_repair_report = report
            self.report_view.setHtml(report_html(report))
            self.tabs.setCurrentWidget(self.report_view)
            self.result_path = Path(report['output'])
            self.reports[str(self.result_path)] = {'info': report['after'], 'checked': True, 'shells': report.get('shells', [])}
            info = report['after']
            self.write(f"Результат: {self.result_path}\nТреугольников: {info['triangles']:,}; оболочек: {info['components']}; дыр: {info['boundary_loops']}; пересекающихся граней: {info['self_intersecting_faces']}.")
            for warning in report['warnings']:
                self.write(warning)
            self.write(f"Время ремонта: {report['elapsed_seconds']:.1f} с. JSON-отчёт сохранён рядом с STL.")
            self.result_button.setEnabled(True)
            QTimer.singleShot(0, lambda: self.show_source(True) if not self.closing else None)

    def preview_loaded(self, data):
        if self.stopping or self.closing:
            return
        try:
            self.view.set_data(data)
            self.update_heatmap()
            if getattr(data, '_heatmap_error', None):
                self.write('Карта отклонения: '+data._heatmap_error)
        except RuntimeError as exc:
            self.preview_failed(str(exc))
            return
        self.progress.setValue(100)
        self.status.setText('Предпросмотр готов. ' + ('Результат ремонта.' if self.view_path == self.result_path else 'Исходная сетка.'))

    def preview_failed(self, message):
        if not self.closing:
            self.write('Предпросмотр: ' + message)
            self.status.setText(message)

    def loader_finished(self):
        if self.loader:
            self.loader.deleteLater()
            self.loader = None
        if not self.closing:
            self.clock.stop()
            self.set_busy(False)

    def finish_ui(self, status):
        self.clock.stop()
        self.set_busy(False)
        self.status.setText(status)

    def update_clock(self):
        seconds = int(time.monotonic() - self.started)
        self.elapsed.setText(f'{seconds // 60}:{seconds % 60:02d}')

    def update_info(self, report):
        info = report.get('info', {})
        for key, widget in self.stat_labels.items():
            value = info.get(key)
            def formatted(v):
                return ('Да' if v else 'Нет') if key == 'closed' and v is not None else (f'{v:,}'.replace(',', ' ') if v is not None else '—')
            if self.last_repair_report and self.view_path == self.result_path:
                widget.setText(formatted(self.last_repair_report['before'].get(key))+' → '+formatted(value))
            else:
                widget.setText(formatted(value))
        checked = report.get('checked', False)
        self.check_state.setText('Проверка выполнена' if checked else 'Пересечения ещё не проверены')
        self.shell_model.set_rows(report.get('shells', []))
        dimensions = info.get('dimensions_mm')
        lines = []
        if dimensions:
            lines.append('Размеры: ' + ' × '.join(f'{x:.3f}' for x in dimensions) + ' мм')
        if 'vertices' in info:
            lines.append(f"Вершин: {info['vertices']:,}".replace(',', ' '))
        if checked:
            lines.append(f"Дыр: {info.get('boundary_loops', '—')} · Пересекающихся граней: {info.get('self_intersecting_faces', '—')} · Вырожденных граней: {info.get('degenerate_faces', '—')}")
            if 'closed_components' in info:
                lines.append(f"Замкнутых оболочек: {info['closed_components']} · Открытых: {info['open_components']}")
        else:
            lines.append('Для полной диагностики нажмите «Проверить».')
        self.detail_text.setText('\n'.join(lines))

    def write_inspection(self, report):
        info = report['info']
        self.write(f"Проверка завершена: {info['triangles']:,} треугольников; {info['components']} связных частей; дыр {info['boundary_loops']}; пересекающихся граней {info['self_intersecting_faces']}. Подробности — вкладка «Отчёт».")

    def cleanup_job(self):
        if self.job_directory:
            shutil.rmtree(self.job_directory, ignore_errors=True)
        self.job_directory = self.job_result_file = None

    def cancel(self):
        if not self.is_busy:
            return
        self.stopping = True
        self.stop_button.setEnabled(False)
        self.status.setText('Остановка…')
        if self.process:
            self.process.kill()
        if self.loader:
            self.loader.requestInterruption()

    def closeEvent(self, event):
        self.closing = True
        self.cancel()
        if self.process:
            self.process.waitForFinished(5000)
        self.cleanup_job()
        if self.loader:
            self.loader.wait()
        if self.preferences:
            for key, value in [('quality', self.quality.value()), ('angle', self.angle.value()),
                               ('planar', self.planar.isChecked()), ('curved', self.curved.isChecked()),
                               ('strict', self.strict.isChecked()), ('weld_close', self.weld_close.isChecked()), ('sharp', self.sharp.isChecked())]:
                self.preferences.setValue(key, value)
        self.view.shutdown()
        self.temp.cleanup()
        event.accept()


def main(smoke=False):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    window = App(preview_enabled=not smoke, persist=not smoke)
    window.show()
    if smoke:
        assets = Path(__file__).with_name('assets')
        assert all((assets / name).is_file() for name in ('check.svg', 'up.svg', 'down.svg'))
        QTimer.singleShot(100, window.close)
        QTimer.singleShot(200, app.quit)
    return app.exec()
