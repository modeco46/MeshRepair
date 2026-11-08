"""One file, one quality setting, Repair and Stop. Native work is isolated."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from .engine import Settings


class App:
    def __init__(self, root):
        self.root = root
        root.title('MeshRepair 0.3')
        root.geometry('850x560')
        root.minsize(680, 440)
        self.process = None
        self.events = queue.Queue()
        self.finished = False
        self.stopping = False
        self.started = 0
        panel = ttk.Frame(root, padding=18)
        panel.pack(fill='both', expand=True)
        ttk.Label(panel, text='Исходная сетка').pack(anchor='w')
        row = ttk.Frame(panel)
        row.pack(fill='x', pady=(5, 14))
        self.source = tk.StringVar()
        self.source_entry = ttk.Entry(row, textvariable=self.source)
        self.source_entry.pack(side='left', fill='x', expand=True)
        self.browse = ttk.Button(row, text='Выбрать файл…', command=self.pick_source)
        self.browse.pack(side='left', padx=(8, 0))
        row = ttk.Frame(panel)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text='Качество, мм:').pack(side='left')
        self.quality = tk.StringVar(value='0.02')
        self.quality_entry = ttk.Combobox(row, textvariable=self.quality, width=10,
            values=['0.001', '0.005', '0.01', '0.02', '0.05', '0.1', '0.2', '0.5'])
        self.quality_entry.pack(side='left', padx=8)
        ttk.Label(row, text='Меньше — подробнее, больше — быстрее.').pack(side='left', padx=8)
        ttk.Label(panel, wraplength=780, text=(
            'Единицы: миллиметры. Результат сохраняется рядом с исходником в новый STL. '
            'Исходный файл не изменяется.')).pack(anchor='w', pady=(7, 10))
        row = ttk.Frame(panel)
        row.pack(fill='x', pady=8)
        self.repair = ttk.Button(row, text='Исправить', command=self.start)
        self.repair.pack(side='left')
        self.stop_button = ttk.Button(row, text='Остановить', command=self.cancel, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        self.elapsed = tk.StringVar(value='')
        ttk.Label(row, textvariable=self.elapsed).pack(side='right')
        self.status = tk.StringVar(value='Выберите файл и нажмите «Исправить».')
        ttk.Label(panel, textvariable=self.status, wraplength=780).pack(anchor='w', pady=(6, 4))
        self.progress = ttk.Progressbar(panel, mode='determinate', maximum=100)
        self.progress.pack(fill='x', pady=(0, 12))
        self.log = tk.Text(panel, height=16, wrap='word', state='disabled')
        self.log.pack(fill='both', expand=True)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_id = root.after(100, self.poll)

    def write(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def pick_source(self):
        path = filedialog.askopenfilename(filetypes=[('Сетка', '*.stl *.obj *.ply')])
        if path:
            self.source.set(path)

    def busy(self, value):
        state = 'disabled' if value else 'normal'
        for widget in (self.source_entry, self.browse, self.quality_entry, self.repair):
            widget.configure(state=state)
        self.stop_button.configure(state='normal' if value else 'disabled')

    def start(self):
        if self.process:
            return
        try:
            value = float(self.quality.get().replace(',', '.'))
            Settings(value).validate()
            if not Path(self.source.get()).is_file():
                raise ValueError('Выберите исходную сетку.')
            command = ([sys.executable, '--worker'] if getattr(sys, 'frozen', False)
                       else [sys.executable, '-u', '-m', 'meshrepair.worker'])
            command += [self.source.get(), '--quality', str(value)]
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except Exception as exc:
            messagebox.showerror('Ошибка запуска', str(exc))
            return
        self.finished = self.stopping = False
        self.started = time.monotonic()
        self.busy(True)
        self.progress['value'] = 0
        self.status.set('Запуск…')
        self.write('\nИсправление: ' + self.source.get())
        threading.Thread(target=self.read_worker, args=(self.process,), daemon=True).start()

    def read_worker(self, process):
        for line in process.stdout:
            self.events.put(line.rstrip())
        process.stdout.close()
        self.events.put({'exit': process.wait()})

    def poll(self):
        if self.process:
            seconds = int(time.monotonic() - self.started)
            self.elapsed.set(f'{seconds // 60}:{seconds % 60:02d}')
        try:
            while True:
                event = self.events.get_nowait()
                if isinstance(event, dict):
                    if self.stopping and not self.finished:
                        self.status.set('Остановлено пользователем.')
                        self.write('Остановлено. Исходный файл не изменён.')
                    elif not self.finished:
                        self.status.set('Процесс завершился без результата.')
                        self.write(f'Процесс завершился (код {event["exit"]}). Возможна ошибка нативной библиотеки или нехватка памяти.')
                    self.process = None
                    self.busy(False)
                    continue
                try:
                    item = json.loads(event)
                except ValueError:
                    self.write(event)
                    continue
                if not isinstance(item, dict) or 'type' not in item:
                    self.write(event)
                    continue
                kind, value = item['type'], item.get('value')
                if kind == 'progress' and not self.stopping:
                    self.progress['value'] = value['percent']
                    self.status.set(f"{value['percent']}% — {value['stage']}")
                elif kind == 'result':
                    self.finished = True
                    self.progress['value'] = 100
                    self.status.set('Готово. Результат сохранён рядом с исходником.')
                    d = value['after']
                    self.write(f"Результат: {value['output']}")
                    self.write(f"Треугольников: {d['triangles']:,}; замкнутых оболочек: {d['components']}; "
                               f"дыр: {d['boundary_loops']}; пересекающихся граней: {d['self_intersecting_faces']}.")
                    for warning in value['warnings']:
                        self.write(warning)
                    self.write(f"Время: {value['elapsed_seconds']:.1f} с. Подробности — в JSON рядом с STL.")
                elif kind == 'error':
                    self.finished = True
                    self.status.set('Не удалось завершить исправление.')
                    self.write(str(value))
                elif kind == 'log':
                    self.write(str(value))
        except queue.Empty:
            pass
        self.poll_id = self.root.after(100, self.poll)

    def cancel(self):
        if self.process and self.process.poll() is None and not self.stopping:
            self.stopping = True
            self.stop_button.configure(state='disabled')
            self.status.set('Остановка…')
            self.process.terminate()

    def close(self):
        self.cancel()
        if self.process:
            self.process.wait(timeout=10)
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
