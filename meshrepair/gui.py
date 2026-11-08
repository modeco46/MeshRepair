"""Small Russian desktop UI. Heavy work never runs on the Tk thread."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from .engine import Settings


class App:
    def __init__(self, root):
        self.root = root
        root.title("MeshRepair 0.1 — проверочная версия")
        root.geometry("850x670")
        self.process = None
        self.events = queue.Queue()
        self.finished = False
        panel = ttk.Frame(root, padding=16)
        panel.pack(fill="both", expand=True)
        self.source = tk.StringVar()
        self.target = tk.StringVar()
        self.tolerance = tk.StringVar(value="0.01")
        self.device = tk.StringVar(value="Авто")
        self.single = tk.BooleanVar(value=True)
        self.units = tk.BooleanVar(value=False)
        for label, variable, command in [("Исходная сетка", self.source, self.pick_source),
                                          ("Новый файл STL", self.target, self.pick_target)]:
            ttk.Label(panel, text=label).pack(anchor="w")
            row = ttk.Frame(panel)
            row.pack(fill="x", pady=(3, 10))
            ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Выбрать…", command=command).pack(side="right", padx=(8, 0))
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Целевой допуск, мм:").pack(side="left")
        ttk.Combobox(row, textvariable=self.tolerance, width=10,
                     values=["0.001", "0.002", "0.005", "0.01", "0.02"]).pack(side="left", padx=8)
        ttk.Label(row, text="Вычисления:").pack(side="left", padx=(18, 0))
        ttk.Combobox(row, textvariable=self.device, state="readonly", width=12,
                     values=["Авто", "CPU", "CUDA"]).pack(side="left", padx=8)
        ttk.Checkbutton(panel, text="Требовать одну связную оболочку", variable=self.single).pack(anchor="w")
        ttk.Checkbutton(panel, text="Подтверждаю: координаты файла заданы в миллиметрах",
                        variable=self.units).pack(anchor="w", pady=5)
        ttk.Label(panel, wraplength=790, text=(
            "Первая версия: диагностика и полное объёмное перестроение. Мелкие детали могут измениться. "
            "Допуск проверяется только по вершинам — это не гарантия точности всей поверхности. "
            "Исходный файл не изменяется. Лимит расчёта: 100 млн ячеек.")).pack(anchor="w", pady=8)
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", pady=8)
        self.analyze = ttk.Button(buttons, text="Проверить сетку", command=lambda: self.start(False))
        self.analyze.pack(side="left")
        self.repair = ttk.Button(buttons, text="Построить кандидата", command=lambda: self.start(True))
        self.repair.pack(side="left", padx=8)
        self.cancel_button = ttk.Button(buttons, text="Отмена", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left")
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x", pady=6)
        self.log = tk.Text(panel, height=19, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def pick_source(self):
        path = filedialog.askopenfilename(filetypes=[("Сетка", "*.stl *.obj *.ply")])
        if path:
            self.source.set(path)
            self.target.set(str(Path(path).with_name(Path(path).stem + "_candidate.stl")))

    def pick_target(self):
        path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL", "*.stl")])
        if path:
            self.target.set(path)

    def start(self, repair):
        if self.process:
            return
        try:
            value = float(self.tolerance.get().replace(",", "."))
            Settings(tolerance_mm=value).validate()
            if not self.units.get():
                raise ValueError("Подтвердите единицы измерения — миллиметры.")
            if not Path(self.source.get()).is_file():
                raise ValueError("Выберите исходную сетку.")
            if repair and (not self.target.get() or Path(self.target.get()).exists()):
                raise ValueError("Выберите новое имя выходного файла.")
            if getattr(sys, "frozen", False):
                command = [sys.executable, "--worker"]
            else:
                command = [sys.executable, "-u", "-m", "meshrepair.worker"]
            command += [self.source.get(), "--tolerance", str(value), "--device",
                        {"Авто": "auto", "CPU": "cpu", "CUDA": "cuda"}[self.device.get()]]
            if repair:
                command += ["--output", self.target.get()]
            if not self.single.get():
                command += ["--allow-components"]
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception as exc:
            messagebox.showerror("Ошибка запуска", str(exc))
            return
        self.finished = False
        self.analyze.configure(state="disabled")
        self.repair.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start()
        self.write("\nЗапуск обработки…")
        threading.Thread(target=self.read_worker, args=(self.process,), daemon=True).start()

    def read_worker(self, process):
        for line in process.stdout:
            self.events.put(line.rstrip())
        process.stdout.close()
        self.events.put({"exit": process.wait()})

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if isinstance(event, dict):
                    if not self.finished:
                        self.write(f"Процесс завершён без результата (код {event['exit']}).")
                    self.process = None
                    self.progress.stop()
                    self.analyze.configure(state="normal")
                    self.repair.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    continue
                try:
                    item = json.loads(event)
                except ValueError:
                    self.write(event)
                    continue
                if item["type"] == "result":
                    self.finished = True
                    r = item["value"]
                    for key, label in [("before", "До"), ("after", "После")]:
                        if key in r:
                            d = r[key]
                            self.write(f"{label}: {d['triangles']:,} треугольников; "
                                f"оболочек {d['components']}; дыр {d['boundary_loops']}; "
                                f"пересекающихся граней {d['self_intersecting_faces']}; замкнута: {d['closed']}")
                    if "output" in r:
                        self.write(f"Отклонение по вершинам: {r['vertex_surface_deviation_mm']:.6g} мм")
                        self.write("Оценка в пределах допуска." if r["sampled_tolerance_pass"] else
                                   "ВНИМАНИЕ: оценка превышает заданный допуск!")
                        self.write("Кандидат: " + r["output"] + "\nОтчёт: " + r["output"] + ".json")
                    for warning in r["warnings"]:
                        self.write(warning)
                    self.write(f"Время: {r['elapsed_seconds']:.1f} с")
                else:
                    self.write(str(item["value"]))
                    if item["type"] == "error":
                        self.finished = True
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def cancel(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.finished = True
            self.write("Обработка отменена. Временные .meshrepair-* файлы можно удалить.")

    def close(self):
        self.cancel()
        if self.process:
            self.process.wait(timeout=10)
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
