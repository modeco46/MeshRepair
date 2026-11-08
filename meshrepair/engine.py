"""Native calls stay in a disposable worker process; no Python triangle loops."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import tempfile
import time


@dataclass(frozen=True)
class Settings:
    tolerance_mm: float = 0.01
    device: str = "auto"
    max_grid_cells: int = 100_000_000
    require_single_component: bool = True

    def validate(self):
        if not math.isfinite(self.tolerance_mm) or not 0.001 <= self.tolerance_mm <= 0.02:
            raise ValueError("Допуск должен быть от 0,001 до 0,02 мм.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Неизвестный режим вычислений.")
        if not isinstance(self.max_grid_cells, int) or self.max_grid_cells < 1:
            raise ValueError("Лимит ячеек должен быть положительным целым числом.")


def grid_estimate(size, step):
    dims = [math.ceil(float(v) / step) + 8 for v in size]
    return {"dimensions": dims, "cells": math.prod(dims),
            "one_float_field_bytes": 4 * math.prod(dims)}


def atomic_json(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".meshrepair-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def diagnostics(mesh, mm):
    part = mm.MeshPart(mesh)
    return {
        "vertices": mesh.topology.numValidVerts(),
        "triangles": mesh.topology.numValidFaces(),
        "closed": mesh.topology.isClosed(),
        "topology_valid": mesh.topology.checkValidity(),
        "boundary_loops": mesh.topology.findNumHoles(),
        "components": mm.MeshComponents.getNumComponents(part),
        "self_intersecting_faces": mm.findSelfCollidingTrianglesBS(part).count(),
        "signed_volume_mm3": mesh.volume(),
    }


def run(source, destination=None, settings=Settings(), emit=lambda message: None):
    """destination=None only diagnoses. Coordinates must already be in millimetres.

    Saved meshes are candidates, NEVER certified to the requested tolerance.
    Vertex-to-surface distances are estimates, not a continuous Hausdorff bound.
    """
    settings.validate()
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError("Входной файл не найден.")
    if source.suffix.lower() not in {".stl", ".obj", ".ply"}:
        raise ValueError("Поддерживаются STL, OBJ и PLY.")
    target = Path(destination).resolve() if destination else None
    if target:
        if target == source or target.exists():
            raise ValueError("Выберите новый выходной файл: перезапись запрещена.")
        if target.suffix.lower() != ".stl":
            raise ValueError("Результат первой версии сохраняется в STL.")
        if not target.parent.is_dir() or Path(str(target) + ".json").exists():
            raise ValueError("Выберите существующую папку и новое имя результата/отчёта.")
    from meshlib import mrmeshpy as mm
    started = time.monotonic()
    emit("Загрузка. Все координаты трактуются как миллиметры.")
    load_settings = mm.MeshLoadSettings()
    skipped, duplicated = mm.int_output(), mm.int_output()
    load_settings.skippedFaceCount = skipped
    load_settings.duplicatedVertexCount = duplicated
    mesh = mm.loadMesh(source, load_settings)
    if mesh.topology.numValidFaces() == 0:
        raise ValueError("Файл не содержит треугольников.")
    # Work near origin. Does not recover precision already lost during file loading.
    box = mesh.computeBoundingBox()
    center = box.center()
    mesh.transform(mm.AffineXf3f.translation(-center))
    step = settings.tolerance_mm / 2
    estimate = grid_estimate((box.size().x, box.size().y, box.size().z), step)
    report = {"version": "0.1.0", "settings": asdict(settings), "source": str(source),
              "units": "mm_assumed", "voxel_step_mm": step,
              "grid_estimate": estimate, "tolerance_certified": False,
              "loader": {"skipped_faces": skipped.value, "duplicated_vertices": duplicated.value},
              "warnings": ["Расстояния проверяются по вершинам в обе стороны, а не по всей поверхности.",
                           "Закрытие дыр и удаление внутренних оболочек изменяют геометрию.",
                           "Размер вокселя не гарантирует заданный допуск."]}
    emit("Проверка исходной сетки и самопересечений…")
    report["before"] = diagnostics(mesh, mm)
    if target is None:
        report["status"] = "analysis_only"
        report["elapsed_seconds"] = time.monotonic() - started
        return report
    if skipped.value:
        raise ValueError(f"Загрузчик пропустил {skipped.value} граней. Автоматический ремонт остановлен.")
    if estimate["cells"] > settings.max_grid_cells:
        raise ValueError(f"Оценка сетки: {estimate['cells']:,} ячеек; лимит {settings.max_grid_cells:,}. "
                         "Полное перестроение при таком допуске слишком дорого. "
                         "Увеличьте допуск или используйте меньшую модель. Точность автоматически не снижается.")
    params = mm.GeneralOffsetParameters()
    params.voxelSize = step
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    params.closeHolesInHoleWindingNumber = True
    params.memoryEfficient = True
    report["compute_device"] = "cpu"
    if settings.device != "cpu":
        try:
            from meshlib import mrcudapy as cuda
            if not cuda.isCudaAvailable():
                raise RuntimeError("CUDA недоступна: GPU, драйвер или модуль не поддерживаются.")
            # CUDA disables memoryEfficient; this is a conservative preflight, not a hard cap.
            available = cuda.getCudaAvailableMemory()
            if estimate["one_float_field_bytes"] * 8 > available * 0.65:
                raise RuntimeError("Недостаточно свободной VRAM по предварительной оценке.")
            params.fwn = cuda.FastWindingNumber(mesh)
            report["compute_device"] = "cuda_winding_number"
        except Exception as exc:
            if settings.device == "cuda":
                raise RuntimeError(f"Запрошена CUDA, но она недоступна: {exc}") from exc
            report["warnings"].append(f"Автоматический переход на CPU: {exc}")
    # A closed, single, consistently inward shell otherwise has negative winding.
    if report["before"]["closed"] and report["before"]["components"] == 1 and mesh.volume() < 0:
        mesh.topology.flipOrientation()
        report["warnings"].append("Развёрнута замкнутая оболочка с отрицательным объёмом.")
    emit(f"Перестроение: {report['compute_device']}; шаг {step:g} мм…")
    repaired = mm.generalOffsetMesh(mm.MeshPart(mesh), 0.0, params)
    if repaired.topology.numValidFaces() == 0:
        raise RuntimeError("Получена пустая поверхность; результат не сохранён.")
    emit("Проверка результата…")
    report["after"] = diagnostics(repaired, mm)
    after = report["after"]
    if after["components"] == 1 and after["closed"] and after["signed_volume_mm3"] < 0:
        repaired.topology.flipOrientation()
        after["signed_volume_mm3"] = repaired.volume()
    valid = after["closed"] and after["topology_valid"] and after["self_intersecting_faces"] == 0
    if settings.require_single_component:
        valid = valid and after["components"] == 1
    if not valid:
        raise RuntimeError("Результат не прошёл проверку замкнутости/пересечений/связности; не сохранён. "
                           + json.dumps(after, ensure_ascii=False))
    emit("Оценка отклонений по вершинам в обе стороны…")
    original_part, result_part = mm.MeshPart(mesh), mm.MeshPart(repaired)
    distance = math.sqrt(mm.findMaxDistanceSq(original_part, result_part))
    if not math.isfinite(distance):
        raise RuntimeError("Не удалось получить конечную оценку отклонения.")
    report["vertex_surface_deviation_mm"] = distance
    report["sampled_tolerance_pass"] = distance <= settings.tolerance_mm
    report["status"] = "candidate_within_sampled_tolerance" if report["sampled_tolerance_pass"] else "candidate_exceeds_tolerance"
    if not report["sampled_tolerance_pass"]:
        report["warnings"].append("Оценка отклонения превышает допуск. Файл — только кандидат для проверки.")
    repaired.transform(mm.AffineXf3f.translation(center))
    # Write same-directory temporary first; exclusive target creation prevents overwrite races.
    fd, tmp = tempfile.mkstemp(prefix=".meshrepair-", suffix=".stl", dir=target.parent)
    os.close(fd)
    published = False
    try:
        mm.saveMesh(repaired, tmp)
        # Atomic, no-overwrite publication on NTFS and POSIX. Fail safely on
        # filesystems without hard links (e.g. FAT32): use a local NTFS folder.
        os.link(tmp, target)
        published = True
        report["output"] = str(target)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(str(target) + ".json", report)
    except BaseException:
        if published:
            target.unlink(missing_ok=True)
        raise
    finally:
        Path(tmp).unlink(missing_ok=True)
    emit("Кандидат сохранён. Откройте отчёт и проверьте геометрию.")
    return report
