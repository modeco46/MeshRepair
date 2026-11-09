"""One-click repair. MeshLib owns the native, multithreaded geometry work."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import tempfile
import time
from .compute import capabilities, cuda_module


@dataclass(frozen=True)
class Settings:
    quality_mm: float = 0.02
    collapse_coplanar: bool = False
    coplanar_angle_deg: float = 1.0
    simplify_curved: bool = False

    def validate(self):
        if not math.isfinite(self.quality_mm) or not .001 <= self.quality_mm <= .5:
            raise ValueError("Качество должно быть от 0,001 до 0,5 мм.")
        if not math.isfinite(self.coplanar_angle_deg) or not 0 <= self.coplanar_angle_deg <= 45:
            raise ValueError("Угол копланарности должен быть от 0 до 45 градусов.")


class Progress:
    """Monotonic stage-weighted progress, driven by actual SDK callbacks."""
    def __init__(self, emit):
        self.emit = emit
        self.percent = 0
        self.last_time = 0
        self.stage = ""

    def update(self, percent, stage):
        percent = max(self.percent, min(100, int(percent)))
        now = time.monotonic()
        if stage != self.stage or percent == 100 or (percent > self.percent and now - self.last_time >= .1):
            self.percent, self.stage, self.last_time = percent, stage, now
            self.emit({"percent": percent, "stage": stage})
        return True

    def callback(self, start, end, stage):
        self.update(start, stage)
        return lambda value: self.update(start + (end - start) * max(0, min(1, value)), stage)


def atomic_json(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".meshrepair-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def output_path(source):
    source = Path(source)
    for number in range(10000):
        suffix = "" if number == 0 else f"_{number}"
        path = source.with_name(source.stem + "_repaired" + suffix + ".stl")
        if not path.exists() and not Path(str(path) + ".json").exists():
            return path
    raise ValueError("В папке слишком много результатов с одинаковым именем.")


def diagnostics(mesh, mm, callback=None):
    part = mm.MeshPart(mesh)
    collisions = mm.findSelfCollidingTrianglesBS(part, callback) if callback else mm.findSelfCollidingTrianglesBS(part)
    return {
        "vertices": mesh.topology.numValidVerts(),
        "triangles": mesh.topology.numValidFaces(),
        "closed": mesh.topology.isClosed(),
        "topology_valid": mesh.topology.checkValidity(),
        "boundary_loops": mesh.topology.findNumHoles(),
        "components": mm.MeshComponents.getNumComponents(part),
        "self_intersecting_faces": collisions.count(),
        "signed_volume_mm3": mesh.volume(),
    }


def valid(info):
    return (info["triangles"] > 0 and info["closed"] and info["topology_valid"]
            and info["self_intersecting_faces"] == 0)


def union_small(mesh, mm, tracker):
    """Small component sets only; larger sets go to sparse reconstruction, never fail by count."""
    regions = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    result = None
    for index, region in enumerate(regions):
        part = mesh.cloneRegion(region)
        if not part.topology.isClosed():
            raise RuntimeError("Открытая оболочка требует объёмного восстановления.")
        if part.volume() < 0:
            part.topology.flipOrientation()
        # Self-intersections within one operand are not valid CSG input.
        if mm.findSelfCollidingTrianglesBS(mm.MeshPart(part)).count():
            raise RuntimeError("Самопересечения внутри оболочки требуют восстановления.")
        if result is None:
            result = part
        else:
            params = mm.BooleanParameters()
            params.cb = tracker.callback(20 + 20 * index / len(regions),
                                         20 + 20 * (index + 1) / len(regions), "Объединение поверхностей")
            operation = mm.boolean(result, part, mm.BooleanOperation.Union, params)
            if not operation.valid():
                raise RuntimeError(operation.errorString)
            result = operation.mesh
    return result


def sparse_repair(mesh, mm, settings, tracker, report, emit):
    """OpenVDB narrow band, NOT a dense bounding-box lattice.

    No fixed component/grid-count limit; selected resolution is never silently reduced.
    Zero offset uses signed winding. CUDA is optional and may fall back to native CPU.
    """
    report["algorithm"] = "sparse_openvdb_winding"
    report["voxel_size_mm"] = settings.quality_mm
    params = mm.OffsetParameters()
    params.voxelSize = settings.quality_mm
    params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
    params.callBack = tracker.callback(42, 65, "Восстановление поверхности")
    report["compute_device"] = "cpu_native_parallel"
    # Winding sign depends on orientation. Closed operands were oriented earlier;
    # for fragmented input this is a global heuristic, not a guarantee of inside/outside.
    if mesh.volume() < 0:
        mesh = mm.Mesh(mesh)
        mesh.topology.flipOrientation()
        report["warnings"].append("Общая ориентация исходной поверхности развёрнута по знаку объёма.")
    try:
        cuda = cuda_module()
        if cuda is not None:
            params.fwn = cuda.FastWindingNumber(mesh)
            report["compute_device"] = "cuda_winding_with_cpu_openvdb"
    except Exception as exc:
        report["warnings"].append(f"CUDA недоступна, используется CPU: {exc}")
    emit(f"Разреженное восстановление; шаг {settings.quality_mm:g} мм. "
         "Мелкий шаг может потребовать много времени и памяти; остановка доступна.")
    try:
        result = mm.offsetMesh(mm.MeshPart(mesh), 0, params)
    except RuntimeError as exc:
        if params.fwn is None:
            raise
        report["warnings"].append(f"Ошибка CUDA; повтор на CPU: {exc}")
        emit("GPU не завершил расчёт. Повтор на CPU с тем же качеством.")
        params.fwn = None
        report["compute_device"] = "cpu_native_parallel"
        result = mm.offsetMesh(mm.MeshPart(mesh), 0, params)
    return result


def run(source, destination=None, settings=Settings(), emit=lambda message: None,
        progress=lambda event: None):
    settings.validate()
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError("Входной файл не найден.")
    if source.suffix.lower() not in {".stl", ".obj", ".ply"}:
        raise ValueError("Поддерживаются STL, OBJ и PLY.")
    target = Path(destination).resolve() if destination else output_path(source)
    if target == source or target.exists() or Path(str(target) + ".json").exists():
        raise ValueError("Выходной файл уже существует. Исходники и результаты не перезаписываются.")
    if not target.parent.is_dir() or target.suffix.lower() != ".stl":
        raise ValueError("Нужна существующая папка и новое имя STL.")
    from meshlib import mrmeshpy as mm
    started = time.monotonic()
    tracker = Progress(progress)
    report = {"version": "0.5.0", "settings": asdict(settings), "source": str(source),
              "units": "mm", "tolerance_certified": False, "warnings": [],
              "compute_device": "cpu_native_parallel", "acceleration": capabilities()}
    load_settings = mm.MeshLoadSettings()
    skipped, duplicated = mm.int_output(), mm.int_output()
    load_settings.skippedFaceCount = skipped
    load_settings.duplicatedVertexCount = duplicated
    load_settings.callback = tracker.callback(0, 10, "Загрузка сетки")
    mesh = mm.loadMesh(source, load_settings)
    if mesh.topology.numValidFaces() == 0:
        raise ValueError("Файл не содержит пригодных треугольников.")
    report["loader"] = {"skipped_faces": skipped.value, "duplicated_vertices": duplicated.value}
    if skipped.value:
        report["warnings"].append(f"Загрузчик пропустил непригодные грани: {skipped.value}. Проверьте форму результата.")
    center = mesh.computeBoundingBox().center()
    mesh.transform(mm.AffineXf3f.translation(-center))
    before = diagnostics(mesh, mm, tracker.callback(10, 18, "Поиск повреждений"))
    report["before"] = before
    emit(f"Треугольников: {before['triangles']:,}; оболочек: {before['components']}; "
         f"дыр: {before['boundary_loops']}; пересекающихся граней: {before['self_intersecting_faces']}.")
    repaired = None
    # A small topology problem can often be solved without resampling any surface.
    if before["components"] <= 32 and before["boundary_loops"] <= 32:
        try:
            work = mm.Mesh(mesh)
            tracker.update(19, "Ремонт поверхностей")
            for edge in list(work.topology.findHoleRepresentiveEdges()):
                mm.fillHole(work, edge)
            if before["components"] == 1 and not before["self_intersecting_faces"]:
                repaired = work
                if repaired.volume() < 0:
                    repaired.topology.flipOrientation()
                report["algorithm"] = "surface_repair"
            else:
                repaired = union_small(work, mm, tracker)
                report["algorithm"] = "solid_union"
                report["warnings"].append("Оболочки объединены как сплошные тела; вложенные поверхности/полости могут быть удалены.")
            candidate_info = diagnostics(repaired, mm, tracker.callback(40, 42, "Проверка поверхностного ремонта"))
            if not valid(candidate_info):
                repaired = None
        except RuntimeError as exc:
            emit("Поверхностного ремонта недостаточно. Перехожу к восстановлению объёма.")
            report["surface_attempt_error"] = str(exc)
            repaired = None
    if repaired is None:
        repaired = sparse_repair(mesh, mm, settings, tracker, report, emit)
    if repaired is None or repaired.topology.numValidFaces() == 0:
        raise RuntimeError("Не удалось восстановить непустую поверхность при выбранном качестве. Файл не сохранён.")
    after = diagnostics(repaired, mm, tracker.callback(65, 70, "Проверка восстановленной сетки"))
    if not valid(after):
        raise RuntimeError("Восстановленная поверхность не прошла проверку. "
                           f"Дыр: {after['boundary_loops']}; пересечений: {after['self_intersecting_faces']}. "
                           "Повреждённый результат не сохранён.")
    if after["components"] == 1 and repaired.volume() < 0:
        repaired.topology.flipOrientation()
    repaired.transform(mm.AffineXf3f.translation(center))
    if report["algorithm"] == "sparse_openvdb_winding":
        cleanup = mm.FixMeshDegeneraciesParams()
        cleanup.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
        cleanup.maxDeviation = settings.quality_mm * .01
        cleanup.tinyEdgeLength = settings.quality_mm * .001
        cleanup.cb = tracker.callback(70, 72, "Удаление вырожденных треугольников")
        mm.fixMeshDegeneracies(repaired, cleanup)
    # Original working copies are no longer needed during STL round-trip validation.
    mesh = None
    work = None
    fd, tmp = tempfile.mkstemp(prefix=".meshrepair-", suffix=".stl", dir=target.parent)
    os.close(fd)
    published = False
    try:
        save_settings = mm.SaveSettings()
        save_settings.progress = tracker.callback(72, 74, "Запись результата")
        mm.saveMesh(repaired, tmp, save_settings)
        # Validate the actual exported STL, not just the in-memory mesh.
        del repaired
        check_settings = mm.MeshLoadSettings()
        check_settings.callback = tracker.callback(74, 75, "Проверка сохранённого файла")
        exported = mm.loadMesh(tmp, check_settings)
        report["after"] = diagnostics(exported, mm, tracker.callback(75, 78, "Контроль результата"))
        if not valid(report["after"]):
            raise RuntimeError("STL после сохранения не прошёл проверку: " + str(report["after"]))
        if settings.collapse_coplanar or settings.simplify_curved:
            from .reduction import propose_reduction, verify_surface_bound
            original_info = report["after"]
            reduction = {"requested": True, "accepted": False,
                         "before": original_info["triangles"], "after": original_info["triangles"],
                         "tolerance_mm": settings.quality_mm}
            report["simplification"] = reduction
            candidate_tmp = None
            try:
                proposal, stages = propose_reduction(exported, mm, settings, tracker)
                reduction["stages"] = stages
                if proposal.topology.numValidFaces() < original_info["triangles"]:
                    fd2, candidate_tmp = tempfile.mkstemp(prefix=".meshrepair-simplified-", suffix=".stl", dir=target.parent)
                    os.close(fd2)
                    proposed_save = mm.SaveSettings()
                    proposed_save.progress = tracker.callback(85, 86, "Запись упрощённой сетки")
                    mm.saveMesh(proposal, candidate_tmp, proposed_save)
                    del proposal
                    candidate = mm.loadMesh(candidate_tmp)
                    candidate_info = diagnostics(candidate, mm, tracker.callback(86, 88, "Проверка топологии после упрощения"))
                    if valid(candidate_info) and candidate_info["components"] == original_info["components"]:
                        reduction["verification"] = verify_surface_bound(exported, candidate, settings.quality_mm,
                            tracker.callback(88, 99, "Проверка допуска упрощения в обе стороны"),
                            acceleration=report['acceleration'], emit=emit)
                        if reduction["verification"]["verified"]:
                            os.replace(candidate_tmp, tmp)
                            candidate_tmp = None
                            report["after"] = candidate_info
                            reduction.update(accepted=True, after=candidate_info["triangles"])
                    else:
                        reduction["verification"] = {"verified": False, "reason": "topology_changed"}
                    if not reduction["accepted"]:
                        emit("Допуск или топология упрощения не подтверждены. Сохраняю исправленную сетку без упрощения.")
                        report["warnings"].append("Упрощение отклонено; сохранён исправленный вариант до упрощения.")
                else:
                    reduction["reason"] = "no_reducible_edges"
                    emit("Избыточных треугольников при выбранных ограничениях не найдено.")
            except (RuntimeError, MemoryError) as exc:
                reduction["reason"] = str(exc)
                report["warnings"].append("Упрощение не завершилось; сохранён исправленный вариант без упрощения.")
            finally:
                if candidate_tmp:
                    Path(candidate_tmp).unlink(missing_ok=True)
            emit(f"Упрощение: {reduction['before']:,} → {reduction['after']:,} треугольников.")
        count = report["after"]["components"]
        if count > 1:
            report["warnings"].append(f"В файле {count} замкнутых оболочек. Раздельные детали сохранены; перемычки не добавлялись.")
        report["status"] = "repaired" if count == 1 else "repaired_multiple_shells"
        report["output"] = str(target)
        report["elapsed_seconds"] = time.monotonic() - started
        # Atomic publication without overwriting another file; same-volume NTFS/POSIX.
        os.link(tmp, target)
        published = True
        atomic_json(str(target) + ".json", report)
    except BaseException:
        if published:
            target.unlink(missing_ok=True)
        raise
    finally:
        Path(tmp).unlink(missing_ok=True)
    tracker.update(100, "Готово")
    return report
