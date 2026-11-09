"""One-click repair. MeshLib owns the native, multithreaded geometry work."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import tempfile
import time
import shutil
import numpy as np
from .compute import capabilities, cuda_module


@dataclass(frozen=True)
class Settings:
    quality_mm: float = 0.02
    collapse_coplanar: bool = False
    coplanar_angle_deg: float = 1.0
    simplify_curved: bool = False
    strict_tolerance: bool = False
    weld_close_vertices: bool = False

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


def atomic_json(path, data, temp_dir=None):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".meshrepair-", suffix=".json", dir=temp_dir or path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean_json(data), f, ensure_ascii=False, indent=2, allow_nan=False)
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
    # Safety pass uses individual CLOSED shells, never a global volume sign.
    from .orientation import orient_closed_components
    report['sparse_closed_orientation_corrections'] = orient_closed_components(mesh, mm)
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



def clean_json(value):
    """Reports remain serializable even when an SDK diagnostic is non-finite."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def publish_stl(source, target):
    # Windows rename is no-replace and works on filesystems without hard links.
    if os.name == 'nt':
        os.rename(source, target)
    else:
        os.link(source, target)


def simplify_attempts(exported, tmp, mm, settings, tracker, report, emit, workdir):
    from . import reduction as rd
    original = report['after']
    summary = {'requested': True, 'accepted': False, 'before': original['triangles'],
               'after': original['triangles'], 'tolerance_mm': settings.quality_mm,
               'strict_tolerance': settings.strict_tolerance, 'attempts': []}
    report['simplification'] = summary
    started = time.monotonic()
    for attempt, factor in enumerate((1., .5, .25), 1):
        item = {'attempt': attempt, 'max_error_mm': settings.quality_mm*.25*factor}
        summary['attempts'].append(item)
        t = time.monotonic()
        candidate_tmp = Path(workdir) / f'reduction-{attempt}.stl'
        try:
            proposal, stages = rd.propose_reduction(exported, mm, settings, tracker, error_scale=factor)
            item['stages'] = stages
            summary['stages'] = stages
            if proposal.topology.numValidFaces() >= original['triangles']:
                item['reason'] = 'no_reducible_edges'
                break
            mm.saveMesh(proposal, candidate_tmp)
            del proposal
            candidate = mm.loadMesh(candidate_tmp)
            candidate_info = diagnostics(candidate, mm, tracker.callback(86, 88, 'Проверка упрощённой сетки'))
            if not valid(candidate_info):
                item['reason'] = 'topology_changed'
            elif candidate_info['components'] != original['components']:
                item['reason'] = 'kept_shell_count_changed'
            else:
                try:
                    verification = rd.verify_surface_bound(exported, candidate, settings.quality_mm,
                        tracker.callback(88, 96, f'Проверка допуска: попытка {attempt}/3'),
                        # In measured mode don't spend unbounded time proving a
                        # result which can be explicitly accepted as sampled.
                        max_checks=None if settings.strict_tolerance else max(100000, 2*(original['triangles']+candidate_info['triangles'])),
                        acceleration=report['acceleration'], emit=emit)
                except RuntimeError as exc:
                    verification = {'verified': False, 'reason': 'projection_failed', 'detail': str(exc)}
                if verification.get('verified'):
                    verification['mode'] = 'proven'
                    verification['max_deviation_mm'] = verification['upper_bound_mm']
                elif not settings.strict_tolerance and verification.get('reason') in {
                        'verification_work_limit', 'distance_not_confirmed', 'projection_failed'}:
                    deviation = rd.measured_distance(exported, candidate, mm)
                    if math.isfinite(deviation) and deviation <= settings.quality_mm:
                        verification = {'verified': False, 'mode': 'measured_not_proven',
                            'max_deviation_mm': deviation, 'strict_reason': verification.get('reason'),
                            'reference': 'repaired_mesh_before_simplification',
                            'method': 'bidirectional_vertices_to_surface'}
                    else:
                        verification = {'verified': False, 'reason': 'sample_exceeds_tolerance',
                                        'max_deviation_mm': deviation}
                item['verification'] = verification
                summary['verification'] = verification
                if verification.get('verified') or verification.get('mode') == 'measured_not_proven':
                    os.replace(candidate_tmp, tmp)
                    report['after'] = candidate_info
                    summary.update(accepted=True, after=candidate_info['triangles'], accepted_attempt=attempt)
                    if verification['mode'] == 'measured_not_proven':
                        report['warnings'].append('Допуск упрощения измерен по вершинам в обе стороны, но не доказан для всей поверхности. Для строгой проверки включите соответствующую галочку.')
                    break
                item['reason'] = verification.get('reason', 'distance_not_confirmed')
        except (RuntimeError, MemoryError) as exc:
            item.update(reason='native_error', detail=str(exc))
        finally:
            item['elapsed_seconds'] = time.monotonic()-t
            candidate_tmp.unlink(missing_ok=True)
        emit(f"Упрощение, попытка {attempt}: {rd.reason_text(item['reason'])}")
    summary['elapsed_seconds'] = time.monotonic()-started
    if not summary['accepted']:
        summary['reason'] = summary['attempts'][-1].get('reason', 'native_error')
        summary['reason_ru'] = rd.reason_text(summary['reason'])
        emit('Сохранена сетка без упрощения. ' + summary['reason_ru'])
        report['warnings'].append(summary['reason_ru'])
    emit(f"Упрощение: {summary['before']:,} → {summary['after']:,} треугольников.")


def run(source, destination=None, settings=Settings(), emit=lambda message: None,
        progress=lambda event: None, work_dir=None):
    settings.validate()
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError('Входной файл не найден.')
    if source.suffix.lower() not in {'.stl', '.obj', '.ply'}:
        raise ValueError('Поддерживаются STL, OBJ и PLY.')
    target = Path(destination).resolve() if destination else output_path(source)
    if target == source or target.exists() or Path(str(target)+'.json').exists():
        raise ValueError('Выходной файл уже существует. Исходники и результаты не перезаписываются.')
    if not target.parent.is_dir() or target.suffix.lower() != '.stl':
        raise ValueError('Нужна существующая папка и новое имя STL.')
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
    from .orientation import prepare_orientation, finite_vertices
    from .inspection import describe_mesh
    started = time.monotonic()
    tracker = Progress(progress)
    report = {'version': '0.6.0', 'settings': asdict(settings), 'source': str(source),
              'units': 'mm', 'tolerance_certified': False, 'warnings': [], 'steps': [],
              'compute_device': 'cpu_native_parallel', 'acceleration': capabilities()}
    load_settings = mm.MeshLoadSettings()
    skipped, duplicated = mm.int_output(), mm.int_output()
    load_settings.skippedFaceCount, load_settings.duplicatedVertexCount = skipped, duplicated
    load_settings.callback = tracker.callback(0, 5, 'Загрузка сетки')
    mesh = mm.loadMesh(source, load_settings)
    finite_vertices(mn.getNumpyVerts(mesh))
    if not mesh.topology.numValidFaces():
        raise ValueError('Файл не содержит пригодных треугольников.')
    mesh.pack()
    report['loader'] = {'skipped_faces': skipped.value, 'duplicated_vertices': duplicated.value,
                        'non_finite_vertices': 0}
    if skipped.value:
        report['warnings'].append(f'Загрузчик пропустил непригодные грани: {skipped.value}. Проверьте форму результата.')
    center = mesh.computeBoundingBox().center()
    mesh.transform(mm.AffineXf3f.translation(-center))
    # Keep the loaded source for final one-way deviation, not just the preconditioned mesh.
    reference = mm.Mesh(mesh)
    mesh, orientation = prepare_orientation(mesh, mm,
        tracker.callback(5, 12, 'Согласование ориентации'),
        settings.quality_mm*.1 if settings.weld_close_vertices else 0)
    report['orientation'] = orientation
    report['steps'].append({'name': 'Согласование ориентации', **orientation})
    emit(f"Ориентация: несогласованных рёбер {orientation['disoriented_edges']}; фрагментов {orientation['components_before']} → {orientation['components_after']}; {orientation['elapsed_seconds']:.2f} с.")
    if not orientation['accepted']:
        report['warnings'].append('Согласование ориентации не принято: ' + orientation.get('reason', 'unknown'))
    before, shells = describe_mesh(reference, mm, tracker.callback(12, 15, 'Диагностика исходника'))
    report['before'], report['source_shells'] = before, shells
    before['disoriented_edges'] = orientation['disoriented_edges']
    before['skipped_faces'], before['non_finite_vertices'] = skipped.value, 0
    info = diagnostics(mesh, mm, tracker.callback(15, 18, 'Поиск оставшихся повреждений'))
    report['prepared'] = info
    emit(f"После согласования: треугольников {info['triangles']:,}; оболочек {info['components']}; дыр {info['boundary_loops']}.")
    repaired = None
    t = time.monotonic()
    if info['components'] <= 32 and info['boundary_loops'] <= 32:
        try:
            work = mm.Mesh(mesh)
            tracker.update(19, 'Ремонт поверхностей')
            for edge in list(work.topology.findHoleRepresentiveEdges()):
                mm.fillHole(work, edge)
            if info['components'] == 1 and not info['self_intersecting_faces']:
                repaired = work
                if repaired.topology.isClosed() and repaired.volume() < 0:
                    repaired.topology.flipOrientation()
                report['algorithm'] = 'surface_repair'
            else:
                repaired = union_small(work, mm, tracker)
                report['algorithm'] = 'solid_union'
            if not valid(diagnostics(repaired, mm)):
                repaired = None
        except RuntimeError as exc:
            report['surface_attempt_error'] = str(exc)
            repaired = None
    # Closed disjoint solids need neither union nor voxel resampling. This also
    # preserves tiny islands in sets of >32 bodies and fixes mixed global winding.
    elif valid(info) and info['boundary_loops'] == 0:
        repaired = mesh
        report['algorithm'] = 'closed_components_preserved'
    if repaired is None:
        repaired = sparse_repair(mesh, mm, settings, tracker, report, emit)
        reason = 'Поверхностного ремонта недостаточно либо слишком много открытых фрагментов/дыр.'
    else:
        reason = 'После согласования ориентации поверхность можно обработать без вокселей.'
    report['algorithm_reason'] = reason
    report['steps'].append({'name': 'Восстановление поверхности', 'elapsed_seconds': time.monotonic()-t,
                            'algorithm': report['algorithm'], 'reason': reason})
    emit(reason)
    if repaired is None or not repaired.topology.numValidFaces():
        raise RuntimeError('Не удалось восстановить непустую поверхность при выбранном качестве. Файл не сохранён.')
    t = time.monotonic()
    after = diagnostics(repaired, mm, tracker.callback(65, 70, 'Проверка восстановленной сетки'))
    if not valid(after):
        raise RuntimeError(f"Результат не прошёл проверку: дыр {after['boundary_loops']}; пересечений {after['self_intersecting_faces']}. Файл не сохранён.")
    report['steps'].append({'name': 'Проверка результата', 'elapsed_seconds': time.monotonic()-t})
    if report['algorithm'] == 'sparse_openvdb_winding':
        cleanup = mm.FixMeshDegeneraciesParams()
        cleanup.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
        cleanup.maxDeviation, cleanup.tinyEdgeLength = settings.quality_mm*.01, settings.quality_mm*.001
        cleanup.cb = tracker.callback(70, 72, 'Удаление вырожденных треугольников')
        mm.fixMeshDegeneracies(repaired, cleanup)
    repaired.transform(mm.AffineXf3f.translation(center))
    reference.transform(mm.AffineXf3f.translation(center))
    mesh = None
    # GUI creates and owns this one directory; after a kill it cleans only its
    # own job, never a glob of temporary files belonging to other instances.
    owned = tempfile.TemporaryDirectory(prefix='.meshrepair-', dir=target.parent) if work_dir is None else None
    folder = Path(owned.name if owned else work_dir)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder/'result.stl'
    try:
        t = time.monotonic()
        save_settings = mm.SaveSettings()
        save_settings.progress = tracker.callback(72, 74, 'Запись результата')
        mm.saveMesh(repaired, tmp, save_settings)
        repaired = None
        exported = mm.loadMesh(tmp)
        report['after'] = diagnostics(exported, mm, tracker.callback(75, 78, 'Контроль сохранённого STL'))
        if not valid(report['after']):
            raise RuntimeError('STL после сохранения не прошёл проверку.')
        report['steps'].append({'name': 'Запись и повторная проверка STL', 'elapsed_seconds': time.monotonic()-t})
        if settings.collapse_coplanar or settings.simplify_curved:
            simplify_attempts(exported, tmp, mm, settings, tracker, report, emit, folder)
        final = mm.loadMesh(tmp)
        t = time.monotonic()
        tracker.update(97, 'Измерение отклонения от исходника')
        deviation = math.sqrt(mm.findMaxDistanceSqOneWay(mm.MeshPart(final), mm.MeshPart(reference)))
        report['source_deviation'] = {'mode': 'measured_not_proven', 'direction': 'source_vertices_to_result_surface',
            'max_deviation_mm': deviation, 'warning_threshold_mm': 2*settings.quality_mm,
            'elapsed_seconds': time.monotonic()-t}
        if not math.isfinite(deviation) or deviation > 2*settings.quality_mm:
            report['warnings'].append(f'Отклонение вершин исходника от результата {deviation:.6g} мм больше двух шагов ({2*settings.quality_mm:g} мм). Проверьте потерянные детали, полости и места ремонта.')
        from .inspection import shell_summary
        summary, report['shells'] = shell_summary(final, mm)
        report['after'].update(summary)
        count = report['after']['components']
        if count > 1:
            report['warnings'].append(f'В файле {count} замкнутых оболочек. Раздельные детали сохранены; перемычки не добавлялись.')
        report.update(status='repaired' if count == 1 else 'repaired_multiple_shells', output=str(target),
                      elapsed_seconds=time.monotonic()-started)
        publish_stl(tmp, target)
        try:
            atomic_json(str(target)+'.json', report, temp_dir=folder)
        except (OSError, ValueError, TypeError) as exc:
            report['warnings'].append('STL сохранён, но не удалось записать JSON-отчёт: '+str(exc))
    finally:
        if owned:
            owned.cleanup()
    tracker.update(100, 'Готово')
    return clean_json(report)
