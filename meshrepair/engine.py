"""One-click repair. MeshLib owns the native, multithreaded geometry work."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import tempfile
import time
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
    restore_sharp_edges: bool = True
    local_fix_max_faces: int = 200
    local_fix_fraction: float = .001
    remove_small_shells: bool = False
    disable_cuda: bool = False
    temp_directory: str = ''
    sharpening_fallback_max_seconds: float = 120.0

    def validate(self):
        if not math.isfinite(self.quality_mm) or not .001 <= self.quality_mm <= .5:
            raise ValueError("Качество должно быть от 0,001 до 0,5 мм.")
        if not math.isfinite(self.coplanar_angle_deg) or not 0 <= self.coplanar_angle_deg <= 45:
            raise ValueError("Угол копланарности должен быть от 0 до 45 градусов.")
        if self.local_fix_max_faces < 0 or not math.isfinite(self.local_fix_fraction) or not 0 <= self.local_fix_fraction <= 1:
            raise ValueError('Некорректный порог локального ремонта.')
        if not math.isfinite(self.sharpening_fallback_max_seconds) or self.sharpening_fallback_max_seconds < 0:
            raise ValueError('Некорректный предел времени повтора без Sharpening.')


class SharpeningRetryRequired(RuntimeError):
    """A costly retry needs one user click; retain the diagnostic report."""
    def __init__(self, report):
        self.report = report
        seconds = report['sharpening']['fallback']['estimated_seconds']
        super().__init__(f'Полный результат Sharpening невалиден. Оценка повтора: {seconds:.0f} с. '
                         'Нажмите «Повторить без Sharpening».')


class Progress:
    """Monotonic stage-weighted progress, driven by actual SDK callbacks."""
    def __init__(self, emit, log=None, timings=None, device=None):
        self.emit = emit
        self.percent = 0
        self.last_time = 0
        self.stage = ""
        self.stage_started = time.monotonic()
        self.log, self.timings, self.device = log, timings, device

    def update(self, percent, stage):
        percent = max(self.percent, min(100, int(percent)))
        now = time.monotonic()
        if stage != self.stage:
            if self.stage and self.timings is not None:
                item = {'name': self.stage, 'elapsed_seconds': now-self.stage_started,
                        'compute_device': self.device(self.stage) if self.device else 'cpu_native_parallel'}
                self.timings.append(item)
                if self.log:
                    self.log(f"{item['name']}: {item['elapsed_seconds']:.2f} с; compute_device={item['compute_device']}.")
            self.stage_started = now
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


def topology_info(mesh, mm):
    """Cheap routing counters; deliberately no self-collision search."""
    return {
        "vertices": mesh.topology.numValidVerts(),
        "triangles": mesh.topology.numValidFaces(),
        "closed": mesh.topology.isClosed(),
        "topology_valid": mesh.topology.checkValidity(),
        "boundary_loops": mesh.topology.findNumHoles(),
        "components": mm.MeshComponents.getNumComponents(mm.MeshPart(mesh)),
        "signed_volume_mm3": mesh.volume(),
    }


def diagnostics(mesh, mm, callback=None):
    part = mm.MeshPart(mesh)
    collisions = mm.findSelfCollidingTrianglesBS(part, callback) if callback else mm.findSelfCollidingTrianglesBS(part)
    return dict(topology_info(mesh, mm), self_intersecting_faces=collisions.count())


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
        # The caller's post-hole diagnostic already excluded intersections in
        # every operand. Rechecking each component repeats that complete search.
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


def sparse_repair(mesh, mm, settings, tracker, report, emit, translation=None):
    """Validate the full result and reuse winding data for a bounded retry."""
    from .orientation import orient_closed_components
    report.update(algorithm='sparse_openvdb_winding', voxel_size_mm=settings.quality_mm)
    report['sparse_closed_orientation_corrections'] = orient_closed_components(mesh, mm)
    sharp = report['sharpening'] = {'requested': settings.restore_sharp_edges, 'accepted': False}
    report['compute_device'] = 'cpu_native_parallel'
    estimate = max(1, int(mesh.area()/settings.quality_mm**2))
    estimated_seconds = estimate * (180.0 / 2000000)
    report['voxel_estimate'] = {'triangles': estimate, 'method': 'area_over_step_squared',
                              'estimated_seconds': estimated_seconds,
                              'seconds_per_million': 90.0}
    emit(f'Оценка воксельной поверхности: ≈ {estimate:,} треугольников (площадь / шаг²); '
         'время зависит от CPU, CUDA и формы. На одном CPU для 2 млн граней — порядка 3 минут.')
    winding = None
    try:
        cuda = None if settings.disable_cuda else cuda_module()
        if cuda is not None:
            winding = cuda.FastWindingNumber(mesh)
    except Exception as exc:
        report['warnings'].append(f'CUDA недоступна, используется CPU: {exc}')
    report['voxel_passes'] = []

    def reconstruct(sharpen, step, probe=False):
        nonlocal winding
        params = mm.GeneralOffsetParameters() if sharpen else mm.OffsetParameters()
        params.voxelSize = step
        params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
        stage = 'Проба острых рёбер (шаг ×4)' if probe else 'Восстановление поверхности'
        params.callBack = tracker.callback(32 if probe else 42, 41 if probe else 65, stage)
        params.fwn = winding
        if sharpen:
            params.mode = mm.GeneralOffsetParametersMode.Sharpening
        operation = mm.generalOffsetMesh if sharpen else mm.offsetMesh
        device = 'cuda_winding_with_cpu_openvdb' if winding is not None else 'cpu_native_parallel'
        report['compute_device'] = device
        item = {'probe': probe, 'sharpening': sharpen, 'voxel_size_mm': step, 'compute_device': device}
        report['voxel_passes'].append(item)
        started = time.monotonic()
        emit(f'{stage}: шаг {step:g} мм; compute_device={device}.')
        try:
            try:
                return operation(mm.MeshPart(mesh), 0, params)
            except RuntimeError as exc:
                if params.fwn is None:
                    raise
                report['warnings'].append(f'Ошибка CUDA; повтор на CPU с тем же шагом: {exc}')
                winding = params.fwn = None
                report['compute_device'] = item['fallback_device'] = 'cpu_native_parallel'
                emit('Повтор после ошибки CUDA: compute_device=cpu_native_parallel.')
                return operation(mm.MeshPart(mesh), 0, params)
        finally:
            item['elapsed_seconds'] = time.monotonic()-started
            emit(f"{stage}: {item['elapsed_seconds']:.2f} с; compute_device={report['compute_device']}.")

    use_sharp = False
    if settings.restore_sharp_edges:
        try:
            probe = reconstruct(True, settings.quality_mm*4, probe=True)
            check = diagnostics(probe, mm)
            sharp['probe_initial'] = check
            # A few coarse-grid seam intersections need not disable a useful
            # sharp reconstruction. Repair only the small, closed probe and
            # validate it; noisy probes with many intersections are rejected.
            threshold = max(settings.local_fix_max_faces, int(check['triangles']*settings.local_fix_fraction))
            if check['closed'] and check['topology_valid'] and 0 < check['self_intersecting_faces'] <= threshold:
                params = mm.FixSelfIntersectionSettings()
                params.method = mm.FixSelfIntersectionMethod.CutAndFill
                mm.SelfIntersections.fix(probe, params)
                probe.pack()
                check = diagnostics(probe, mm)
                sharp['probe_local_fix'] = True
            sharp['probe'] = check
            if not valid(check):
                raise RuntimeError(f"дыр {check['boundary_loops']}, пересекающихся граней {check['self_intersecting_faces']}")
            use_sharp = True
            emit('Грубая проба Sharpening прошла проверку.')
        except (RuntimeError, AttributeError, TypeError) as exc:
            sharp['reason'] = str(exc)
            warning = f'Sharpening отклонён на грубой пробе: {exc}. Полный расчёт — обычные воксели.'
            report['warnings'].append(warning)
            emit(warning)
    def full_result(sharpen):
        result = reconstruct(sharpen, settings.quality_mm)
        if translation is not None:
            result.transform(mm.AffineXf3f.translation(translation))
        # Validate in output coordinates, once per full candidate. Never try
        # CutAndFill on a failed full-resolution Sharpening reconstruction.
        check = diagnostics(result, mm)
        return result, check

    result, check = full_result(use_sharp)
    if use_sharp:
        sharp['full_validation'] = check
    if use_sharp and not valid(check):
        reason = (f"Полный результат Sharpening невалиден: дыр {check['boundary_loops']}, "
                  f"пересекающихся граней {check['self_intersecting_faces']}, "
                  f"топология {'корректна' if check['topology_valid'] else 'повреждена'}.")
        fallback = sharp['fallback'] = {'reason': 'full_sharpening_invalid', 'reason_ru': reason,
            'estimated_seconds': estimated_seconds, 'limit_seconds': settings.sharpening_fallback_max_seconds,
            'automatic': estimated_seconds <= settings.sharpening_fallback_max_seconds,
            'attempted': False, 'elapsed_seconds': 0.0}
        sharp['reason'] = reason
        report['warnings'].append(reason)
        emit(reason)
        if not fallback['automatic']:
            report['retry_without_sharpening'] = True
            raise SharpeningRetryRequired(report)
        emit(f'Автоматический повтор без Sharpening; оценка {estimated_seconds:.1f} с.')
        started = time.monotonic()
        fallback['attempted'] = True
        # Drop the rejected dense mesh before allocating its replacement.
        del result
        try:
            result, check = full_result(False)
            fallback.update(accepted=valid(check), after=check)
        finally:
            fallback['elapsed_seconds'] = time.monotonic()-started
            emit(f"Повтор без Sharpening: {fallback['elapsed_seconds']:.2f} с.")
        use_sharp = False
    sharp['accepted'] = use_sharp
    report['reconstruction_validation'] = check
    return result


def local_repair(mesh, info, mm, settings, tracker, report, emit):
    threshold = max(settings.local_fix_max_faces, int(info['triangles']*settings.local_fix_fraction))
    item = report['local_repair'] = {'threshold_faces': threshold, 'attempted': False, 'accepted': False}
    if not 0 < info['self_intersecting_faces'] <= threshold:
        return None, None
    tracker.update(31, 'Локальное устранение самопересечений')
    started = time.monotonic()
    item.update(attempted=True, before=info)
    try:
        candidate = mm.Mesh(mesh)
        params = mm.FixSelfIntersectionSettings()
        params.method = mm.FixSelfIntersectionMethod.CutAndFill
        mm.SelfIntersections.fix(candidate, params)
        candidate.pack()
        after = diagnostics(candidate, mm)
        item['after'] = after
        if valid(after):
            item['accepted'] = True
            from .inspection import shell_summary
            _, item['shells'] = shell_summary(candidate, mm)
            return candidate, after
        item['reason'] = 'local_fix_invalid'
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='local_fix_failed', detail=str(exc))
    finally:
        item['elapsed_seconds'] = time.monotonic()-started
        report['steps'].append({'name': 'Локальное устранение самопересечений', **item})
        emit(f"CutAndFill: {'принят' if item['accepted'] else 'отклонён'}; "
             f"{info['self_intersecting_faces']} → {item.get('after', {}).get('self_intersecting_faces', '—')}; "
             f"{item['elapsed_seconds']:.2f} с; compute_device=cpu_native_parallel.")
    return None, None



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


def cleanup_exported(exported, path, mm, settings, tracker, report, folder, emit):
    """Compatibility name; cleanup and validation now stay entirely in memory."""
    started = time.monotonic()
    item = {'accepted': False, 'before': report['after']['triangles']}
    try:
        candidate = mm.Mesh(exported)
        params = mm.FixMeshDegeneraciesParams()
        params.mode = mm.FixMeshDegeneraciesParams.Mode.Decimate
        params.maxDeviation = settings.quality_mm*.01
        params.tinyEdgeLength = settings.quality_mm*.001
        params.cb = tracker.callback(78, 79, 'Очистка вырожденных граней')
        mm.fixMeshDegeneracies(candidate, params)
        candidate.invalidateCaches()
        candidate.pack()
        info = diagnostics(candidate, mm)
        if not valid(info):
            item['reason'] = 'cleanup_topology_changed'
        elif info['components'] != report['after']['components']:
            item['reason'] = 'cleanup_shell_count_changed'
        else:
            exported = candidate
            report['after'] = info
            item.update(accepted=True, after=info['triangles'])
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='cleanup_native_error', detail=str(exc))
    item['elapsed_seconds'] = time.monotonic()-started
    report['degenerate_cleanup'] = item
    report['steps'].append({'name': 'Очистка вырожденных граней', **item})
    emit(f"Очистка: {item['elapsed_seconds']:.2f} с; compute_device=cpu_native_parallel.")
    if not item['accepted']:
        message = 'Очистка вырожденных граней отклонена. Сохранена сетка до очистки. ['+item['reason']+']'
        emit(message)
        report['warnings'].append(message)
    return exported


def simplify_attempts(exported, tmp, mm, settings, tracker, report, emit, workdir):
    """Validate small candidates in memory; return the accepted mesh or original."""
    from . import reduction as rd
    original = report['after']
    mode = 'proven' if settings.strict_tolerance else 'measured_not_proven'
    summary = {'requested': True, 'accepted': False, 'before': original['triangles'],
               'after': original['triangles'], 'tolerance_mm': settings.quality_mm,
               'strict_tolerance': settings.strict_tolerance, 'mode': mode,
               'input_triangles': report.get('before', {}).get('triangles'), 'attempts': []}
    report['simplification'] = summary
    emit('Режим допуска: '+('доказано (строгая проверка).' if settings.strict_tolerance else 'измерено (вершины в обе стороны).'))
    started = time.monotonic()
    factor = .25
    accepted = exported
    for attempt in range(1, 4):
        item = {'attempt': attempt, 'error_scale': factor, 'max_error_mm': settings.quality_mm*factor}
        summary['attempts'].append(item)
        t = time.monotonic()
        try:
            candidate, stages = rd.propose_reduction(exported, mm, settings, tracker, error_scale=factor)
            item['stages'] = summary['stages'] = stages
            if candidate.topology.numValidFaces() >= original['triangles']:
                item['reason'] = 'no_reducible_edges'
                break
            candidate_info = diagnostics(candidate, mm, tracker.callback(86, 88, 'Проверка упрощённой сетки'))
            if not valid(candidate_info) and mm.findDegenerateFaces(mm.MeshPart(candidate)).count():
                cleanup_report = {'after': candidate_info, 'steps': [], 'warnings': []}
                candidate = cleanup_exported(candidate, None, mm, settings, tracker, cleanup_report, None, emit)
                candidate_info = cleanup_report['after']
                item['cleanup'] = cleanup_report['degenerate_cleanup']
            item['candidate'] = candidate_info
            if not valid(candidate_info):
                item['reason'] = 'topology_changed'
            elif candidate_info['components'] != original['components']:
                item['reason'] = 'kept_shell_count_changed'
            else:
                if settings.strict_tolerance:
                    verification = rd.verify_surface_bound(exported, candidate, settings.quality_mm,
                        tracker.callback(88, 96, f'Строгая проверка допуска: {attempt}/3'),
                        acceleration=report['acceleration'], emit=emit)
                    verification['mode'] = mode
                    if verification.get('verified'):
                        verification['max_deviation_mm'] = verification['upper_bound_mm']
                    passed = verification.get('verified', False)
                else:
                    tracker.update(88, f'Измерение допуска: {attempt}/3')
                    deviation = rd.measured_distance(exported, candidate, mm)
                    passed = math.isfinite(deviation) and deviation <= settings.quality_mm
                    verification = {'verified': False, 'mode': mode, 'max_deviation_mm': deviation,
                                    'reference': 'repaired_mesh_before_simplification',
                                    'method': 'bidirectional_vertices_to_surface'}
                    if not passed:
                        verification['reason'] = 'sample_exceeds_tolerance'
                item['verification'] = summary['verification'] = verification
                if passed:
                    accepted = candidate
                    report['after'] = candidate_info
                    summary.update(accepted=True, after=candidate_info['triangles'], accepted_attempt=attempt)
                    break
                item['reason'] = verification.get('reason', 'distance_not_confirmed')
        except (RuntimeError, MemoryError) as exc:
            item.update(reason='native_error', detail=str(exc))
        finally:
            item['elapsed_seconds'] = time.monotonic()-t
            emit(f"Упрощение, попытка {attempt}: {item['elapsed_seconds']:.2f} с; compute_device=cpu_native_parallel.")
        emit(f"Упрощение, попытка {attempt}: {rd.reason_text(item['reason'])}")
        measured = item.get('verification', {}).get('max_deviation_mm')
        ratio = .5
        if measured and math.isfinite(measured) and measured > settings.quality_mm:
            ratio = max(.1, min(.8, .8*settings.quality_mm/measured))
        factor *= ratio
    summary['elapsed_seconds'] = time.monotonic()-started
    if not summary['accepted']:
        summary['reason'] = summary['attempts'][-1].get('reason', 'native_error')
        summary['reason_ru'] = rd.reason_text(summary['reason'])
        emit('Сохранена сетка без упрощения. ' + summary['reason_ru'])
        report['warnings'].append(summary['reason_ru'])
    emit(f"Упрощение: {summary['before']:,} → {summary['after']:,} треугольников.")
    report['steps'].append({'name': 'Упрощение', 'elapsed_seconds': summary['elapsed_seconds']})
    return accepted



def run(source, destination=None, settings=Settings(), emit=lambda message: None,
        progress=lambda event: None, work_dir=None):
    from .storage import create_job, remove_job
    settings.validate()
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError('Входной файл не найден.')
    if source.suffix.lower() not in {'.stl', '.obj', '.ply'}:
        raise ValueError('Поддерживаются STL, OBJ и PLY.')
    target = Path(destination).resolve() if destination else None
    if target:
        if target == source or any(Path(str(target)+suffix).exists() for suffix in ('', '.json', '.deviation.npz')):
            raise ValueError('Выходной файл уже существует. Исходники и результаты не перезаписываются.')
        if not target.parent.is_dir() or target.suffix.lower() != '.stl':
            raise ValueError('Нужна существующая папка и новое имя STL.')
    owned = work_dir is None
    folder = create_job(settings.temp_directory) if owned else Path(work_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        report = _run(source, settings, emit, progress, folder)
        if target:
            _publish_cli(report, target)
        return clean_json(report)
    except BaseException:
        if owned:
            remove_job(folder)
        raise
    finally:
        if owned and target:
            remove_job(folder)


def _publish_cli(report, target):
    """Explicit --output retains STL + JSON + deviation and no-overwrite semantics."""
    from .storage import save_artifacts
    # Sidecar errors must not hide a successfully published STL (legacy contract).
    saved, _ = save_artifacts(report, target, {'stl': True}, no_replace=True)
    report.update(output=str(target), saved_at=saved['saved_at'])
    published = {'stl': str(target)}
    try:
        if report['source_deviation'].get('heatmap_error'):
            raise OSError(report['source_deviation']['heatmap_error'])
        save_artifacts(report, target, {'deviation': True}, no_replace=True)
        report['source_deviation']['heatmap_file'] = str(target)+'.deviation.npz'
        published['deviation'] = str(target)+'.deviation.npz'
    except OSError as exc:
        report['source_deviation']['heatmap_file'] = None
        report['warnings'].append('STL сохранён, но карту отклонения записать не удалось: '+str(exc))
    report['artifacts'] = published
    report.pop('work_directory', None)
    try:
        atomic_json(str(target)+'.json', report)
    except (OSError, ValueError, TypeError) as exc:
        report['warnings'].append('STL сохранён, но не удалось записать JSON-отчёт: '+str(exc))


def _run(source, settings, emit, progress, folder):
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
    from .orientation import prepare_orientation, finite_vertices
    from .inspection import describe_mesh
    started = time.monotonic()
    report = {'version': '0.9.1', 'settings': asdict(settings), 'source': str(source),
              'units': 'mm', 'tolerance_certified': False, 'warnings': [], 'steps': [],
              'compute_device': 'cpu_native_parallel', 'acceleration': capabilities(settings.disable_cuda)}
    report['stage_timings'] = []
    tracker = Progress(progress, emit, report['stage_timings'],
        lambda stage: report['compute_device'] if stage in {'Проба острых рёбер (шаг ×4)', 'Восстановление поверхности'}
        else ','.join(report['acceleration'].get('projection_backends', ['cpu_native_parallel']))
        if stage == 'Карта отклонения от исходника' or stage.startswith('Строгая проверка') else 'cpu_native_parallel')
    emit('compute_device=cpu_native_parallel; CUDA '+('выключена настройкой.' if settings.disable_cuda else 'авто.'))
    load_settings = mm.MeshLoadSettings()
    skipped, duplicated = mm.int_output(), mm.int_output()
    load_settings.skippedFaceCount, load_settings.duplicatedVertexCount = skipped, duplicated
    load_settings.callback = tracker.callback(0, 5, 'Загрузка сетки')
    mesh = mm.loadMesh(source, load_settings)
    finite_vertices(mn.getNumpyVerts(mesh))
    if not mesh.topology.numValidFaces():
        raise ValueError('Файл не содержит пригодных треугольников.')
    mesh.pack()
    from .storage import check_space
    report['temporary_storage'] = check_space(folder, mesh.topology.numValidFaces())
    report['loader'] = {'skipped_faces': skipped.value, 'duplicated_vertices': duplicated.value,
                        'non_finite_vertices': 0}
    if skipped.value:
        report['warnings'].append(f'Загрузчик пропустил непригодные грани: {skipped.value}. Проверьте форму результата.')
    # Keep the loaded source for final one-way deviation, not just the preconditioned mesh.
    reference = mm.Mesh(mesh)
    mesh, orientation = prepare_orientation(mesh, mm,
        tracker.callback(5, 12, 'Согласование ориентации'),
        settings.quality_mm*.1 if settings.weld_close_vertices else 0)
    report['orientation'] = orientation
    report['steps'].append({'name': 'Согласование ориентации', **orientation})
    emit(f"Ориентация: несогласованных рёбер {orientation['disoriented_edges']}; фрагментов {orientation['components_before']} → {orientation['components_after']}; {orientation['elapsed_seconds']:.2f} с.")
    if orientation.get('conflicting_faces'):
        report['warnings'].append(f"Неориентируемых компонентов: {orientation['unoriented_components']}, граней: {orientation['conflicting_faces']}. Их ориентация пропущена; остальные компоненты обработаны.")
    if not orientation['accepted']:
        report['warnings'].append('Согласование ориентации не принято: ' + orientation.get('reason', 'unknown'))
    before, shells = describe_mesh(reference, mm, tracker.callback(12, 15, 'Диагностика исходника'))
    report['before'], report['source_shells'] = before, shells
    before['disoriented_edges'] = orientation['disoriented_edges']
    before['skipped_faces'], before['non_finite_vertices'] = skipped.value, 0
    info = topology_info(mesh, mm)
    report['prepared'] = info
    emit(f"После согласования: треугольников {info['triangles']:,}; оболочек {info['components']}; дыр {info['boundary_loops']}.")
    from .holes import fill_simple_holes
    before_holes = mesh
    mesh, hole_report = fill_simple_holes(mesh, mm, settings.quality_mm,
        tracker.callback(18, 30, 'Заполнение простых дыр'), emit, defer_intersections=True)
    report['hole_filling'] = hole_report
    tracker.update(30, 'Проверка после заполнения дыр')
    info = diagnostics(mesh, mm)
    report['after_hole_filling'] = info
    repaired, after = local_repair(mesh, info, mm, settings, tracker, report, emit)
    pending = hole_report['pending_local_fix']
    if pending:
        hole_report['intersections_before_local_fix'] = info['self_intersecting_faces']
        if repaired is not None:
            hole_report.update(accepted_with_local_fix=pending, pending_local_fix=0,
                               intersections_after_local_fix=after['self_intersecting_faces'])
            for item in hole_report['details']:
                if item.pop('pending_local_fix', False):
                    item['accepted_with_local_fix'] = True
            emit(f'Заплаток принято после локального исправления: {pending}.')
        else:
            # Patches can move old rim vertices, so deleting added faces is not
            # a rollback. Replay only safe patches from the untouched input.
            attempt_report = dict(hole_report)
            started_rollback = time.monotonic()
            mesh, hole_report = fill_simple_holes(before_holes, mm, settings.quality_mm, emit=emit)
            hole_report.update(rolled_back_local_fix=pending, conditional_attempt=attempt_report,
                rollback_reason=report['local_repair'].get('reason', 'local_fix_threshold_exceeded'),
                rollback_seconds=time.monotonic()-started_rollback)
            report['hole_filling'] = hole_report
            info = diagnostics(mesh, mm)
            report['after_hole_filling'] = info
            emit(f'Локальное исправление заплаток не принято; отменено заплаток: {pending}.')
    report['steps'].append({'name': 'Заполнение простых дыр', **hole_report})
    del before_holes
    if hole_report['holes_after']:
        report['warnings'].append(f"После безопасного заполнения осталось дыр: {hole_report['holes_after']}. Они переданы объёмному восстановлению.")
    t = time.monotonic()
    if repaired is not None:
        report['algorithm'] = 'local_cut_and_fill'
        from .deviation import source_deviation
        report['local_repair']['source_deviation'] = source_deviation(reference, repaired, mm,
            settings.quality_mm, None, report['acceleration'], emit=emit)
    elif valid(info):
        if info['components'] == 1:
            repaired, after = mesh, info
            report['algorithm'] = 'surface_repair'
        elif info['components'] <= 32:
            try:
                repaired = union_small(mesh, mm, tracker)
                after = diagnostics(repaired, mm)
                if not valid(after):
                    repaired = None
                report['algorithm'] = 'solid_union'
            except RuntimeError as exc:
                report['surface_attempt_error'] = str(exc)
                repaired = None
        else:
            repaired, after = mesh, info
            report['algorithm'] = 'closed_components_preserved'
    if repaired is None:
        center = mesh.computeBoundingBox().center()
        centered = mm.Mesh(mesh)
        centered.transform(mm.AffineXf3f.translation(-center))
        repaired = sparse_repair(centered, mm, settings, tracker, report, emit, translation=center)
        after = report['reconstruction_validation']
        reason = 'После локального ремонта остались повреждения; выполнено воксельное восстановление.'
    elif report['algorithm'] == 'local_cut_and_fill':
        reason = 'Самопересечения устранены локально методом CutAndFill, без вокселизации модели.'
    else:
        reason = 'Ориентация и заполнение простых дыр позволили сохранить поверхность без вокселей.'
    report['algorithm_reason'] = reason
    report['steps'].append({'name': 'Восстановление поверхности', 'elapsed_seconds': time.monotonic()-t,
                            'algorithm': report['algorithm'], 'reason': reason})
    emit(reason)
    if repaired is None or not repaired.topology.numValidFaces():
        raise RuntimeError('Не удалось восстановить непустую поверхность при выбранном качестве.')
    tracker.update(66, 'Проверка восстановленной сетки')
    if after is None:
        after = diagnostics(repaired, mm, tracker.callback(66, 70, 'Проверка восстановленной сетки'))
    if not valid(after):
        raise RuntimeError(f"Результат не прошёл проверку: дыр {after['boundary_loops']}; пересечений {after['self_intersecting_faces']}. "
                           'Файл не сохранён. Попробуйте выключить восстановление острых рёбер.')
    report['after'] = after
    # Retain valid native stages for a failed final STL round-trip, never disk copies.
    fallbacks = []
    if settings.remove_small_shells:
        from .inspection import shell_summary
        _, shells = shell_summary(repaired, mm)
        regions = mm.MeshComponents.getAllComponents(mm.MeshPart(repaired))
        small = mm.FaceBitSet()
        for region in regions:
            if region.count() < 50:
                small |= region
        removed = [shell for shell in shells if shell['triangles'] < 50]
        report['removed_shells'] = removed
        if small.count() and small.count() < repaired.topology.numValidFaces():
            fallbacks.append((repaired, after, 'small_shells'))
            repaired = repaired.cloneRegion(repaired.topology.getValidFaces()-small)
            after = dict(topology_info(repaired, mm), self_intersecting_faces=0)
            report['after'] = after
            emit(f'Удалено мелких оболочек: {len(removed)}.')
        elif small.count():
            report['warnings'].append('Все оболочки мелкие; удаление отменено, чтобы не получить пустой результат.')
            report['removed_shells'] = []
    emit(f"После ремонта: {info['triangles']:,} → {after['triangles']:,} треугольников.")
    tmp = folder/'result.stl'
    final = repaired
    if settings.collapse_coplanar or settings.simplify_curved:
        final = simplify_attempts(repaired, None, mm, settings, tracker, report, emit, folder)
        if final is not repaired:
            fallbacks.append((repaired, after, 'simplification'))
    else:
        count = mm.findDegenerateFaces(mm.MeshPart(final)).count()
        report['degenerate_faces_before_cleanup'] = count
        if count:
            final = cleanup_exported(repaired, None, mm, settings, tracker, report, folder, emit)
            if final is not repaired:
                fallbacks.append((repaired, after, 'cleanup'))
    t = time.monotonic()
    roundtrips = []
    while True:
        save_settings = mm.SaveSettings()
        save_settings.progress = tracker.callback(96, 97, 'Запись итогового STL')
        mm.saveMesh(final, tmp, save_settings)
        exported = mm.loadMesh(tmp)
        final_info = diagnostics(exported, mm, tracker.callback(97, 98, 'Контроль итогового STL'))
        roundtrips.append(final_info)
        if valid(final_info):
            shells_changed = final_info['components'] != report['after']['components']
            if not shells_changed or not fallbacks:
                if shells_changed:
                    # STL stores triangle soup. Its loader can split coincident
                    # voxel seams even when both meshes are closed and valid.
                    # Optional changes are rolled back first; a valid base is
                    # still usable, with the actual exported topology reported.
                    report['export_topology'] = {'before': report['after'], 'after': final_info}
                    report['warnings'].append(f"STL изменил связность при загрузке: {report['after']['components']} → "
                        f"{final_info['components']} оболочек. Замкнутость, топология и отсутствие пересечений проверены; "
                        'проверьте раздельные фрагменты. [stl_shell_count_changed]')
                break
        if not fallbacks:
            raise RuntimeError('Итоговый STL не прошёл проверку; проверенные стадии исчерпаны.')
        final, restored, stage = fallbacks.pop()
        report['after'] = restored
        report['warnings'].append(f'Итоговый STL не прошёл проверку; откат стадии [{stage}] и повторная запись.')
        if stage == 'simplification':
            from .reduction import reason_text
            reason_code = 'export_validation_failed'
            report['simplification'].update(accepted=False, after=restored['triangles'], reason=reason_code,
                                             reason_ru=reason_text(reason_code))
        elif stage == 'small_shells':
            report['removed_shells'] = []
        elif stage == 'cleanup':
            report['degenerate_cleanup'].update(accepted=False, reason='export_validation_failed')
    report['after'] = final_info
    report['final_roundtrips'] = len(roundtrips)
    report['steps'].append({'name': 'Запись и повторная проверка итогового STL', 'elapsed_seconds': time.monotonic()-t})
    from .deviation import source_deviation
    map_tmp = folder/'deviation.npz'
    report['source_deviation'] = source_deviation(reference, exported, mm, settings.quality_mm,
        map_tmp, report['acceleration'], tracker.callback(98, 99, 'Карта отклонения от исходника'), emit)
    report['source_deviation']['heatmap_file'] = str(map_tmp)
    deviation = report['source_deviation']['max_deviation_mm']
    if not math.isfinite(deviation) or deviation > 2*settings.quality_mm:
        report['warnings'].append(f'Отклонение вершин исходника от результата {deviation:.6g} мм больше двух шагов ({2*settings.quality_mm:g} мм). Проверьте места ремонта.')
    from .inspection import shell_summary
    summary, report['shells'] = shell_summary(exported, mm)
    report['after'].update(summary)
    count = report['after']['components']
    if count > 1:
        report['warnings'].append(f'В файле {count} замкнутых оболочек. Раздельные детали сохранены; перемычки не добавлялись.')
    report.update(status='repaired' if count == 1 else 'repaired_multiple_shells', output=str(tmp),
                  artifacts={'stl': str(tmp), 'deviation': str(map_tmp)}, work_directory=str(folder),
                  elapsed_seconds=time.monotonic()-started)
    emit(f"Карта отклонения: {report['source_deviation']['elapsed_seconds']:.2f} с; "
         f"проекции={','.join(report['acceleration'].get('projection_backends', []))}.")
    tracker.update(100, 'Готово')
    return report
