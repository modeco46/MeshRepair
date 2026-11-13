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


DENSE_WINDING_VOXEL_LIMIT = 64_000_000


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
    union_max_seconds: float = 240.0
    voxel_warn_seconds: float = 600.0
    voxel_max_cells: float = 6e9

    def validate(self):
        if not math.isfinite(self.quality_mm) or not .001 <= self.quality_mm <= .5:
            raise ValueError("Качество должно быть от 0,001 до 0,5 мм.")
        if not math.isfinite(self.coplanar_angle_deg) or not 0 <= self.coplanar_angle_deg <= 45:
            raise ValueError("Угол копланарности должен быть от 0 до 45 градусов.")
        if self.local_fix_max_faces < 0 or not math.isfinite(self.local_fix_fraction) or not 0 <= self.local_fix_fraction <= 1:
            raise ValueError('Некорректный порог локального ремонта.')
        if not math.isfinite(self.sharpening_fallback_max_seconds) or self.sharpening_fallback_max_seconds < 0:
            raise ValueError('Некорректный предел времени повтора без Sharpening.')
        if not math.isfinite(self.union_max_seconds) or self.union_max_seconds < 0:
            raise ValueError('Некорректный предел времени объединения оболочек.')
        if not math.isfinite(self.voxel_warn_seconds) or self.voxel_warn_seconds < 0:
            raise ValueError('Некорректный порог предупреждения о времени вокселизации.')
        if not math.isfinite(self.voxel_max_cells) or self.voxel_max_cells <= 0:
            raise ValueError('Некорректный предел размера воксельной сетки.')


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


def union_small(mesh, mm, tracker, deadline=None):
    """Small component sets only; larger sets go to sparse reconstruction, never fail by count.

    deadline is an optional time.monotonic() limit; the native boolean is cancelled through its callback."""
    regions = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    result = None
    for index, region in enumerate(regions):
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError('Превышен лимит времени объединения оболочек.')
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
            progress_callback = tracker.callback(20 + 20 * index / len(regions),
                                                 20 + 20 * (index + 1) / len(regions), "Объединение поверхностей")
            params.cb = progress_callback if deadline is None else \
                (lambda value, cb=progress_callback: bool(cb(value)) and time.monotonic() < deadline)
            operation = mm.boolean(result, part, mm.BooleanOperation.Union, params)
            if not operation.valid():
                raise RuntimeError(operation.errorString)
            result = operation.mesh
    return result


def shell_intersection_split(mesh, mm):
    """Colliding triangle pairs inside one shell versus between different shells."""
    pairs = mm.findSelfCollidingTriangles(mm.MeshPart(mesh))
    if not len(pairs):
        return 0, 0
    mapping, _ = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(mesh))
    labels = np.fromiter((int(value) for value in mapping), dtype=np.int64)
    first = np.fromiter((int(pair.aFace) for pair in pairs), dtype=np.int64, count=len(pairs))
    second = np.fromiter((int(pair.bFace) for pair in pairs), dtype=np.int64, count=len(pairs))
    same = int((labels[first] == labels[second]).sum())
    return same, len(pairs) - same


def union_between_shells(mesh, info, mm, settings, tracker, report, emit):
    """Closed shells that only cross each other (bail, prongs, stones) are merged exactly by a boolean union.

    Voxelising the whole model for this case is slow, inflates the triangle count and leaves internal cavities."""
    item = report['union_attempt'] = {'attempted': False, 'accepted': False,
                                      'self_intersecting_faces': info['self_intersecting_faces']}
    if info['self_intersecting_faces'] > 500_000:
        item['reason'] = 'too_many_intersections'
        return None, None
    started = time.monotonic()
    inside, between = shell_intersection_split(mesh, mm)
    item.update(pairs_inside_shell=inside, pairs_between_shells=between, split_seconds=time.monotonic()-started)
    if inside or not between:
        item['reason'] = 'intersections_inside_shell' if inside else 'no_pairs'
        emit(f'Пересечения внутри одной оболочки ({inside}); булево объединение оболочек не применяется.' if inside
             else 'Пары пересекающихся граней не найдены; булево объединение не применяется.')
        return None, None
    item['attempted'] = True
    emit(f"Пересечения только между оболочками ({between:,} пар, оболочек {info['components']}): булево объединение без вокселей.")
    started = time.monotonic()
    try:
        repaired = union_small(mesh, mm, tracker, deadline=started + settings.union_max_seconds)
        if time.monotonic()-started >= settings.union_max_seconds:
            raise RuntimeError('Превышен лимит времени объединения оболочек.')
        after = diagnostics(repaired, mm)
    except (RuntimeError, MemoryError) as exc:
        item.update(reason='union_failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        report['warnings'].append(f'Булево объединение оболочек не удалось ({exc}); выполнено воксельное восстановление.')
        emit('Булево объединение не удалось; воксельное восстановление.')
        return None, None
    item['elapsed_seconds'] = time.monotonic()-started
    if not valid(after):
        item.update(reason='union_invalid', after=after)
        emit('Результат объединения не прошёл проверку; воксельное восстановление.')
        return None, None
    item.update(accepted=True, after=after)
    report['algorithm'] = 'solid_union'
    return repaired, after


def drop_cavities(mesh, after, mm, report, emit, quality_mm):
    """Internal voids and zero-thickness shells are removed after voxel reconstruction.

    They appear where self-intersecting or doubled internal surfaces give a zero winding number. Their signed
    volume is negative (void) or numerically zero (flat sheet thinner than a voxel). They make decimation change
    the shell count and an STL round trip split coincident layers into holes. Cavities inside the part are not
    preserved by design; at least one shell with real positive volume must remain."""
    regions = mm.MeshComponents.getAllComponents(mm.MeshPart(mesh))
    if len(regions) < 2:
        return mesh, after, None
    volumes = [(region, float(mesh.volume(region))) for region in regions]
    negligible = 1e-3*quality_mm**3
    voids = [(region, volume) for region, volume in volumes if volume <= negligible]
    if not voids or len(voids) == len(volumes):
        return mesh, after, None
    drop = mm.FaceBitSet()
    for region, _ in voids:
        drop |= region
    cleaned = mesh.cloneRegion(mesh.topology.getValidFaces()-drop)
    report['removed_cavities'] = [{'triangles': region.count(), 'volume_mm3': volume} for region, volume in voids]
    emit(f"Удалено внутренних полостей и пластин нулевой толщины: {len(voids)} (граней {drop.count():,}).")
    return cleaned, dict(topology_info(cleaned, mm), self_intersecting_faces=0), mesh


def sparse_repair(mesh, mm, settings, tracker, report, emit, translation=None):
    """Validate the full result and reuse winding data for a bounded retry."""
    from .orientation import orient_closed_components
    report.update(algorithm='sparse_openvdb_winding', voxel_size_mm=settings.quality_mm)
    report['sparse_closed_orientation_corrections'] = orient_closed_components(mesh, mm)
    sharp = report['sharpening'] = {'requested': settings.restore_sharp_edges, 'accepted': False}
    report['compute_device'] = 'cpu_native_parallel'
    estimate = max(1, int(mesh.area()/settings.quality_mm**2))
    bbox = mesh.computeBoundingBox()
    extent = bbox.max - bbox.min

    def grid_for(step):
        return [max(1, int(math.ceil(float(value)/step))+2) for value in (extent.x, extent.y, extent.z)]

    grid = grid_for(settings.quality_mm)
    cells = float(grid[0])*grid[1]*grid[2]
    report['voxel_estimate'] = {'triangles': estimate, 'method': 'bbox_cells_timed_probe',
                              'grid': grid, 'cells': cells, 'estimated_seconds': None}
    estimated_seconds = cells*1.3e-6   # conservative CPU figure until the timed probe calibrates it
    emit(f'Воксельная сетка {grid[0]}×{grid[1]}×{grid[2]} ≈ {cells/1e6:,.0f} млн ячеек; '
         f'поверхность ≈ {estimate:,} треугольников (площадь / шаг²).')
    if cells > settings.voxel_max_cells:
        limit_step = settings.quality_mm*(cells/settings.voxel_max_cells)**(1/3)
        raise RuntimeError(f'Воксельная сетка слишком велика ({cells/1e9:.1f} млрд ячеек): расчёт займёт часы. '
                           f'Увеличьте допуск до {math.ceil(limit_step*1000)/1000:g} мм или больше.')
    winding = None
    try:
        cuda = None if settings.disable_cuda else cuda_module()
        if cuda is not None:
            winding = cuda.FastWindingNumber(mesh)
    except Exception as exc:
        report['warnings'].append(f'CUDA недоступна, используется CPU: {exc}')
    report['voxel_passes'] = []

    def reconstruct(sharpen, step, probe=False, stage=None, span=None):
        nonlocal winding
        params = mm.GeneralOffsetParameters() if sharpen else mm.OffsetParameters()
        params.voxelSize = step
        params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
        stage = stage or ('Проба острых рёбер (шаг ×4)' if probe else 'Восстановление поверхности')
        start, end = span or ((32, 41) if probe else (42, 65))
        grid_cells = math.prod(grid_for(step))
        streaming = grid_cells > DENSE_WINDING_VOXEL_LIMIT
        # Non-chunked external winding disables the SDK's FunctionVolume.
        params.fwn = (winding if not streaming or isinstance(winding, mm.IFastWindingNumberByParts)
                      else None)
        params.memoryEfficient = True
        narrow_band = streaming and not sharpen and params.fwn is None and mesh.topology.isClosed()
        params.closeHolesInHoleWindingNumber = not narrow_band
        params.callBack = tracker.callback(start, end, stage+(' (по слоям)' if streaming else ''))
        if sharpen:
            params.mode = mm.GeneralOffsetParametersMode.Sharpening
        operation = mm.generalOffsetMesh if sharpen else mm.mcOffsetMesh if streaming else mm.offsetMesh
        backend = 'sharpening' if sharpen else 'marching_cubes_by_parts' if streaming else 'smooth_openvdb'
        device = ('cuda_winding_with_cpu_marching_cubes' if streaming else 'cuda_winding_with_cpu_openvdb') if params.fwn is not None else 'cpu_native_parallel'
        report['compute_device'] = device
        item = {'probe': probe, 'sharpening': sharpen, 'voxel_size_mm': step, 'compute_device': device,
                'timing': stage.startswith('Оценка времени'), 'backend': backend, 'streaming': streaming,
                'grid_cells': grid_cells, 'cpu_closed_surface_band': narrow_band}
        report['voxel_passes'].append(item)
        started = time.monotonic()
        emit(f'{stage}: шаг {step:g} мм; backend={backend}; compute_device={device}.')
        try:
            try:
                result = operation(mm.MeshPart(mesh), 0, params)
                if narrow_band and not result.topology.isClosed():
                    item['closed_surface_band_fallback'] = True
                    params.closeHolesInHoleWindingNumber = True
                    del result
                    emit('Узкая область расчёта оставила отверстия; повтор с полным полем и тем же шагом.')
                    result = operation(mm.MeshPart(mesh), 0, params)
                return result
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

    if cells >= 30e6:
        # Voxel time follows the number of grid cells of the bounding box, not the surface area, so a thin
        # ring with an empty centre is far slower than area/step² suggests. A 64× cheaper pass calibrates it.
        probe_step = settings.quality_mm*4
        probe_grid = grid_for(probe_step)
        probe_cells = float(probe_grid[0])*probe_grid[1]*probe_grid[2]
        started_probe = time.monotonic()
        reconstruct(False, probe_step, probe=True, stage='Оценка времени (шаг ×4)', span=(41, 42))
        probe_seconds = time.monotonic()-started_probe
        eta = estimated_seconds = probe_seconds*cells/probe_cells
        report['voxel_estimate'].update(estimated_seconds=eta, probe_seconds=probe_seconds, probe_cells=probe_cells)
        emit(f'Оценка времени воксельного расчёта по пробе (шаг ×4): ≈ {eta:,.0f} с.')
        if eta > settings.voxel_warn_seconds:
            target = math.ceil(settings.quality_mm*(eta/300.0)**(1/3)*1000)/1000
            message = (f'Воксельный расчёт займёт около {eta/60:.0f} мин при допуске {settings.quality_mm:g} мм. '
                       f'Допуск {target:g} мм даст примерно 5 минут.')
            report['warnings'].append(message)
            emit(message)

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
        if not sharpen and not valid(check):
            result, check = repair_reconstruction_intersections(result, check, mm, settings, tracker, report, emit)
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


def repair_reconstruction_intersections(mesh, info, mm, settings, tracker, report, emit):
    """Repair a few plain-mesher collisions, with rollback and a surface bound.

    This is deliberately separate from input repair: a reconstructed shell may
    neither disappear nor move outside the small local correction budget.
    Full Sharpening output never enters this path.
    """
    from . import reduction as rd
    threshold = max(settings.local_fix_max_faces, int(info['triangles']*settings.local_fix_fraction))
    item = report['reconstruction_local_repair'] = {
        'attempted': False, 'accepted': False, 'before': info, 'threshold_faces': threshold,
        'tolerance_mm': settings.quality_mm*.25}
    if not (info['closed'] and info['topology_valid'] and 0 < info['self_intersecting_faces'] <= threshold):
        item['reason'] = 'outside_local_repair_limits'
        return mesh, info
    item['attempted'] = True
    started = time.monotonic()
    tracker.update(65, 'Локальное исправление воксельной поверхности')
    emit(f"Воксельная поверхность: локальное исправление {info['self_intersecting_faces']} пересекающихся граней.")
    item['attempts'] = []
    try:
        for method in (mm.FixSelfIntersectionMethod.Relax, mm.FixSelfIntersectionMethod.CutAndFill):
            attempt = {'method': str(method)}
            item['attempts'].append(attempt)
            try:
                candidate = mm.Mesh(mesh)
                params = mm.FixSelfIntersectionSettings()
                params.method = method
                params.mimicPatch = True
                # Match diagnostics. The SDK's touch=True prepass remeshes
                # degeneracies across the ENTIRE mesh and can delete unrelated
                # tiny shells before it reaches the actual intersection patch.
                params.touchIsIntersection = False
                mm.SelfIntersections.fix(candidate, params)
                candidate.pack()
                after = attempt['after'] = diagnostics(candidate, mm)
                if not valid(after):
                    attempt['reason'] = 'local_fix_invalid'
                elif after['components'] != info['components']:
                    attempt['reason'] = 'local_fix_shell_count_changed'
                else:
                    deviation = attempt['measured_deviation_mm'] = rd.measured_distance(mesh, candidate, mm)
                    if not math.isfinite(deviation) or deviation > item['tolerance_mm']:
                        attempt['reason'] = 'local_fix_exceeds_tolerance'
                    else:
                        proof = attempt['verification'] = rd.verify_surface_bound(mesh, candidate, item['tolerance_mm'],
                            tracker.callback(65, 66, 'Проверка локального исправления'),
                            acceleration=report.get('acceleration'), emit=emit)
                        proof['reference'] = 'reconstructed_mesh_before_local_fix'
                        if proof.get('verified'):
                            item.update(attempt, accepted=True)
                            return candidate, after
                        attempt['reason'] = proof.get('reason', 'distance_not_confirmed')
            except (RuntimeError, MemoryError) as exc:
                attempt.update(reason='local_fix_native_error', detail=str(exc))
        item.update(item['attempts'][-1])
    finally:
        item['elapsed_seconds'] = time.monotonic()-started
        emit(f"Локальное исправление воксельной поверхности: {'принято' if item['accepted'] else 'отклонено'}; "
             f"{item['elapsed_seconds']:.2f} с" + (f" [{item['reason']}]" if item.get('reason') else '.'))
    return mesh, info


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
    """Search measured error against the unchanged repaired reference; prove once."""
    from . import reduction as rd
    original = report['after']
    mode = 'proven' if settings.strict_tolerance else 'measured_not_proven'
    target = .95 * settings.quality_mm
    summary = {'requested': True, 'accepted': False, 'before': original['triangles'],
               'after': original['triangles'], 'tolerance_mm': settings.quality_mm,
               'measured_target_mm': target, 'scale_range': [0., .8],
               'strict_tolerance': settings.strict_tolerance, 'mode': mode,
               'input_triangles': report.get('before', {}).get('triangles'), 'attempts': []}
    report['simplification'] = summary
    emit('Режим допуска: '+('доказано (строгая проверка).' if settings.strict_tolerance else 'измерено (вершины в обе стороны).'))
    started = time.monotonic()
    shellwise = False
    prepared = None

    def propose(factor, phase):
        nonlocal shellwise, prepared
        item = {'attempt': len(summary['attempts'])+1, 'phase': phase, 'shellwise': shellwise,
                'error_scale': factor, 'max_error_mm': settings.quality_mm*factor, 'checkpoints': []}
        summary['attempts'].append(item)
        t = time.monotonic()
        last_valid = None

        def checkpoint(candidate, stages):
            nonlocal last_valid
            check = {'pass': len(stages), 'stages': stages}
            item['checkpoints'].append(check)
            if candidate.topology.numValidFaces() >= original['triangles']:
                check['reason'] = 'no_reducible_edges'
                return False
            info = diagnostics(candidate, mm, tracker.callback(86, 88, 'Проверка упрощённой сетки'))
            if not valid(info) and mm.findDegenerateFaces(mm.MeshPart(candidate)).count():
                cleanup_report = {'after': info, 'steps': [], 'warnings': []}
                candidate = cleanup_exported(candidate, None, mm, settings, tracker, cleanup_report, None, emit)
                info = cleanup_report['after']
                check['cleanup'] = cleanup_report['degenerate_cleanup']
            check['candidate'] = info
            if not valid(info):
                check['reason'] = 'topology_changed'
            elif info['components'] != original['components']:
                check['reason'] = 'kept_shell_count_changed'
            else:
                tracker.update(88, f"Подбор допуска: {item['attempt']}")
                deviation = rd.measured_distance(exported, candidate, mm)
                check['verification'] = {'verified': False, 'mode': 'measured_not_proven',
                    'max_deviation_mm': deviation, 'reference': 'repaired_mesh_before_simplification',
                    'method': 'bidirectional_vertices_to_surface'}
                if math.isfinite(deviation) and deviation <= target:
                    if last_valid is None or info['triangles'] < last_valid[1]['triangles']:
                        last_valid = candidate, info, check
                    return True
                check['reason'] = 'sample_exceeds_tolerance'
            emit(f"Попытка {item['attempt']}, проход {len(stages)} отклонён: {rd.reason_text(check['reason'])}")
            return False

        try:
            reduce = rd.propose_reduction_by_shell if shellwise else rd.propose_reduction
            # Keep the whole-mesh -> per-shell fallback. A single body also
            # retains safe intermediate QEM passes against the same reference.
            per_pass = original['components'] == 1 and not shellwise
            options = {'checkpoint': checkpoint} if per_pass else {}
            if not shellwise:
                if prepared is None:
                    prepared = rd.prepare_reduction(exported, mm)
                    summary['preparation_seconds'] = time.monotonic()-t
                options['prepared'] = prepared
            candidate, stages = reduce(exported, mm, settings, tracker, error_scale=factor,
                preserve_planar_corners=report.get('algorithm') in {
                    'surface_repair', 'solid_union', 'closed_components_preserved'}, **options)
            if not item['checkpoints']:
                checkpoint(candidate, stages)
            if not item['checkpoints']:
                item.update(reason='no_reducible_edges', stages=stages)
            elif last_valid is None:
                item.update(item['checkpoints'][-1])
        except (RuntimeError, MemoryError) as exc:
            item.update(reason='native_error', detail=str(exc))
        finally:
            item['elapsed_seconds'] = time.monotonic()-t
            emit(f"Упрощение, попытка {item['attempt']}: {item['elapsed_seconds']:.2f} с; compute_device=cpu_native_parallel.")
            if item.get('reason') and last_valid is None:
                emit(f"Попытка {item['attempt']} отклонена: {rd.reason_text(item['reason'])}")
        if last_valid is not None:
            candidate, info, selected = last_valid
            if item['checkpoints'][-1].get('reason') or item.get('reason'):
                item['stopped_reason'] = item.get('reason') or item['checkpoints'][-1]['reason']
                emit(f"Сохранён проверенный проход {selected['pass']} попытки {item['attempt']}.")
            item.pop('reason', None)
            item.update(selected)
            return candidate, info, item
        if item.get('reason') in {'topology_changed', 'kept_shell_count_changed'} and not shellwise \
                and original['components'] > 1:
            shellwise = True
            summary['shellwise'] = True
            emit('Общее упрощение меняет оболочки; упрощение каждой оболочки отдельно.')
            return propose(factor, phase)
        return None, None, item

    best = None
    low, high = 0., .8
    for iteration in range(6):
        # Preserve the first two historical probes for stable cases. If both
        # fail, approach zero instead of hitting the old hard floor of 0.15.
        factor = .475 if iteration == 0 else .3125 if iteration == 1 and best is None else (low+high)/2
        candidate, info, item = propose(factor, 'search')
        if candidate is None:
            high = factor
            if item.get('reason') == 'no_reducible_edges':
                break
        else:
            low = factor
            # QEM is not strictly monotone in scale. Retain the smallest valid
            # candidate, even if a later, larger scale gives a worse mesh.
            if best is None or info['triangles'] < best[1]['triangles']:
                best = candidate, info, item
            if item['verification']['max_deviation_mm'] < settings.quality_mm*1e-4:
                # Exact planar collapse has no unused curved error budget.
                break
            if item['verification']['max_deviation_mm'] >= target*.99:
                summary['search_stop'] = 'within_one_percent_of_measured_target'
                break

    # Strict proof is intentionally outside the binary search. Each proof and
    # every retry uses the original repaired surface, never the previous pass.
    if best is not None and settings.strict_tolerance:
        for proof_attempt in range(3):
            candidate, info, item = best
            t = time.monotonic()
            try:
                verification = rd.verify_surface_bound(exported, candidate, settings.quality_mm,
                    tracker.callback(88, 96, f'Строгая проверка допуска: {proof_attempt+1}/3'),
                    acceleration=report['acceleration'], emit=emit)
            except (RuntimeError, MemoryError) as exc:
                verification = {'verified': False, 'reason': 'native_error', 'detail': str(exc)}
            verification['mode'] = mode
            verification['reference'] = 'repaired_mesh_before_simplification'
            item['measured_verification'] = item['verification']
            item['verification'] = verification
            item['proof_seconds'] = time.monotonic()-t
            if verification.get('verified'):
                verification['max_deviation_mm'] = verification['upper_bound_mm']
                break
            item['reason'] = verification.get('reason', 'distance_not_confirmed')
            best = None
            if proof_attempt < 2:
                candidate, info, item = propose(item['error_scale']*.8, 'proof_retry')
                if candidate is not None:
                    best = candidate, info, item
                else:
                    break

    accepted = exported
    if best is not None:
        accepted, info, item = best
        report['after'] = info
        summary.update(accepted=True, after=info['triangles'], accepted_attempt=item['attempt'],
                       error_scale=item['error_scale'], stages=item['stages'], verification=item['verification'])
    else:
        item = summary['attempts'][-1]
        summary['reason'] = item.get('reason', 'native_error')
        summary['reason_ru'] = rd.reason_text(summary['reason'])
        summary['verification'] = item.get('verification', {})
        emit('Сохранена сетка без упрощения. ' + summary['reason_ru'])
        report['warnings'].append(summary['reason_ru'])
    summary['elapsed_seconds'] = time.monotonic()-started
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
    report = {'version': '1.0.0', 'settings': asdict(settings), 'source': str(source),
              'units': 'mm', 'tolerance_certified': False, 'warnings': [], 'steps': [],
              'compute_device': 'cpu_native_parallel', 'acceleration': capabilities(settings.disable_cuda)}
    report['stage_timings'] = []
    tracker = Progress(progress, emit, report['stage_timings'],
        lambda stage: report['compute_device'] if stage in {'Проба острых рёбер (шаг ×4)', 'Оценка времени (шаг ×4)', 'Восстановление поверхности'}
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
    if repaired is None and not valid(info) and info['closed'] and info['topology_valid'] \
            and info['boundary_loops'] == 0 and 1 < info['components'] <= 32 and info['self_intersecting_faces'] > 0:
        repaired, after = union_between_shells(mesh, info, mm, settings, tracker, report, emit)
    if repaired is None:
        center = mesh.computeBoundingBox().center()
        centered = mm.Mesh(mesh)
        centered.transform(mm.AffineXf3f.translation(-center))
        repaired = sparse_repair(centered, mm, settings, tracker, report, emit, translation=center)
        after = report['reconstruction_validation']
        reason = 'После локального ремонта остались повреждения; выполнено воксельное восстановление.'
    elif report['algorithm'] == 'local_cut_and_fill':
        reason = 'Самопересечения устранены локально методом CutAndFill, без вокселизации модели.'
    elif report['algorithm'] == 'solid_union' and report.get('union_attempt', {}).get('accepted'):
        reason = 'Оболочки пересекались только друг с другом; объединены булевой операцией без вокселей.'
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
                           'Оставшиеся дефекты не удалось безопасно исправить. Файл не сохранён.')
    report['after'] = after
    # Retain valid native stages for a failed final STL round-trip, never disk copies.
    fallbacks = []
    if report['algorithm'] == 'sparse_openvdb_winding':
        repaired, cleaned_after, before_cavities = drop_cavities(repaired, after, mm, report, emit, settings.quality_mm)
        if before_cavities is not None:
            fallbacks.append((before_cavities, after, 'cavities'))
            after = cleaned_after
            report['after'] = after
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
        elif stage == 'cavities':
            report['removed_cavities'] = []
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
    from .planes import triangle_distribution
    tracker.update(99, 'Где остались треугольники')
    report['triangle_distribution'] = triangle_distribution(reference, exported, mm, settings.quality_mm)
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
