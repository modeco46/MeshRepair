"""Optional native simplification with a conservative whole-surface distance check.

The check is relative to the repaired mesh, not the damaged input. It bounds every
triangle (both directions), rather than merely checking distances at mesh vertices.
"""
import math
import os
import ctypes
import time
import numpy as np
from .compute import Projector, cpu_count


def prepare_reduction(mesh, mm):
    """Split the immutable reference once, rather than once per search attempt."""
    from meshlib import mrmeshnumpy as mn
    candidate = mm.Mesh(mesh)
    candidate.pack()
    # Parallel decimation must not even receive the tiny shells. bdVerts alone
    # did not reliably preserve them in the pinned SDK's partitioned path.
    mapping, count = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(candidate))
    groups = np.fromiter((int(x) for x in mapping), dtype=np.int32, count=mapping.size())
    sizes = np.bincount(groups, minlength=count)
    small = mn.faceBitSetFromBools(sizes[groups] < 50)
    protected_count = int(np.count_nonzero(sizes < 50))
    protected = candidate.cloneRegion(small) if small.count() else None
    # A face-count threshold alone cannot protect shells: a physically tiny
    # shell can have thousands of faces and disappear in the FIRST QEM pass.
    # Give each connected component its own deletion budget, including on
    # subsequent passes. Four faces must remain in every closed component.
    large_ids = np.flatnonzero(sizes >= 50)
    components = ([candidate] if count == 1 and len(large_ids) else
                  [candidate.cloneRegion(mn.faceBitSetFromBools(groups == i)) for i in large_ids])
    return {'components': components, 'protected': protected, 'protected_count': protected_count}


def propose_reduction(mesh, mm, settings, progress, error_scale=1.0, *, preserve_planar_corners=False,
                      emit=lambda message: None, prepared=None, checkpoint=None, component_tolerance=None):
    from meshlib import mrmeshnumpy as mn
    progress.update(78, 'Подготовка оболочек к упрощению')
    prepared = prepare_reduction(mesh, mm) if prepared is None else prepared
    components = [mm.Mesh(part) for part in prepared['components']]
    protected, protected_count = prepared['protected'], prepared['protected_count']
    if not components:
        return mm.Mesh(mesh), []
    largest = max(range(len(components)), key=lambda i: components[i].topology.numValidFaces())
    frozen = {}
    protected_faces = protected.topology.numValidFaces() if protected is not None else 0
    first_pass_counts = []
    stages = []
    requested = []
    # A general QEM pass already reduces planar faces. Running a separate
    # planar pass first resets the metric and spends the error budget twice.
    if settings.simplify_curved:
        requested.extend([('curved', None)] * 3)
    elif settings.collapse_coplanar:
        requested.extend([('coplanar', settings.coplanar_angle_deg)] * 3)
    for index, (kind, angle) in enumerate(requested):
        before = sum(part.topology.numValidFaces() for part in components)
        started = time.monotonic()
        start, end = 78 + 7 * index / len(requested), 78 + 7 * (index + 1) / len(requested)
        stage = 'Схлопывание плоских участков' if kind == 'coplanar' else 'Упрощение всей поверхности в пределах допуска'
        stage += f': проход {index + 1}/{len(requested)}'
        progress.update(start, stage)
        emit(f'{stage}; треугольников {before:,}; оболочек {len(components)}.')
        details, completed = [], 0
        for component_index, candidate in enumerate(components):
            part_before = candidate.topology.numValidFaces()
            preserve_corners = (preserve_planar_corners and kind == 'curved' and index > 0 and
                                first_pass_counts[component_index][1] < first_pass_counts[component_index][0]*.1)
            # A thin satellite can drift far along its plane despite a small
            # QEM estimate. Its failure must not discard a good main body.
            # The largest component and inter-component collisions are checked
            # by the complete-mesh checkpoint below, against the same reference.
            check_component = component_tolerance is not None and component_index != largest
            previous = mm.Mesh(candidate) if check_component else None
            if component_index in frozen:
                detail = dict(frozen[component_index], before=part_before, after=part_before)
            else:
                detail = _decimate_component(candidate, mm, mn, settings, kind, angle, error_scale,
                    preserve_corners, progress, start + (end-start)*completed/before,
                    start + (end-start)*(completed+part_before)/before, stage)
                if check_component:
                    reference = prepared['components'][component_index]
                    reason = None
                    if (candidate.topology.isClosed() != reference.topology.isClosed() or
                            not candidate.topology.checkValidity() or
                            mm.MeshComponents.getNumComponents(mm.MeshPart(candidate)) != 1 or
                            mm.findSelfCollidingTrianglesBS(mm.MeshPart(candidate)).count()):
                        reason = 'component_topology_changed'
                    else:
                        deviation = measured_distance(reference, candidate, mm)
                        detail['measured_deviation_mm'] = deviation
                        if not math.isfinite(deviation) or deviation > component_tolerance:
                            reason = 'component_exceeds_tolerance'
                    if reason:
                        components[component_index] = previous
                        detail.update(reverted_reason=reason, proposed_after=detail['after'], after=part_before)
                        frozen[component_index] = detail.copy()
                        emit(f'Оболочка {component_index+1}: сохранён предыдущий проход [{reason}].')
            details.append(detail)
            completed += part_before
            if index == 0:
                first_pass_counts.append((part_before, detail['after']))
        after = sum(part.topology.numValidFaces() for part in components)
        progress.update(end, stage)
        stages.append({'kind': kind, 'angle_deg': angle, 'before': before, 'after': after,
                       'pass': index + 1, 'error_scale': error_scale,
                       'preserved_reduced_corners': any(d['preserved_reduced_corners'] for d in details),
                       'after_with_protected': after + protected_faces,
                       'native_error_estimate_mm': max(d['native_error_estimate_mm'] for d in details),
                       'angle_weighted_metric': True, 'protected_components': protected_count,
                       'component_count': len(components) + protected_count,
                       'component_passes': details, 'elapsed_seconds': time.monotonic()-started,
                       'includes_planar_surfaces': kind == 'curved',
                       'feature_policy': 'ridge_corners' if kind == 'coplanar' else 'surface_tolerance'})
        emit(f'{stage}: {before:,} → {after:,}; {stages[-1]["elapsed_seconds"]:.2f} с; '
             f'участков на оболочку до {max(d["subdivide_parts"] for d in details)}.')
        if checkpoint is not None:
            # Own the snapshot: later native passes mutate components in place.
            snapshot = _assemble_components(components, protected, mm)
            if not checkpoint(snapshot, list(stages)):
                break
        if after > before * .97:
            stages[-1]['stop_reason'] = 'removed_less_than_three_percent'
            break
    return _assemble_components(components, protected, mm), stages


def _assemble_components(components, protected, mm):
    candidate = mm.Mesh(components[0])
    for part in components[1:]:
        candidate.addMesh(part)
    if protected is not None:
        candidate.addMesh(protected)
    candidate.pack()
    return candidate


def _decimate_component(candidate, mm, mn, settings, kind, angle, error_scale,
                        preserve_corners, progress, start, end, stage):
    before = candidate.topology.numValidFaces()
    candidate.packOptimally(False)
    params = mm.DecimateSettings()
    params.maxDeletedFaces = max(0, before - 4)
    # The SDK's maxError is an estimate. Acceptance is decided by the
    # independent distance check on the candidate in memory.
    params.maxError = settings.quality_mm * error_scale
    params.maxAngleChange = -1  # this property controls edge FLIPS, not coplanarity
    params.tinyEdgeLength = 0
    params.optimizeVertexPos = kind != 'coplanar'
    # Angle weighting reduces artificial dependence of QEM on local
    # triangulation density (in particular the dense voxel seam bands).
    params.angleWeightedDistToPlane = True
    params.stabilizer = 0.0 if kind == 'coplanar' else 1e-6
    params.touchBdVerts = False
    params.touchNearBdEdges = True
    params.bdVerts = candidate.topology.findBdVerts()
    if kind == 'coplanar':
        creases = mm.findCreaseEdges(candidate.topology, candidate.points,
                                    max(math.radians(angle), 1e-7))
        # Keep junctions/corners, but allow collinear ridge vertices to
        # disappear. Locking every crease vertex leaves a dense seam belt.
        params.bdVerts |= crease_corners(candidate, creases, mm, mn)
        params.notFlippable = creases
        params.collapseNearNotFlippable = True
    if preserve_corners:
        # A >90% first-pass collapse is typical of triangulated planar
        # solids. Reinitializing QEM on their few remaining faces can let
        # sharp junctions drift even when the tolerance is much larger.
        # The caller enables this for surface repairs only; voxel seam
        # vertices must remain free to remove reconstruction artifacts.
        creases = mm.findCreaseEdges(candidate.topology, candidate.points, math.radians(45))
        params.bdVerts |= crease_corners(candidate, creases, mm, mn)
    # General simplification is controlled by the distance to the complete
    # repaired surface, not by locking voxel-generated normal discontinuities.
    # In 3.1.4.297 the parallel path replaces bdVerts with partition
    # boundaries, ignoring caller-specified corners and open boundaries.
    # Keep constrained passes global. Unconstrained closed shells still
    # use native parallelism, with its final inter-part cleanup enabled.
    constrained = kind == 'coplanar' or params.bdVerts.count() > 0
    # The pinned partition setup scans the whole mesh for each part. Bound
    # that overhead on many-core machines instead of growing it with size.
    params.subdivideParts = (1 if constrained else
        min(8, os.cpu_count() or 1, cpu_count(), max(1, before // 25_000)))
    params.decimateBetweenParts = True
    # MeshLib 3.1.4.297 Python callbacks from parallel decimation workers
    # deadlock on the GIL. Keep native parallelism, but report only stage
    # boundaries here. The separate UI process still supports immediate Stop.
    if params.subdivideParts == 1:
        params.progressCallback = progress.callback(start, end, stage)
    result = mm.decimateMesh(candidate, params)
    if result.cancelled:
        raise RuntimeError('Упрощение прервано.')
    return {'before': before, 'after': candidate.topology.numValidFaces(),
            'subdivide_parts': params.subdivideParts, 'fixed_vertices': params.bdVerts.count(),
            'preserved_reduced_corners': preserve_corners,
            'native_error_estimate_mm': float(result.errorIntroduced)}



def crease_corners(mesh, creases, mm, mn):
    """Freeze only branch/end/corner vertices of crease chains.

    On a straight ridge a degree-two vertex can be removed. At a real corner
    the turn angle is nonzero, so we preserve it. No Python collapse callback
    is installed (parallel MeshLib callbacks can deadlock the GIL).
    """
    verts = mn.getNumpyVerts(mesh)
    if not creases.count():
        return mm.VertBitSet()
    edges = np.fromiter((int(e)*2 for e in creases), dtype=np.int32, count=creases.count())
    first = np.fromiter((int(mesh.topology.org(mm.EdgeId(int(e)))) for e in edges), dtype=np.int32, count=len(edges))
    second = np.fromiter((int(mesh.topology.dest(mm.EdgeId(int(e)))) for e in edges), dtype=np.int32, count=len(edges))
    delta = verts[second].astype(float)-verts[first]
    length = np.linalg.norm(delta,axis=1)
    direction = delta/np.maximum(length[:,None],np.finfo(float).tiny)
    degree = np.bincount(np.r_[first,second],minlength=len(verts))
    directions = np.zeros((len(verts),3))
    np.add.at(directions,first,direction)
    np.add.at(directions,second,-direction)
    # Only almost-collinear chains are relaxed here. Broader removal of voxel
    # stair-steps belongs to the tolerance-checked general simplification pass.
    turn = np.linalg.norm(directions,axis=1)
    pinned = (degree > 0) & ((degree != 2) | (turn > 2*math.sin(math.radians(1)/2)))
    return mn.vertBitSetFromBools(pinned)

def triangle_upper_bounds(triangles, targets):
    """For each source triangle, an upper bound to ONE target triangle.

    Choose three points q_i on the same target triangle, one for each source
    vertex p_i. Convexity implies that all points of the source triangle are at
    distance <= max_i ||p_i-q_i|| from that target triangle (and thus target mesh).
    No assumption that a triangle's interior follows its vertices' nearest faces.
    """
    return np.max(triangle_vertex_distances(triangles, targets), axis=1)


def triangle_vertex_distances(triangles, targets):
    """Distances from source vertices to explicitly constructed target points."""
    a, b, c = targets[:, 0, :], targets[:, 1, :], targets[:, 2, :]
    ab, ac = b - a, c - a
    delta = triangles - a[:, None, :]
    d00 = np.einsum('ij,ij->i', ab, ab)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d20 = np.einsum('ikj,ij->ik', delta, ab)
    d21 = np.einsum('ikj,ij->ik', delta, ac)
    det = d00 * d11 - d01 * d01
    stable = det > 1e-12 * np.maximum(d00 * d11, np.finfo(float).tiny)
    divisor = np.where(stable, det, 1)[:, None]
    u = (d11[:, None] * d20 - d01[:, None] * d21) / divisor
    v = (d00[:, None] * d21 - d01[:, None] * d20) / divisor
    # Clipping produces actual convex combinations even for outside projections.
    u = np.clip(np.where(stable[:, None], u, 0), 0, 1)
    v = np.minimum(np.maximum(np.where(stable[:, None], v, 0), 0), 1 - u)
    points = a[:, None, :] + u[:, :, None] * ab[:, None, :] + v[:, :, None] * ac[:, None, :]
    best = np.sum((triangles - points) ** 2, axis=2)
    for first, second in ((a, b), (b, c), (c, a)):
        edge = second - first
        length_sq = np.einsum('ij,ij->i', edge, edge)
        t = np.einsum('ikj,ij->ik', triangles - first[:, None, :], edge)
        t /= np.maximum(length_sq[:, None], np.finfo(float).tiny)
        t = np.clip(t, 0, 1)
        on_edge = first[:, None, :] + t[:, :, None] * edge[:, None, :]
        best = np.minimum(best, np.sum((triangles - on_edge) ** 2, axis=2))
    return np.sqrt(best)


def quad_upper_bounds(triangles, quads):
    """Bound to two adjacent triangles ABC/ABD, including any fold height.

    Project D onto ABC's plane. If ABC union ABD_projected is convex, convex
    combinations of three chosen target points stay in that union. Lifting any
    such point back onto ABC or ABD adds at most |D-D_projected| to the distance.
    This avoids subdividing to the tolerance along artificial triangulation edges.
    A concave/degenerate projection supplies no bound and returns infinity.
    """
    a, b, c, d = (quads[:, i] for i in range(4))
    normal = np.cross(b - a, c - a)
    area_sq = np.sum(normal * normal, axis=1)
    safe_area = np.maximum(area_sq, np.finfo(float).tiny)
    dot = np.sum((d - a) * normal, axis=1)
    projected = d - (dot / safe_area)[:, None] * normal
    height = np.abs(dot) / np.sqrt(safe_area)
    side_d = np.sum(np.cross(b - a, projected - a) * normal, axis=1)
    side_a = np.sum(np.cross(projected - c, a - c) * normal, axis=1)
    side_b = np.sum(np.cross(projected - c, b - c) * normal, axis=1)
    eps = 1e-12 * area_sq
    crossing = (((side_a < -eps) & (side_b > eps)) | ((side_b < -eps) & (side_a > eps))
                | ((side_a == 0) & (np.abs(side_b) > eps))
                | ((side_b == 0) & (np.abs(side_a) > eps)))
    convex = (area_sq > 0) & (side_d < -eps) & crossing
    result = np.full(len(triangles), np.inf)
    if convex.any():
        first = triangle_vertex_distances(triangles[convex], quads[convex, :3])
        second = triangle_vertex_distances(triangles[convex],
            np.stack([a[convex], b[convex], projected[convex]], axis=1))
        result[convex] = np.max(np.minimum(first, second), axis=1) + height[convex]
    return result


def verify_surface_bound(reference, candidate, tolerance, progress=lambda value: None,
                         max_depth=32, max_checks=None, acceleration=None, emit=lambda message: None):
    """Conservative adaptive bidirectional bound; uncertainty REJECTS simplification.

    Native parallel projection chooses useful reference triangles; bound arithmetic
    is float64 with a numerical margin. This is not exact-arithmetic certification.
    At a depth/work limit we report 'unconfirmed', never a false success based on samples.
    """
    from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
    # Packing makes native face IDs correspond to the NumPy face rows.
    reference.pack()
    candidate.pack()
    meshes = (reference, candidate)
    faces = [mn.getNumpyFaces(m.topology) for m in meshes]
    vertices = [mn.getNumpyVerts(m).astype(np.float64) for m in meshes]
    if any(not np.isfinite(v).all() for v in vertices):
        return {'verified': False, 'reason': 'non_finite_coordinates'}
    total_faces = len(faces[0]) + len(faces[1])
    if not total_faces:
        return {'verified': False, 'reason': 'empty_surface'}
    max_checks = max_checks if max_checks is not None else max(1_000_000, total_faces * 16)
    scale = max(1.0, *(float(np.abs(v).max()) for v in vertices))
    margin = 256 * np.finfo(float).eps * scale
    limit = tolerance - margin
    if limit <= 0:
        return {'verified': False, 'reason': 'insufficient_numeric_resolution'}
    checked = 0
    completed = 0
    maximum = 0.0
    measured_maximum = 0.0
    for direction in range(2):
        source_faces, source_verts = faces[direction], vertices[direction]
        target_faces, target_verts = faces[1 - direction], vertices[1 - direction]
        projector = Projector(meshes[1 - direction], mm, acceleration, emit)
        batch_size = 16384
        neighbors = face_neighbors(target_faces)

        for start in range(0, len(source_faces), batch_size):
            initial = source_verts[source_faces[start:start + batch_size]]
            stack = [(initial, 0)]
            while stack:
                tris, depth = stack.pop()
                checked += len(tris)
                if checked > max_checks:
                    return {'verified': False, 'reason': 'verification_work_limit', 'checked_subtriangles': checked,
                            'max_deviation_mm': measured_maximum}
                centers = tris.mean(axis=1).astype(np.float32)
                projections = mm.std_vector_MeshProjectionResult()
                projector.findProjections(projections, mn.fromNumpyArray(centers), None, None,
                                          float(np.finfo(np.float32).max), 0)
                ids, distances = projection_arrays(projections, mm)
                measured_maximum = max(measured_maximum, math.sqrt(float(distances.max())))
                if (ids < 0).any() or (ids >= len(target_faces)).any():
                    return {'verified': False, 'reason': 'projection_failed'}
                # A sampled violation may safely reject a proposal; samples alone
                # are never used to ACCEPT it.
                if (distances > tolerance * tolerance).any():
                    return {'verified': False, 'reason': 'sample_exceeds_tolerance', 'checked_subtriangles': checked,
                            'max_deviation_mm': measured_maximum}
                bounds = triangle_upper_bounds(tris, target_verts[target_faces[ids]])
                unresolved = np.flatnonzero(bounds > limit)
                if len(unresolved):
                    for edge in range(3):
                        unresolved = np.flatnonzero(bounds > limit)
                        if not len(unresolved):
                            break
                        ids0 = ids[unresolved]
                        # Edge slots are (0,1), (1,2), (2,0). Adjacent opposite
                        # vertices are precomputed once, no Python topology walks.
                        vertices4 = np.column_stack((target_faces[ids0, edge],
                            target_faces[ids0, (edge+1)%3], target_faces[ids0, (edge+2)%3],
                            neighbors[ids0, edge]))
                        bounds[unresolved] = np.minimum(bounds[unresolved],
                            quad_upper_bounds(tris[unresolved], target_verts[vertices4]))
                good = np.isfinite(bounds) & (bounds <= limit)
                if good.any():
                    maximum = max(maximum, float(bounds[good].max()) + margin)
                bad = tris[~good]
                if len(bad):
                    if depth >= max_depth:
                        return {'verified': False, 'reason': 'distance_not_confirmed', 'checked_subtriangles': checked,
                                'max_deviation_mm': measured_maximum}
                    # Bisect longest edge. Both children cover their whole parent.
                    lengths = np.stack([np.sum((bad[:, 0] - bad[:, 1]) ** 2, axis=1),
                                        np.sum((bad[:, 1] - bad[:, 2]) ** 2, axis=1),
                                        np.sum((bad[:, 2] - bad[:, 0]) ** 2, axis=1)], axis=1)
                    which = np.argmax(lengths, axis=1)
                    rows = np.arange(len(bad))
                    a, b, c = bad[rows, which], bad[rows, (which + 1) % 3], bad[rows, (which + 2) % 3]
                    middle = (a + b) * .5
                    children = np.concatenate([np.stack([a, middle, c], axis=1),
                                               np.stack([middle, b, c], axis=1)])
                    # Keep useful batch sizes instead of recursively splitting
                    # into six tiny groups by edge index; bound temporary memory.
                    for offset in range(0, len(children), batch_size):
                        stack.append((children[offset:offset + batch_size], depth + 1))
                progress(completed / total_faces)
            completed += len(initial)
            progress(completed / total_faces)
    return {'verified': True, 'upper_bound_mm': maximum, 'tolerance_mm': tolerance,
            'checked_subtriangles': checked, 'numeric_margin_mm': margin,
            'reference': 'repaired_mesh_before_simplification',
            'method': 'bidirectional_adaptive_convex_patch_bounds'}


def face_neighbors(faces):
    """Opposite vertex across each triangle edge; own opposite at boundaries."""
    from .orientation import edge_pairs
    n = len(faces)
    x, y, _, _, _ = edge_pairs(faces, int(faces.max())+1)
    result = faces[:, [2, 0, 1]].copy()
    result[x % n, x // n] = faces[y % n, (y // n + 2) % 3]
    result[y % n, y // n] = faces[x % n, (x // n + 2) % 3]
    return result


class ProjectionLayoutError(ValueError):
    """Pinned native ABI differs; never silently degrade to a Python loop."""


def projection_arrays(projections, mm):
    """Guard the SDK's 32-byte POD records before exposing NumPy views."""
    message = ('Несовместимая раскладка проекций MeshLib. Требуется версия '
               '3.1.4.297; переустановите зависимости или обновите приложение. '
               '[projection_layout_mismatch]')
    try:
        layout = (mm.MeshProjectionResult._offsetof_proj, mm.PointOnFace._offsetof_face,
                  mm.MeshProjectionResult._offsetof_mtp, mm.MeshProjectionResult._offsetof_distSq)
        if layout != (0, 0, 16, 28):
            raise ProjectionLayoutError(message)
        n = len(projections)
        if not n:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        pointer = projections.data_pointer()
        if not pointer:
            raise ProjectionLayoutError(message)
        buffer = (ctypes.c_ubyte * (32*n)).from_address(pointer)
        ids = np.ndarray((n,), dtype=np.int32, buffer=buffer, offset=0, strides=(32,))
        distances = np.ndarray((n,), dtype=np.float32, buffer=buffer, offset=28, strides=(32,))
        for i in {0, n-1}:
            if int(ids[i]) != int(projections[i].proj.face) or not np.isclose(
                    distances[i], projections[i].distSq, rtol=0, atol=0, equal_nan=True):
                raise ProjectionLayoutError(message)
        return ids, distances
    except (AttributeError, TypeError) as exc:
        raise ProjectionLayoutError(message) from exc


def measured_distance(reference, candidate, mm):
    # First argument is the TARGET surface. This samples vertices, not interiors.
    a, b = mm.MeshPart(reference), mm.MeshPart(candidate)
    return math.sqrt(max(mm.findMaxDistanceSqOneWay(a, b), mm.findMaxDistanceSqOneWay(b, a)))


REASONS = {
    'export_validation_failed': 'Итоговый STL не прошёл проверку; восстановлена предыдущая стадия из памяти.',
    'topology_changed': 'После упрощения появились дыры или самопересечения.',
    'kept_shell_count_changed': 'После упрощения изменилось число оболочек.',
    'sample_exceeds_tolerance': 'Измеренное отклонение превышает заданный допуск.',
    'distance_not_confirmed': 'Не удалось подтвердить допуск на всей поверхности.',
    'verification_work_limit': 'Достигнут предел объёма строгой проверки.',
    'projection_failed': 'Не удалось вычислить проекции для строгой проверки.',
    'no_reducible_edges': 'При этих ограничениях нет рёбер, доступных для упрощения.',
    'non_finite_coordinates': 'В координатах обнаружены NaN/inf.',
    'insufficient_numeric_resolution': 'Недостаточная численная точность координат.',
    'native_error': 'Нативная библиотека не завершила упрощение.',
}


def reason_text(code):
    return REASONS.get(code, 'Упрощение не завершено.') + f' [{code}]'
