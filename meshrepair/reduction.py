"""Optional native simplification with a conservative whole-surface distance check.

The check is relative to the repaired mesh, not the damaged input. It bounds every
triangle (both directions), rather than merely checking distances at mesh vertices.
"""
import math
import os
import ctypes
import numpy as np
from .compute import Projector, cpu_count


def propose_reduction(mesh, mm, settings, progress, error_scale=1.0):
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
    if small.count() == candidate.topology.numValidFaces():
        return candidate, []
    if protected is not None:
        candidate = candidate.cloneRegion(candidate.topology.getValidFaces() - small)
    stages = []
    requested = []
    # A general QEM pass already reduces planar faces. Running a separate
    # planar pass first resets the metric and spends the error budget twice.
    if settings.simplify_curved:
        requested.append(('curved', None))
    elif settings.collapse_coplanar:
        requested.append(('coplanar', settings.coplanar_angle_deg))
    for index, (kind, angle) in enumerate(requested):
        before = candidate.topology.numValidFaces()
        candidate.packOptimally(False)
        params = mm.DecimateSettings()
        # The SDK's maxError is an estimate. Acceptance is decided by the
        # independent surface bound after the actual exported STL is reloaded.
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
        # General simplification is controlled by the distance to the complete
        # repaired surface, not by locking voxel-generated normal discontinuities.
        params.subdivideParts = min(os.cpu_count() or 1, cpu_count(), max(1, before // 25_000))
        start, end = 78 + 7 * index / len(requested), 78 + 7 * (index + 1) / len(requested)
        stage = 'Схлопывание плоских участков' if kind == 'coplanar' else 'Упрощение всей поверхности в пределах допуска'
        progress.update(start, stage)
        # MeshLib 3.1.4.297 Python callbacks from parallel decimation workers
        # deadlock on the GIL. Keep native parallelism, but report only stage
        # boundaries here. The separate UI process still supports immediate Stop.
        if params.subdivideParts == 1:
            params.progressCallback = progress.callback(start, end, stage)
        result = mm.decimateMesh(candidate, params)
        if result.cancelled:
            raise RuntimeError('Упрощение прервано.')
        progress.update(end, stage)
        stages.append({'kind': kind, 'angle_deg': angle, 'before': before,
                       'after': candidate.topology.numValidFaces(),
                       'native_error_estimate_mm': float(result.errorIntroduced),
                       'angle_weighted_metric': True, 'protected_components': protected_count,
                       'includes_planar_surfaces': kind == 'curved',
                       'feature_policy': 'ridge_corners' if kind == 'coplanar' else 'surface_tolerance'})
    if protected is not None:
        candidate.addMesh(protected)
    candidate.pack()
    return candidate, stages



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
                    return {'verified': False, 'reason': 'verification_work_limit', 'checked_subtriangles': checked}
                centers = tris.mean(axis=1).astype(np.float32)
                projections = mm.std_vector_MeshProjectionResult()
                projector.findProjections(projections, mn.fromNumpyArray(centers), None, None,
                                          float(np.finfo(np.float32).max), 0)
                ids, distances = projection_arrays(projections, mm)
                if (ids < 0).any() or (ids >= len(target_faces)).any():
                    return {'verified': False, 'reason': 'projection_failed'}
                # A sampled violation may safely reject a proposal; samples alone
                # are never used to ACCEPT it.
                if (distances > tolerance * tolerance).any():
                    return {'verified': False, 'reason': 'sample_exceeds_tolerance', 'checked_subtriangles': checked}
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
                        return {'verified': False, 'reason': 'distance_not_confirmed', 'checked_subtriangles': checked}
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
