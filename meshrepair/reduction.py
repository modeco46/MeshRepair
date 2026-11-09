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
    candidate = mm.Mesh(mesh)
    stages = []
    requested = []
    if settings.collapse_coplanar:
        requested.append(('coplanar', settings.coplanar_angle_deg))
    if settings.simplify_curved:
        requested.append(('curved', 30.0))
    for index, (kind, angle) in enumerate(requested):
        before = candidate.topology.numValidFaces()
        candidate.packOptimally(False)
        params = mm.DecimateSettings()
        # The SDK's maxError is an estimate. Acceptance is decided by the
        # independent surface bound after the actual exported STL is reloaded.
        params.maxError = settings.quality_mm * .25 * error_scale
        params.maxAngleChange = -1  # this property controls edge FLIPS, not coplanarity
        params.tinyEdgeLength = 0
        params.optimizeVertexPos = kind != 'coplanar'
        if kind == 'coplanar':
            params.stabilizer = 0.0
        params.touchBdVerts = False
        params.touchNearBdEdges = True
        creases = mm.findCreaseEdges(candidate.topology, candidate.points,
                                    max(math.radians(angle), 1e-7))
        params.bdVerts = mm.getIncidentVerts(candidate.topology, creases) | candidate.topology.findBdVerts()
        # Preserve tiny islands completely, including their connectivity.
        from meshlib import mrmeshnumpy as mn
        mapping, count = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(candidate))
        groups = np.fromiter((int(x) for x in mapping), dtype=np.int32, count=mapping.size())
        sizes = np.bincount(groups, minlength=count)
        small_faces = mn.faceBitSetFromBools(sizes[groups] < 50)
        params.bdVerts |= mm.getIncidentVerts(candidate.topology, small_faces)
        if kind == 'coplanar':
            params.edgesToCollapse = candidate.topology.findNotLoneUndirectedEdges() - creases
        params.subdivideParts = min(os.cpu_count() or 1, cpu_count(), max(1, before // 25_000))
        start, end = 78 + 7 * index / len(requested), 78 + 7 * (index + 1) / len(requested)
        stage = 'Схлопывание плоских участков' if kind == 'coplanar' else 'Упрощение криволинейных участков'
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
                       'native_error_estimate_mm': float(result.errorIntroduced)})
    candidate.pack()
    return candidate, stages


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


def projection_arrays(projections, mm):
    """Bulk view of the pinned SDK POD vector, guarded by exported layout offsets.

    MeshLib 3.1.4.297 exports data_pointer and C++ offsetof values. Its record is
    32 bytes (int32 face, 3 floats, MeshTriPoint, float distance). Unsupported
    layouts safely use public properties. The vector must outlive these views.
    """
    n = len(projections)
    layout = (mm.MeshProjectionResult._offsetof_proj, mm.PointOnFace._offsetof_face,
              mm.MeshProjectionResult._offsetof_mtp, mm.MeshProjectionResult._offsetof_distSq)
    if layout == (0, 0, 16, 28) and n:
        buffer = (ctypes.c_ubyte * (32*n)).from_address(projections.data_pointer())
        ids = np.ndarray((n,), dtype=np.int32, buffer=buffer, offset=0, strides=(32,))
        distances = np.ndarray((n,), dtype=np.float32, buffer=buffer, offset=28, strides=(32,))
        # Two public accesses check the ABI without a per-projection Python loop.
        if int(ids[-1]) == int(projections[-1].proj.face) and float(distances[-1]) == float(projections[-1].distSq):
            return ids, distances
    return (np.fromiter((int(p.proj.face) for p in projections), dtype=np.int32, count=n),
            np.fromiter((p.distSq for p in projections), dtype=float, count=n))


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
