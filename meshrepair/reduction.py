"""Optional native simplification with a conservative whole-surface distance check.

The check is relative to the repaired mesh, not the damaged input. It bounds every
triangle (both directions), rather than merely checking distances at mesh vertices.
"""
import math
import os
import numpy as np


def propose_reduction(mesh, mm, settings, progress):
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
        params.maxError = settings.quality_mm * .25
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
        if kind == 'coplanar':
            params.edgesToCollapse = candidate.topology.findNotLoneUndirectedEdges() - creases
        params.subdivideParts = min(8, os.cpu_count() or 1) if before >= 50_000 else 1
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
    return np.sqrt(np.max(best, axis=1))


def verify_surface_bound(reference, candidate, tolerance, progress=lambda value: None,
                         max_depth=32, max_checks=None):
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
        projector = mm.PointsToMeshProjector()
        projector.updateMeshData(meshes[1 - direction])
        for start in range(0, len(source_faces), 2048):
            initial = source_verts[source_faces[start:start + 2048]]
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
                ids = np.fromiter((int(p.proj.face) for p in projections), dtype=np.int64, count=len(tris))
                if (ids < 0).any() or (ids >= len(target_faces)).any():
                    return {'verified': False, 'reason': 'projection_failed'}
                # A sampled violation may safely reject a proposal; samples alone
                # are never used to ACCEPT it.
                distances = np.fromiter((p.distSq for p in projections), dtype=float, count=len(tris))
                if (distances > tolerance * tolerance).any():
                    return {'verified': False, 'reason': 'sample_exceeds_tolerance', 'checked_subtriangles': checked}
                bounds = triangle_upper_bounds(tris, target_verts[target_faces[ids]])
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
                    for edge, (i, j, k) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1))):
                        subset = bad[which == edge]
                        if not len(subset):
                            continue
                        middle = (subset[:, i] + subset[:, j]) * .5
                        stack.append((np.stack([subset[:, i], middle, subset[:, k]], axis=1), depth + 1))
                        stack.append((np.stack([middle, subset[:, j], subset[:, k]], axis=1), depth + 1))
                progress(completed / total_faces)
            completed += len(initial)
            progress(completed / total_faces)
    return {'verified': True, 'upper_bound_mm': maximum, 'tolerance_mm': tolerance,
            'checked_subtriangles': checked, 'numeric_margin_mm': margin,
            'reference': 'repaired_mesh_before_simplification',
            'method': 'bidirectional_adaptive_triangle_convex_bounds'}
