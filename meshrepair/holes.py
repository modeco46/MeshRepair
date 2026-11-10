"""Transactional, bounded filling of simple boundary loops before voxelization."""
import time
import numpy as np


def planar_boundary(mesh, mm, loop, points, quality_mm):
    # A planar boundary alone is insufficient (a latitude hole on a sphere).
    # Require support from at least one coplanar neighboring face as well.
    centered = points-points.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] < 1e-10:
        return False
    normal = axes[-1]
    if np.max(np.abs(centered @ normal)) > max(1e-6, quality_mm*.001):
        return False
    for edge in loop:
        face = mesh.topology.right(edge)
        if face.valid():
            n = mesh.normal(face)
            values = np.array([n.x, n.y, n.z], float)
            length = np.linalg.norm(values)
            if length and abs(values @ normal)/length > 1-1e-6:
                return True
    return False


def fill_simple_holes(mesh, mm, quality_mm, progress=lambda value: None, emit=lambda text: None,
                      max_diameter_mm=10.0, max_boundary_edges=2048,
                      max_patch_faces=50000, time_budget_seconds=60.0):
    from meshlib import mrmeshnumpy as mn
    started = time.monotonic()
    work = mm.Mesh(mesh)
    loops = list(work.topology.findHoleRepresentiveEdges())
    report = {'holes_before': len(loops), 'filled': 0, 'rejected': 0, 'skipped': 0,
              'max_diameter_mm': max_diameter_mm, 'max_boundary_edges': max_boundary_edges,
              'max_patch_faces': max_patch_faces, 'time_budget_seconds': time_budget_seconds,
              'details': []}
    if loops:
        existing_collisions = mm.findSelfCollidingTrianglesBS(mm.MeshPart(work))
    target_edge = max(2*quality_mm, .05)
    for index, edge in enumerate(loops):
        progress(index/max(1, len(loops)))
        item = {'hole': index+1}
        report['details'].append(item)
        if time.monotonic()-started >= time_budget_seconds:
            report['skipped'] += len(loops)-index
            item['reason'] = 'hole_fill_time_budget'
            break
        # Accepted edits do not pack/reindex the mesh. A previous patch may have
        # closed a touching loop too; check its representative before proceeding.
        if work.topology.left(edge).valid():
            item['reason'] = 'already_closed'
            continue
        loop = mm.trackRightBoundaryLoop(work.topology, edge)
        ids = np.fromiter((int(work.topology.org(e)) for e in loop), dtype=np.int32)
        pts = mn.getNumpyVerts(work)[ids].astype(float)
        diameter = float(np.linalg.norm(np.ptp(pts, axis=0)))
        estimated = int(np.ceil(diameter**2 / target_edge**2 * 2))
        item.update(boundary_edges=len(ids), diameter_mm=diameter, estimated_patch_faces=estimated)
        if len(ids) > max_boundary_edges or diameter > max_diameter_mm or estimated > max_patch_faces:
            item['reason'] = 'hole_size_limit'
            report['skipped'] += 1
            continue
        if len(ids) < 3 or len(np.unique(ids)) != len(ids):
            item['reason'] = 'non_simple_boundary'
            report['skipped'] += 1
            continue
        attempt = time.monotonic()
        try:
            # This copy also rolls back modified boundary triangles/coordinates,
            # not merely newly created faces. No partial failed patch can leak.
            candidate = mm.Mesh(work)
            before_holes = candidate.topology.findNumHoles()
            settings = mm.FillHoleNicelySettings()
            flat = planar_boundary(work, mm, loop, pts, quality_mm)
            settings.triangulateOnly = flat
            item['method'] = 'planar_triangulation' if flat else 'curvature_patch'
            settings.subdivideSettings.maxEdgeLen = target_edge
            settings.subdivideSettings.maxEdgeSplits = min(20000, max_patch_faces//2)
            settings.smoothCurvature = True
            patch = mm.fillHoleNicely(candidate, edge, settings)
            remaining = candidate.topology.findNumHoles()
            collisions = mm.findSelfCollidingTrianglesBS(mm.MeshPart(candidate))
            if (not patch.count() or remaining >= before_holes or not candidate.topology.checkValidity()
                    or not np.isfinite(mn.getNumpyVerts(candidate)).all()):
                item['reason'] = 'patch_not_closed'
            elif (collisions-existing_collisions).count():
                item['reason'] = 'patch_self_intersections'
            elif patch.count() > max_patch_faces:
                item['reason'] = 'patch_work_limit'
            else:
                work = candidate
                existing_collisions = collisions
                item.update(accepted=True, patch_triangles=patch.count())
                report['filled'] += before_holes-remaining
        except RuntimeError as exc:
            item.update(reason='patch_native_error', detail=str(exc))
        if not item.get('accepted'):
            report['rejected'] += 1
        item['elapsed_seconds'] = time.monotonic()-attempt
    report.update(holes_after=work.topology.findNumHoles(), closed_after=work.topology.isClosed(),
                  elapsed_seconds=time.monotonic()-started)
    progress(1)
    if loops:
        emit(f"Заполнение дыр: {len(loops)} → {report['holes_after']}; принято {report['filled']}, "
             f"отклонено {report['rejected']}, пропущено {report['skipped']}; {report['elapsed_seconds']:.2f} с.")
    return work, report
