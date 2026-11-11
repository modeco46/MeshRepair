"""Bounded, vectorized normal-cluster statistics; never modifies geometry."""
import time
import numpy as np


def triangle_distribution(reference, result, mm, tolerance, angle_deg=8., minimum_area=3.):
    from meshlib import mrmeshnumpy as mn
    from .reduction import projection_arrays
    started = time.monotonic()
    result.pack()
    verts = mn.getNumpyVerts(result).astype(float)
    faces = mn.getNumpyFaces(result.topology)
    triangles = verts[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(lengths[:, None], np.finfo(float).tiny)
    areas = lengths*.5
    centers = triangles.mean(axis=1)
    # Quantization supplies at most 64 dominant seeds. Every actual assignment
    # uses the exact angular criterion, not the quantized bins.
    bins, inverse = np.unique(np.rint(normals/.1).astype(np.int16), axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=areas)
    assigned = np.full(len(faces), -1, dtype=np.int32)
    groups = []
    cosine = np.cos(np.radians(angle_deg))
    for index in np.argsort(weights)[::-1][:64]:
        direction = bins[index].astype(float)
        norm = np.linalg.norm(direction)
        if not norm:
            continue
        direction /= norm
        selected = (assigned < 0) & (np.einsum('ij,j->i', normals, direction) >= cosine)
        ids = np.flatnonzero(selected)
        if areas[ids].sum() <= minimum_area:
            continue
        # Separate parallel surfaces by their position; overlapping offset bins
        # are not merged into one fictitious plane through several shells.
        offsets = np.einsum('ij,j->i', centers[ids], direction)
        pitch = max(.1, tolerance*4)
        order = np.argsort(offsets)
        splits = np.flatnonzero(np.diff(offsets[order]) > pitch)+1
        for subset in np.split(ids[order], splits):
            area = float(areas[subset].sum())
            if area <= minimum_area:
                continue
            normal = np.sum(normals[subset]*areas[subset, None], axis=0)
            normal /= np.linalg.norm(normal)
            center = np.average(centers[subset], axis=0, weights=areas[subset])
            assigned[subset] = len(groups)
            groups.append({'id': len(groups)+1, 'normal': normal.tolist(), 'center_mm': center.tolist(),
                           'area_mm2': area, 'triangles': len(subset),
                           'fraction': len(subset)/max(1, len(faces))})
    if groups:
        # One native projector for all original vertices. Nearest result faces
        # identify the corresponding original region without axis assumptions.
        projector = mm.PointsToMeshProjector()
        projector.updateMeshData(result)
        original = mn.getNumpyVerts(reference)
        distances = [[] for _ in groups]
        for start in range(0, len(original), 131072):
            batch = original[start:start+131072]
            projections = mm.std_vector_MeshProjectionResult()
            projector.findProjections(projections, mn.fromNumpyArray(batch), None, None,
                                      float(np.finfo(np.float32).max), 0)
            face_ids, _ = projection_arrays(projections, mm)
            valid = (face_ids >= 0) & (face_ids < len(assigned))
            labels = np.full(len(batch), -1, np.int32)
            labels[valid] = assigned[face_ids[valid]]
            for i, group in enumerate(groups):
                points = batch[labels == i].astype(float)
                if len(points):
                    distances[i].append(np.abs(np.einsum('ij,j->i', points-np.array(group['center_mm']), np.array(group['normal']))))
        for group, chunks in zip(groups, distances):
            values = np.concatenate(chunks) if chunks else np.empty(0)
            group['source_vertices'] = len(values)
            group['source_residual_p50_mm'] = float(np.quantile(values, .5)) if len(values) else None
            group['source_residual_p90_mm'] = float(np.quantile(values, .9)) if len(values) else None
            group['rougher_than_tolerance'] = bool(len(values) and group['source_residual_p90_mm'] > tolerance)
    planar = int(np.count_nonzero(assigned >= 0))
    return {'method': 'dominant_normal_clusters_and_nearest_source_vertices', 'angle_deg': angle_deg,
            'minimum_area_mm2': minimum_area, 'planes': groups, 'total_triangles': len(faces),
            'large_planes_triangles': planar, 'other_triangles': len(faces)-planar,
            'large_planes_fraction': planar/max(1, len(faces)), 'other_fraction': 1-planar/max(1, len(faces)),
            'elapsed_seconds': time.monotonic()-started,
            'tolerance_hint': {'kind': 'heuristic_not_prediction', 'tolerance_multiplier': 2.5,
                               'approximate_triangle_reduction_fraction': .25}}
