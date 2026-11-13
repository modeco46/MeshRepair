"""Chunked source-vertex distances, full sidecar map and compact JSON hotspots."""
import time
import numpy as np
from .compute import Projector
from .reduction import projection_arrays


def inside_flags(vertices, distances, target, mm, quality_mm, limit=400_000):
    """True for source vertices that lie inside the solid of the result and deeper than the tolerance.

    Such vertices belong to internal or doubled source surfaces that a union or voxel repair removes on purpose,
    so they are not a shape deviation. The sign comes from MeshLib's pseudo-normal signed distance, which stays
    correct at concave edges (a nearest-triangle plane test does not). Only vertices farther than the tolerance
    are tested; if there are more than `limit`, the farthest ones are tested and the rest stay counted."""
    flags = np.zeros(len(vertices), dtype=bool)
    candidates = np.flatnonzero(distances > quality_mm)
    if len(candidates) > limit:
        candidates = candidates[np.argsort(-distances[candidates], kind='stable')[:limit]]
    for index in candidates:
        value = target.signedDistance(mm.Vector3f(*(float(c) for c in vertices[index])))
        if value is not None and value < 0:
            flags[index] = True
    return flags


def source_deviation(source, target, mm, quality_mm, map_path, acceleration=None,
                     progress=lambda value: None, emit=lambda text: None):
    from meshlib import mrmeshnumpy as mn
    started = time.monotonic()
    source.pack()
    vertices = mn.getNumpyVerts(source)
    distances = np.empty(len(vertices), dtype=np.float32)
    projector = Projector(target, mm, acceleration, emit)
    for start in range(0, len(vertices), 131072):
        points = mn.fromNumpyArray(np.ascontiguousarray(vertices[start:start+131072], dtype=np.float32))
        projections = mm.std_vector_MeshProjectionResult()
        projector.findProjections(projections, points, None, None, float(np.finfo(np.float32).max), 0)
        ids, squared = projection_arrays(projections, mm)
        if len(squared) != len(vertices[start:start+131072]) or (ids < 0).any() or not np.isfinite(squared).all():
            raise RuntimeError('Не удалось вычислить карту отклонения: отсутствует проекция на поверхность.')
        distances[start:start+len(squared)] = np.sqrt(np.maximum(squared, 0))
        progress((start+len(squared))/len(vertices))
    threshold = 2*quality_mm
    inside = inside_flags(vertices, distances, target, mm, quality_mm)
    classified = True
    outer = np.where(inside, 0, distances).astype(np.float32)     # internal source vertices do not count
    count = min(10, len(distances))
    top = np.argpartition(outer, len(outer)-count)[-count:]
    top = top[np.argsort(-outer[top], kind='stable')]
    # No millions of JSON entries: the map is a separate, lossless NumPy file.
    error = None
    try:
        if map_path is not None:
            with open(map_path, 'wb') as stream:
                np.savez(stream, vertices=vertices, distances_mm=outer, all_distances_mm=distances,
                         interior=inside, threshold_mm=threshold)
    except OSError as exc:
        error = str(exc)
    return {'heatmap_error': error,'mode': 'measured_not_proven', 'direction': 'source_vertices_to_result_surface',
            'max_deviation_mm': float(outer.max()), 'max_deviation_all_mm': float(distances.max()),
            'warning_threshold_mm': threshold, 'interior_classified': classified,
            'interior_vertices': int(inside.sum()),
            'interior_max_deviation_mm': float(distances[inside].max()) if inside.any() else 0.0,
            'vertex_count': len(vertices), 'exceeding_vertices': int(np.count_nonzero(outer > threshold)),
            'vertices_over_quality': int(np.count_nonzero(outer > quality_mm)),
            'top10': [{'vertex_id': int(i), 'coordinates_mm': vertices[i].tolist(),
                       'deviation_mm': float(outer[i])} for i in top],
            'elapsed_seconds': time.monotonic()-started}
