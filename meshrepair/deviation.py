"""Chunked source-vertex distances, full sidecar map and compact JSON hotspots."""
import time
import numpy as np
from .compute import Projector
from .reduction import projection_arrays


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
    count = min(10, len(distances))
    top = np.argpartition(distances, len(distances)-count)[-count:]
    top = top[np.argsort(-distances[top], kind='stable')]
    # No millions of JSON entries: the map is a separate, lossless NumPy file.
    error = None
    try:
        if map_path is not None:
            with open(map_path, 'wb') as stream:
                np.savez(stream, vertices=vertices, distances_mm=distances, threshold_mm=threshold)
    except OSError as exc:
        error = str(exc)
    return {'heatmap_error': error,'mode': 'measured_not_proven', 'direction': 'source_vertices_to_result_surface',
            'max_deviation_mm': float(distances.max()), 'warning_threshold_mm': threshold,
            'vertex_count': len(vertices), 'exceeding_vertices': int(np.count_nonzero(distances > threshold)),
            'vertices_over_quality': int(np.count_nonzero(distances > quality_mm)),
            'top10': [{'vertex_id': int(i), 'coordinates_mm': vertices[i].tolist(),
                       'deviation_mm': float(distances[i])} for i in top],
            'elapsed_seconds': time.monotonic()-started}
