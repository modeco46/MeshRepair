"""Read-only inspection and preview export, isolated in the cancellable worker."""
from pathlib import Path
import time
import numpy as np
from .engine import Progress, diagnostics
from .compute import capabilities


def load(source, tracker):
    from meshlib import mrmeshpy as mm
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() not in {'.stl', '.obj', '.ply'}:
        raise ValueError('Выберите существующий STL, OBJ или PLY.')
    settings = mm.MeshLoadSettings()
    skipped, duplicated = mm.int_output(), mm.int_output()
    settings.skippedFaceCount, settings.duplicatedVertexCount = skipped, duplicated
    settings.callback = tracker.callback(0, 30, 'Загрузка сетки')
    mesh = mm.loadMesh(source, settings)
    if not mesh.topology.numValidFaces():
        raise ValueError('Файл не содержит пригодных треугольников.')
    mesh.pack()
    return source, mesh, mm, {'skipped_faces': skipped.value, 'duplicated_vertices': duplicated.value}


def basic_info(mesh, mm):
    box = mesh.computeBoundingBox()
    size = box.size()
    return {'vertices': mesh.topology.numValidVerts(), 'triangles': mesh.topology.numValidFaces(),
            'components': mm.MeshComponents.getNumComponents(mm.MeshPart(mesh)),
            'closed': mesh.topology.isClosed(), 'dimensions_mm': [size.x, size.y, size.z]}


def inspect_mesh(source, emit=lambda message: None, progress=lambda event: None):
    started = time.monotonic()
    tracker = Progress(progress)
    source, mesh, mm, loader = load(source, tracker)
    emit('Проверка топологии и пересечений: нативный CPU (CUDA для этого этапа в SDK нет).')
    info = diagnostics(mesh, mm, tracker.callback(30, 70, 'Проверка самопересечений и топологии'))
    info.update(basic_info(mesh, mm))
    info['area_mm2'] = mesh.area()
    info['degenerate_faces'] = mm.findDegenerateFaces(mm.MeshPart(mesh), cb=tracker.callback(70, 78, 'Поиск вырожденных граней')).count()
    tracker.update(79, 'Проверка отдельных оболочек')
    mapping, count = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(mesh))
    groups = np.fromiter((int(x) for x in mapping), dtype=np.int32, count=mapping.size())
    sizes = np.bincount(groups, minlength=count)
    boundary = mesh.topology.findLeftBdEdges()
    adjacent = np.fromiter((int(mesh.topology.right(e)) for e in boundary), dtype=np.int32, count=boundary.count())
    open_edges = np.bincount(groups[adjacent], minlength=count)
    info['boundary_edges'] = int(open_edges.sum())
    info['closed_components'] = int(np.count_nonzero(open_edges == 0))
    info['open_components'] = int(np.count_nonzero(open_edges))
    # The summary always covers every component; only the UI detail table is capped.
    order = np.argsort(-sizes)[:2000]
    shells = [{'id': int(i) + 1, 'triangles': int(sizes[i]), 'closed': bool(open_edges[i] == 0),
               'boundary_edges': int(open_edges[i])} for i in order]
    warnings = []
    if loader['skipped_faces']:
        warnings.append(f"Загрузчик пропустил непригодные грани: {loader['skipped_faces']}.")
    if count > len(shells):
        warnings.append(f'В таблице показаны {len(shells)} крупнейших оболочек из {count}; общие числа включают все.')
    tracker.update(100, 'Проверка завершена')
    return {'operation': 'check', 'source': str(source), 'units': 'mm', 'checked': True,
            'info': info, 'shells': shells, 'loader': loader, 'warnings': warnings,
            'acceleration': capabilities(), 'elapsed_seconds': time.monotonic() - started}


def export_preview(source, destination, progress=lambda event: None):
    from meshlib import mrmeshnumpy as mn
    started = time.monotonic()
    tracker = Progress(progress)
    source, mesh, mm, loader = load(source, tracker)
    tracker.update(35, 'Подготовка предпросмотра')
    info = basic_info(mesh, mm)
    center = mesh.computeBoundingBox().center()
    # Translation only; keep every triangle and do not simplify the displayed mesh.
    mesh.transform(mm.AffineXf3f.translation(-center))
    vertices = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    tracker.update(65, 'Передача геометрии в 3D-окно')
    with open(destination, 'wb') as stream:
        np.savez(stream, vertices=vertices, faces=faces)
    tracker.update(100, 'Предпросмотр подготовлен')
    return {'operation': 'preview', 'source': str(source), 'preview': str(destination),
            'checked': False, 'info': info, 'loader': loader, 'acceleration': capabilities(),
            'elapsed_seconds': time.monotonic() - started}
