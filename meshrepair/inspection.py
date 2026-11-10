"""Read-only inspection and preview export, isolated in the cancellable worker."""
from pathlib import Path
import time
import numpy as np
from .engine import Progress, diagnostics
from .compute import capabilities
from .orientation import finite_vertices, prepare_orientation


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
    from meshlib import mrmeshnumpy as mn
    finite_vertices(mn.getNumpyVerts(mesh))
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


def shell_summary(mesh, mm):
    from meshlib import mrmeshnumpy as mn
    mesh.pack()
    mapping, count = mm.MeshComponents.getAllComponentsMap(mm.MeshPart(mesh))
    groups = np.fromiter((int(x) for x in mapping), dtype=np.int32, count=mapping.size())
    sizes = np.bincount(groups, minlength=count)
    boundary = mesh.topology.findLeftBdEdges()
    adjacent = np.fromiter((int(mesh.topology.right(e)) for e in boundary), dtype=np.int32, count=boundary.count())
    open_edges = np.bincount(groups[adjacent], minlength=count)
    minimum, maximum = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
    verts, faces = mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology)
    for start in range(0, len(faces), 131072):
        sl = slice(start, start+131072)
        tri = verts[faces[sl]]
        np.minimum.at(minimum, groups[sl], tri.min(axis=1))
        np.maximum.at(maximum, groups[sl], tri.max(axis=1))
    order = np.argsort(-sizes)[:2000]
    largest = int(np.argmax(sizes))
    shells = [{'id': int(i)+1, 'triangles': int(sizes[i]), 'closed': bool(open_edges[i] == 0),
               'boundary_edges': int(open_edges[i]), 'bbox_mm': (maximum[i]-minimum[i]).tolist(),
               'kind': 'основное тело' if i == largest else ('мелкий остров' if sizes[i] < 50 else 'оболочка')}
              for i in order]
    info = {'boundary_edges': int(open_edges.sum()), 'closed_components': int(np.count_nonzero(open_edges == 0)),
            'open_components': int(np.count_nonzero(open_edges)), 'small_components': int(np.count_nonzero(sizes < 50)),
            'largest_component_triangles': int(sizes[largest])}
    return info, shells


def describe_mesh(mesh, mm, callback=None):
    info = diagnostics(mesh, mm, callback)
    info.update(basic_info(mesh, mm))
    info['area_mm2'] = mesh.area()
    info['degenerate_faces'] = mm.findDegenerateFaces(mm.MeshPart(mesh)).count()
    summary, shells = shell_summary(mesh, mm)
    info.update(summary)
    return info, shells


def inspect_mesh(source, emit=lambda message: None, progress=lambda event: None):
    started = time.monotonic()
    tracker = Progress(progress)
    source, mesh, mm, loader = load(source, tracker)
    info, shells = describe_mesh(mesh, mm, tracker.callback(30, 65, 'Диагностика исходника'))
    # Probe a working copy: Check does not replace the original mesh or file.
    _, orientation = prepare_orientation(mesh, mm, tracker.callback(65, 95, 'Проверка согласованности ориентации'))
    info.update(disoriented_edges=orientation['disoriented_edges'], non_finite_vertices=0,
                skipped_faces=loader['skipped_faces'])
    warnings = []
    if orientation.get('conflicting_faces'):
        warnings.append(f"Неориентируемых компонентов: {orientation['unoriented_components']}; граней пропущено при согласовании: {orientation['conflicting_faces']}. Остальные компоненты обработаны.")
    if loader['skipped_faces']:
        warnings.append(f"Загрузчик пропустил непригодные грани: {loader['skipped_faces']}. Проверьте исходную модель.")
    if orientation['disoriented_edges']:
        warnings.append(f"Есть несогласованные рёбра. После согласования: {orientation['components_before']} → {orientation['components_after']} связных частей. Нажмите «Исправить сетку».")
    if info['components'] > len(shells):
        warnings.append(f"В таблице {len(shells)} крупнейших частей; общие числа включают все.")
    tracker.update(100, 'Проверка завершена')
    emit(f"Проверка: {info['triangles']:,} треугольников; {info['components']} фрагментов; несогласованных рёбер {orientation['disoriented_edges']}.")
    return {'operation': 'check', 'source': str(source), 'units': 'mm', 'checked': True,
            'info': info, 'shells': shells, 'loader': loader, 'warnings': warnings, 'orientation': orientation,
            'acceleration': capabilities(), 'elapsed_seconds': time.monotonic()-started}


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
            'center_mm': [center.x, center.y, center.z],
            'elapsed_seconds': time.monotonic() - started}
