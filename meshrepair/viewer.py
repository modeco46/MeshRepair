"""Qt + VTK viewport. Geometry transfer is bulk, rendering uses OpenGL."""
import os
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


def polydata_from_arrays(vertices, faces):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Некорректные данные предпросмотра.')
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('Предпросмотр содержит пустую или некорректную геометрию.')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Некорректные индексы треугольников.')
    points = vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=False))
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int32)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtk(offsets, deep=False), numpy_to_vtk(faces.ravel(), deep=False))
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(cells)
    data._buffers = (vertices, faces, offsets)
    return data


def heatmap_polydata(path, center):
    """Original-space vertex cloud, aligned with either preview's own center."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    with np.load(path, allow_pickle=False) as archive:
        vertices = archive['vertices'].astype(np.float32)-np.asarray(center, np.float32)
        distances = archive['distances_mm']
        threshold = float(archive['threshold_mm'])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or distances.shape != (len(vertices),)
            or not np.isfinite(vertices).all() or not np.isfinite(distances).all() or threshold <= 0):
        raise ValueError('Некорректная карта отклонения.')
    ratio = np.clip(distances/threshold, 0, 1)
    colors = np.column_stack((40+215*ratio, 110+100*ratio, 220-150*ratio)).astype(np.uint8)
    colors[distances > threshold] = [255, 45, 45]
    vertices = np.ascontiguousarray(vertices)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=False))
    offsets = np.arange(len(vertices)+1, dtype=np.int32)
    ids = np.arange(len(vertices), dtype=np.int32)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtk(offsets, deep=False), numpy_to_vtk(ids, deep=False))
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetVerts(cells)
    data.GetPointData().SetScalars(numpy_to_vtk(colors, deep=False))
    data._buffers = (vertices, offsets, ids, colors)
    data._threshold_mm = threshold
    return data


class PreviewLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, parent=None, heatmap=None, center=(0, 0, 0)):
        super().__init__(parent)
        self.path = Path(path)
        self.heatmap, self.center = heatmap, center

    def run(self):
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                vertices, faces = archive['vertices'], archive['faces']
            if self.isInterruptionRequested():
                self.failed.emit('Загрузка предпросмотра остановлена.')
                return
            data = polydata_from_arrays(vertices, faces)
            data._heatmap = None
            if self.heatmap:
                try:
                    data._heatmap = heatmap_polydata(self.heatmap, self.center)
                except (OSError, ValueError, KeyError) as exc:
                    data._heatmap_error = str(exc)
            if self.isInterruptionRequested():
                self.failed.emit('Загрузка предпросмотра остановлена.')
            else:
                self.loaded.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.path.unlink(missing_ok=True)


def make_scene():
    # Explicit imports also make the frozen application include the required factories.
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    import vtkmodules.vtkInteractionStyle  # noqa: F401
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
    renderer = vtkRenderer()
    renderer.SetBackground(.055, .08, .12)
    renderer.SetBackground2(.105, .15, .21)
    renderer.GradientBackgroundOn()
    mapper = vtkPolyDataMapper()
    mapper.ScalarVisibilityOff()
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    actor = vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(.73, .84, .88)
    prop.SetAmbient(.22)
    prop.SetDiffuse(.75)
    prop.SetSpecular(.25)
    prop.SetSpecularPower(28)
    prop.SetEdgeColor(.12, .25, .30)
    renderer.AddActor(actor)
    return renderer, mapper, actor


class MeshView(QWidget):
    renderer_info = Signal(str)

    def __init__(self, parent=None, enabled=True):
        super().__init__(parent)
        self.data = None
        self.surface, self.edges = True, False
        self.heatmap_data = None
        self.heatmap_visible = False
        self.enabled = enabled
        self.vtk = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if not enabled:
            label = QLabel('3D-предпросмотр\n\nПеретащите STL, OBJ или PLY в окно')
            label.setAlignment(Qt.AlignCenter)
            label.setObjectName('emptyView')
            layout.addWidget(label)
            return
        import vtkmodules.qt
        vtkmodules.qt.PyQtImpl = 'PySide6'
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        self.renderer, self.mapper, self.actor = make_scene()
        from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
        self.heatmap_mapper = vtkPolyDataMapper()
        self.heatmap_mapper.SetColorModeToDirectScalars()
        self.heatmap_actor = vtkActor()
        self.heatmap_actor.SetMapper(self.heatmap_mapper)
        self.heatmap_actor.GetProperty().SetPointSize(4)
        self.heatmap_actor.GetProperty().LightingOff()
        self.heatmap_actor.SetVisibility(False)
        self.renderer.AddActor(self.heatmap_actor)
        self.vtk = QVTKRenderWindowInteractor(self)
        self.vtk.setAcceptDrops(False)
        self.vtk.GetRenderWindow().SetMultiSamples(0)
        self.vtk.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        layout.addWidget(self.vtk)
        self.initialized = False

    def initialize(self):
        if self.vtk and not self.initialized:
            self.vtk.Initialize()
            if not self.vtk.GetRenderWindow().SupportsOpenGL():
                raise RuntimeError('Для 3D-предпросмотра нужен OpenGL 3.2 и совместимый видеодрайвер. Проверка и ремонт доступны.')
            self.initialized = True

    def set_data(self, data):
        self.data = data  # Keep NumPy buffers alive as long as VTK uses them.
        self.heatmap_data = getattr(data, '_heatmap', None)
        if not self.enabled:
            return
        self.initialize()
        self.mapper.SetInputData(data)
        self.heatmap_mapper.SetInputData(self.heatmap_data)
        self.set_heatmap(self.heatmap_visible)
        self.set_mode(self.surface, self.edges)
        self.fit('iso')
        try:
            window = self.vtk.GetRenderWindow()
            description = window.ReportCapabilities()
            renderer = next((line.split(':', 1)[-1].strip() for line in description.splitlines()
                             if 'OpenGL renderer string' in line), 'OpenGL')
            self.renderer_info.emit(renderer)
        except Exception:
            self.renderer_info.emit('OpenGL')

    def clear(self):
        self.data = None
        self.heatmap_data = None
        if self.enabled:
            self.heatmap_mapper.SetInputData(None)
            self.heatmap_actor.SetVisibility(False)
        if self.enabled:
            self.mapper.SetInputData(None)
            if self.initialized:
                self.vtk.GetRenderWindow().Render()

    def set_heatmap(self, visible):
        self.heatmap_visible = bool(visible)
        if self.enabled:
            self.heatmap_actor.SetVisibility(visible and self.heatmap_data is not None)
            if self.initialized:
                self.vtk.GetRenderWindow().Render()

    def set_mode(self, surface, edges):
        self.surface, self.edges = surface, edges
        if not self.enabled:
            return
        prop = self.actor.GetProperty()
        self.actor.SetVisibility(surface or edges)
        if surface:
            prop.SetRepresentationToSurface()
            prop.SetEdgeVisibility(edges)
            prop.SetColor(.73, .84, .88)
        else:
            prop.SetRepresentationToWireframe()
            prop.EdgeVisibilityOff()
            prop.SetColor(.35, .85, .75)
        if self.initialized:
            self.vtk.GetRenderWindow().Render()

    def fit(self, view=None):
        if not self.enabled or self.data is None:
            return
        camera = self.renderer.GetActiveCamera()
        if view:
            direction, up = {'iso': ((1, -1, 1), (0, 0, 1)),
                             'top': ((0, 0, 1), (0, 1, 0)),
                             'front': ((0, -1, 0), (0, 0, 1))}[view]
            bounds = self.data.GetBounds()
            center = tuple((bounds[2*i] + bounds[2*i+1]) / 2 for i in range(3))
            camera.SetFocalPoint(*center)
            camera.SetPosition(*(center[i] + direction[i] for i in range(3)))
            camera.SetViewUp(*up)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.vtk.GetRenderWindow().Render()

    def shutdown(self):
        if self.vtk:
            self.vtk.Finalize()


def render_offscreen(data, filename, edges=False):
    """Real renderer smoke test and reproducible development screenshot."""
    from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkWindowToImageFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    renderer, mapper, actor = make_scene()
    mapper.SetInputData(data)
    actor.GetProperty().SetEdgeVisibility(edges)
    window = vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(800, 600)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(1, -1, 1)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()
    window.Render()
    image = vtkWindowToImageFilter()
    image.SetInput(window)
    image.Update()
    writer = vtkPNGWriter()
    writer.SetFileName(str(filename))
    writer.SetInputConnection(image.GetOutputPort())
    writer.Write()
    window.Finalize()
