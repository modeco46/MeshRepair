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


class PreviewLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)

    def run(self):
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                vertices, faces = archive['vertices'], archive['faces']
            if self.isInterruptionRequested():
                self.failed.emit('Загрузка предпросмотра остановлена.')
                return
            data = polydata_from_arrays(vertices, faces)
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
        if not self.enabled:
            return
        self.initialize()
        self.mapper.SetInputData(data)
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
        if self.enabled:
            self.mapper.SetInputData(None)
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
