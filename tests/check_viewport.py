"""Exercise the real Qt/VTK bridge with an X11 GLX context (CI uses Xvfb/Mesa)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtWidgets import QApplication
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
from vtkmodules.util.numpy_support import vtk_to_numpy
from meshrepair.gui import App
from meshrepair.viewer import polydata_from_arrays


def pixels(window):
    image = vtkWindowToImageFilter()
    image.SetInput(window)
    image.Update()
    return vtk_to_numpy(image.GetOutput().GetPointData().GetScalars()).copy()


if __name__ == '__main__':
    app = QApplication([])
    app.setStyle('Fusion')
    ui = App(persist=False)
    ui.show()
    app.processEvents()
    try:
        mesh = mm.makeTorus(10, 2, 64, 32)
        data = polydata_from_arrays(mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology))
        ui.view.set_data(data)
        app.processEvents()
        window = ui.view.vtk.GetRenderWindow()
        assert window.SupportsOpenGL()
        solid = pixels(window)
        assert solid.max() > 100
        ui.edge_button.click()
        app.processEvents()
        edges = pixels(window)
        assert np.count_nonzero(solid != edges) > 1000
        ui.surface_button.click()
        app.processEvents()
        wire = pixels(window)
        assert np.count_nonzero(wire != edges) > 1000
        ui.view.fit('top')
        app.processEvents()
        assert np.count_nonzero(pixels(window) != wire) > 1000
        ui.view.clear()
        ui.view.set_data(data)
        app.processEvents()
        print('Embedded Qt/VTK viewport: solid, edges, wireframe, camera and reload passed.')
        print(ui.render_label.toolTip())
    finally:
        ui.close()
        app.processEvents()
