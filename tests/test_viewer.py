import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.viewer import polydata_from_arrays


def test_vtk_receives_exact_triangle_buffer():
    mesh = mm.makeCube()
    data = polydata_from_arrays(mn.getNumpyVerts(mesh), mn.getNumpyFaces(mesh.topology))
    assert data.GetNumberOfPoints() == 8
    assert data.GetNumberOfCells() == 12
    assert data.GetBounds() == (-.5, .5, -.5, .5, -.5, .5)


def test_invalid_preview_indices_are_rejected():
    with pytest.raises(ValueError):
        polydata_from_arrays(np.zeros((3, 3)), np.array([[0, 1, 6]]))


@pytest.mark.skipif(sys.platform == 'win32', reason='CI Windows runner has no OpenGL GPU; real rendering is tested with Linux EGL')
def test_real_opengl_renderer_and_wireframe(tmp_path):
    script = '''
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from meshlib import mrmeshpy as mm, mrmeshnumpy as mn
from meshrepair.viewer import polydata_from_arrays,render_offscreen
m=mm.makeTorus()
d=polydata_from_arrays(mn.getNumpyVerts(m),mn.getNumpyFaces(m.topology))
root=Path(sys.argv[1])
render_offscreen(d,root/'solid.png',False)
render_offscreen(d,root/'edges.png',True)
a=np.array(Image.open(root/'solid.png'));b=np.array(Image.open(root/'edges.png'))
assert a.shape[:2] == (600,800)
assert np.count_nonzero(a != b) > 1000
assert a[300,200].max() > a[0,0].max()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True,
                            env=dict(os.environ, QT_QPA_PLATFORM='offscreen'), timeout=60)
    assert result.returncode == 0, result.stderr
