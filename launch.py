"""Source and PyInstaller entrypoint."""
import sys

if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        from meshrepair.worker import main
        raise SystemExit(main())
    if "--smoke-preview" in sys.argv:
        import numpy as np
        from meshrepair.viewer import polydata_from_arrays, make_scene
        with np.load(sys.argv[-1], allow_pickle=False) as data:
            mesh = polydata_from_arrays(data['vertices'], data['faces'])
        renderer, mapper, actor = make_scene()
        mapper.SetInputData(mesh)
        assert mapper.GetInput().GetNumberOfCells() > 0
        from PySide6.QtWidgets import QApplication
        from meshrepair.viewer import MeshView
        app = QApplication(sys.argv[:1])
        view = MeshView(enabled=True)
        view.shutdown()  # Load the actual Qt/VTK bridge without requiring CI GPU hardware.
    elif "--smoke-ui" in sys.argv:
        from meshrepair.gui import main
        raise SystemExit(main(smoke=True))
    else:
        from meshrepair.gui import main
        raise SystemExit(main())
