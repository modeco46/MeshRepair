"""Source and PyInstaller entrypoint."""
import sys
from meshrepair.runtime import configure

configure()

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
        if '--console-state' in sys.argv:
            import ctypes
            import json
            from pathlib import Path
            window = int(ctypes.windll.kernel32.GetConsoleWindow()) if sys.platform == 'win32' else 0
            Path(sys.argv[sys.argv.index('--console-state')+1]).write_text(json.dumps({'console_window': window}))
        from meshrepair.gui import main
        raise SystemExit(main(smoke=True))
    else:
        from meshrepair.gui import main
        raise SystemExit(main())
