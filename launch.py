"""Source and PyInstaller entrypoint."""
import sys

if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        from meshrepair.worker import main
        raise SystemExit(main())
    if "--smoke-ui" in sys.argv:
        import tkinter as tk
        from meshrepair.gui import App
        root = tk.Tk()
        app = App(root)
        root.update()
        app.close()
    else:
        from meshrepair.gui import main
        main()
