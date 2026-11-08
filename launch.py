"""Source and PyInstaller entrypoint."""
import sys

if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        from meshrepair.worker import main
        raise SystemExit(main())
    from meshrepair.gui import main
    main()
