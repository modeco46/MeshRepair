"""Line-delimited JSON protocol shared by desktop UI and CLI."""
import argparse
import json
import sys
import traceback
from .engine import Settings, run


def send(kind, value):
    print(json.dumps({"type": kind, "value": value}, ensure_ascii=False), flush=True)


def main():
    # Stable protocol even in frozen Windows builds with a legacy console locale.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="MeshRepair 0.2 — native mesh repair prototype")
    parser.add_argument("source")
    parser.add_argument("--output")
    parser.add_argument("--tolerance", type=float, default=.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-grid-cells", type=int, default=100_000_000)
    parser.add_argument("--allow-components", action="store_true")
    parser.add_argument("--method", choices=["surface", "union", "voxel"], default="surface")
    parser.add_argument("--max-hole-perimeter", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.source, args.output,
                     Settings(args.tolerance, args.device, args.max_grid_cells, not args.allow_components, args.method, args.max_hole_perimeter),
                     lambda text: send("progress", text))
        send("result", result)
        return 0
    except Exception as exc:
        send("error", str(exc))
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
