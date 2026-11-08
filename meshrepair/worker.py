"""Line-delimited JSON protocol shared by desktop UI and CLI."""
import argparse
import json
import sys
import traceback
from .engine import Settings, run


def send(kind, value):
    print(json.dumps({"type": kind, "value": value}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="MeshRepair 0.1 — native mesh repair prototype")
    parser.add_argument("source")
    parser.add_argument("--output")
    parser.add_argument("--tolerance", type=float, default=.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-grid-cells", type=int, default=100_000_000)
    parser.add_argument("--allow-components", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.source, args.output,
                     Settings(args.tolerance, args.device, args.max_grid_cells, not args.allow_components),
                     lambda text: send("progress", text))
        send("result", result)
        return 0
    except Exception as exc:
        send("error", str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
