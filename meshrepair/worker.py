"""Line-delimited UTF-8 JSON protocol for the disposable native worker."""
import argparse
import json
import sys
import traceback
from .engine import Settings, run


def send(kind, value):
    print(json.dumps({"type": kind, "value": value}, ensure_ascii=False), flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='MeshRepair 0.5 — automatic repair')
    parser.add_argument('source')
    parser.add_argument('--output')
    parser.add_argument('--quality', type=float, default=.02)
    parser.add_argument('--collapse-coplanar', action='store_true')
    parser.add_argument('--coplanar-angle', type=float, default=1.0)
    parser.add_argument('--simplify-curved', action='store_true')
    parser.add_argument('--debug', action='store_true')
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument('--check', action='store_true')
    operations.add_argument('--preview-output')
    args = parser.parse_args()
    try:
        if args.check:
            from .inspection import inspect_mesh
            send('result', inspect_mesh(args.source, lambda text: send('log', text), lambda event: send('progress', event)))
            return 0
        if args.preview_output:
            from .inspection import export_preview
            send('result', export_preview(args.source, args.preview_output, lambda event: send('progress', event)))
            return 0
        result = run(args.source, args.output, Settings(args.quality, args.collapse_coplanar, args.coplanar_angle, args.simplify_curved),
                     lambda text: send('log', text), lambda event: send('progress', event))
        send('result', result)
        return 0
    except MemoryError:
        send('error', 'Недостаточно памяти для выбранного качества. Результат не создан.')
        return 1
    except Exception as exc:
        send('error', str(exc))
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
