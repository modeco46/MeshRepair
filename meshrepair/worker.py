"""UTF-8 JSON-lines; optional atomic result file avoids native stdout interleaving."""
import argparse
import json
import sys
import traceback
from .engine import Settings, run, atomic_json, clean_json


def send(kind, value):
    print(json.dumps({'type': kind, 'value': clean_json(value)}, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='MeshRepair 0.8.1 — ремонт сеток')
    parser.add_argument('source')
    parser.add_argument('--output')
    parser.add_argument('--quality', type=float, default=.02)
    parser.add_argument('--collapse-coplanar', action='store_true')
    parser.add_argument('--coplanar-angle', type=float, default=1.)
    parser.add_argument('--simplify-curved', action='store_true')
    parser.add_argument('--strict-tolerance', action='store_true')
    parser.add_argument('--weld-close-vertices', action='store_true')
    parser.add_argument('--no-restore-sharp-edges', action='store_true')
    parser.add_argument('--result-file')
    parser.add_argument('--work-dir')
    parser.add_argument('--debug', action='store_true')
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument('--check', action='store_true')
    operations.add_argument('--preview-output')
    args = parser.parse_args()
    def terminal(kind, value):
        if args.result_file:
            atomic_json(args.result_file, {'type': kind, 'value': value})
            send('result_file', args.result_file)
        else:
            send(kind, value)
    try:
        log, progress = lambda text: send('log', text), lambda event: send('progress', event)
        if args.check:
            from .inspection import inspect_mesh
            result = inspect_mesh(args.source, log, progress)
        elif args.preview_output:
            from .inspection import export_preview
            result = export_preview(args.source, args.preview_output, progress)
        else:
            settings = Settings(args.quality, args.collapse_coplanar, args.coplanar_angle,
                                args.simplify_curved, args.strict_tolerance, args.weld_close_vertices, not args.no_restore_sharp_edges)
            result = run(args.source, args.output, settings, log, progress, work_dir=args.work_dir)
        terminal('result', result)
        return 0
    except Exception as exc:
        message = 'Недостаточно памяти для выбранного качества.' if isinstance(exc, MemoryError) else str(exc)
        terminal('error', message)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
