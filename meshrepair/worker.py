"""UTF-8 JSON-lines; optional atomic result file avoids native stdout interleaving."""
import argparse
import json
import sys
import traceback
from .engine import Settings, run, atomic_json, clean_json, SharpeningRetryRequired


def send(kind, value, progress_stream=None):
    line = json.dumps({'type': kind, 'value': clean_json(value)}, ensure_ascii=False, allow_nan=False)
    if progress_stream is not None:
        progress_stream.write(line+'\n')
        progress_stream.flush()
    if sys.stdout is not None and not getattr(sys, '_meshrepair_redirected_stdout', False):
        print(line, flush=True)


def main():
    if sys.stdout is None or sys.stderr is None:
        from .runtime import configure
        configure(worker=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='MeshRepair 0.9.1 — ремонт сеток')
    parser.add_argument('source')
    parser.add_argument('--output')
    parser.add_argument('--quality', type=float, default=.02)
    parser.add_argument('--collapse-coplanar', action='store_true')
    parser.add_argument('--coplanar-angle', type=float, default=1.)
    parser.add_argument('--simplify-curved', action='store_true')
    parser.add_argument('--strict-tolerance', action='store_true')
    parser.add_argument('--weld-close-vertices', action='store_true')
    parser.add_argument('--no-restore-sharp-edges', action='store_true')
    parser.add_argument('--remove-small-shells', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--temp-directory', default='')
    parser.add_argument('--local-fix-max-faces', type=int, default=200)
    parser.add_argument('--local-fix-fraction', type=float, default=.001)
    parser.add_argument('--result-file')
    parser.add_argument('--progress-file')
    parser.add_argument('--work-dir')
    parser.add_argument('--debug', action='store_true')
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument('--check', action='store_true')
    operations.add_argument('--preview-output')
    args = parser.parse_args()
    channel = open(args.progress_file, 'a', encoding='utf-8') if args.progress_file else None
    def event(kind, value):
        send(kind, value, channel)
    def terminal(kind, value):
        if args.result_file:
            atomic_json(args.result_file, {'type': kind, 'value': value})
            event('result_file', args.result_file)
        else:
            event(kind, value)
    try:
        log, progress = lambda text: event('log', text), lambda value: event('progress', value)
        if args.check:
            from .inspection import inspect_mesh
            result = inspect_mesh(args.source, log, progress)
        elif args.preview_output:
            from .inspection import export_preview
            result = export_preview(args.source, args.preview_output, progress)
        else:
            settings = Settings(args.quality, args.collapse_coplanar, args.coplanar_angle,
                                args.simplify_curved, args.strict_tolerance, args.weld_close_vertices, not args.no_restore_sharp_edges,
                                args.local_fix_max_faces, args.local_fix_fraction, args.remove_small_shells,
                                args.no_cuda, args.temp_directory)
            result = run(args.source, args.output, settings, log, progress, work_dir=args.work_dir)
            if not args.output and not args.work_dir:
                # Standalone worker artifacts remain usable by its caller. GUI
                # supplies its own job and owns cleanup; orphan cleanup handles CLI.
                from .storage import release_job
                release_job(result['work_directory'])
        terminal('result', result)
        return 0
    except SharpeningRetryRequired as exc:
        terminal('retry_without_sharpening', {'message': str(exc), 'report': exc.report})
        return 1
    except Exception as exc:
        message = 'Недостаточно памяти для выбранного качества.' if isinstance(exc, MemoryError) else str(exc)
        terminal('error', message)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if channel is not None:
            channel.close()


if __name__ == '__main__':
    raise SystemExit(main())
