"""Incremental UTF-8 JSON-lines tail, including split writes and final draining."""
from pathlib import Path


class ProgressTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b''

    def read(self):
        try:
            with self.path.open('rb') as stream:
                if self.path.stat().st_size < self.offset:
                    self.offset, self.pending = 0, b''
                stream.seek(self.offset)
                chunk = stream.read(1024*1024)
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        self.pending += chunk
        lines = self.pending.split(b'\n')
        self.pending = lines.pop()
        return [line.decode('utf-8', errors='replace').rstrip() for line in lines]
