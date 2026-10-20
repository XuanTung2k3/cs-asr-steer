"""Atomic durable state and immutable artifact seals; no ML imports."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        d = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(d)
        finally: os.close(d)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def event(root, kind, **kw):
    with (Path(root)/'events.jsonl').open('a') as f:
        f.write(json.dumps({'utc': now(), 'event': kind, **kw}, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


@contextlib.contextmanager
def lock(path):
    with Path(path).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def seal(root, paths):
    root = Path(root)
    return {str(Path(p).relative_to(root)): sha(p) for p in paths}


def verify(root, hashes):
    return bool(hashes) and all((Path(root)/p).is_file() and sha(Path(root)/p) == h for p,h in hashes.items())
