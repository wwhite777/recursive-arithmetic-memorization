"""Atomic owned outputs and identity checks; no torch or hardware import."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parents[1]
PROJECT = WORKSPACE
PYTHON = sys.executable
FROZEN = PROJECT/'configs/base.json'
INTERVAL = 2048
CAP_SECONDS = 24 * 3600
PLAN_SECONDS = 21 * 3600
MAX_SEGMENTS = 10
POLL_SECONDS = 600
FOREIGN_GRACE_SECONDS = 60
FOREIGN_MEMORY_LIMIT_BYTES = 2_147_483_648


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path, limit=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining = limit
        while remaining is None or remaining > 0:
            data = stream.read(1024*1024 if remaining is None else min(1024*1024, remaining))
            if not data:
                break
            h.update(data)
            if remaining is not None:
                remaining -= len(data)
        require(remaining in (None, 0), 'short file: '+str(path))
    return h.hexdigest()


def owned(path):
    raw = Path(os.path.abspath(path))
    require(raw.is_relative_to(WORKSPACE) and raw != WORKSPACE, 'output outside owned directory')
    current = WORKSPACE
    for part in raw.relative_to(WORKSPACE).parts:
        current /= part
        require(not current.is_symlink(), 'symlink output forbidden')
    return raw


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data, new=False):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if new:
        try:
            os.link(temp, path)
        finally:
            temp.unlink()
    else:
        os.replace(temp, path)
    fsync_dir(path.parent)


def save_json(path, value, new=False):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()+b'\n', new)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def verify_inputs():
    inventory = read_json(WORKSPACE/'data/datasets.json')
    for entry in inventory:
        path = WORKSPACE/entry['path']
        if path.exists():
            require(sha(path) == entry['sha256'], 'input changed: '+entry['path'])
    return inventory


def code_identity():
    return {str(p.relative_to(WORKSPACE)): sha(p) for p in sorted(PACKAGE.glob('*.py'))}


def identity(config, manifest_sha):
    return {'config_sha256': object_sha(config), 'manifest_sha256': manifest_sha,
            'inputs_sha256': sha(WORKSPACE/'data/datasets.json'), 'source_sha256': object_sha(code_identity())}


def setup_cpu():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU test requires CUDA_VISIBLE_DEVICES empty')
    require(sys.executable == PYTHON, 'designated interpreter required')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CUDA must be unavailable and uninitialized')
    return torch
