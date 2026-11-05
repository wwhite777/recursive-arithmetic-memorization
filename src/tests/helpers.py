import copy
import json
import os
from pathlib import Path
from qdriver.common import PROJECT, require
import torch

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CPU fixtures require CUDA hidden')


def tiny_config(steps=8, total=32):
    from qdriver.workload import config_for
    config = copy.deepcopy(config_for(1024, 8, 2048))
    config['model'].update(hidden_size=8, layers=1, heads=2, expansion=1.0, ff_multiple=8,
                           forward_dtype='float64', engineering_fixture=True)
    config['training'].update(total_updates=total, physical_batch=2, act_steps=steps, warmup_updates=8, engineering_fixture=True)
    return config


def rows():
    # Exact released sample rows make small CPU fixtures self-contained.
    train = json.loads((PROJECT/'records/scoring_samples/original_train_rows.json').read_text())
    with (PROJECT/'data/id_dev.jsonl').open() as stream:
        dev = [json.loads(next(stream)) for _ in range(8)]
    return [item['row'] for item in train[:8]], dev


def run_root():
    return Path(os.environ['FIXTURE_OUTPUT'])


def exact(a, b, path='root'):
    import torch
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), 'tensor mismatch at '+path)
    elif isinstance(a, dict):
        require(type(a) == type(b) and a.keys() == b.keys(), 'dict mismatch at '+path)
        for key in a:
            exact(a[key], b[key], path+'.'+str(key))
    elif isinstance(a, (tuple, list)):
        require(type(a) == type(b) and len(a) == len(b), 'sequence mismatch at '+path)
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path+f'[{i}]')
    else:
        require(type(a) == type(b) and a == b, 'value mismatch at '+path+': '+repr(a)+' / '+repr(b))
