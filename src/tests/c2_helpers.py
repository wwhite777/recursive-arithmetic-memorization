import copy
import torch
from .helpers import tiny_config


def matched_tiny(k=1, total=40):
    config = tiny_config(16, total)
    config['schema'] = 'experiment'
    config['training'].update(physical_batch=8//k, effective_batch=8)
    return config


def close(a, b, path='root', atol=2e-11, rtol=2e-10):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        if a.is_floating_point():
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=lambda detail: path+'\n'+detail)
        else:
            assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            close(a[key], b[key], path+'.'+str(key), atol, rtol)
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, path+f'[{i}]', atol, rtol)
    elif isinstance(a, float):
        assert abs(a-b) <= atol+rtol*abs(a), (path, a, b)
    else:
        assert type(a) == type(b) and a == b, (path, a, b)


def compared_state(trainer):
    state = trainer.state_dict()
    state.pop('training_config')  # Only micro-batch/effective-batch declarations differ.
    return state


def max_error(a, b):
    if isinstance(a, torch.Tensor):
        return (a-b).abs().max().item() if a.is_floating_point() and a.numel() else 0.
    if isinstance(a, dict):
        return max((max_error(a[key], b[key]) for key in a), default=0.)
    if isinstance(a, (tuple, list)):
        return max((max_error(x, y) for x, y in zip(a, b)), default=0.)
    if isinstance(a, float):
        return abs(a-b)
    return 0.
