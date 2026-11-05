"""CPU-only verifier; run with the same Torch version as calibration/main."""
import argparse
import json
import os
import resource
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='curve/ or calibration probe directory')
    parser.add_argument('--updates', type=int, help='explicit partial record length (default: complete run)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from qdriver.fresh_record import verify_record
    try:
        result = verify_record(args.root, args.updates)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print('FAIL: '+str(error), file=sys.stderr)
        return 1
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status': 'PASS', **result}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
