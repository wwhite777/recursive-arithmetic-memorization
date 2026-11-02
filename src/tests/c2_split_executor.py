"""Fresh-process C2 resume fixture, with CPU data and real accumulated updates."""
import argparse
from pathlib import Path
from qdriver.common import setup_cpu, verify_inputs


def main():
    torch = setup_cpu()
    verify_inputs()
    from qdriver.engine import Engine
    from .c2_helpers import matched_tiny
    from .helpers import rows
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('straight', 'first', 'second', 'resume'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--k', required=True, type=int)
    args = parser.parse_args()
    train, id_rows = rows()
    engine = Engine(matched_tiny(args.k, 64), train, id_rows, Path(args.output), 'a'*64)
    if args.phase in ('second', 'resume'):
        engine.resume()
    end = 16 if args.phase == 'first' else 32 if args.phase == 'second' else 64
    while engine.trainer.updates < end:
        engine.step()
        if engine.trainer.updates % 4 == 0:
            engine.evaluate()
    engine.checkpoint()
    assert not torch.cuda.is_initialized()
    print(f'PASS {args.phase}, K={args.k}, updates={end}, epoch={engine.sampler.epoch}')


if __name__ == '__main__':
    main()
