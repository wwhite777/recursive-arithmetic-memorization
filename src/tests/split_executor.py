"""CPU-only training phases run in distinct interpreter processes."""
import argparse
from pathlib import Path
from qdriver.common import setup_cpu, verify_inputs


def main():
    torch = setup_cpu()
    verify_inputs()
    from qdriver.engine import Engine
    from .helpers import tiny_config, rows
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('straight', 'first', 'resume'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--total', type=int, required=True)
    args = parser.parse_args()
    train, id_rows = rows()
    config = tiny_config(args.steps, args.total)
    engine = Engine(config, train, id_rows, Path(args.output), 'a'*64)
    if args.phase == 'resume':
        engine.resume()
    end = args.total//2 if args.phase == 'first' else args.total
    while engine.trainer.updates < end:
        engine.step()
        if engine.trainer.updates % 4 == 0:
            engine.evaluate()
    engine.checkpoint()
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    print(f'PASS {args.phase}: updates={engine.trainer.updates}, steps={args.steps}, epoch={engine.sampler.epoch}, cursor={engine.sampler.cursor}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
