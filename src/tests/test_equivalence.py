import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qdriver.common import read_json, sha
from .helpers import exact, rows, run_root, tiny_config


class EquivalenceTests(unittest.TestCase):
    def state(self, root):
        import torch
        ref = read_json(root/'LATEST.json')
        self.assertEqual(sha(root/ref['path']), ref['sha256'])
        return torch.load(root/ref['path'], map_location='cpu', weights_only=False)

    def test_fresh_process_split_exact(self):
        for steps, total in ((8, 32), (8, 48), (16, 64)):
            # Boundary splits; cross shuffled epochs after restoration.
            a = run_root()/f's{steps}-n{total}-straight'
            b = run_root()/f's{steps}-n{total}-split'
            for phase, output in (('straight', a), ('first', b), ('resume', b)):
                argv = [sys.executable, '-B', '-m', 'tests.split_executor', phase, '--output', str(output), '--steps', str(steps), '--total', str(total)]
                process = subprocess.run(argv, capture_output=True, text=True, env=os.environ)
                (run_root()/f's{steps}-n{total}-{phase}.log').write_text(process.stdout+process.stderr)
                self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            first, second = self.state(a), self.state(b)
            for key in ('trainer', 'gradients', 'sampler', 'evaluations', 'rng', 'model_training', 'logs'):
                exact(first[key], second[key], key)
            for key in ('updates', 'optimizer_steps', 'ema_steps', 'act_step', 'trajectory'):
                exact(first['clocks'][key], second['clocks'][key], key)
            # Manifest/config/log identities have no output-path dependence.
            exact(first['identities'], second['identities'], 'identities')
            for name in ('metrics', 'samples', 'evaluations'):
                self.assertEqual((a/(name+'.jsonl')).read_bytes(), (b/(name+'.jsonl')).read_bytes())
            print(f'EXACT: ACT{steps}, N={total}; model/optimizer/EMA/carry/masks/metrics/order/cursor/epoch/RNG')

    def test_restore_tail_and_identity_failures(self):
        from qdriver.engine import Engine
        train, id_rows = rows()
        path = run_root()/'tails'
        engine = Engine(tiny_config(), train, id_rows, path, 'b'*64)
        for _ in range(8):
            engine.step()
        ref = engine.checkpoint()
        saved_metrics = (path/'metrics.jsonl').read_bytes()
        engine.step()
        with (path/'metrics.jsonl').open('ab') as stream:
            stream.write(b'{torn')
        restored = Engine(tiny_config(), train, id_rows, path, 'b'*64)
        restored.resume()
        self.assertEqual((path/'metrics.jsonl').read_bytes(), saved_metrics)
        self.assertTrue(list((path/'discarded_tails').glob('metrics.*.jsonl')))
        changed = Engine(tiny_config(), train, id_rows, path, 'c'*64)
        with self.assertRaisesRegex(ValueError, 'identity'):
            changed.resume()
        p = path/ref['path']
        body = p.read_bytes()
        p.write_bytes(body+b'corrupt')
        with self.assertRaisesRegex(ValueError, 'digest'):
            restored.resume(ref)
        p.write_bytes(body)
        with (path/'metrics.jsonl').open('r+b') as stream:
            stream.write(b'!')
        with self.assertRaisesRegex(ValueError, 'log identity'):
            restored.resume(ref)

    def test_atomic_pointer_survives_interrupted_save(self):
        from qdriver.engine import Engine
        train, id_rows = rows()
        path = run_root()/'atomic'
        engine = Engine(tiny_config(), train, id_rows, path, 'd'*64)
        for _ in range(8):
            engine.step()
        first = engine.checkpoint()
        for _ in range(8):
            engine.step()
        import qdriver.engine as module
        original = module.save_json
        def fail_pointer(path, value, new=False):
            if Path(path).name == 'LATEST.json':
                raise OSError('synthetic interrupted pointer publication')
            return original(path, value, new)
        with patch.object(module, 'save_json', side_effect=fail_pointer):
            with self.assertRaises(OSError):
                engine.checkpoint()
        self.assertEqual(read_json(path/'LATEST.json'), first)
        restored = Engine(tiny_config(), train, id_rows, path, 'd'*64)
        restored.resume()
        self.assertEqual(restored.trainer.updates, 8)

    def test_experiment_evaluation_and_training_are_unchanged_for_act8(self):
        import torch
        from reasoning.trainer import PilotTrainer
        from qdriver.training import ArithmeticModel, PilotConfig, Trainer, TrainingConfig, tensors, evaluate, evaluator
        train, id_rows = rows()
        config = tiny_config()
        torch.manual_seed(101)
        model = ArithmeticModel(PilotConfig(**config['model'])).double()
        original = PilotTrainer(copy.deepcopy(model), TrainingConfig(**config['training']), 'S22')
        adapted = Trainer(copy.deepcopy(model), TrainingConfig(**config['training']), 'S22')
        batch = tensors(train[:2])
        for _ in range(8):
            exact(original.step(batch, collect_displacement=True), adapted.step(batch, collect_displacement=True))
        before = adapted.state_dict()
        actual = evaluate(adapted, id_rows, 'cpu')
        expected = evaluator().evaluate_split(original, id_rows, 512, 'cpu')
        actual.pop('scope'); expected.pop('scope')
        exact(actual, expected)
        exact(before, adapted.state_dict(), 'eval leaves training state unchanged')
        exact(original.state_dict(), adapted.state_dict(), 'experiment trainer state')
